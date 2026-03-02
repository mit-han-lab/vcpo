import asyncio
import json
import os
import requests

import hydra
import numpy as np
import ray

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from recipe.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager
from verl.utils import hf_tokenizer

from recipe.fully_async_policy.detach_utils import (
    RolloutSample,
)

SYSTEM_PROMPT = """
You are a helpful assistant.

Solve the following problem step by step. You now have the ability to
selectively write executable Python code to enhance your reasoning
process. The Python code will be executed by an external sandbox,
and the output (after “Code execution result: ”) is returned to aid your
reasoning and help you arrive at the final answer. The Python code
should be complete scripts, including necessary imports.

Code Format:
Each code snippet is wrapped between ```python and ```. You need to use print()
to output intermediate results. For example

```python
print(10 * 3)
```

will return 30. Do not write anything else after the tool call. Only your final tool call with be executed.
"""

USER_PROMPT = """
Alice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either $1$ token or $4$ tokens from the stack. Whoever removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play. Let's think step by step and output the final answer within \\boxed{}.
"""


def _decode_valid_tokens(tokenizer, token_ids, token_mask) -> str:
    valid_ids = [int(token_id) for token_id, mask in zip(token_ids.tolist(), token_mask.tolist(), strict=False) if int(mask) == 1]
    if not valid_ids:
        return ""
    return tokenizer.decode(valid_ids, skip_special_tokens=True)


def _print_final_trajectories(batch: DataProto, tokenizer, max_print: int = 3):
    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]

    prompt_len = prompts.size(1)
    num_trajectories = responses.size(0)
    print_count = min(max_print, num_trajectories)

    print(f"\n========== Final trajectories ({print_count}/{num_trajectories}) ==========")
    for i in range(print_count):
        prompt_text = _decode_valid_tokens(tokenizer, prompts[i], attention_mask[i, :prompt_len])
        response_text = _decode_valid_tokens(tokenizer, responses[i], attention_mask[i, prompt_len:])

        print(f"\n----- Trajectory {i} -----")
        print("[prompt]")
        print(prompt_text)
        print("\n[response]")
        print(response_text)

        if "tool_extra_fields" in batch.non_tensor_batch:
            print("\n[tool_extra_fields]")
            print(batch.non_tensor_batch["tool_extra_fields"][i])


def _save_final_trajectories(batch: DataProto, tokenizer, output_dir: str, step: int, max_save: int = 3):
    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    attention_mask = batch.batch["attention_mask"]

    prompt_len = prompts.size(1)
    num_trajectories = responses.size(0)
    save_count = min(max_save, num_trajectories)

    os.makedirs(output_dir, exist_ok=True)
    for i in range(save_count):
        prompt_text = _decode_valid_tokens(tokenizer, prompts[i], attention_mask[i, :prompt_len])
        response_text = _decode_valid_tokens(tokenizer, responses[i], attention_mask[i, prompt_len:])

        tool_extra_fields = ""
        if "tool_extra_fields" in batch.non_tensor_batch:
            tool_extra_fields = str(batch.non_tensor_batch["tool_extra_fields"][i])

        log_text = (
            f"step={step}\ntrajectory_index={i}\n\n"
            f"[prompt]\n{prompt_text}\n\n"
            f"[response]\n{response_text}\n\n"
            f"[tool_extra_fields]\n{tool_extra_fields}\n"
        )
        file_path = os.path.join(output_dir, f"step_{step:06d}_traj_{i:02d}.txt")
        with open(file_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(log_text)


def query_model(model_name: str, url: str = "http://127.0.0.1:8001/v1/chat/completions"):
    payload = {
        "model": model_name,
        "messages": [
            {
            "role": "system",
            "content": "You are a helpful assistant."
            },
            {
            "role": "user",
            "content": "PLACEHOLDER"
            }
        ],
        "temperature": 0.8
    }

    req = requests.post(
        url = url,
        headers = {"Content-Type" : "application/json"},
        json = payload,
        timeout = 480,
    )

    req.raise_for_status()
    print(req.text)

    print(json.dumps(req.json(), indent=2, sort_keys=True))



async def _run_test(config):
    print("=======================================================")
    print("========== Initializing Model + Tokenizer... ==========")
    print("=======================================================")

    # Tokenizer
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )
    tokenizer = hf_tokenizer(local_path)
    
    strategy = config.actor_rollout_ref.actor.strategy
    if strategy == "megatron":
        from recipe.fully_async_policy.megatron_worker import DetachAsyncRolloutWorker
    else:
        from recipe.fully_async_policy.fsdp_workers import DetachAsyncRolloutWorker

    # Resource pool manager
    role_worker_mapping = {
        Role.Rollout: ray.remote(DetachAsyncRolloutWorker),
    }

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            "rollout_pool": [config.rollout.n_gpus_per_node] * config.rollout.nnodes
        },
        mapping={
            Role.Rollout: "rollout_pool"
        },
    )

    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    rollout_resource_pool = resource_pool_manager.get_resource_pool(Role.Rollout)
    role_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.Rollout],
        config=config.actor_rollout_ref,
        role=str(Role.Rollout),
    )

    resource_pool_to_cls[rollout_resource_pool][str(Role.Rollout)] = role_cls

    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
        all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

    # Initialize Rollout WG
    rollout_wg = all_wg[str(Role.Rollout)]
    rollout_wg.init_model()

    # Async Rollout Manager
    async_rollout_manager = await FullyAsyncAgentLoopManager.create(
        config=config,
        worker_group=rollout_wg,
    )

    print("=================================")
    print("========== Sampling... ==========")
    print("=================================")

    traj_output_dir = os.environ.get(
        "VERL_DEBUG_MULTITURN", "/home/lukh23/vcpo-dev-v2/recipe/fully_async_policy/code_sandbox/traj"
    )

    step = 0 
    while (True):

        print(f"=================================")
        print(f"============ Step {step} ============")
        print(f"=================================")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": USER_PROMPT.strip()},
        ]
        
        agent_name = "async_partial_tool_agent" if config.actor_rollout_ref.rollout.multi_turn.enable else "partial_single_turn_agent"
        sample = DataProto(
            non_tensor_batch={
                "raw_prompt": np.array([np.asarray(messages)], dtype=object),
                "agent_name": np.array([agent_name], dtype=object),
                "data_source": np.array(["deepscaler"], dtype=object),
                "reward_model": np.array([{"style": "rule", "ground_truth": ""}], dtype=object),
            }
        )

        rollout_n = config.actor_rollout_ref.rollout.n
        rollout_sample = RolloutSample(
            full_batch=sample.repeat(rollout_n),
            agent_loop_output_list=[None] * rollout_n,
            sample_id="debug_sample",
            epoch=0,
            processing_times=[],
            tool_calls=[],
            param_version=step,
            param_version_start=[],
            param_version_end=[],
            rollout_status={},
        )

        ret, is_cancel = await async_rollout_manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        if is_cancel:
            print("[debug] sample was cancelled; waiting for resumed rollout")
        else:
            _print_final_trajectories(ret, tokenizer, max_print=min(3, rollout_n))
            _save_final_trajectories(ret, tokenizer, output_dir=traj_output_dir, step=step, max_save=min(3, rollout_n))

        step += 1



@hydra.main(config_path="config", config_name="fully_async_ppo_trainer", version_base=None)
def main(config):
    ray.init(ignore_reinit_error=True)
    try:
        asyncio.run(_run_test(config))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
