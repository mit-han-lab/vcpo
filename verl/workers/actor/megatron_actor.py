# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Megatron Actor.
In megatron actor, the differences are:
1. We only make minibatch

Note that our model doesn't have to be `MegatronModule` because we don't share embedding in the last layer
"""

import itertools
import logging
import os
from contextlib import ExitStack
from functools import partial
from typing import Dict, Iterable, Tuple
from dataclasses import asdict

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32

# from megatron.core.optimizer import DistributedOptimizer
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_model_config
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.profiler.profile import Profiler
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor import BasePPOActor
from recipe.fully_async_policy.staleness_utils import (
    TrajRecord,
    compute_ess_info,
    compute_grad_info,
    compute_is_info,
    compute_opob_baseline,
    compute_staleness_statistics,
)
from verl.workers.utils.vcpo import (
    _get_local_model_grads_for_norm,
    accumulate_grad_buffers,
    allocate_grad_accum_buffers,
    copy_accum_buffers_to_grad_buffers,
    disable_dp_sync,
    move_grad_buffers,
    restore_dp_sync,
    zero_grad_accum_buffers,
)

__all__ = ["MegatronPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MegatronPPOActor(BasePPOActor):
    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
    ):
        """MeagtronPPOActor class. This class implements the simple PPO logics when the model is built with Megatron.

        Args:
            config (OmegaConf): the basic config that contains the hyper-parameters of PPO Actor. It must contain

                ``ppo_micro_batch_size_per_gpu``: micro batch size when updating ppo.

                ``ppo_mini_batch_size``: minibatch size when updating ppo using the batch data.

                ``ppo_epochs``: number of epochs to update the actor using the batch data.

                ``shuffle``: whether to shuffle the data after each ppo epoch.

                ``clip_ratio``: clip ratio of the ppo algorithm. See https://arxiv.org/abs/1707.06347.

                ``entropy_coeff``: entropy coefficient of the PPO loss. See https://arxiv.org/abs/1707.06347.
            model_config (OmegaConf): model configuration. It must contains ``model_config.vocab_size`` and
                ``model_config.hidden_size``
            hf_config (PretrainedConfig): huggingface config
            tf_config (TransformerConfig): mcore transformer config
            actor_module (nn.ModuleList): actor module is a ModuleList that contains a list of nn.Module in this
                pp stage.
                each nn.Module in this rank holds a vpp module chunk. See https://arxiv.org/pdf/2104.04473.pdf for
                more details.
                The actor module has some constraints to follow in order to use the updating logics implemented here

                1. It must implement unpad_input before any computation and pad_input after all the computation.
                Remove padding is an
                optimization that removes the padding tokens. See unpad_input and pad_input function in flash-attn
                (https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py).

                2. Each pp stage must return the hidden state with the same shape [total_nnz, 1, hidden_size],
                where total_nnz is the number of valid tokens in this batch. If sequence parallel is enabled, the size
                of the hidden state is [total_nnz // tp, 1, hidden_size].
            actor_optimizer (DistributedOptimizer): currently, we only support DistributedOptimizer in Megatron.
                It implements
                zero1 optimizer that shards the optimizer state across dp ranks.

        >>> from megatron.training import get_model
        >>> from megatron.optimizer import get_megatron_optimizer
        >>> actor_module = get_model(megatron_actor_model_provider, wrap_with_ddp=True)
        >>> actor_module = nn.ModuleList(actor_module)
        >>> actor_optimizer = get_megatron_optimizer(actor_module)
        >>> actor = MegatronPPOActor(config=config,
        >>>                          model_config=actor_model_config,
        >>>                          hf_config=hf_config,
        >>>                          tf_config=tf_config,
        >>>                          actor_module=actor_module,
        >>>                          actor_optimizer=actor_optimizer)
        """
        super().__init__(config)
        self._validate_config(config)
        self.model_config = model_config
        self.hf_config = hf_config
        self.tf_config = tf_config
        self.actor_module = actor_module
        self.actor_optimizer: DistributedOptimizer = actor_optimizer
        self.use_distributed_opt = bool(self.actor_module[0].ddp_config.use_distributed_optimizer)
        self.use_torch_profiler = self.config.profiler.get("tool") == "torch"
        if self.use_torch_profiler:
            self.prof = Profiler(
                self.config.profiler, tool_config=self.config.profiler.get("tool_config", {}).get("torch", {})
            )
        else:
            self.prof = None
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if self.use_fused_kernels and not getattr(self.config, "overlap_moe_expert_parallel_comm", False):
            # do not patch if overlap_moe_expert_parallel_comm is enabled
            from verl.models.mcore.model_forward_fused import patch_fused_forward

            for model in self.actor_module:
                patch_fused_forward(model)

        self.optimizer_step_args = OmegaConf.create(
            {
                "skip_grad": None,
                "overlap_dp_param_comm": False,
                "overlap_dp_grad_comm": False,
                "gradient_accumulation_steps": 1,
                "sequence_parallel": self.tf_config.sequence_parallel,
                "DDP_impl": "local",
                "layernorm_allreduce_bucket_threshold": 0,
                "reduce_grads_use_alltoall": False,
            }
        )

        config = get_model_config(self.actor_module[0])
        print(config)
        config.finalize_model_grads_func = finalize_model_grads

    def _validate_config(self, config) -> None:
        """Validate config options not implemented for Megatron backend"""
        assert config.get("ulysses_sequence_parallel_size", 1) == 1
        if config.get("shuffle", False):
            assert config.data_loader_seed is not None, "If shuffle dataloader, seed must be manually set"
        if config.megatron.tensor_model_parallel_size == 1:
            print("[Warining] Because actor tp size == 1, set sp to False")
            config.megatron.sequence_parallel = False
        self.config = config

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            DataProto: torch.Tensor: the log_prob tensor
        """
        use_dynamic_bsz = data.meta_info.get("use_dynamic_bsz", False)
        micro_batch_size = data.meta_info.get("micro_batch_size", None)
        max_token_len = data.meta_info.get("max_token_len", None)
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            max_token_len = max_token_len * self.config.megatron.context_parallel_size
        else:
            assert micro_batch_size is not None, (
                "micro batch size is needed for forward compute when use_dynamic_bsz is False"
            )

        def compute_logprobs_fn(output, data, use_dynamic_bsz=False, indices=None):
            response = data["responses"]
            response_length = response.size(1)
            log_probs = output["log_probs"][:, -response_length - 1 : -1].contiguous()
            return {"log_probs": log_probs}

        # We make recompute_old_log_prob by default here.
        # TODO (zhangchi.usc1992): actually, this function should only return log_prob and this logic should be
        # handled by user outside
        recompute_old_log_prob = self.config.get("recompute_old_log_prob", True)

        entropys = torch.Tensor()
        if recompute_old_log_prob:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            input_ids = batch["input_ids"]
            batch_size = input_ids.size(0)
            response = batch["responses"]
            response_length = response.size(1)
            with torch.no_grad():
                output = self.forward_backward_batch(
                    data,
                    forward_only=True,
                    post_process_fn=compute_logprobs_fn,
                    calculate_entropy=calculate_entropy,
                    use_dynamic_bsz=use_dynamic_bsz,
                    micro_batch_size=micro_batch_size,
                    max_token_len=max_token_len,
                )
                if mpu.is_pipeline_last_stage(ignore_virtual=True):
                    # only on last rank. It should be on every tp rank
                    if calculate_entropy:
                        log_probs = [o[0]["log_probs"] for o in output["output"]]  # (bs, seq_size)
                    else:
                        log_probs = [o["log_probs"] for o in output["output"]]  # (bs, seq_size)
                    log_probs = torch.cat(log_probs, dim=0).to(torch.float32)
                    if use_dynamic_bsz:
                        indices = output["indices"]
                        indices = list(itertools.chain.from_iterable(indices))
                        assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
                        revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                        log_probs = log_probs[revert_indices]
                else:
                    log_probs = torch.empty(
                        size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                    )
                log_probs = log_probs.to(get_device_id())
                # broadcast across pp ranks
                torch.distributed.broadcast(
                    tensor=log_probs,
                    src=mpu.get_pipeline_model_parallel_last_rank(),
                    group=mpu.get_pipeline_model_parallel_group(),
                    async_op=False,
                )
                log_probs = log_probs.to("cpu")
                if calculate_entropy:
                    # Note that o[0] is metrics, o[1] is entropy
                    if mpu.is_pipeline_last_stage(ignore_virtual=True):
                        entropys = torch.cat([o[1] for o in output["output"]], dim=0)
                        entropys = entropys.to(torch.float32)
                        if use_dynamic_bsz:
                            indices = output["indices"]
                            indices = list(itertools.chain.from_iterable(indices))
                            assert len(indices) == entropys.size(0), f"{len(indices)} vs. {entropys.size()}"
                            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                            entropys = entropys[revert_indices]
                    else:
                        entropys = torch.empty(
                            size=(batch_size, response_length), dtype=torch.float32, device=input_ids.device
                        )
                    # broadcast across pp ranks
                    entropys = entropys.to(get_device_id())
                    torch.distributed.broadcast(
                        tensor=entropys,
                        src=mpu.get_pipeline_model_parallel_last_rank(),
                        group=mpu.get_pipeline_model_parallel_group(),
                        async_op=False,
                    )
                    entropys = entropys.to("cpu")

        # add empty cache after each compute
        get_torch_device().empty_cache()

        return log_probs, entropys

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
                ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
                responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
                of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of
                responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "response_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        skip_recompute_old_log_prob = bool(data.meta_info.get("skip_recompute_old_log_prob", False))
        if skip_recompute_old_log_prob:
            select_keys = [k for k in select_keys if k != "old_log_probs"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        non_tensor_select_keys = []
        for key in [
            "reward_scalar",
            "advantage_scalar",
            "uid",
            "traj_uid",
            "param_version_start",
            "param_version_end",
            "trainer_param_version",
        ]:
            if key in data.non_tensor_batch:
                non_tensor_select_keys.append(key)

        self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")

        if non_tensor_select_keys:
            data = data.select(select_keys, non_tensor_select_keys)
        else:
            data = data.select(batch_keys=select_keys)
        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        data.to(get_device_id())
        data.batch = data.batch.contiguous()
        mini_batch = data
        with torch.no_grad():
            broadcast_dict_tensor(
                mini_batch.batch,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][
                :, 0
            ]  # mcore patch recompute qwen2vl's pos ids during forward

        indices = None
        temperature = data.meta_info["temperature"]
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {micro_batches} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output, data, meta_info):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            log_probs = None
            entropy = None
            if isinstance(output, dict):
                log_probs = output["log_probs"]
                if "entropy" in output:
                    entropy = output["entropy"]
            else:
                assert isinstance(output, torch.Tensor)
                log_probs = output

            device = log_probs.device
            metrics = {}
            if forward_only:
                if post_process_fn is None:
                    pass
                    # metrics["logits"] = output
                else:
                    stats = post_process_fn(output, data)
                    metrics.update(stats)
                if not calculate_entropy:
                    return torch.tensor(1.0, device=device), metrics

            skip_recompute_old_log_prob = bool(meta_info.get("skip_recompute_old_log_prob", False))
            responses = data["responses"]
            response_length = responses.size(1)
            response_mask = data["response_mask"].to(bool)
            loss_agg_mode = self.config.loss_agg_mode
            # compute policy loss
            log_prob = log_probs[:, -response_length - 1 : -1].contiguous()
            ret_entropy = None
            stats = {}
            old_log_prob = None
            rollout_log_prob = None
            if not forward_only:
                if not skip_recompute_old_log_prob:
                    old_log_prob = data["old_log_probs"]
                else:
                    # Compute rollout-policy IS weights in backward pass using policy log-probs.
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

                    rollout_corr_cfg = meta_info.get("rollout_corr_config")
                    if rollout_corr_cfg is None:
                        rollout_corr_cfg = self.config.policy_loss.get("rollout_correction", None)
                    if rollout_corr_cfg is None:
                        raise ValueError(
                            "skip_recompute_old_log_prob=True requires rollout_corr_config in meta_info "
                            "or policy_loss.rollout_correction in config."
                        )
                    old_log_prob = log_prob.detach().clone()
                    rollout_log_prob = data["rollout_log_probs"]
                    rollout_is_weights_proto, modified_response_mask, rollout_corr_metrics = (
                        compute_rollout_correction_and_rejection_mask(
                            old_log_prob=old_log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=data["response_mask"],
                            rollout_is=rollout_corr_cfg.get("rollout_is", "token"),
                            rollout_is_threshold=rollout_corr_cfg.get("rollout_is_threshold", 2.0),
                            rollout_rs=rollout_corr_cfg.get("rollout_rs", None),
                            rollout_rs_threshold=rollout_corr_cfg.get("rollout_rs_threshold", None),
                            rollout_rs_threshold_lower=rollout_corr_cfg.get("rollout_rs_threshold_lower", None),
                            rollout_token_veto_threshold=rollout_corr_cfg.get("rollout_token_veto_threshold", None),
                            rollout_is_batch_normalize=rollout_corr_cfg.get("rollout_is_batch_normalize", None),
                        )
                    )
                    stats.update(rollout_corr_metrics)
                    if rollout_is_weights_proto is not None:
                        data["rollout_is_weights"] = rollout_is_weights_proto.batch["rollout_is_weights"]
                    response_mask = modified_response_mask.bool()
                advantages = data["advantages"]

                entropy_coeff = self.config.entropy_coeff
                loss_agg_mode = self.config.loss_agg_mode

                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                policy_loss_fn = get_policy_loss_fn(loss_mode)

                # Extract pre-computed rollout correction weights if present
                # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                rollout_is_weights = data.get("rollout_is_weights", None)
                pg_loss, pg_metrics = policy_loss_fn(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                    config=self.config,
                    rollout_is_weights=rollout_is_weights,
                )
                stats.update(pg_metrics)

                # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                rollout_log_prob = data.get("rollout_log_probs", None)
                if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                    # Compute metrics using CURRENT policy π_θ vs π_rollout
                    # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                    rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                        log_prob=log_prob,
                        rollout_log_prob=rollout_log_prob,
                        response_mask=response_mask,
                    )
                    stats.update(rollout_corr_metrics)

                stats["actor/pg_loss"] = pg_loss.detach().item()
                policy_loss = pg_loss

            if calculate_entropy:
                entropy = output["entropy"][:, -response_length - 1 : -1].contiguous()
                if not forward_only:
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                    entropy_coeff = meta_info["entropy_coeff"]
                    policy_loss = pg_loss - entropy_coeff * entropy_loss
                else:
                    ret_entropy = entropy

            if forward_only:
                policy_loss = torch.tensor(1.0, device=device)
            else:
                if self.config.use_kl_loss:
                    ref_log_prob = data["ref_log_prob"]
                    # compute kl loss
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics["actor/kl_loss"] = kl_loss.detach().item()
                    metrics["actor/kl_coef"] = self.config.kl_loss_coef

                # return loss and stats

            append_to_dict(metrics, stats)
            if forward_only:
                return policy_loss, [metrics, ret_entropy, None, None]
            loss_multiplier = float(meta_info.get("loss_multiplier", 1.0) or 1.0)
            if loss_multiplier != 1.0:
                policy_loss = policy_loss * loss_multiplier
            return policy_loss, [metrics, ret_entropy, rollout_log_prob, old_log_prob]

        def forward_step(batch_iter, model, return_schedule_plan: bool = False):
            """
            Args:
                batch_iter: the batch iterator
                model: the model
                return_schedule_plan: whether to return the schedule plan, for 1f1b overlap
            """
            if return_schedule_plan:
                assert self.tf_config.overlap_moe_expert_parallel_comm, (
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                )
                # TODO: Fix this
                assert not calculate_entropy, "calculate_entropy must be disabled to return the schedule plan"
                from megatron.core.models.gpt.gpt_model import GPTModel

                assert isinstance(model, GPTModel), "model must be a GPTModel"
                assert self.use_fused_kernels, "use_fused_kernels must be enabled to return the schedule plan"
                # TODO: support VLM with MoE
                from verl.models.mcore.model_forward_1f1b_overlap import gptmodel_forward_1f1b_overlap

            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()
            skip_recompute_old_log_prob = bool(data.meta_info.get("skip_recompute_old_log_prob", False))
            rollout_corr_cfg = data.meta_info.get("rollout_corr_config")

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)
            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                if return_schedule_plan:
                    forward_fn = gptmodel_forward_1f1b_overlap
                # return dict of [logits, entropy]
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=label,
                    labels_mask=label_mask,
                    temperature=temperature,
                    multi_modal_inputs=multi_modal_inputs,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    logits.div_(temperature)
                    ret = {}
                    if calculate_entropy:
                        logits_bak = logits.clone()
                        # # disable the hint until the fused_kernel is optimized for triton>=3.3
                        # logger.warning_once(
                        #     "For memory-efficient computation, enable fused kernels via "
                        #     "`actor_rollout_ref.model.use_fused_kernels=True`. "
                        #     "The current `clone()` operation ensures correctness but increases memory usage."
                        # )
                        entropy = vocab_parallel_entropy(logits)
                        ret["entropy"] = entropy
                    else:
                        logits_bak = logits
                    log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    data_format="thd" if self.config.megatron.use_remove_padding else "bshd",
                )

            if forward_only:
                meta_info = {
                    "skip_recompute_old_log_prob": skip_recompute_old_log_prob,
                    "rollout_corr_config": rollout_corr_cfg,
                    "loss_multiplier": float(data.meta_info.get("loss_multiplier", 1.0) or 1.0),
                }
            else:
                clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                meta_info = {
                    "clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                    "clip_ratio_c": clip_ratio_c,
                    "skip_recompute_old_log_prob": skip_recompute_old_log_prob,
                    "rollout_corr_config": rollout_corr_cfg,
                    "loss_multiplier": float(data.meta_info.get("loss_multiplier", 1.0) or 1.0),
                }
            return output, partial(loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # no use when input_shapes was set
                micro_batch_size=1,  # no use when input_shapes was set
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,  # in use for pp = 1
                micro_batch_size=1,  # in use for pp = 1
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        return losses_reduced

    def get_lr(self) -> list[float] | None:
        """Return current actor optimizer lrs (one per param group), or None if no optimizer is attached."""
        param_groups = getattr(self.actor_optimizer, "param_groups", None)
        if not param_groups:
            return None
        return [float(pg.get("lr", 0.0)) for pg in param_groups]

    def _compute_grad_norms(
        self,
        response_len: int,
        loss_agg_mode: str,
        adv_scalar: float,
        *,
        microbatch_loss_scale: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Returns unscaled_grad_norm (not normalized by length), grad_norm.
        """
        loss_scale = 1.0
        if hasattr(self.actor_optimizer, "get_loss_scale"):
            loss_scale = float(self.actor_optimizer.get_loss_scale().item())
        found_inf_flag = self.actor_optimizer.prepare_grads()

        if found_inf_flag:
            unscaled_grad_norm = grad_norm = float("inf")
        else:
            # Use full local grads before any DP sharding to keep per-replica norms.
            grads = _get_local_model_grads_for_norm(self.actor_module)
            unscaled_grad_norm = get_grad_norm_fp32(grads, grad_stats_parallel_group=mpu.get_tensor_model_parallel_group())
            if loss_agg_mode in ["seq-mean-token-mean"]:
                unscaled_grad_norm *= response_len

            if loss_scale not in (0.0, 1.0):
                unscaled_grad_norm = unscaled_grad_norm / loss_scale

            # [NOTE] Loss in backward pass already scaled by 1 / minibatch_size
            unscaled_grad_norm = unscaled_grad_norm / microbatch_loss_scale
            grad_norm = unscaled_grad_norm * abs(adv_scalar)

        return unscaled_grad_norm, grad_norm

    def _update_grad_buffers(
        self,
        accum_buffers: list[torch.Tensor],
        score_gradient_buffers: list[torch.Tensor] | None,
        local_traj_records: list[TrajRecord],
        reward_scalar: float,
        reward_std: float,
        adv_scalar: float,
        group_uid: int,
        microbatch_loss_scale: float,
        norm_by_std: bool = False,
        is_last_traj_in_scope: bool = False,
        grad_baselining: bool = False,
    ):
        """Gradient buffer update."""
        if grad_baselining:
            assert score_gradient_buffers is not None
            scale_reward = reward_scalar
            scale_baseline = 1
            if self.config.grad_baselining.scope == "group":
                if norm_by_std and reward_std is not None and reward_std > 1e-8:
                    scale_reward = (reward_scalar / reward_std) 
                    scale_baseline = (1.0 / reward_std)
            else:
                scale_reward = adv_scalar
                scale_baseline = 1

            accumulate_grad_buffers(self.actor_module, accum_buffers, scale=scale_reward)
            accumulate_grad_buffers(self.actor_module, score_gradient_buffers, scale=scale_baseline)
        else:
            accumulate_grad_buffers(self.actor_module, accum_buffers, scale=adv_scalar)

        if grad_baselining and is_last_traj_in_scope:
            assert score_gradient_buffers is not None
            # Compute grad-norm-weighted reward baseline (group/minibatch scope).
            opob_baseline = compute_opob_baseline(
                local_traj_records,
                group_uid,
                use_is_weights=self.config.grad_baselining.use_is_weights,
                use_clipped_is_ratios=self.config.grad_baselining.use_clipped_is_ratios,
                normalize_by_length=self.config.grad_baselining.normalize_by_length,
                agg_mode=self.config.grad_baselining.agg_mode,
                scope=self.config.grad_baselining.scope,
            )

            move_grad_buffers(src=score_gradient_buffers, dest=accum_buffers, scale=-opob_baseline)
            zero_grad_accum_buffers(score_gradient_buffers)

    def _optimizer_step_with_buffer(
        self,
        accum_buffers,
        local_traj_records: list[TrajRecord],
        rollout_is_threshold: float | None,
        minibatch_idx: int = 0,
        do_grad_sync: bool = True,
    ) -> Tuple[bool, Dict]:
        staleness_metrics = compute_ess_info(local_traj_records, rollout_is_threshold)
        minibatch_ess = staleness_metrics.get("ess")
        ess_ratio = staleness_metrics.get("ess_ratio")
        minibatch_ess_clipped = staleness_metrics["ess_clipped"]
        ess_ratio_clipped = staleness_metrics["ess_ratio_clipped"]

        ess_ratio_for_scaling = ess_ratio_clipped if self.config.ess_scaling.use_clipped else ess_ratio
        if ess_ratio_for_scaling is None:
            ess_ratio_for_scaling = 0.0
        lrs_now = self.get_lr()
        lr = lrs_now[0] if lrs_now else None

        # ================ Optimizer Step ================
        copy_accum_buffers_to_grad_buffers(self.actor_module, accum_buffers)

        # ===== Optional finalize with DDP sync of gradients =====
        #
        # `forward_backward_batch()` already calls `config.finalize_model_grads_func`, which
        # performs DP×CP gradient synchronization via `finish_grad_sync()`. If we did NOT
        # disable that finalize path, then calling `finish_grad_sync()` again here is redundant.
        #
        # Only run this sync when we intentionally disabled Megatron's DP×CP sync during
        # backward (DP>1 in the per-traj path).
        if do_grad_sync:
            for chunk in self.actor_module:
                if chunk.ddp_config.overlap_grad_reduce:
                    chunk.start_grad_sync()
            for chunk in self.actor_module:
                if chunk.ddp_config.overlap_grad_reduce:
                    chunk.finish_grad_sync()
                else:
                    # finish_grad_sync will call start_grad_sync internally for non-overlap
                    chunk.finish_grad_sync()

        base_lrs = self.get_lr()
        if self.config.ess_scaling.enable and base_lrs is not None:
            base_ess_ratio = max(float(self.config.ess_scaling.base_ess_ratio), 1e-8)
            lr_scale = min(1.0, float(ess_ratio_for_scaling) / base_ess_ratio)
            scaling_rule = self.config.ess_scaling.scaling_rule
            for pg, base_lr in zip(self.actor_optimizer.param_groups, base_lrs, strict=True):
                if scaling_rule == "sqrt":
                    pg["lr"] = float(base_lr) * (lr_scale**0.5)
                elif scaling_rule == "linear":
                    pg["lr"] = float(base_lr) * lr_scale
                else:
                    raise NotImplementedError(f"{scaling_rule} not implemented for ESS scaling")

            lrs_now = self.get_lr()
            lr = lrs_now[0] if lrs_now else None

        update_successful, minibatch_grad_norm, num_zeros_in_grad = self.actor_optimizer.step()

        if self.config.ess_scaling.enable and base_lrs is not None:
            for pg, base_lr in zip(self.actor_optimizer.param_groups, base_lrs, strict=True):
                pg["lr"] = base_lr

        minibatch_metrics = {
            "actor/grad_norm": minibatch_grad_norm,
            "staleness/ess": [
                {
                    "minibatch_idx": minibatch_idx,
                    "minibatch_ess": minibatch_ess,
                    "minibatch_ess_clipped": minibatch_ess_clipped,
                    "minibatch_ess_ratio": ess_ratio,
                    "minibatch_ess_ratio_clipped": ess_ratio_clipped,
                    "ess_scaled_lr": lr,
                }
            ],
        }

        return update_successful, minibatch_metrics

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy_per_traj(self, dataloader: Iterable[DataProto], grad_baselining: bool = False) -> dict:
        """Update the policy with per-trajectory gradient norm capture.
        
        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

            "actor/local_traj_records": list[dict]
            "staleness/ess": list[dict]
        """
        metrics = {}

        accum_buffers = allocate_grad_accum_buffers(self.actor_module)
        score_gradient_buffers = None
        if grad_baselining:
            assert self.config.loss_agg_mode in [
                "seq-mean-token-mean",
                "seq-mean-token-sum",
                "seq-mean-token-sum-norm",
            ]
            score_gradient_buffers = allocate_grad_accum_buffers(self.actor_module)

        # [NOTE]: Megatron's DDP grad sync is over the DP×CP domain if using distributed optimizer.
        with_context_parallel = self.use_distributed_opt
        dp_world_size = mpu.get_data_parallel_group(with_context_parallel=with_context_parallel).size()
        if dp_world_size > 1:
            orig_no_sync, orig_grad_sync, orig_finalize = disable_dp_sync(self.actor_module)

        local_traj_records = []
        for minibatch_idx, minibatch in enumerate(dataloader):
            self.actor_optimizer.zero_grad()
            for chunk in self.actor_module:
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.entropy_coeff != 0
            if minibatch.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = minibatch.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            skip_recompute_old_log_prob = bool(minibatch.meta_info.get("skip_recompute_old_log_prob", False))
            skip_recompute_old_log_prob = skip_recompute_old_log_prob or bool(
                minibatch.meta_info.get("calculate_rollout_policy_is", False)
            )

            assert (self.config.use_dynamic_bsz is False) and micro_batch_size == 1, (
                "Must use micro batch size 1 for accurate gradient norm statistics per trajectory"
            )

            rollout_corr_cfg = minibatch.meta_info.get("rollout_corr_config", None)
            if rollout_corr_cfg is None:
                rollout_corr_cfg = self.config.policy_loss.get("rollout_correction", {})
            rollout_is_threshold = rollout_corr_cfg.get("rollout_is_threshold", None)
            minibatch_size = len(minibatch)
            microbatch_loss_scale = 1 / len(minibatch)

            local_traj_records, _ = compute_staleness_statistics(
                minibatch, minibatch_idx, rollout_is_threshold, not skip_recompute_old_log_prob
            )

            if grad_baselining:
                minibatch = compute_grad_info(minibatch, scope=self.config.grad_baselining.scope)

            zero_grad_accum_buffers(accum_buffers)
            if grad_baselining:
                zero_grad_accum_buffers(score_gradient_buffers)

            # Emulate Megatron schedule's loss scaling by num_microbatches=len(minibatch) 
            # Scaling loss instead of gradients avoids extra numeric/rounding differences
            minibatch.meta_info["loss_multiplier"] = microbatch_loss_scale

            # Per-trajectory updates and gradient statistics.
            for microbatch_idx, microbatch in enumerate(minibatch.split(1)):
                response_mask = microbatch.batch["response_mask"]
                response_len = int(response_mask.sum().item())
                group_uid = microbatch.non_tensor_batch["uid"][0]
                traj_uid = microbatch.non_tensor_batch["traj_uid"][0]
                traj_record = local_traj_records[traj_uid]
                reward_scalar = traj_record.reward_scalar
                adv_scalar = traj_record.advantage_scalar
                # Use raw score gradients g_i <- w_i * grad(log pi) for per-traj accumulation.
                microbatch.batch["advantages"] = torch.ones_like(microbatch.batch["advantages"])

                with ExitStack() as stack:
                    if dp_world_size > 1:
                        for model_chunk in self.actor_module:
                            stack.enter_context(model_chunk.no_sync())
                    metric_micro_batch = self.forward_backward_batch(
                        microbatch,
                        calculate_entropy=calculate_entropy,
                        use_dynamic_bsz=False,
                        micro_batch_size=1,
                        max_token_len=None,
                        mini_batch_size=self.config.ppo_mini_batch_size,
                    )

                metric_micro_batch = metric_micro_batch["output"]
                for metric in metric_micro_batch:
                    # o[0] metrics, o[1] entropy, o[2] rollout_log_probs, o[3] old_log_probs/policy_log_probs
                    append_to_dict(metrics, metric[0])
                    if skip_recompute_old_log_prob:
                        rollout_log_probs = metric[2]
                        policy_log_probs = metric[3]
                        traj_record = compute_is_info(
                            traj_record,
                            rollout_log_probs,
                            policy_log_probs,
                            response_mask,
                            rollout_is_threshold,
                        )

                # Compute gradient norm statistics.
                unscaled_grad_norm, grad_norm = self._compute_grad_norms(response_len, self.config.loss_agg_mode, adv_scalar, microbatch_loss_scale=microbatch_loss_scale)

                traj_record = local_traj_records[traj_uid]
                traj_record.grad_norm = grad_norm
                traj_record.grad_norm_unscaled = unscaled_grad_norm

                last_traj_in_scope = microbatch_idx == minibatch_size - 1
                reward_std = 0.0
                if grad_baselining:
                    last_traj_in_scope = minibatch.meta_info["is_last_traj_in_scope"][traj_uid]
                    reward_std = minibatch.meta_info["reward_std_by_traj_uid"][traj_uid]

                self._update_grad_buffers(
                    accum_buffers=accum_buffers,
                    score_gradient_buffers=score_gradient_buffers,
                    local_traj_records=local_traj_records,
                    reward_scalar=reward_scalar,
                    reward_std=reward_std,
                    adv_scalar=adv_scalar,
                    group_uid=group_uid,
                    microbatch_loss_scale=microbatch_loss_scale,
                    norm_by_std=self.config.grad_baselining.norm_by_std,
                    is_last_traj_in_scope=last_traj_in_scope,
                    grad_baselining=grad_baselining,
                )

                self.actor_optimizer.zero_grad()
                for chunk in self.actor_module:
                    chunk.zero_grad_buffer()

            # Minibatch updates with accumulation buffers.
            update_successful, minibatch_metrics = self._optimizer_step_with_buffer(
                accum_buffers,
                local_traj_records,
                rollout_is_threshold,
                minibatch_idx,
                do_grad_sync=(dp_world_size > 1),
            )

            if not update_successful:
                raise NotImplementedError

            minibatch_metrics["actor/minibatch_grad_info"] = [
                {
                    "epoch_idx": 0,
                    "minibatch_idx": minibatch_idx,
                    "grad_norm": minibatch_metrics["actor/grad_norm"],
                    "trainer_global_step": minibatch.meta_info.get("trainer_global_step", -1),
                    "trainer_local_step": minibatch.meta_info.get("trainer_local_step", -1),
                }
            ]

            append_to_dict(metrics, minibatch_metrics)

        metrics["actor/local_traj_records"] = [asdict(rec) for rec in local_traj_records]

        if dp_world_size > 1:
            restore_dp_sync(self.actor_module, orig_no_sync, orig_grad_sync, orig_finalize)

        self.actor_optimizer.zero_grad()
        get_torch_device().empty_cache()
        return metrics

    @GPUMemoryLogger(role="megatron actor", logger=logger)
    def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            dataloader (Iterable[DataProto]): an iterator over the DataProto that returns by ``make_minibatch_iterator``
                The keys of each data batch is described in the make_minibatch_iterator.

        Returns:
            Dict: a dictionary containing the statistics. Note that the statistics are only valid in the last pp stage
            and users have to combine the output in each dp rank manually.

        """
        metrics = {}
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.start()
        for data in dataloader:
            self.actor_optimizer.zero_grad()
            # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
            for chunk in self.actor_module:
                # if use distributed optimizer, zero grad buffer will be handled by optimizer
                chunk.zero_grad_buffer()

            calculate_entropy = self.config.entropy_coeff != 0
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size
            metric_micro_batch = self.forward_backward_batch(
                data,
                calculate_entropy=calculate_entropy,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )
            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that o[0] is metrics, o[1] is entropy, o[2] is response_mask
                append_to_dict(metrics, metric[0])  # append the metric from this micro-batch to global metrics.

            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, data)

            if update_successful:
                # allgather already execute in optimizer.step in new megatron
                pass
            else:
                raise NotImplementedError
            if self.use_torch_profiler and self.prof and self.prof.enable:
                self.prof.step()
        # add empty cache after each compute
        if self.use_torch_profiler and self.prof and self.prof.enable:
            self.prof.stop_and_save()
            self.prof.stop_trace()
        get_torch_device().empty_cache()
        return metrics
