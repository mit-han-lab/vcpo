from __future__ import annotations
import torch
import torch.distributed as dist
from typing import Iterable, Optional, Sequence, List

from verl.utils.device import get_device_id, get_device_name
from verl.utils.megatron_utils import get_model_config

from megatron.core import parallel_state as mpu, tensor_parallel
from megatron.core.transformer.module import param_is_not_shared
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_conditional_embedding_grads,
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_embedding_grads,
    _update_router_expert_bias,
    _get_main_grad_attr,
    get_attr_wrapped_model,
    _unshard_if_dtensor,
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
    _reshard_if_dtensor
)
from verl.utils.metric import reduce_metrics
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from megatron.core.transformer.module import param_is_not_shared
from contextlib import nullcontext

def _iter_grad_buffers(modules: Iterable[torch.nn.Module]) -> Iterable[torch.Tensor]:
    """
    Iterator over megatron grad buffers
    """
    for module in modules:
        buffers = []
        if hasattr(module, "buffers"):
            buffers.extend(module.buffers)
        if hasattr(module, "expert_parallel_buffers"):
            buffers.extend(module.expert_parallel_buffers)
        if not buffers and hasattr(module, "param_and_grad_buffer"):
            buffers.append(module.param_and_grad_buffer)
        for buffer in buffers:
            yield buffer.grad_data

def allocate_grad_accum_buffers(
    modules: Iterable[torch.nn.Module],
) -> list[torch.Tensor]:
    """
    Allocated memory for grad buffer on GPU, initialized to zeros
    """
    with torch.no_grad():
        accum_buffers: list[torch.Tensor] = []
        for grad_data in _iter_grad_buffers(modules):
            accum_buffers.append(torch.zeros_like(grad_data))
    return accum_buffers

def snapshot_grad_buffers(
    modules: Iterable[torch.nn.Module],
    to_cpu: bool = True,
    pin_memory: bool = False,
) -> list[torch.Tensor]:
    """
    Move grad buffers to CPU for inspection
    """
    snapshots: list[torch.Tensor] = []
    for grad_data in _iter_grad_buffers(modules):
        if to_cpu:
            cpu_copy = torch.empty_like(grad_data, device="cpu", pin_memory=pin_memory)
            cpu_copy.copy_(grad_data.detach(), non_blocking=pin_memory)
            snapshots.append(cpu_copy)
        else:
            snapshots.append(grad_data.detach().clone())
    return snapshots

def apply_scaled_grad_delta(
    modules: Iterable[torch.nn.Module],
    snapshots: Sequence[torch.Tensor],
    scale: float,
) -> None:
    """
    In place scale addition
        grad <- base + (grad - base) * scale
    """
    with torch.no_grad():
        for grad_data, base in zip(_iter_grad_buffers(modules), snapshots):
            if base.device != grad_data.device:
                base = base.to(device=grad_data.device, non_blocking=base.is_pinned())
            grad_data.sub_(base).mul_(scale).add_(base)

def zero_grad_accum_buffers(accum_buffers: Sequence[torch.Tensor]) -> None:
    """
    Zero grad buffer in place
    """
    with torch.no_grad():
        for buffer in accum_buffers:
            buffer.zero_()

def accumulate_grad_buffers(
    modules: Iterable[torch.nn.Module],
    accum_buffers: Sequence[torch.Tensor],
    scale: float,
) -> None:
    """
    Move accum_buffers into grad buffers of modules
    """
    grad_buffers = list(_iter_grad_buffers(modules))
    with torch.no_grad():
        try:
            torch._foreach_add_(accum_buffers, grad_buffers, alpha=scale)
            return
        except Exception as e:
            print(f"[MegatronUtils] _foreach_add failed: {e}")
            pass
        for grad_data, buffer in zip(grad_buffers, accum_buffers):
            buffer.add_(grad_data, alpha=scale)


def copy_accum_buffers_to_grad_buffers(
    modules: Iterable[torch.nn.Module],
    accum_buffers: Sequence[torch.Tensor],
) -> None:
    """
    Copy accum_buffers into grad buffers of modules
    """
    with torch.no_grad():
        for grad_data, buffer in zip(_iter_grad_buffers(modules), accum_buffers):
            grad_data.copy_(buffer)

def move_grad_buffers(
    src: Sequence[torch.Tensor],
    dest: Sequence[torch.Tensor],
    scale: float = 1,
) -> None:
    """
    Move into src into dest grad buffer
    """
    with torch.no_grad():
        try:
            torch._foreach_add_(dest, src, alpha=scale)
            return
        except Exception as e:
            print(f"[MegatronUtils] _foreach_add failed: {e}")
            pass
        for src_grad, dest_grad in zip(src, dest):
            dest_grad.add_(src_grad, alpha = scale)

def _get_local_model_grads_for_norm(actor_modules: List[torch.nn.Module]) -> List[torch.Tensor]:
    grads_for_norm: List[torch.Tensor] = []
    for model_chunk in actor_modules:
        ddp_config = getattr(model_chunk, "ddp_config", None)
        use_custom_fsdp = bool(getattr(ddp_config, "use_custom_fsdp", False))
        for param in model_chunk.parameters():
            if not param.requires_grad:
                continue
            if not param_is_not_shared(param):
                continue
            if not tensor_parallel.param_is_not_tensor_parallel_duplicate(param):
                continue
            grad_attr = _get_main_grad_attr(param, use_custom_fsdp=use_custom_fsdp)
            grad = getattr(param, grad_attr, None)
            if grad is not None:
                grads_for_norm.append(grad)
    return grads_for_norm

def _allreduce_grads_cp(model):
    """
    Based on Megatron Core's _allreduce_non_tensor_model_parallel_grads
    """
    if mpu.get_context_parallel_world_size() <= 1:
        return

    params_avg = []
    grads_avg = []

    for model_chunk in model:
        ddp_config = model_chunk.ddp_config
        for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
            if param.requires_grad:
                grad_attr = _get_main_grad_attr(param, ddp_config.use_custom_fsdp)
                grad = getattr(param, grad_attr)
                if grad is None:
                    continue
                grad = _unshard_if_dtensor(grad)
                
                grads_avg.append(grad.data)
                params_avg.append(param)

    # Loop grads and perform correct all-reduce
    if grads_avg:
        coalesced = _flatten_dense_tensors(grads_avg)
        torch.distributed.all_reduce(
            coalesced, op=torch.distributed.ReduceOp.AVG, group=mpu.get_context_parallel_group()
        )
        for param, buf, synced in zip(
            params_avg, grads_avg, _unflatten_dense_tensors(coalesced, grads_avg)
        ):
            buf.copy_(synced)
            grad_attr = _get_main_grad_attr(param, ddp_config.use_custom_fsdp)
            orig_grad = getattr(param, grad_attr)
            setattr(param, grad_attr, _reshard_if_dtensor(buf, orig_grad))


def finalize_model_grads_ignore_dp(model: Sequence[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads within DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    # if config.timers is not None:
    #     config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    # for model_chunk in model:
    #     model_chunk.finish_grad_sync()
    # if config.timers is not None:
    #     config.timers('all-grads-sync').stop()

    # All-reduce t_embedder grads (for pp & vpp of DiT).
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_conditional_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('conditional-embedder-grads-all-reduce').stop()

    # Do CP all reduce grads
    _allreduce_grads_cp(model)

    # All-reduce layer-norm grads (for sequence parallelism) and non-tensor parallel modules.
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_non_tensor_model_parallel_grads(model, config)
    if config.timers is not None:
        config.timers('non-tensor-parallel-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    assert not config.moe_router_enable_expert_bias, f"Expert Bias not supported"

    # Disable all reduce across TP x CP x DP
    # if config.moe_router_enable_expert_bias:
    #     _update_router_expert_bias(model, config)

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:

        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        last_rank = mpu.get_pipeline_model_parallel_last_rank()
        pp_group = mpu.get_pipeline_model_parallel_group()

        if not isinstance(last_rank, list):
            assert not isinstance(last_rank, list)
            last_rank = [last_rank]
            assert not isinstance(pp_group, list)
            pp_group = [pp_group]

        # need to do a broadcast for every pp group, even though num_tokens should be the same.
        num_tokens_list = []
        for lr, group in zip(last_rank, pp_group):
            torch.distributed.broadcast(num_tokens, src=lr, group=group)
            num_tokens_list.append(torch.clone(num_tokens))
        assert all(x.item() == num_tokens_list[0] for x in num_tokens_list)

        # all-reduce across DP ranks.
        # torch.distributed.all_reduce(
        #     num_tokens, group=mpu.get_data_parallel_group(with_context_parallel=True)
        # )
        # for model_chunk in model:
        #     if num_tokens > 0:
        #         scaling = 1.0 / num_tokens
        #         model_chunk.scale_gradients(scaling)


def disable_dp_sync(actor_modules: Iterable[torch.nn.Module]) -> Tuple:
    config = get_model_config(actor_modules[0])
    orig_no_sync = config.no_sync_func
    orig_grad_sync = config.grad_sync_func
    orig_finalize = config.finalize_model_grads_func

    config.no_sync_func = nullcontext
    config.grad_sync_func = None
    config.finalize_model_grads_func = finalize_model_grads_ignore_dp

    return orig_no_sync, orig_grad_sync, orig_finalize

def restore_dp_sync(actor_modules: Iterable[torch.nn.Module], orig_no_sync, orig_grad_sync, orig_finalize):
    config = get_model_config(actor_modules[0])
    config.no_sync_func = orig_no_sync
    config.grad_sync_func = orig_grad_sync
    config.finalize_model_grads_func = orig_finalize