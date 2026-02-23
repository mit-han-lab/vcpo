import torch
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Literal
from verl import DataProto
from megatron.core import parallel_state as mpu
from verl.utils.torch_functional import allgather_dict_into_list
from collections import defaultdict


@dataclass
class TrajRecord:
    """Record of a single trajectory, populated incrementally during training.

    Core fields are set in ``compute_staleness_statistics``.  Optional fields
    (IS weights, log-prob lists, grad norms, loss) are filled in later by the
    actor or trainer.
    """
    # --- identity ---
    uid: Any
    group_uid: Any

    # --- training position ---
    epoch_idx: int
    minibatch_idx: int
    trainer_global_step: int
    trainer_local_step: int

    # --- staleness / versioning ---
    param_version_start: Any
    param_version_end: Any
    trainer_param_version: Any

    # --- trajectory statistics ---
    response_length: int
    prompt_length: Optional[int]
    advantage_scalar: float
    reward_scalar: float

    # --- filled in after loss / grad computation ---
    traj_loss: Optional[float] = None
    grad_norm: Optional[float] = None
    grad_norm_unscaled: Optional[float] = None

    # --- IS / staleness fields (set by compute_is_info) ---
    old_log_probs: Optional[List[float]] = None
    rollout_log_probs: Optional[List[float]] = None
    kl_rollout_old: Optional[float] = None          # KL(rollout||old), K3 f-divergence form
    rollout_is_geom_mean: Optional[float] = None
    rollout_seq_is: Optional[float] = None
    rollout_seq_is_clipped: Optional[float] = None


class TrajRecordList(list):
    """List of :class:`TrajRecord` objects with optional lookup by ``traj_uid``."""

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        key_str = str(key)
        for record in self:
            if str(record.uid) == key_str:
                return record
        raise KeyError(key)


def mean_nonzero(x: torch.Tensor, dim=None, keepdim: bool = False):
    """
    Mean over nonzero entries of `x`.
    - If dim is None: mean over all nonzero elements.
    - If dim is int/tuple: mean over nonzero elements along that dim.
    - If there are no nonzero elements in the reduction, returns 0 there.
    """
    mask = x != 0
    x_ = x if torch.is_floating_point(x) else x.float()

    sum_ = (x_ * mask).sum(dim=dim, keepdim=keepdim)
    cnt = mask.sum(dim=dim, keepdim=keepdim)

    return torch.where(cnt > 0, sum_ / cnt.clamp_min(1), torch.zeros_like(sum_))

def compute_global_ess_ratio(
    local_is_sum: float,
    local_is_sq_sum: float,
    local_minibatch_size: int,
    eps: float = 1e-8,
):
    """All-reduce IS sums and counts over the data-parallel group to compute global ESS."""
    if local_minibatch_size is None:
        local_minibatch_size = 0

    global_is_sum = float(local_is_sum or 0.0)
    global_is_sq_sum = float(local_is_sq_sum or 0.0)
    global_minibatch_size = int(local_minibatch_size or 0)

    if torch.distributed.is_initialized():
        try:
            dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        except Exception:
            dp_group = None

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor(
            [global_is_sum, global_is_sq_sum, float(global_minibatch_size)],
            device=device,
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        global_is_sum, global_is_sq_sum, global_minibatch_size = tensor.tolist()
        global_minibatch_size = int(global_minibatch_size)

    if global_minibatch_size > 0:
        global_ess = (global_is_sum ** 2) / (global_is_sq_sum + eps)
        global_ess_ratio = global_ess / global_minibatch_size
    else:
        global_ess = 0.0
        global_ess_ratio = 0.0

    return global_ess, global_ess_ratio, global_is_sum, global_is_sq_sum, global_minibatch_size

def compute_staleness_statistics(
    batch: DataProto, 
    minibatch_idx: int, 
    rollout_is_threshold: float | None,
    use_old_log_probs: bool = False,
) -> Tuple[list[TrajRecord], Dict]:
    """
    Computes the local per-traj :class:`TrajRecord` list as well as ESS info.

    Core fields populated here (see :class:`TrajRecord` for the full schema):
        uid, group_uid, epoch_idx, minibatch_idx,
        trainer_global_step, trainer_local_step,
        param_version_start, param_version_end, trainer_param_version,
        response_length, prompt_length, advantage_scalar, reward_scalar

    The following fields are filled in later by the actor/trainer:
        grad_norm, grad_norm_unscaled, traj_loss

    When ``use_old_log_probs=True``, IS-related fields are also populated via
    :func:`compute_is_info`.
    """
    traj_uids = batch.non_tensor_batch["traj_uid"]
    meta_info = getattr(batch, "meta_info", {}) or {}
    non_tensor_batch = batch.non_tensor_batch

    trainer_global_step = int(meta_info.get("trainer_global_step", -1) or -1)
    trainer_local_step = int(meta_info.get("trainer_local_step", -1) or -1)
    trainer_param_version = meta_info.get("trainer_param_version", None)

    # Scalars are required in non_tensor_batch for per-trajectory baselining.
    reward_scalars = non_tensor_batch.get("reward_scalar")
    if reward_scalars is None:
        raise KeyError("Missing non_tensor key 'reward_scalar' required by compute_staleness_statistics.")
    advantage_scalars = non_tensor_batch.get("advantage_scalar")
    if advantage_scalars is None:
        raise KeyError("Missing non_tensor key 'advantage_scalar' required by compute_staleness_statistics.")

    param_version_start_all = non_tensor_batch.get("param_version_start")
    param_version_end_all = non_tensor_batch.get("param_version_end")
    if param_version_start_all is None or param_version_end_all is None:
        raise KeyError(
            "Missing param version metadata. Expected non_tensor keys "
            "'param_version_start'/'param_version_end'."
        )

    local_records = TrajRecordList()

    for idx, traj_uid in enumerate(traj_uids):
        group_uid = batch.non_tensor_batch["uid"][idx]
        response_mask = batch.batch["response_mask"][idx]
        reward_scalar = float(reward_scalars[idx])
        adv_scalar = float(advantage_scalars[idx])

        param_version_start = param_version_start_all[idx]
        param_version_end = param_version_end_all[idx]

        attention_mask = batch.batch["attention_mask"][idx]
        response_len = int(response_mask.sum().item())
        prompt_len = None
        if attention_mask is not None:
            prompt_len = max(int(attention_mask.sum().item()) - response_len, 0)

        record = TrajRecord(
            uid=traj_uid,
            group_uid=group_uid,
            epoch_idx=0,
            minibatch_idx=int(minibatch_idx),
            trainer_global_step=trainer_global_step,
            trainer_local_step=trainer_local_step,
            param_version_start=param_version_start,
            param_version_end=param_version_end,
            trainer_param_version=trainer_param_version,
            response_length=response_len,
            prompt_length=prompt_len,
            advantage_scalar=adv_scalar,
            reward_scalar=reward_scalar,
        )

        if use_old_log_probs:
            old_log_prob = batch.batch["old_log_probs"][idx]
            rollout_log_prob = batch.batch["rollout_log_probs"][idx]
            
            record = compute_is_info(
                record,
                rollout_log_prob,
                old_log_prob,
                response_mask,
                rollout_is_threshold,
            )

        local_records.append(record)

    if not use_old_log_probs:
        return local_records, {}
    
    staleness_info = compute_ess_info(list(local_records), rollout_is_threshold)

    return local_records, staleness_info

def compute_is_info(
    record: TrajRecord,
    rollout_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is_threshold: float | None,
) -> TrajRecord:
    """
    Update record with IS / KL statistics:
        old_log_probs
        rollout_log_probs
        kl_rollout_old  (K3 f-divergence form)
        rollout_is_geom_mean
        rollout_seq_is
        rollout_seq_is_clipped
    """
    mask_device = old_log_prob.device
    response_mask = response_mask.to(device=mask_device)
    mask_float = response_mask.to(dtype=old_log_prob.dtype)
    token_count = mask_float.sum()

    # Add per-token logprobs to the record so the trainer can plot token-level scatters.
    # Store *all* tokens where response_mask == 1 (can be large).
    with torch.no_grad():
        mask = response_mask.reshape(-1).bool()
        old_tokens = old_log_prob.reshape(-1)[mask]
        record.old_log_probs = old_tokens.detach().float().cpu().tolist()
        if rollout_log_prob is not None:
            rollout_tokens = rollout_log_prob.reshape(-1)[mask]
            record.rollout_log_probs = rollout_tokens.detach().float().cpu().tolist()

    # NOTE: The expressions below use the K3 f-divergence form:
    #   K3(P||Q) = E_P[exp(log(P/Q)) - log(P/Q) - 1]
    if rollout_log_prob is not None:
        # KL(rollout_log_prob || old_log_prob)
        with torch.no_grad():
            log_ratio = old_log_prob - rollout_log_prob
            k3_matrix = torch.exp(log_ratio) - log_ratio - 1
            k3_value = (k3_matrix * mask_float).sum() / (token_count + 1e-8)

            record.kl_rollout_old = float(k3_value.detach().item())

            # Geometric Mean of IS ratios
            geom_mean = torch.exp(((log_ratio * mask_float).sum()) / (token_count + 1e-8))
            record.rollout_is_geom_mean = float(geom_mean.detach().item())

            seq_is = torch.exp(((log_ratio * mask_float).sum()))
            seq_is_value = float(seq_is.detach().item())
            record.rollout_seq_is = seq_is_value
            if rollout_is_threshold is not None and rollout_is_threshold > 0:
                record.rollout_seq_is_clipped = min(seq_is_value, float(rollout_is_threshold))
    return record

def compute_ess_info(local_records_list: List[TrajRecord], rollout_is_threshold: float | None, eps: float = 1e-8):
    """
    ESS Calculations:
        ess = (sum w_i)^2 / (sum w_i^2) 
        ess_ratio = ess / minibatch_size
        
        ess_clipped = ess with clipped w_i
        ess_ratio_clipped = ess_clipped / minibatch_size

    """
    dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
    is_leader = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == 0
    )

    # allgather_dict_into_list requires plain dicts; serialize TrajRecord objects first.
    local_dicts: list[dict] = [asdict(r) for r in local_records_list] if is_leader else []
    global_records: list[dict] = allgather_dict_into_list(local_dicts, group=dp_group)

    IS_sum = 0.0
    IS_sq_sum = 0.0
    IS_sum_unclipped = 0.0
    IS_sq_sum_unclipped = 0.0
    ESS = 0.0
    ESS_unclipped = 0.0
    minibatch_size = 0

    for rec in global_records:
        seq_is_unclipped = rec.get("rollout_seq_is")
        if seq_is_unclipped is None:
            continue
        seq_is_clipped = rec.get("rollout_seq_is_clipped")
        if seq_is_clipped is None:
            seq_is_clipped = float(seq_is_unclipped)
            if rollout_is_threshold is not None and rollout_is_threshold > 0:
                seq_is_clipped = min(seq_is_clipped, float(rollout_is_threshold))

        IS_sum_unclipped += float(seq_is_unclipped)
        IS_sq_sum_unclipped += float(seq_is_unclipped) ** 2
        IS_sum += float(seq_is_clipped)
        IS_sq_sum += float(seq_is_clipped) ** 2
        minibatch_size += 1

    if minibatch_size > 0:
        ESS = (IS_sum) ** 2 / (IS_sq_sum + eps)
        ess_ratio = ESS / minibatch_size
        ESS_unclipped = (IS_sum_unclipped) ** 2 / (IS_sq_sum_unclipped + eps)
        ess_ratio_unclipped = ESS_unclipped / minibatch_size
    else:
        ESS = 0.0
        ess_ratio = 0.0
        ESS_unclipped = 0.0
        ess_ratio_unclipped = 0.0

    staleness_info = {
        "ess": ESS_unclipped,
        "ess_ratio": ess_ratio_unclipped,
        "ess_clipped": ESS,
        "ess_ratio_clipped": ess_ratio,
    }

    return staleness_info

def rearrange_minibatch(batch: DataProto) -> DataProto:
    """
    Rearrange minibatch to make trajectories of same group contiguous
    """
    group_uids = batch.non_tensor_batch.get("uid")
    if group_uids is None:
        return batch

    group_to_indices: dict[Any, list[int]] = {}
    group_order: list[Any] = []
    for idx, group_uid in enumerate(list(group_uids)):
        if group_uid not in group_to_indices:
            group_to_indices[group_uid] = []
            group_order.append(group_uid)
        group_to_indices[group_uid].append(idx)

    new_indices = [idx for group_uid in group_order for idx in group_to_indices[group_uid]]
    if new_indices == list(range(len(group_uids))):
        return batch

    return batch[new_indices]

def compute_grad_info(batch: DataProto, scope: Literal["group", "minibatch"] = "group", eps: float = 1e-8):
    n_resp_per_rollout = batch.meta_info["n_resp_per_rollout"]
    batch = rearrange_minibatch(batch)
    traj_uids = batch.non_tensor_batch["traj_uid"]
    raw_rewards = batch.non_tensor_batch["reward_scalar"]

    group_traj_counts = defaultdict(int)
    group_to_trajs = defaultdict(list)
    group_rewards = defaultdict(list)
    is_last_traj_in_scope = {}
    reward_by_traj_uid = {}
    reward_std_by_traj_uid = {}
    grpo_adv_by_traj_uid = {}

    for idx, traj_uid in enumerate(traj_uids):
        # Is last traj in group
        group_uid = batch.non_tensor_batch["uid"][idx]
        group_traj_counts[group_uid] += 1

        if scope == "group":
            is_last_traj_in_scope[traj_uid] = (group_traj_counts[group_uid] == n_resp_per_rollout)
        else:
            is_last_traj_in_scope[traj_uid] = (idx == len(traj_uids) - 1)

        group_to_trajs[group_uid].append(traj_uid)

        raw_reward = batch.non_tensor_batch["reward_scalar"][idx]
        reward = float(raw_reward)
        group_rewards[group_uid].append(reward)
        reward_by_traj_uid[traj_uid] = reward

    for traj_count in group_traj_counts.values():
        assert traj_count == n_resp_per_rollout

    for group_uid, traj_list in group_to_trajs.items():
        rewards = group_rewards.get(group_uid, [])
        if rewards:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            reward_std = float(torch.std(rewards_tensor, unbiased=False).item())
            reward_mean = float(torch.mean(rewards_tensor).item())
        else:
            reward_std = 0.0
            reward_mean = 0.0

        for traj_uid in traj_list:
            reward_std_by_traj_uid[traj_uid] = reward_std
            reward = reward_by_traj_uid[traj_uid]
            grpo_adv_by_traj_uid[traj_uid] = (reward - reward_mean) / (reward_std + eps)

    batch.meta_info["is_last_traj_in_scope"] = is_last_traj_in_scope
    if raw_rewards is not None:
        batch.meta_info["reward_std_by_traj_uid"] = reward_std_by_traj_uid
        batch.meta_info["grpo_adv_by_traj_uid"] = grpo_adv_by_traj_uid

    return batch

    
def compute_opob_baseline(
    local_traj_records: List[TrajRecord],
    group_uid: int,
    eps: float = 1e-8,
    use_is_weights: bool = True,
    use_clipped_is_ratios: bool = False,
    normalize_by_length: bool = False,
    agg_mode: str = "mean",
    scope: str = "group",
):
    """
    Optimal Off Policy Baseline:
        b = (sum_i W_i * R_i) / (sum_i W_i)
    where
        W_i = ||g_i||^2 * (ratio_i^2) * (1/L_i^2 if enabled)
    """
    weights = []
    values = []

    def _in_scope(rec: TrajRecord):
        if scope == "minibatch":
            return True
        return rec.group_uid == group_uid
    
    with torch.no_grad():
        for rec in local_traj_records:
            if _in_scope(rec):
                if scope == "minibatch":
                    rwd = rec.advantage_scalar
                else:
                    rwd = rec.reward_scalar
    
                if use_clipped_is_ratios:
                    seq_is_ratio = rec.rollout_seq_is_clipped
                else:
                    seq_is_ratio = rec.rollout_seq_is
                
                grad_norm = rec.grad_norm_unscaled
                weight = grad_norm ** 2
                if use_is_weights:
                    weight *= seq_is_ratio ** 2

                if normalize_by_length:
                    length = rec.response_length
                    weight = weight / (length ** 2)

                weights.append(weight)
                values.append(rwd)

    if agg_mode == "mean":
        baseline = get_weighted_mean(values, weights, eps=eps)
    elif agg_mode == "median":
        baseline = get_weighted_median(values, weights)
    elif agg_mode == "winsorized_mean":
        baseline = get_weighted_winsorized_mean(values, weights, eps=eps)
    else:
        raise NotImplementedError(f"Unsupported agg_mode: {agg_mode}")

    return baseline


def get_weighted_mean(values, weights, eps: float = 1e-8):
    """Compute a weighted mean from value/weight sequences."""
    if torch.is_tensor(values) or torch.is_tensor(weights):
        values_t = values if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
        weights_t = weights if torch.is_tensor(weights) else torch.tensor(weights, dtype=torch.float32)
        if values_t.numel() == 0:
            return values_t.new_tensor(0.0)
        return (values_t * weights_t).sum() / (weights_t.sum() + eps)
    if not values:
        return 0.0
    numer = 0.0
    denom = 0.0
    for value, weight in zip(values, weights):
        numer += float(value) * float(weight)
        denom += float(weight)
    return numer / (denom + eps)


def get_weighted_median(values, weights):
    """Compute a weighted median from value/weight sequences."""
    if torch.is_tensor(values) or torch.is_tensor(weights):
        values_t = values if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
        weights_t = weights if torch.is_tensor(weights) else torch.tensor(weights, dtype=torch.float32)
        if values_t.numel() == 0:
            return values_t.new_tensor(0.0)
        total_w = weights_t.sum()
        if total_w == 0:
            return values_t.new_tensor(0.0)
        sort_idx = torch.argsort(values_t)
        sorted_vals = values_t[sort_idx]
        sorted_w = weights_t[sort_idx]
        cum_w = torch.cumsum(sorted_w, dim=0)
        cutoff = 0.5 * total_w
        median_idx = torch.searchsorted(cum_w, cutoff, right=False)
        median_idx = torch.clamp(median_idx, max=sorted_vals.numel() - 1)
        return sorted_vals[median_idx]

    if not values:
        return 0.0
    total_w = 0.0
    for weight in weights:
        total_w += float(weight)
    if total_w == 0.0:
        return 0.0
    sorted_pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
    cum_w = 0.0
    cutoff = 0.5 * total_w
    for value, weight in sorted_pairs:
        cum_w += float(weight)
        if cum_w >= cutoff:
            return float(value)
    return float(sorted_pairs[-1][0])


def get_weighted_winsorized_mean(values, weights, lower_q: float = 0.05, upper_q: float = 0.95, eps: float = 1e-8):
    """Compute a weighted winsorized mean by clipping values to weighted quantiles."""
    lower_q = min(max(lower_q, 0.0), 1.0)
    upper_q = min(max(upper_q, 0.0), 1.0)
    if upper_q < lower_q:
        lower_q, upper_q = upper_q, lower_q

    if torch.is_tensor(values) or torch.is_tensor(weights):
        values_t = values if torch.is_tensor(values) else torch.tensor(values, dtype=torch.float32)
        weights_t = weights if torch.is_tensor(weights) else torch.tensor(weights, dtype=torch.float32)
        if values_t.numel() == 0:
            return values_t.new_tensor(0.0)
        total_w = weights_t.sum()
        if total_w == 0:
            return values_t.new_tensor(0.0)

        sort_idx = torch.argsort(values_t)
        sorted_vals = values_t[sort_idx]
        sorted_w = weights_t[sort_idx]
        cum_w = torch.cumsum(sorted_w, dim=0)

        lower_cut = lower_q * total_w
        upper_cut = upper_q * total_w
        lower_idx = torch.searchsorted(cum_w, lower_cut, right=False)
        upper_idx = torch.searchsorted(cum_w, upper_cut, right=False)
        lower_idx = torch.clamp(lower_idx, max=sorted_vals.numel() - 1)
        upper_idx = torch.clamp(upper_idx, max=sorted_vals.numel() - 1)

        lower_val = sorted_vals[lower_idx]
        upper_val = sorted_vals[upper_idx]
        clipped = torch.clamp(values_t, min=lower_val, max=upper_val)
        return (clipped * weights_t).sum() / (weights_t.sum() + eps)

    if not values:
        return 0.0
    total_w = 0.0
    for weight in weights:
        total_w += float(weight)
    if total_w == 0.0:
        return 0.0

    sorted_pairs = sorted(zip(values, weights), key=lambda pair: pair[0])
    cum_w = 0.0
    lower_cut = lower_q * total_w
    upper_cut = upper_q * total_w
    lower_val = sorted_pairs[0][0]
    upper_val = sorted_pairs[-1][0]
    for value, weight in sorted_pairs:
        cum_w += float(weight)
        if cum_w >= lower_cut:
            lower_val = value
            break
    cum_w = 0.0
    for value, weight in sorted_pairs:
        cum_w += float(weight)
        if cum_w >= upper_cut:
            upper_val = value
            break

    numer = 0.0
    denom = 0.0
    for value, weight in zip(values, weights):
        clipped = min(max(float(value), float(lower_val)), float(upper_val))
        numer += clipped * float(weight)
        denom += float(weight)
    return numer / (denom + eps)
