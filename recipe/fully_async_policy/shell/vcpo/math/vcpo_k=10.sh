#!/usr/bin/env bash
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurm/%A_%x.out
#SBATCH --error=./slurm/%A_%x.err
#SBATCH --job-name=vcpo

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DISABLE_IMPORT_WARNING=1
export VLLM_USE_V1=1
export RAY_ADDRESS="local"

# ================= Paths =================
MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-7B"}
TRAIN_FILE=${TRAIN_FILE:-"data/math-500/math-train.parquet"}
TEST_FILE=${TEST_FILE:-"data/math-500/math-500.parquet"}

project_name='vcpo'

# ================= GPU Layout =================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
n_gpus_rollout=4
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))

# ================= Rollout =================
rollout_mode="async"
rollout_name="vllm"
return_raw_chat="True"
gen_tp=2
n_resp_per_prompt=16
gpu_memory_utilization=0.8
enable_chunked_prefill=True
calculate_log_probs=True

# ================= Sequence Lengths =================
max_prompt_length=1024
max_response_length=2048
max_num_batched_tokens=$((max_prompt_length + max_response_length))

# ================= Megatron Parallelism =================
train_tp=4
train_pp=1
train_cp=1
sequence_parallel=True
use_remove_padding=True
precision_dtype="bfloat16"

# ================= Batch Sizes =================
train_prompt_bsz=0
gen_prompt_bsz=1
train_prompt_mini_bsz=32
micro_bsz_per_gpu=1
use_dynamic_bsz=False
log_prob_micro_bsz_per_gpu=1

bsz_per_dp_rank=32 # Rollout Bsz

# ================= Algorithm =================
adv_estimator=grpo
loss_agg_mode="seq-mean-token-mean"
clip_ratio_low=1.0
clip_ratio_high=1e9
clip_ratio_c=1e9
use_kl_loss=False
kl_loss_coef=0.0
use_kl_in_reward=False
kl_coef=0.0
entropy_coeff=0
grad_clip=1.0

# ================= Optimizer =================
lr=1e-6
lr_warmup_steps=0
weight_decay=0.1

# ================= IS / Rollout Correction =================
rollout_is="sequence"
rollout_is_threshold="8.0"
rollout_rs=null
rollout_rs_threshold=null

skip_recompute_old_log_prob=True
compute_prox_log_prob=False

# ================= Async Training =================
staleness_threshold=10.0
updates_per_param_sync=1
num_minibatches_per_update=1
partial_rollout=True
use_rollout_log_probs=True

# ================= VCPO-specific =================
update_policy_per_traj=True

# ESS Scaling
ess_scaling=True
ess_scaling_scaling_rule="sqrt"
ess_scaling_base_ess_ratio=1.0
ess_scaling_use_clipped=False

# OPOB baselining
grad_baselining=True
grad_baselining_scope="group"
grad_baselining_agg_mode="mean"
grad_baselining_use_is_weights=True
grad_baselining_norm_by_std=True

grad_baselining_use_clipped_is_ratios=False
grad_baselining_normalize_by_length=False

# ================= Training/Rollout Steps =================
total_rollout_steps=$((500 * num_minibatches_per_update * updates_per_param_sync * train_prompt_mini_bsz))
epochs=10000000
test_freq=10
save_freq=-1

# ================= Logging =================
exp_name="VCPO k-${staleness_threshold} MATH Qwen2.5-7B B-${train_prompt_mini_bsz} ${loss_agg_mode} ${max_response_length}-len ${weight_decay}-wd"
exp_name_safe=${exp_name//\//_}
log_dir="logs/${exp_name_safe}"
CKPTS_DIR="${log_dir}"

trainer_logger="['console','wandb']"
log_val_generations=0
wandb_entity=${wandb_entity:-""}
wandb_group=${wandb_group:-"vcpo"}
val_before_train=False

# ================= LR decay =================
lr_decay_style="constant"
lr_decay_steps=${total_rollout_steps}

# ================= Run =================
python -m recipe.fully_async_policy.fully_async_main \
    --config-name=fully_async_ppo_megatron_trainer.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    actor_rollout_ref.actor.strategy=megatron \
    critic.strategy=megatron \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.sequence_parallel=${sequence_parallel} \
    actor_rollout_ref.actor.megatron.dtype=${precision_dtype} \
    actor_rollout_ref.actor.megatron.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.optim.lr_decay_style=${lr_decay_style} \
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
    actor_rollout_ref.actor.optim.weight_decay=${weight_decay} \
    actor_rollout_ref.actor.optim.clip_grad=${grad_clip} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_rollout_log_probs=${use_rollout_log_probs} \
    actor_rollout_ref.actor.update_policy_per_traj=${update_policy_per_traj} \
    actor_rollout_ref.actor.ess_scaling.enable=${ess_scaling} \
    actor_rollout_ref.actor.ess_scaling.scaling_rule=${ess_scaling_scaling_rule} \
    actor_rollout_ref.actor.ess_scaling.base_ess_ratio=${ess_scaling_base_ess_ratio} \
    actor_rollout_ref.actor.ess_scaling.use_clipped=${ess_scaling_use_clipped} \
    actor_rollout_ref.actor.grad_baselining.enable=${grad_baselining} \
    actor_rollout_ref.actor.grad_baselining.scope=${grad_baselining_scope} \
    actor_rollout_ref.actor.grad_baselining.agg_mode=${grad_baselining_agg_mode} \
    actor_rollout_ref.actor.grad_baselining.use_is_weights=${grad_baselining_use_is_weights} \
    actor_rollout_ref.actor.grad_baselining.use_clipped_is_ratios=${grad_baselining_use_clipped_is_ratios} \
    actor_rollout_ref.actor.grad_baselining.normalize_by_length=${grad_baselining_normalize_by_length} \
    actor_rollout_ref.actor.grad_baselining.norm_by_std=${grad_baselining_norm_by_std} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.sequence_parallel=${sequence_parallel} \
    actor_rollout_ref.ref.megatron.dtype=${precision_dtype} \
    actor_rollout_ref.ref.megatron.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.dtype=${precision_dtype} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=3 \
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_bsz_per_gpu} \
    critic.megatron.tensor_model_parallel_size=${train_tp} \
    critic.megatron.pipeline_model_parallel_size=${train_pp} \
    critic.megatron.context_parallel_size=${train_cp} \
    critic.megatron.sequence_parallel=${sequence_parallel} \
    critic.megatron.dtype=${precision_dtype} \
    trainer.logger=${trainer_logger} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    +trainer.wandb_entity="${wandb_entity}" \
    +trainer.wandb_group="${wandb_group}" \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=${save_freq} \
    trainer.rollout_data_dir="${log_dir}" \
    trainer.log_val_generations=${log_val_generations} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.total_epochs="${epochs}" \
    rollout.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${updates_per_param_sync}" \
    async_training.require_batches="${num_minibatches_per_update}" \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.compute_prox_log_prob="${compute_prox_log_prob}" \
    async_training.use_rollout_log_probs="${use_rollout_log_probs}" \
    async_training.skip_recompute_old_log_prob="${skip_recompute_old_log_prob}" \
    +async_training.bsz_per_dp_rank="${bsz_per_dp_rank}"
