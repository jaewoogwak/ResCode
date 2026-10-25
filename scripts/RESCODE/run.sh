#!/usr/bin/env bash
# ============================================================================
# RESCODE - generic launcher.
# Runs the RESCODE pipeline (I1 + I2 + I3) on any SMoE model and compression
# ratio.
#
# The user-facing shell variables follow the paper's notation; they are mapped
# to the internal Python flags at the bottom of this file.
#
#   I1  Saliency-aware restoration
#       Step 1: cluster formation with output similarity + singleton pin rho
#       Step 2: saliency-weighted base construction
#               alpha_i^G = (a_i + delta) / sum_j (a_j + delta)
#   I2  Diagonal Residual-code:    D_i ~= B_G * diag(z_i) * A_G
#   I3  Output-guided fitting:     L = lambda_mse * L_rec + lambda_lm * L_LM  (paper: L_rec + lambda L_LM)
#
# Override anything via env vars. Examples:
#   MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat NUM_EXPERTS=30 GPU=0 bash scripts/RESCODE/run.sh
#   MODEL=mistralai/Mixtral-8x7B-v0.1 NUM_EXPERTS=4 SALIENCY_TOP_TAU=0 bash ...
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."   # to repo root

# ---------------------------------------------------------------------------
# 1. Experiment identity
# ---------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
NUM_EXPERTS="${NUM_EXPERTS:-30}"          # compressed expert count per layer
GPU="${GPU:-0}"
TAG="${TAG:-}"
SEED="${SEED:-0}"                         # optimization seed
CALIB_SEED="${CALIB_SEED:-42}"            # calibration-sampling seed; the paper's
                                          # multi-seed runs (App. F.5) vary SEED
                                          # and CALIB_SEED jointly

# ---------------------------------------------------------------------------
# 2. Data / calibration
# ---------------------------------------------------------------------------
TASK="${TASK:-winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte}"

# CALIB_DATASET picks a registered calibration corpus by short name (paper
# default: dclm). Prepare a corpus once with
#   python scripts/RESCODE/prepare_calib_datasets.py --dataset <name>
# (`c4` instead expects the pre-downloaded shard below). TRAIN_DATA overrides
# the resolved path directly.
CALIB_DATASET="${CALIB_DATASET:-dclm}"
case "$CALIB_DATASET" in
  c4)           _DEFAULT_TRAIN="rescode/data/c4-train.00000-of-01024.json" ;;
  wikipedia)    _DEFAULT_TRAIN="rescode/data/wikipedia-calib.jsonl" ;;
  slimpajama)   _DEFAULT_TRAIN="rescode/data/slimpajama-calib.jsonl" ;;
  dclm)         _DEFAULT_TRAIN="rescode/data/dclm-calib.jsonl" ;;
  starcoder)    _DEFAULT_TRAIN="rescode/data/starcoder-calib.jsonl" ;;
  *) echo "[RESCODE] unknown CALIB_DATASET=$CALIB_DATASET" >&2; exit 1 ;;
esac
TRAIN_DATA="${TRAIN_DATA:-$_DEFAULT_TRAIN}"
if [ ! -f "$TRAIN_DATA" ]; then
  echo "[RESCODE] calibration file missing: $TRAIN_DATA" >&2
  echo "[RESCODE]   run: python scripts/RESCODE/prepare_calib_datasets.py --dataset $CALIB_DATASET" >&2
  exit 1
fi

# VAL_DATA: held-out validation calibration corpus for joint LM-loss val.
# Point it to a separately prepared jsonl file. When empty (the default),
# joint validation (and best-val checkpoint restore) is disabled and
# training proceeds normally.
VAL_DATA="${VAL_DATA:-}"
if [ -n "$VAL_DATA" ] && [ ! -f "$VAL_DATA" ]; then
  echo "[RESCODE] val calibration file missing: $VAL_DATA - joint validation disabled." >&2
  VAL_DATA=""
fi
N_SENTENCES="${N_SENTENCES:-2400}"        # |D|: calibration sequences (saliency, dist_out, fitting)
CLUSTER_N_SENTENCES="${CLUSTER_N_SENTENCES:-32}"   # only used by CLUSTERING=descriptor
TRAIN_BATCH="${TRAIN_BATCH:-1}"           # sequences per forward pass
EVAL_BATCH="${EVAL_BATCH:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"             # >0: evaluate only the first N examples per task (quick checks)
MAX_TOKENS_PER_LAYER="${MAX_TOKENS_PER_LAYER:-32768}"
MAX_TOKENS_PER_CLUSTER="${MAX_TOKENS_PER_CLUSTER:-32768}"

# ---------------------------------------------------------------------------
# 3. Clustering - Idea 1 Step 1 (saliency-aware cluster formation)
# ---------------------------------------------------------------------------
CLUSTERING="${CLUSTERING:-dist}"                 # dist       = dist_out(i,j) = mean_x ||E_i(x)-E_j(x)||^2 over D
                                                  # descriptor = distance between mean expert outputs on
                                                  #              CLUSTER_N_SENTENCES sequences (HC-SMoE)
CALIB_STATS_CACHE="${CALIB_STATS_CACHE:-}"        # cache file for the statistics (auto when empty)
LINKAGE="${LINKAGE:-average}"                     # agglomerative linkage
BUDGET_AWARE="${BUDGET_AWARE:-True}"              # count residual bases/codes toward the expert-memory
                                                  # budget |Theta_hat_E| <= tau|Theta_E| (extra merges spread over layers)
SALIENCY_TOP_TAU="${SALIENCY_TOP_TAU:-15.0}"     # singleton ratio rho, in %: the top-ceil(rho*N) experts per
                                                  # layer by saliency stay singletons (paper: rho = 0.15)
DESCRIPTOR_TOKENS="${DESCRIPTOR_TOKENS:-all}"    # tokens averaged by CLUSTERING=descriptor
                                                  #   all    = every calibration token (default)
                                                  #   routed = tokens routed to each expert

# ---------------------------------------------------------------------------
# 4. Base construction - Idea 1 Step 2 (saliency-weighted base)
# ---------------------------------------------------------------------------
BASE_WEIGHT="${BASE_WEIGHT:-saliency}"           # saliency | freq
                                                  #   saliency = alpha_i = (a_i+delta)/sum(a_j+delta)  (paper)
                                                  #   freq     = usage-frequency weighting (HC-SMoE-style)
DELTA="${DELTA:-1e-3}"                           # delta floor in alpha_i (Step 2)

# ---------------------------------------------------------------------------
# 5. Residual structure - Idea 2 (diagonal residual-code)
#    D_i ~= B_G * diag(z_i) * A_G
# ---------------------------------------------------------------------------
PREACT_STAR="${PREACT_STAR:-True}"               # Idea 2 on/off (False = no residual code)
RANK="${RANK:-128}"                              # r - A_G in R^{r x d_in}, B_G in R^{d_out x r}, z_i in R^r
ALPHA="${ALPHA:-2.0}"                            # residual scale
INIT="${INIT:-covariance}"                       # covariance | lora_zero_B | expert_svd_align
INIT_ALIGN="${INIT_ALIGN:-hungarian}"            # only used by expert_svd_align init
INIT_STD="${INIT_STD:-1e-3}"                     # only used by lora_zero_B / zero_L / zero_R inits
DOWN_INIT="${DOWN_INIT:-covariance}"             # down-proj residual init: covariance | zero_L | zero_R
TARGET="${TARGET:-gate_up_down}"                 # which projection branches get a residual
                                                  # gate | up | down | gate_up | gate_up_down

# Aggregation choice for the cluster forward (Idea 2 path)
#   postact (default) : POST-activation aggregation matching paper Eq.(4)
#                       exactly: for each cluster member j in C cap T(x), evaluate
#                       SwiGLU with W_hat_j = W_bar+B*diag(z_j)*A and take a
#                       pi_j-weighted sum. Costs Kx SwiGLU evaluations per
#                       cluster per token (memory/throughput scales with
#                       cluster size).
#   preact            : router-conditioned PRE-activation modulation.
#                       SwiGLU is evaluated once per cluster on the
#                       rho-aggregated pre-activations. Fast (1x SwiGLU
#                       per cluster per token), first-order approximation
#                       of paper Eq.(4) under nonlinear SwiGLU.
AGGREGATION="${AGGREGATION:-postact}"            # postact | preact
case "$AGGREGATION" in
  preact)  POSTACT_FLAG="--paper_postact_aggregation=False" ;;
  postact) POSTACT_FLAG="--paper_postact_aggregation=True"  ;;
  *) echo "[RESCODE] unknown AGGREGATION=$AGGREGATION (expected preact|postact)" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# 6. Residual training - Idea 3 (output-guided fitting)
#    L = lambda_mse * L_rec + lambda_lm * L_LM   (paper: L_rec + lambda L_LM, i.e. lambda_mse = 1, lambda = lambda_lm)
#    L_rec = 1/(|D| L) sum_x sum_l ||y_hat^l(x) - y^l(x)||^2  over every MoE layer
# ---------------------------------------------------------------------------
FIT_PREACT="${FIT_PREACT:-True}"                 # Idea 3 on/off (False = no LM-guided fitting)
JOINT_EPOCHS="${JOINT_EPOCHS:-3}"                # n_ep = 3 (paper App. E)
JOINT_GRAD_ACCUM="${JOINT_GRAD_ACCUM:-8}"        # with TRAIN_BATCH=1: 8 sequences per optimizer step
LAMBDA_LM="${LAMBDA_LM:-1.0}"                    # lambda - LM cross-entropy weight
LAMBDA_MSE="${LAMBDA_MSE:-1.0}"                  # weight of L_rec (0 = LM-only)
REC_MODE="${REC_MODE:-local}"                    # local      = original layer on the compressed
                                                  #              model's layer input (Sec. 4.4),
                                                  #              each layer fits its own residuals
                                                  # trajectory = original-model layer outputs (Alg. 1)
                                                  # cluster    = per-cluster MSE, online teacher (TEACHER_MODE)
                                                  # trajectory/local keep frozen copies of the merged
                                                  # experts only (no second model copy)
TEACHER_MODE="${TEACHER_MODE:-online}"           # REC_MODE=cluster only: online | offline
JOINT_VAL_N="${JOINT_VAL_N:-200}"
JOINT_VAL_EVERY="${JOINT_VAL_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-1}"

# ---------------------------------------------------------------------------
# 7. Optimizer / scheduler / regularization
# ---------------------------------------------------------------------------
LR="${LR:-1.5e-4}"                               # learning rate for A,B,z
LR_DOWN="${LR_DOWN:-$LR}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"           # constant | cosine | linear
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
L2="${L2:-0}"                                    # extra L2 on A,B,z params (0 = AdamW weight decay only, as in the paper)
L2_DOWN="${L2_DOWN:-$L2}"

# ---------------------------------------------------------------------------
# 8. Output / load-mode
# ---------------------------------------------------------------------------
RESULT_NAME="${RESULT_NAME:-result.txt}"
SAVE_MODEL="${SAVE_MODEL:-False}"                # also dump the full model state_dict (~16 GB for Qwen)

# Pick model family: detect from MODEL string
if [[ "$MODEL" == *Mixtral* || "$MODEL" == *mixtral* ]]; then
  MODEL_FAMILY="mixtral"
  PY_ENTRY="rescode/rescode_mixtral.py"
elif [[ "$MODEL" == *Qwen3* || "$MODEL" == *qwen3* ]]; then
  MODEL_FAMILY="qwen3"
  PY_ENTRY="rescode/rescode_qwen3.py"
elif [[ "$MODEL" == *eepSeek* || "$MODEL" == *deepseek* ]]; then
  MODEL_FAMILY="deepseek"
  PY_ENTRY="rescode/rescode_deepseek.py"
else
  MODEL_FAMILY="qwen"
  PY_ENTRY="rescode/rescode_qwen.py"
fi

OUT_SUFFIX="rescode-${MODEL_FAMILY}-${NUM_EXPERTS}e"
[ -n "$TAG" ] && OUT_SUFFIX="${OUT_SUFFIX}-${TAG}"
OUT="results/${MODEL_FAMILY}/${OUT_SUFFIX}"
mkdir -p "$OUT"
if [ -z "$CALIB_STATS_CACHE" ]; then
  CALIB_STATS_CACHE="results/${MODEL_FAMILY}/calib_stats/${CALIB_DATASET}-n${N_SENTENCES}-seed${CALIB_SEED}.pt"
fi

# Optional reload mode: LOAD=auto -> reuse $OUT/rescode_state.pt and skip fitting
LOAD="${LOAD:-}"
LOAD_ARGS=""
if [ -n "$LOAD" ]; then
  if [ "$LOAD" = "auto" ] || [ "$LOAD" = "1" ]; then
    LOAD_PATH="$OUT/rescode_state.pt"
  else
    LOAD_PATH="$LOAD"
  fi
  LOAD_ARGS="--load_rescode_path=$LOAD_PATH"
fi

# ---------------------------------------------------------------------------
# 0. Environment (HF cache, accelerate)
#    User-specific paths - override via env vars before running.
#    ACCELERATE: path to the accelerate executable inside your virtualenv
#                (e.g. $HOME/venvs/rescode/bin/accelerate)
#    HF_HOME   : HuggingFace cache (where models are downloaded)
# ---------------------------------------------------------------------------
ACC="${ACCELERATE:-accelerate}"          # 'accelerate' on PATH, or a virtualenv path
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE=0
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

FIRST_GPU="${GPU%%,*}"                   # GPU may list several ids (e.g. "0,1") for large models
PORT="${PORT:-$((30000 + 10#$FIRST_GPU * 100 + NUM_EXPERTS))}"

# warmup_steps = 5 % of optimizer steps
BATCHES_PER_EPOCH=$(( (N_SENTENCES + TRAIN_BATCH - 1) / TRAIN_BATCH ))
EST_OPT_STEPS=$(( (BATCHES_PER_EPOCH * JOINT_EPOCHS + JOINT_GRAD_ACCUM - 1) / JOINT_GRAD_ACCUM ))
WARMUP_STEPS="${WARMUP_STEPS:-$((EST_OPT_STEPS / 20))}"
[ "$WARMUP_STEPS" -lt 1 ] && WARMUP_STEPS=1

# ---------------------------------------------------------------------------
# Summary echo
# ---------------------------------------------------------------------------
echo "[RESCODE] ============================================================"
echo "[RESCODE]   model=$MODEL  family=$MODEL_FAMILY"
echo "[RESCODE]   num_experts=$NUM_EXPERTS  gpu=$GPU  out=$OUT"
echo "[RESCODE]   calib_dataset=$CALIB_DATASET  train_data=$TRAIN_DATA"
echo "[RESCODE]   ----- Idea 1 (saliency-aware restoration) -----"
echo "[RESCODE]   clustering=$CLUSTERING  linkage=$LINKAGE  budget_aware=$BUDGET_AWARE  saliency_top_tau=$SALIENCY_TOP_TAU  base_weight=$BASE_WEIGHT  delta=$DELTA"
echo "[RESCODE]   ----- Idea 2 (diagonal residual-code) ---------"
echo "[RESCODE]   preact_star=$PREACT_STAR  rank=$RANK  alpha=$ALPHA  init=$INIT  down_init=$DOWN_INIT  target=$TARGET  aggregation=$AGGREGATION"
echo "[RESCODE]   ----- Idea 3 (output-guided fitting) ----------"
echo "[RESCODE]   fit_preact=$FIT_PREACT  epochs=$JOINT_EPOCHS  grad_accum=$JOINT_GRAD_ACCUM  est_opt_steps=$EST_OPT_STEPS"
echo "[RESCODE]   lambda_lm=$LAMBDA_LM  lambda_mse=$LAMBDA_MSE  rec_mode=$REC_MODE  train_batch=$TRAIN_BATCH"
echo "[RESCODE]   lr=$LR  sched=$LR_SCHEDULER  warmup=$WARMUP_STEPS  wd=$WEIGHT_DECAY  l2=$L2"
echo "[RESCODE] ============================================================"

# ---------------------------------------------------------------------------
# Launch
# Internal Python flags keep their own names; user-facing knobs are
# remapped above.
# ---------------------------------------------------------------------------
"$ACC" launch --config_file static/single_gpu.yaml \
  --gpu_ids "$GPU" \
  --main_process_port "$PORT" "$PY_ENTRY" \
  --model_name="$MODEL" \
  --task="$TASK" \
  --seed="$SEED" \
  --calib_seed="$CALIB_SEED" \
  --train_data_path="$TRAIN_DATA" \
  --val_data_path="$VAL_DATA" \
  --n_sentences="$N_SENTENCES" \
  --cluster_n_sentences="$CLUSTER_N_SENTENCES" \
  --train_batch_size="$TRAIN_BATCH" \
  --eval_batch_size="$EVAL_BATCH" \
  --max_fit_tokens_per_layer="$MAX_TOKENS_PER_LAYER" \
  --max_fit_tokens_per_cluster="$MAX_TOKENS_PER_CLUSTER" \
  --num_average_groups="$NUM_EXPERTS" \
  --exclude_top_percent="$SALIENCY_TOP_TAU" \
  --cluster=hierarchical \
  --linkage="$LINKAGE" \
  --clustering="$CLUSTERING" \
  --calib_stats_cache="$CALIB_STATS_CACHE" \
  --budget_aware="$BUDGET_AWARE" \
  --descriptor_tokens="$DESCRIPTOR_TOKENS" \
  --hc_merge="$BASE_WEIGHT" \
  --rescode_delta="$DELTA" \
  --preact_star=$PREACT_STAR \
  --preact_mode=token_adaptive_lowrank \
  --preact_lowrank_target="$TARGET" \
  --preact_lowrank_rank="$RANK" \
  --preact_lowrank_alpha="$ALPHA" \
  --preact_lowrank_init="$INIT" \
  --preact_lowrank_init_alignment="$INIT_ALIGN" \
  --preact_lowrank_init_std="$INIT_STD" \
  --preact_lowrank_dual=False \
  $POSTACT_FLAG \
  --preact_down_residual_rank="$RANK" \
  --preact_down_residual_alpha="$ALPHA" \
  --preact_down_residual_init="$DOWN_INIT" \
  --preact_lowrank_l2_lambda="$L2" \
  --preact_down_l2_lambda="$L2_DOWN" \
  --preact_normalize_by_std=False \
  --preact_grad_clip_norm="$GRAD_CLIP" \
  --preact_lr_lowrank="$LR" \
  --preact_lr_down="$LR_DOWN" \
  --code_type=none \
  --fit_codes=False \
  --fit_preact=$FIT_PREACT \
  --fit_mode=joint \
  --teacher_mode="$TEACHER_MODE" \
  --rec_mode="$REC_MODE" \
  --joint_epochs="$JOINT_EPOCHS" \
  --joint_grad_accum="$JOINT_GRAD_ACCUM" \
  --lambda_lm="$LAMBDA_LM" \
  --lambda_mse="$LAMBDA_MSE" \
  --joint_val_n_sentences="$JOINT_VAL_N" \
  --joint_val_every="$JOINT_VAL_EVERY" \
  --lr_scheduler="$LR_SCHEDULER" \
  --warmup_steps="$WARMUP_STEPS" \
  --fit_weight_decay="$WEIGHT_DECAY" \
  --log_every="$LOG_EVERY" \
  --do_eval=True \
  --eval_limit="$EVAL_LIMIT" \
  --save_model="$SAVE_MODEL" \
  --output_path="$OUT" \
  --save_rescode_path="$OUT/rescode_state.pt" \
  --result_path="$OUT/$RESULT_NAME" \
  $LOAD_ARGS \
  |& tee "$OUT/log.txt"
