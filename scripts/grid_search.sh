#!/usr/bin/env bash
# scripts/grid_search.sh
# Grid search for speed/accuracy on MNIST

# ---------- strict mode (toggle with GRID_DEBUG=1) ----------
if [[ "${GRID_DEBUG:-0}" == "1" ]]; then
  set -euxo pipefail
else
  set -euo pipefail
fi

# ---------- paths ----------
IMG_TR="data/MNIST/raw/train-images-idx3-ubyte"
LBL_TR="data/MNIST/raw/train-labels-idx1-ubyte"
IMG_TE="data/MNIST/raw/t10k-images-idx3-ubyte"
LBL_TE="data/MNIST/raw/t10k-labels-idx1-ubyte"

# subset for iteration; tune as needed; max 60000 (full MNIST)
LIMIT_TR=${LIMIT_TR:-10000}
VAL_FRAC=${VAL_FRAC:-0.1}

# ---------- sweeps ----------
THREADS_LIST=(${THREADS_LIST:-8})

MODEL_LIST=(
  "1  --units 64"
  "1  --units 128"
  "1  --units 256"
  "2  --units 64,32"
  "2  --units 128,32"
  "2  --units 256,32"
  "2  --units 128,64"
  "2  --units 256,64"
)

BATCH_LIST=(${BATCH_LIST:-128})
LR_LIST=(${LR_LIST:-0.05 0.075 0.1 0.125 0.15})
MOM_LIST=(${MOM_LIST:-0.9 0.92 0.94 0.96})
DECAY_LIST=(${DECAY_LIST:-0.9 0.95 0.99})
STEP_LIST=(${STEP_LIST:-4})


EPOCHS=${EPOCHS:-60}
SEED=${SEED:-1337}
PATIENCE=${PATIENCE:-10}

# ---------- tools ----------
PERCEPTRON_BIN="./perceptron"
[[ -x "$PERCEPTRON_BIN" ]] || { echo "[grid][ERR] $PERCEPTRON_BIN not found or not executable"; exit 2; }

# Optional stdbuf to keep line-buffered logs; fall back to no-op if missing
if command -v stdbuf >/dev/null 2>&1; then
  STDBUF=(stdbuf -oL -eL)
else
  STDBUF=()
fi

# ---------- checks ----------
for f in "$IMG_TR" "$LBL_TR" "$IMG_TE" "$LBL_TE"; do
  [[ -f "$f" ]] || { echo "[grid][ERR] missing MNIST file: $f"; exit 2; }
done

[[ ${#THREADS_LIST[@]} -gt 0 ]] || { echo "[grid][ERR] THREADS_LIST empty"; exit 2; }
[[ ${#MODEL_LIST[@]}   -gt 0 ]] || { echo "[grid][ERR] MODEL_LIST empty"; exit 2; }

echo "[grid] starting grid search"
echo "[grid] perceptron: $PERCEPTRON_BIN"
echo "[grid] train imgs: $IMG_TR"
echo "[grid] train lbls: $LBL_TR"
echo "[grid] test  imgs: $IMG_TE"
echo "[grid] test  lbls: $LBL_TE"
echo "[grid] limits: limit_tr=$LIMIT_TR val_frac=$VAL_FRAC"
echo "[grid] sweeps: threads=${THREADS_LIST[*]} batches=${BATCH_LIST[*]} lr=${LR_LIST[*]} mom=${MOM_LIST[*]} decay=${DECAY_LIST[*]} step=${STEP_LIST[*]}"
echo "[grid] models: ${#MODEL_LIST[@]} configs"
echo

# ---------- output dirs ----------
ts="$(date +%Y%m%d-%H%M%S)"
OUTDIR="runs/grid_${ts}"
LOGDIR="${OUTDIR}/logs"
MODELDIR="${OUTDIR}/models"
PLOTDIR="${OUTDIR}/plots"
mkdir -p "$OUTDIR" "$LOGDIR" "$MODELDIR" "$PLOTDIR"

CSV="${OUTDIR}/results.csv"
echo "model_key,threads,layers,units,batch,lr,momentum,decay,step,epochs,train_time_s,best_val_pct,test_acc_pct,model_path,log_path" > "$CSV"
echo "[grid] results -> $CSV"
echo

# ---------- helpers ----------
parse_train_log () {
  local log="$1"
  # total time (e.g., "[train] total time: 21.0s")
  local total
  total="$(grep -Eo '\[train\] total time: *[0-9.]+s' "$log" | awk '{print $(NF)}' | sed 's/s//')" || true
  # best val (or last val)
  local bestv
  bestv="$(grep -Eo 'Best=[0-9.]+%' "$log" | tail -n1 | sed 's/Best=//; s/%//')" || true
  if [[ -z "${bestv:-}" ]]; then
    bestv="$(grep -Eo 'val=[0-9.]+%' "$log" | tail -n1 | sed 's/val=//; s/%//')" || true
  fi
  echo "${total:-NA},${bestv:-NA}"
}

parse_eval_log () {
  local log="$1"
  local acc=""

  # 1) Canonical: "[eval] acc=98.12%"
  acc="$(grep -Eo '\[eval\][^%]*acc=[0-9.]+%' "$log" 2>/dev/null | tail -n1 | sed -E 's/.*acc=([0-9.]+)%.*/\1/')" || true

  # 2) Fallbacks that might appear in other builds:
  #    "acc=98.12%" anywhere, but prefer lines containing eval/test/accuracy
  if [[ -z "$acc" ]]; then
    acc="$(grep -Ei 'eval|test|accuracy' "$log" 2>/dev/null \
        | grep -Eo 'acc(uracy)?=[0-9.]+%' \
        | tail -n1 | sed -E 's/.*=([0-9.]+)%.*/\1/')" || true
  fi

  # 3) Another fallback used in some logs: "test=98.12%"
  if [[ -z "$acc" ]]; then
    acc="$(grep -Eo 'test=[0-9.]+%' "$log" 2>/dev/null | tail -n1 | sed -E 's/test=([0-9.]+)%.*/\1/')" || true
  fi

  # 4) Last resort: any percentage on a line with "eval" or "test"
  if [[ -z "$acc" ]]; then
    acc="$(grep -Ei 'eval|test' "$log" 2>/dev/null \
        | grep -Eo '[0-9]+\.[0-9]+%' \
        | tail -n1 | sed -E 's/([0-9.]+)%.*/\1/')" || true
  fi

  [[ -n "$acc" ]] && echo "$acc" || echo "NA"
}



# ---------- main loops ----------
i=0
for th in "${THREADS_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    # split "L --units a,b,c"
    LAYERS="$(awk '{print $1}' <<<"$model")"
    UNITS="$(sed -E 's/^[0-9]+\s+--units\s+//' <<<"$model")"

    if [[ -z "${LAYERS}" || -z "${UNITS}" ]]; then
      echo "[grid][WARN] bad model entry: '$model' (skipping)"
      continue
    fi

    for bs in "${BATCH_LIST[@]}"; do
      for lr in "${LR_LIST[@]}"; do
        for mom in "${MOM_LIST[@]}"; do
          for decay in "${DECAY_LIST[@]}"; do
            for step in "${STEP_LIST[@]}"; do
              ((++i))
              key="m${i}_L${LAYERS}_U${UNITS//,/x}_B${bs}_lr${lr}_m${mom}_d${decay}_s${step}_T${th}"

              model_path="${MODELDIR}/${key}.bin"
              train_log="${LOGDIR}/${key}.train.log"

              echo "====[ ${key} ]===="
              echo "[grid] training..."
              # Allow failures within the pipeline; capture rc reliably
              set +e
              "${STDBUF[@]}" "$PERCEPTRON_BIN" train \
                --dataset mnist \
                --mnist-images "$IMG_TR" \
                --mnist-labels "$LBL_TR" \
                --limit "$LIMIT_TR" --val "$VAL_FRAC" \
                --layers "$LAYERS" --units "$UNITS" \
                --epochs "$EPOCHS" --threads "$th" \
                --lr "$lr" --batch "$bs" --momentum "$mom" \
                --lr-decay "$decay" --lr-step "$step" \
                --patience "$PATIENCE" \
                --seed "$SEED" \
                --out "$model_path" \
                | tee "$train_log"
              tr_rc=${PIPESTATUS[0]}
              set -e

              if [[ $tr_rc -ne 0 ]]; then
                echo "[grid][WARN] train failed (rc=$tr_rc); skipping eval."
                echo "${key},${th},${LAYERS},\"${UNITS}\",${bs},${lr},${mom},${decay},${step},${EPOCHS},NA,NA,NA,${model_path},${train_log}" >> "$CSV"
                echo
                continue
              fi

              echo "[grid] evaluating..."
              set +e
              "${STDBUF[@]}" "$PERCEPTRON_BIN" eval \
                --model "$model_path" \
                --dataset mnist \
                --mnist-images "$IMG_TE" \
                --mnist-labels "$LBL_TE" \
                --threads "$th" \
                | tee -a "$train_log"
              ev_rc=${PIPESTATUS[0]}
              set -e

              IFS=',' read -r total_time best_val <<<"$(parse_train_log "$train_log")"
              test_acc="NA"
              if [[ $ev_rc -eq 0 ]]; then
                test_acc="$(parse_eval_log "$train_log" || echo NA)"
              else
                echo "[grid][WARN] eval failed (rc=$ev_rc)"
              fi

              echo "[grid] result: time=${total_time}s best_val=${best_val}% test=${test_acc}%"
              echo "${key},${th},${LAYERS},\"${UNITS}\",${bs},${lr},${mom},${decay},${step},${EPOCHS},${total_time},${best_val},${test_acc},${model_path},${train_log}" >> "$CSV"
              echo
            done
          done
        done
      done
    done
  done
done

echo "[grid] Done."
echo "[grid] Results CSV: ${CSV}"
echo "[grid] Logs: ${LOGDIR}"
echo "[grid] Models: ${MODELDIR}"
