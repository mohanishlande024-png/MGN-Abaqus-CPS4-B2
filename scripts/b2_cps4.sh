#!/usr/bin/env bash
# Run B2 (Kt target) on the CPS4 SI dataset - every step is one short command (Ubuntu, from the repo folder):
#
#   bash scripts/b2_cps4.sh quick       20-epoch timing run -> results_si_cps4_b2_quick/ (delete it afterwards)
#   bash scripts/b2_cps4.sh train       the full 10000-epoch run in the background (safe to close SSH)
#   EPOCHS=4000 bash scripts/b2_cps4.sh train     a shorter run (B1's best validation epoch was 3050)
#   bash scripts/b2_cps4.sh status      is it running? last lines of both logs
#   bash scripts/b2_cps4.sh curves      R2 vs epoch plot, train + validation (works while training runs)
#   bash scripts/b2_cps4.sh resume      continue a stopped run, in the background
#   bash scripts/b2_cps4.sh eval        the three tests, 8 edge draws, high-res contours + scatter plots
#   bash scripts/b2_cps4.sh fitcheck    over/under-fitting verdict + plots (after eval)
#   bash scripts/b2_cps4.sh eval-b1     the same new plots and NRMSE column for your finished B1 run
#   bash scripts/b2_cps4.sh fitcheck-b1 fit check of B1 with the new plots (after eval-b1)
#
# B2 uses B1's training graphs (data/si_cps4/dataset_si_cps4_aug20.pt) - no rebuild.
# Activate the environment first:   source ~/mgn/.venv/bin/activate

set -e
cd "$(dirname "$0")/.."

D=data/si_cps4
DATA=$D/dataset_si_cps4_aug20.pt
RES=results_si_cps4_b2
CKPT=$RES/mgn_best_val.pt
EPOCHS=${EPOCHS:-10000}
B1CKPT=results_si_cps4_b1/mgn_best_val.pt
TESTS="--train-data $DATA --loads-dir $D/unseen_loads --interp-dir $D/unseen_interp --extrap-dir $D/unseen_extrap --manifest $D/si_manifest.csv --units MPa --draws 8"

if [ -z "$1" ] || [ "$1" = "help" ]; then sed -n '2,16p' "$0"; exit 0; fi
if ! python -c "import torch" 2>/dev/null; then
    echo "python cannot import torch - activate the environment first:  source ~/mgn/.venv/bin/activate"
    exit 1
fi
if [ ! -f "$DATA" ]; then
    echo "No $DATA - build it once (the same graphs as B1):  python -m mgn.dataset --raw $D/train --aug-perc 0.2 --out $DATA"
    exit 1
fi
PAT='^[^ ]*python[^ ]* (-u )?scripts/train_b2_val\.py'          # a python process running the B2 trainer
running() { pgrep -f "$PAT" > /dev/null; }

case "$1" in
  quick)
    python scripts/train_b2_val.py --data $DATA --val-dir $D/val --epochs 20 --eval-every 10 --results ${RES}_quick ;;
  train|resume)
    if running; then echo "A B2 training run is already going (bash scripts/b2_cps4.sh status). Not starting a second one."; exit 1; fi
    mkdir -p $RES
    EXTRA=""; [ "$1" = "resume" ] && EXTRA="--resume"
    nohup python -u scripts/train_b2_val.py --data $DATA --val-dir $D/val --results $RES --epochs $EPOCHS $EXTRA >> $RES/train.out 2>&1 &
    sleep 5
    if running; then echo "started (PID $!). Output: $RES/train.out   Check with: bash scripts/b2_cps4.sh status";
    else echo "it stopped straight away - the error:"; tail -n 20 $RES/train.out; exit 1; fi ;;
  status)
    if running; then echo "RUNNING:"; pgrep -af "$PAT"; else echo "not running"; fi
    echo "--- last epochs: epoch, loss, seconds"
    if [ -f $RES/loss_log.csv ]; then tail -n 3 $RES/loss_log.csv; else echo "none yet"; fi
    echo "--- last R2 check: epoch, train R2, validation R2"
    if [ -f $RES/r2_log.csv ]; then tail -n 2 $RES/r2_log.csv | cut -d, -f1,3,8; else echo "none yet"; fi ;;
  curves)
    python scripts/check_fit.py --curves-only --ckpt $CKPT
    echo "plot: $RES/fit_check/r2_vs_epoch.png" ;;
  eval)
    python scripts/eval_unseen_v2.py --ckpt $CKPT $TESTS ;;
  fitcheck)
    python scripts/check_fit.py --ckpt $CKPT --train-data $DATA --val-dir $D/val --draws 8 ;;
  eval-b1)
    python scripts/eval_unseen_v2.py --ckpt $B1CKPT $TESTS ;;
  fitcheck-b1)
    python scripts/check_fit.py --ckpt $B1CKPT --train-data $DATA --val-dir $D/val --draws 8 ;;
  *)
    echo "unknown step '$1'"; sed -n '2,16p' "$0"; exit 1 ;;
esac
