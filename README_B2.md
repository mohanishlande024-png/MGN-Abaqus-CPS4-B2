# Run B2 - Kt target (B1 with one change)

B2 is B1 with **one** change: the network learns **Kt = von Mises / load** and predicts
**stress = load x Kt**. Graph, anchors, random edges, model size, learning rate and training
step are B1's. It uses B1's training graphs (`data/si_cps4/dataset_si_cps4_aug20.pt`), so
nothing needs rebuilding and no new Abaqus runs are needed.

The R2 formula does not change: R2 = 1 - sum(FE - MGN)^2 / sum(FE - mean FE)^2, per file, on
the stress. For one file, R2 on stress equals R2 on Kt (load^2 cancels), so B1 and B2 are
scored the same way. B2 changes the model, not the score. Every file now also gets
**NRMSE = RMSE / load (%)**.

## Files

| File | What | B1 affected? |
|---|---|---|
| `mgn/kt_trainer.py` | NEW - `KtMGN`, the B2 model (subclass of B1's `AnchorMGN`) | no |
| `scripts/train_b2_val.py` | NEW - B2 training with validation (copy of `train_b1_val.py`, target = Kt) | no |
| `scripts/b2_cps4.sh` | NEW - one-word steps: quick, train, status, curves, resume, eval, fitcheck, eval-b1, fitcheck-b1 | no |
| `scripts/eval_unseen_v2.py` | NEW - evaluation for B1 **and** B2: high-res contours, Arial, scatter plots separately and together, NRMSE column | no |
| `scripts/plot_style.py` | NEW - Arial + 300 dpi for every figure | no |
| `scripts/check_fit.py` | **REPLACES** the old one - reads B1 and B2 checkpoints, Arial, 300 dpi, fixed the misleading "(a) loss: still falling = under-trained" title | still works for B1 |

Nothing else is touched: `trainer.py`, `anchor_trainer.py`, `train_b1_val.py`,
`eval_own_unseen.py` and `b1_cps4.sh` stay as they are, so the B1 results stay reproducible.

**Evaluate B2 only with `eval_unseen_v2.py` / `check_fit.py` (or `b2_cps4.sh`).** The old
`eval_own_unseen.py` does not know the Kt target and would read a B2 model as B1.

## Steps (Ubuntu, from the repo folder, environment active)

    bash scripts/b2_cps4.sh quick
    bash scripts/b2_cps4.sh train
    bash scripts/b2_cps4.sh status
    bash scripts/b2_cps4.sh curves
    bash scripts/b2_cps4.sh eval
    bash scripts/b2_cps4.sh fitcheck

The same new plots for the finished B1 run (for a side-by-side comparison):

    bash scripts/b2_cps4.sh eval-b1
    bash scripts/b2_cps4.sh fitcheck-b1

Time: B1 took 36.9 h for 10000 epochs (its best validation epoch, 3050, came after 11.2 h).
A shorter B2 run: `EPOCHS=4000 bash scripts/b2_cps4.sh train` (about 15 h). If the validation
curve is still rising at the end, continue: `EPOCHS=10000 bash scripts/b2_cps4.sh resume`.

Optional second ablation - the network cannot see the load at all:

    python scripts/train_b2_val.py --no-load-input --results results_si_cps4_b2_noload

## What `eval` writes  (`results_si_cps4_b2/unseen_eval_v2_draws8/`)

- `metrics_all.csv` - R2, RMSE, MAE, max error, peak truth / prediction, peak error %, **NRMSE %**
- `summary_unseen_loads.png`, `summary_r2_vs_load.png`, `summary_unseen_geometries_heatmap.png`
- `contours/<set>/<geometry>__<load>_plate.png` - whole plate, FE / MGN / error stacked
- `contours/<set>/<geometry>__<load>_zoom.png` - zoom on the hole, FE / MGN / error side by side
- `scatter/<set>/<geometry>__<load>.png` - **separate** parity plot per file
- `scatter_grid_<group>_<load>.png` - **together**: one panel per geometry
- `scatter_overlay_<group>_<load>.png` - **together**: all geometries on one axes, coloured by family

All PNGs are 300 dpi. Options: `--dpi 600`, `--formats png,pdf` (vector PDF, fonts embedded),
`--contour-loads all`, `--contour-sets interp,extrap`, `--scatter-loads 60`, `--overlay-log`.
For these, run `python scripts/eval_unseen_v2.py --ckpt results_si_cps4_b2/mgn_best_val.pt --draws 8 <options>`.

## Arial on Ubuntu

Arial is not installed on Ubuntu by default (the scripts fall back to Liberation Sans, which
has the same letter widths, and print a note). To get real Arial, either:

    sudo apt install ttf-mscorefonts-installer

or copy it from Windows (Windows CMD, after `mkdir -p ~/.local/share/fonts` on Ubuntu):

    scp C:\Windows\Fonts\arial*.ttf USER@HOST:~/.local/share/fonts/

The scripts find the font by themselves; no cache clearing is needed. The `eval` printout
shows which font was used (`figures   font Arial, 300 dpi, png`).

## Tested

- `KtMGN`: prediction = load x Kt exactly; with `--no-load-input` Kt is identical at every load;
  save / load round trip; a B1 checkpoint is refused when loaded as B2.
- `train_b2_val.py`: a 2-epoch run on the full 540 + 80 file dataset (small model) runs, logs and saves.
- `eval_unseen_v2.py` on your **B1** `mgn_best_val.pt` + Abaqus files reproduces your
  numbers exactly (P3 @ 60 MPa R2 0.8039, R_H8 0.7203 / 0.9684 / 0.9537, P7 0.9912).
- `check_fit.py` runs on a B2 checkpoint (train R2 equals the value logged by training).
