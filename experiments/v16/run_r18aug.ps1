# V16: ResNet18 refiner with augmentation ON, same 2-stage recipe as V12 base -> V13 hard, then PQ sweep vs hard_tta.
# Usage: run_resnet34.ps1 [-From 2]   (resume from a step; earlier outputs must exist)
# No ErrorActionPreference=Stop: PS 5.1 turns native stderr (tqdm bars) into terminating errors.
# Failures are caught via $LASTEXITCODE instead.
param([int]$From = 1)
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$py = ".\.venv\Scripts\python.exe"

function Step($n, $msg) { Write-Host "`n=== [$n/4] $msg ===" -ForegroundColor Cyan }

if ($From -le 1) {
Step 1 "Train ResNet18-aug base on v12 crops (<=40 epochs, patience 12)"
& $py -u scripts\train_local_refiner.py --config experiments\v16\configs\r18aug_refiner.yaml
if ($LASTEXITCODE) { throw "base training failed" }
}

Step 2 "Mine hard train crops (refine IoU 0.30-0.50, x3)"
& $py -u scripts\v13_mine_hard_refine_crops.py --config experiments\v16\configs\r18aug_refiner_hard.yaml `
    --checkpoint experiments\v16\r18aug_refiner\best.pt --src-crops experiments\v12\crops `
    --out-crops experiments\v16\r18aug_hard\crops
if ($LASTEXITCODE) { throw "hard mining failed" }

Step 3 "Hard fine-tune (lr 5e-5, <=20 epochs, patience 8)"
& $py -u scripts\train_local_refiner.py --config experiments\v16\configs\r18aug_refiner_hard.yaml `
    --init-weights experiments\v16\r18aug_refiner\best.pt
if ($LASTEXITCODE) { throw "hard fine-tune failed" }

Step 4 "PQ sweep: hard_tta vs r18aug_tta vs r18aughard_tta, single and both YOLO"
& $py -u scripts\v13_inference_sweep.py --refiner hard_tta,r18aug_tta,r18aughard_tta --conf 0.35 --thr 0.50 `
    --supp-iou 0.25 --score conf_prob --det single,both `
    --yolo-weights experiments/v12/yolo11s/weights/best.pt --yolo-weights2 experiments/v14/yolo11s/weights/best.pt `
    --out experiments/v16/sweep_r18aug
if ($LASTEXITCODE) { throw "sweep failed" }

Write-Host "`nDONE" -ForegroundColor Green

