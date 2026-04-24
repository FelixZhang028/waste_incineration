# waste_incineration
End-to-end MTSC pipeline for incinerator operation-state classification.
=======
﻿# MTSC Preprocess

## Run

```bash
python run_preprocess.py --config configs/preprocess.json
```

Current `configs/preprocess.json` is configured as:
- train source: `2024-05` (full month)
- test source: `2026-01` (full month)
- split strategy: `source_holdout`
- transition buffer: `10` minutes
- drop label: `故障`
- temporal features: diff + rolling stats
- window samples: enabled (`window_index_train/test.csv`)

## Full-train mode

```bash
python run_preprocess.py --config configs/preprocess.json --mode full
```

Outputs are written into `data/processed` including train/val/test datasets, feature list, label map, scaler stats, quality report, run manifest, and window sample files.

## Train

```bash
python run_train.py --config configs/train.json
```

If your `torch` is installed in conda env `DL`, run training via:

```powershell
conda run -n DL python run_train.py --config configs/train.json
```

Switching model only requires changing `model.name` and `model.params` in `configs/train.json`.
Currently supported models:
- `lstm`
- `gru`

Useful debug command (small subset):

```bash
python run_train.py --config configs/train.json --max-train-windows 20000 --max-eval-windows 5000
```

Resume from checkpoint:

```bash
python run_train.py --config configs/train.json --resume artifacts/train/last.pt
```

## Evaluate (Raw vs Rule)

Raw model output only:

```bash
python scripts/eval_with_rules.py --config configs/train.json --checkpoint artifacts/train_lstm/best.pt --split test --out-dir artifacts/eval_lstm
```

Apply optional rule postprocess (`configs/post_rules.json`):

```bash
python scripts/eval_with_rules.py --config configs/train.json --checkpoint artifacts/train_lstm/best.pt --split test --with-rules --rules-config configs/post_rules.json --out-dir artifacts/eval_lstm_rule
```

