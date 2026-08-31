# Calibration data

The compression pipeline needs a small calibration corpus (a few thousand
short documents) and an optional validation corpus. The corpora are **not**
shipped with this repo — download them yourself and place the files here, or
override the path via the `TRAIN_DATA` / `VAL_DATA` environment variables when
launching the shell scripts.

The shell launcher (`scripts/RESCODE/run.sh`) selects a calibration corpus
via `CALIB_DATASET=c4|wikipedia|slimpajama|dclm|starcoder`. Each name maps to
one file in this directory:

| `CALIB_DATASET` | Expected file                                    |
| --------------- | ------------------------------------------------ |
| `c4`            | `c4-train.00000-of-01024.json`                   |
| `wikipedia`     | `wikipedia-calib.jsonl`                          |
| `slimpajama`    | `slimpajama-calib.jsonl`                         |
| `dclm` (default)| `dclm-calib.jsonl`                               |
| `starcoder`     | `starcoder-calib.jsonl`                          |

## C4

Download one shard of the English C4 stream from
[allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz),
gunzip it, and place it as `rescode/data/c4-train.00000-of-01024.json`.

## Other corpora

Use the helper to download & format any of the alternative calibration
corpora:

```bash
python scripts/RESCODE/prepare_calib_datasets.py --dataset wikipedia
python scripts/RESCODE/prepare_calib_datasets.py --dataset slimpajama
python scripts/RESCODE/prepare_calib_datasets.py --dataset dclm
python scripts/RESCODE/prepare_calib_datasets.py --dataset starcoder
```

Each call writes one `*-calib.jsonl` file directly into this directory.

## Validation corpus (optional)

Joint LM-loss fitting can use a held-out validation file (`VAL_DATA` env
var). It is independent of `CALIB_DATASET`. If you do not provide one,
joint validation is skipped.

```
VAL_DATA=PATH/TO/YOUR/val-calib.json bash scripts/RESCODE/run_mixtral_4x.sh
```
