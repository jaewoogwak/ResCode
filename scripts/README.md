# Launch scripts

Per-experiment shell scripts live under `scripts/RESCODE/`. Every script
just sets env vars and `exec`s the shared launcher `scripts/RESCODE/run.sh`,
which then calls one of the Python entry points in `rescode/`.

| Script                          | Model                          | Compression |
| ------------------------------- | ------------------------------ | ----------- |
| `run_mixtral_4x.sh`             | Mixtral-8x7B-v0.1              | 8 → 4 (50%) |
| `run_mixtral_6x.sh`             | Mixtral-8x7B-v0.1              | 8 → 6 (25%) |
| `run_qwen_30e.sh`               | Qwen1.5-MoE-A2.7B-Chat         | 60 → 30 (50%) |
| `run_qwen_45e.sh`               | Qwen1.5-MoE-A2.7B-Chat         | 60 → 45 (25%) |
| `run_dsmoe16b_32e.sh`           | DeepSeek-MoE 16B               | 64 → 32 (50%) |
| `prepare_calib_datasets.py`     | helper — downloads + formats a calibration corpus |

All knobs are documented inline in `run.sh`. Common overrides:

```bash
GPU=0 NUM_EXPERTS=4 TAG=run01 bash scripts/RESCODE/run_mixtral_4x.sh
CALIB_DATASET=c4 bash scripts/RESCODE/run_mixtral_4x.sh   # calibrate on C4 instead of DCLM (default)
LOAD=auto bash scripts/RESCODE/run_mixtral_4x.sh   # reuse pre-fitted residuals
```

See the top-level [README.md](../README.md) for the full pipeline.
