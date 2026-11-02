#!/usr/bin/env bash
# Recreate the tested ResCode Python environment on a Linux x86_64 GPU instance.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_dir="${RESCODE_ENV_DIR:-${repo_dir}/.venv}"
if [[ "${env_dir}" != /* ]]; then
    env_dir="${repo_dir}/${env_dir}"
fi
lock_file="${repo_dir}/requirements-rescode.lock"
uv_version="0.13.4"
python_version="3.10.22"
pytorch_index="https://download.pytorch.org/whl/cu128"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    echo "setup.sh requires Linux x86_64 (the platform of the pinned CUDA wheels)." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "setup.sh requires an NVIDIA GPU driver and nvidia-smi." >&2
    exit 1
fi

uv_bin="$(command -v uv || true)"
if [[ -z "${uv_bin}" || "$("${uv_bin}" --version | cut -d ' ' -f 2)" != "${uv_version}" ]]; then
    uv_dir="${XDG_DATA_HOME:-${HOME}/.local/share}/rescode-setup/uv-${uv_version}"
    uv_bin="${uv_dir}/uv"
    if [[ ! -x "${uv_bin}" ]]; then
        if ! command -v curl >/dev/null 2>&1; then
            echo "curl is needed to install uv ${uv_version}." >&2
            exit 1
        fi
        installer="$(mktemp)"
        trap 'rm -f "${installer}"' EXIT
        curl -fsSL "https://astral.sh/uv/${uv_version}/install.sh" -o "${installer}"
        UV_UNMANAGED_INSTALL="${uv_dir}" sh "${installer}"
        rm -f "${installer}"
        trap - EXIT
    fi
fi
if [[ "$("${uv_bin}" --version | cut -d ' ' -f 2)" != "${uv_version}" ]]; then
    echo "Expected uv ${uv_version} at ${uv_bin}." >&2
    exit 1
fi

if [[ -e "${env_dir}" && ! -f "${env_dir}/pyvenv.cfg" ]]; then
    echo "${env_dir} exists but is not a virtual environment; choose RESCODE_ENV_DIR." >&2
    exit 1
fi
if [[ ! -e "${env_dir}/bin/python" ]]; then
    "${uv_bin}" venv --no-config --managed-python --python "${python_version}" \
        --seed --prompt rescode "${env_dir}"
fi
actual_python="$("${env_dir}/bin/python" -c 'import platform; print(platform.python_version())')"
if [[ "${actual_python}" != "${python_version}" ]]; then
    echo "${env_dir} has Python ${actual_python}; expected ${python_version}." >&2
    exit 1
fi

"${uv_bin}" pip sync --no-config --python "${env_dir}/bin/python" --strict \
    --default-index https://pypi.org/simple --index "${pytorch_index}" \
    --index-strategy unsafe-best-match "${lock_file}"
"${env_dir}/bin/python" -m pip check

"${env_dir}/bin/python" - "${lock_file}" <<'PY'
import importlib.metadata as metadata
import pathlib
import sys

for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    name, expected = line.split("==", 1)
    actual = metadata.version(name)
    if actual != expected:
        raise RuntimeError(f"{name}: installed {actual}, expected {expected}")

import accelerate
import datasets
import lm_eval
import peft
import torch
import transformers
import triton

print(f"Python {sys.version.split()[0]}, torch {torch.__version__}, CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access an NVIDIA GPU on this instance")
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
print(f"GPU check passed: {torch.cuda.get_device_name(0)}")
PY

echo "ResCode environment ready: ${env_dir}"
echo "Activate it with: source \"${env_dir}/bin/activate\""
