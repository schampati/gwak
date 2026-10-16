#!/usr/bin/env bash
# Run inside an environment with the GWAK training dependencies and visible data/GPU.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: bash $0 DATA_DIR OUTPUT_DIR [smoke|train] [--dry-run]" >&2
    exit 2
fi
data_dir="$1"
output_dir="$2"
mode="${3:-smoke}"
preview="${4:-}"
if [[ "$mode" != smoke && "$mode" != train ]]; then
    echo "Mode must be smoke or train." >&2; exit 2
fi
if [[ -n "$preview" && "$preview" != --dry-run ]]; then
    echo "Fourth argument must be --dry-run, if supplied." >&2; exit 2
fi
# Interpret relative user paths before changing to the checkout.
[[ "$data_dir" = /* ]] || data_dir="$PWD/$data_dir"
[[ "$output_dir" = /* ]] || output_dir="$PWD/$output_dir"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
# No hardcoded home directory; shipped checkout and dependencies can live anywhere.
export PYTHONPATH="$repo_root:$repo_root/ml4gw:$repo_root/gwak/train${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-180}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
cmd=("${GWAK_PYTHON:-python}" "$repo_root/gwak/train/cli.py" fit
     --config "$repo_root/examples/sky_position/sg150_directional_event.json"
     --data.init_args.data_dir "$data_dir"
     --data.init_args.num_workers 0
     --trainer.default_root_dir "$output_dir"
     --trainer.logger.init_args.save_dir "$output_dir"
     --trainer.num_sanity_val_steps 0)
if [[ "$mode" = smoke ]]; then
    # Keep normal logging/checkpoint hooks: Lightning fast_dev_run skips those.
    cmd+=(--trainer.max_epochs 1 --trainer.limit_train_batches 2
          --trainer.limit_val_batches 2 --data.init_args.batch_size 8)
else
    cmd+=(--trainer.limit_train_batches 640)
fi
printf 'Mode: %s\nCommand:' "$mode"
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "$preview" = --dry-run ]]; then exit 0; fi
if [[ ! -d "$data_dir" ]]; then
    echo "Background data directory does not exist: $data_dir" >&2; exit 2
fi
# An exclusive new output folder avoids colliding with previous runs.
mkdir -p "$(dirname "$output_dir")"
if ! mkdir "$output_dir"; then
    echo "Choose a new output directory for this run." >&2; exit 2
fi
exec "${cmd[@]}"
