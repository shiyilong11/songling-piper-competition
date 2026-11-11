#!/usr/bin/env bash
set -eo pipefail
field_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$field_root"
if [[ "${ROS_DISTRO:-}" != "humble" ]]; then
  if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "Use the organiser's Ubuntu 22.04 / ROS 2 Humble image" >&2
    exit 2
  fi
  source /opt/ros/humble/setup.bash
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./setup.sh first" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PIPER_SAM2_PYTHON="$field_root/.venv/bin/python"
export PYTHONPATH="$field_root/src:$field_root/vendor${PYTHONPATH:+:$PYTHONPATH}"
if ! .venv/bin/python -c 'import rclpy; from piper_msgs.msg import PiperStatusMsg' 2>/dev/null; then
  echo "Source the organiser's PiPER ROS 2 driver workspace (piper_msgs) before running" >&2
  exit 2
fi
exec .venv/bin/python -m piper_sim.field_cli "$@"
