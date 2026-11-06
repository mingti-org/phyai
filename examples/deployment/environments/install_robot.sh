#!/usr/bin/env bash
# Install only the client framework and profile dependencies, never the workspace.
set -euo pipefail
# Resolve the source/config root consistently; pass WHEEL as an absolute path.
# ROS overlays are sourced later in the client shell, not in this installer.
cd "$(dirname "$0")/../../.."
profile=${1:?Usage: install_robot.sh PROFILE WHEEL}
wheel=${2:?Supply the reviewed phyai-robot wheel}
case "$profile" in
  agilex-humble-py310-x86_64) minor=3.10; arch=x86_64 ;;
  tianji-jazzy-py312-aarch64) minor=3.12; arch=aarch64 ;;
  *) echo "Unsupported profile: $profile" >&2; exit 1 ;;
esac
# A ROS binary package must be loaded by a compatible native interpreter.
# Selecting another Python is allowed only when it still matches this profile.
python=${ROBOT_PYTHON:-/usr/bin/python3}
actual=$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[[ $actual == "$minor" && $(uname -m) == "$arch" ]] || {
  echo "Profile needs native Python $minor on $arch" >&2; exit 1;
}
[[ -f $wheel && $wheel == *phyai_robot*py3-none-any.whl ]] || {
  echo 'Expected an existing phyai-robot py3-none-any wheel' >&2; exit 1;
}
if [[ -d .venv-robot ]]; then
  existing=$(.venv-robot/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  [[ $existing == "$minor" ]] || { echo '.venv-robot has the wrong Python version' >&2; exit 1; }
else
  # uv pip does not require pip/ensurepip inside the environment.
  "$python" -m venv --without-pip --system-site-packages .venv-robot
fi
# Resolve/install only the target profile. The reviewed wheel is then installed
# without dependency resolution, so it cannot pull in PhyAI or CUDA packages.
uv pip install --python .venv-robot/bin/python --require-hashes \
  -r "examples/deployment/environments/$profile/requirements.lock"
uv pip install --python .venv-robot/bin/python --no-deps \
  --reinstall-package phyai-robot "$wheel"
sha256sum "$wheel" > .venv-robot/phyai-robot-wheel.sha256
.venv-robot/bin/python -c 'import phyai_robot; from phyai_robot.remote import RemotePolicy; print(phyai_robot.__file__)'
