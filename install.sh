#!/usr/bin/env bash
# LARYNX installer (Linux / WSL). Idempotent: every stage skips what is already done.
#
#   ./install.sh                 system deps, venv, build + tests, golden set, pycore check, plugin install
#   ./install.sh --data          + download corpora and build shards
#   ./install.sh --train         + --data, train Live (D=0) and Studio (D=16), export the weights
#   ./install.sh --all           same as --train
#
# Options: --no-apt (skip apt; deps must already be there), --hours N (per-corpus cap for shards,
# default 10), --steps N (training steps, default 60000), --jobs N (shard workers).
# Env: LARYNX_DATA (default data/) for corpora, shards and runs.
set -euo pipefail
cd "$(dirname "$0")"

APT=1 DATA=0 TRAIN=0 HOURS=10 STEPS=60000 JOBS=""
while [ $# -gt 0 ]; do
    case $1 in
        --no-apt) APT=0 ;;
        --data) DATA=1 ;;
        --train | --all) DATA=1 TRAIN=1 ;;
        --hours) HOURS=$2; shift ;;
        --steps) STEPS=$2; shift ;;
        --jobs) JOBS="--jobs $2"; shift ;;
        -h | --help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
    shift
done
export LARYNX_DATA=${LARYNX_DATA:-data}
PY=.venv/bin/python
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "system packages"
PKGS="build-essential cmake ninja-build pkg-config git curl python3-venv python3-dev pybind11-dev
      libasound2-dev libfreetype-dev libfontconfig1-dev libx11-dev libxext-dev libxrender-dev
      libxrandr-dev libxinerama-dev libxcursor-dev libxcomposite-dev libgl-dev libglu1-mesa-dev mesa-common-dev
      pulseaudio-utils"
missing=$(for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || echo "$p"; done)
if [ -n "$missing" ]; then
    if [ $APT = 1 ]; then
        sudo apt-get update && sudo apt-get install -y $missing
    else
        echo "missing packages (--no-apt): $missing"; exit 1
    fi
fi

step "python venv"
[ -x $PY ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q numpy scipy soundfile pyloudnorm praat-parselmouth
if ! $PY -c "import torch" 2>/dev/null; then
    if command -v nvidia-smi >/dev/null; then
        .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121
    else
        .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    fi
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu')"

step "build (Release) + tests"
# pycore must be built for the venv's interpreter.
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(realpath $PY)" >/dev/null
cmake --build build
ctest --test-dir build --output-on-failure

step "golden set"
python3 ml/data/fetch_golden.py

step "pycore == C++ (§9 gate)"
PYTHONPATH=build/ml/pycore $PY ml/pycore/check_equivalence.py golden/8_vozes/*/*.wav | tail -1

step "install plugin (~/.vst3, ~/.clap)"
ART=build/plugin/Larynx_artefacts/Release
mkdir -p ~/.vst3 ~/.clap
[ -d $ART/VST3/LARYNX.vst3 ] && rm -rf ~/.vst3/LARYNX.vst3 && cp -r $ART/VST3/LARYNX.vst3 ~/.vst3/
[ -e $ART/CLAP/LARYNX.clap ] && cp -r $ART/CLAP/LARYNX.clap ~/.clap/
ls -d ~/.vst3/LARYNX.vst3 ~/.clap/LARYNX.clap 2>/dev/null || true

if [ $DATA = 1 ] || [ ! -f plugin/source/neural/profiles.data ]; then
    step "morph profiles (12 GTSinger singers outside the golden set)"
    [ -f plugin/source/neural/profiles.data ] || $PY ml/data/build_morph.py
fi

if [ $DATA = 1 ]; then
    step "corpora → $LARYNX_DATA (VocalSet, VCTK, LibriTTS-R, PTDB-TUG; JVS-MuSiC is manual)"
    ml/data/download_corpora.sh
    step "shards (≤ $HOURS h per corpus)"
    $PY ml/data/build_shards.py vocalset vctk libritts_r ptdb jvs_music --max-hours "$HOURS" $JOBS
fi

if [ $TRAIN = 1 ]; then
    for D in 0 16; do
        name=$([ $D = 0 ] && echo live || echo studio)
        step "train $name (D=$D, $STEPS steps)"
        $PY ml/train/train.py --name "$name" --delay $D --steps "$STEPS"
        mkdir -p plugin/source/neural
        $PY ml/train/model.py "$LARYNX_DATA/runs/$name/best.pt" "plugin/source/neural/model_$name.json"
        $PY ml/train/check_export.py "plugin/source/neural/model_$name.json" "$LARYNX_DATA/runs/$name/best.pt"
        $PY ml/eval/eval_golden.py "$LARYNX_DATA/runs/$name/best.pt" | tail -1
    done
    step "rebuild with embedded weights"
    cmake -B build >/dev/null && cmake --build build && ctest --test-dir build --output-on-failure
    cp -r $ART/VST3/LARYNX.vst3 ~/.vst3/ && cp -r $ART/CLAP/LARYNX.clap ~/.clap/
fi

step "done"
echo "Standalone: $ART/Standalone/LARYNX   ABX: python3 ml/eval/abx.py prepare abx/identidade"
