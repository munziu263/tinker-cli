#!/bin/bash
# Set this project up on a new machine: a Python environment in .venv holding what the
# model build and the survey tools need. Safe to run again.
#
#     bash setup.sh            # set up, then say what is in place
#     bash setup.sh --check    # only say what is and is not in place; change nothing
#
# It installs nothing outside this folder, apart from what uv keeps in its own cache in
# your home folder (the Python build and the packages it downloads). It does not install
# uv, Homebrew or ffmpeg for you: it says what is missing and how to get it.
set -euo pipefail
cd "$(dirname "$0")"

check_only=0
[ "${1:-}" = "--check" ] && check_only=1
say() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------- the machine
os=$(uname -s)
arch=$(uname -m)
if [ "$os" = "Darwin" ]; then
  ver=$(sw_vers -productVersion)
  major=${ver%%.*}
  say "macOS $ver on $arch"
  if [ "$arch" != "arm64" ] || [ "$major" -lt 14 ]; then
    say "NOTE: the ready-made builds of COLMAP (pycolmap) and Blender (bpy) need Apple Silicon and"
    say "      macOS 14 or later. On this machine those two may not install; the viewer, the marker"
    say "      tools and the photo intake do not need them."
  fi
else
  say "$os on $arch"
fi

# ---------------------------------------------------------------- Python and the packages
if [ "$check_only" = 0 ]; then
  if ! command -v uv >/dev/null 2>&1; then
    say ""
    say "uv is not installed. It fetches Python 3.13 and the packages. Install it one of these"
    say "ways, open a new terminal, and run this script again:"
    say "    brew install uv"
    say "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
  if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.13 .venv
  fi
  if ! uv pip install --python .venv/bin/python -r requirements.txt; then
    say ""
    say "The packages would not install all together. One at a time, to see which is the trouble:"
    sed -e 's/#.*//' -e 's/[[:space:]]*$//' requirements.txt | while read -r pkg; do
      [ -z "$pkg" ] && continue
      if uv pip install --python .venv/bin/python "$pkg" >/dev/null 2>&1; then
        say "   ok      $pkg"
      else
        say "   FAILED  $pkg"
      fi
    done
  fi
fi

say ""
if [ -x .venv/bin/python ]; then
  .venv/bin/python -I - <<'EOF' || true
import importlib
import sys

print("Python", sys.version.split()[0], "in .venv")
parts = (("numpy", "arrays"), ("scipy", "fitting"), ("PIL", "pictures"), ("pillow_heif", "the HEIC photographs phones take"),
         ("cv2", "printed markers, picture handling (OpenCV)"), ("pycolmap", "reconstruction from photographs (COLMAP)"),
         ("bpy", "the model build (Blender as a module)"))
for name, what in parts:
    try:
        m = importlib.import_module(name)
        v = m.app.version_string if name == "bpy" else getattr(m, "__version__", "")
        if name == "cv2":
            m.aruco.getPredefinedDictionary(m.aruco.DICT_APRILTAG_36h11)
        print(f"   ok       {name:12s} {str(v):12s} {what}")
    except Exception as e:
        print(f"   MISSING  {name:12s} {'':12s} {what}  ({type(e).__name__}: {str(e)[:80]})")
EOF
else
  say "no Python environment yet (.venv): run  bash setup.sh"
fi

# ---------------------------------------------------------------- the rest
if command -v ffmpeg >/dev/null 2>&1; then
  say "   ok       ffmpeg                    cutting frames out of video"
else
  say "   missing  ffmpeg                    only needed to cut frames out of video:  brew install ffmpeg"
fi
if [ -f viewer/model/garden.gltf.json ]; then
  say "   ok       the built model           viewer/model/"
else
  say "   missing  the built model           make it with:  .venv/bin/python build.py   (about a minute and a half)"
fi
if [ -f private/README.md ]; then
  say "   ok       the private working files private/  (start with private/README.md)"
else
  say "   absent   the private working files private/  (the survey, the drawings, the model's generators: the owner has them)"
fi

say ""
say "The viewer:   .venv/bin/python tools/serve.py     then open http://localhost:8765"
