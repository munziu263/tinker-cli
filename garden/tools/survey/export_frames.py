"""Export reference frames from the pan and their camera poses for the viewer.

    export_frames.py <survey.npz> <frames_dir> <viewer_dir> [plan_rotation_deg] [count]

Writes <viewer_dir>/frames/NN.jpg (640 px wide) and <viewer_dir>/frames.json with, for
each frame, the camera position and orientation in three.js coordinates (Y up), so the
3D view can be put exactly where the phone was.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey  # noqa: E402


def mat_to_quat(M):
    t = np.trace(M)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        w, x, y, z = (M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s
    elif M[1, 1] > M[2, 2]:
        s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        w, x, y, z = (M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s
    else:
        s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        w, x, y, z = (M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def to_three(v):
    return np.array([v[0], v[2], -v[1]])


def main():
    npz, frames_dir, viewer_dir = sys.argv[1:4]
    rot = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    count = int(sys.argv[5]) if len(sys.argv) > 5 else 44
    sv = Survey(npz, frames_dir)
    sv.set_plan_frame(rot)
    nums = np.array([int(n[2:6]) for n in sv.names])

    def sharp(i):
        im = np.asarray(sv.image(i).convert("L").resize((480, 360)), np.float32)
        lap = im[1:-1, 1:-1] * 4 - im[:-2, 1:-1] - im[2:, 1:-1] - im[1:-1, :-2] - im[1:-1, 2:]
        return float(lap.var())

    yaw = sv.yaw
    targets = np.linspace(yaw[0], yaw[-1], count)
    picked = []
    for t in targets:
        k = int(np.argmin(np.abs(yaw - t)))
        cand = [j for j in range(max(0, k - 2), min(len(yaw), k + 3)) if j not in picked]
        if not cand:
            continue
        picked.append(max(cand, key=sharp))
    picked = sorted(set(picked))

    out_dir = os.path.join(viewer_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    frames = []
    for n, i in enumerate(picked):
        R = sv.R[i] @ sv.T.T            # plan frame -> camera
        right, down, fwd = R[0], R[1], R[2]
        M = np.stack([to_three(right), to_three(-down), to_three(-fwd)], axis=1)
        c = sv.centre(i)
        name = f"{n:02d}.jpg"
        sv.image(i).resize((640, 480), Image.LANCZOS).save(os.path.join(out_dir, name), quality=72, optimize=True)
        frames.append({
            "file": f"frames/{name}", "src": int(nums[i]), "t": round(float(nums[i]) / 30.0, 2),
            "pos": [round(float(v), 4) for v in to_three(c)],
            "quat": [round(v, 6) for v in mat_to_quat(M)],
            "heading": round(float(np.degrees(np.arctan2(fwd[1], fwd[0]))), 1),
        })
    meta = {
        "vfov": round(float(np.degrees(2 * np.arctan((sv.H / 2) / sv.f))), 3),
        "aspect": sv.W / sv.H,
        "frames": frames,
    }
    with open(os.path.join(viewer_dir, "frames.json"), "w") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"{len(frames)} frames, {size / 1e6:.2f} MB, vfov {meta['vfov']} deg")


if __name__ == "__main__":
    main()
