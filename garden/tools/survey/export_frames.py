"""Export reference stills from the clips, with their camera poses, for the viewer.

    export_frames.py <survey.npz> <viewer_dir> <plan_rotation_deg> <frames_dir> [<frames_dir> ...]

Writes <viewer_dir>/frames/NN.jpg (640 px wide) and <viewer_dir>/frames.json with, for
each still, the camera position, orientation and field of view in three.js coordinates
(Y up), so the 3D view can be put exactly where the phone was.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey  # noqa: E402

# clip prefix -> (title shown in the viewer, how many stills, how to spread them[, last frame number to use])
CLIPS = {
    "f": ("Pan from the lawn", 44, "yaw"),
    "g": ("Walk east across the lawn", 28, "time"),
    "h": ("Walk round the patio", 28, "time"),
    "i": ("Along the road hedge", 22, "time"),
    "j": ("East side of the house", 8, "time"),
    "k": ("Turn by the stump", 22, "time", 284),          # stops before someone walks into shot
}


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


def sharp(sv, i):
    im = sv.image(i).convert("L")
    im = np.asarray(im.resize((480, 360) if im.width >= im.height else (360, 480)), np.float32)
    lap = im[1:-1, 1:-1] * 4 - im[:-2, 1:-1] - im[2:, 1:-1] - im[1:-1, :-2] - im[1:-1, 2:]
    return float(lap.var())


def choose(sv, ids, count, how):
    if how == "yaw":
        key = np.degrees(np.unwrap(np.radians(sv.yaw[ids])))      # a pan can pass through +-180
    else:
        key = np.arange(len(ids), dtype=float)
    picked = []
    for t in np.linspace(key[0], key[-1], min(count, len(ids))):
        k = int(np.argmin(np.abs(key - t)))
        cand = [j for j in range(max(0, k - 2), min(len(ids), k + 3)) if ids[j] not in picked]
        if cand:
            picked.append(ids[max(cand, key=lambda j: sharp(sv, ids[j]))])
    return sorted(set(picked))


def main():
    npz, viewer_dir, rot = sys.argv[1], sys.argv[2], float(sys.argv[3])
    sv = Survey(npz, sys.argv[4:])
    sv.set_plan_frame(rot)
    out_dir = os.path.join(viewer_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    frames, clips = [], []
    for prefix, (title, count, how, *last) in CLIPS.items():
        ids = sv.clip(prefix)
        if last:
            ids = [i for i in ids if int(sv.names[i][2:6]) <= last[0]]
        if not ids:
            continue
        picked = choose(sv, ids, count, how)
        clips.append({"title": title, "start": len(frames), "count": len(picked)})
        for i in picked:
            R = sv.R[i] @ sv.T.T                  # plan frame -> camera
            right, down, fwd = R[0], R[1], R[2]
            M = np.stack([to_three(right), to_three(-down), to_three(-fwd)], axis=1)
            name = f"{len(frames):02d}.jpg" if len(frames) < 100 else f"{len(frames)}.jpg"
            f_px, cx, cy = (float(v) for v in sv.intr[i][:3])
            w_px, h_px = 2 * cx, 2 * cy                           # this frame's own shape: some clips are filmed upright
            size = (640, 480) if w_px >= h_px else (480, 640)
            sv.image(i).resize(size, Image.LANCZOS).save(os.path.join(out_dir, name), quality=72, optimize=True)
            num = int(sv.names[i][2:6])
            frames.append({
                "file": f"frames/{name}", "clip": len(clips) - 1, "src": num, "t": round(num / 30.0, 2),
                "pos": [round(float(v), 4) for v in to_three(sv.centre(i))],
                "quat": [round(v, 6) for v in mat_to_quat(M)],
                "vfov": round(float(np.degrees(2 * np.arctan((h_px / 2) / f_px))), 3),
                "aspect": round(w_px / h_px, 4),
            })
    meta = {"aspect": sv.W / sv.H, "vfov": frames[0]["vfov"], "clips": clips, "frames": frames}
    with open(os.path.join(viewer_dir, "frames.json"), "w") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"{len(frames)} stills in {len(clips)} clips, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
