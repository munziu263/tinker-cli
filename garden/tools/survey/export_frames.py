"""Export reference stills from the clips, with their camera poses, for the viewer.

    export_frames.py <survey.npz> <viewer_dir> <clips.json> [--buildings garden.json] <frames_dir> [<frames_dir> ...]

Writes <viewer_dir>/frames/NN.jpg (640 px on the long side) and <viewer_dir>/frames.json
with, for each still, the camera position, orientation and field of view in three.js
coordinates (Y up), so the 3D view can be put exactly where the phone was.

clips.json says what to show (see control/clips.json):

  walks   the entries of the viewer's clip list, in order. Each has a title and one or more
          clips: {"clip": name, "stills": how many, "spread": "time" | "yaw",
          "first" / "last": frame numbers to stay within, "skip": [[from, to], ...] frame
          ranges to leave out (someone walking through the shot, say)}.
  blur    things that should not be readable in a published still: {"what", "at": [x, y, z]
          in the site grid, "size": metres across, "clips": optional list}. Wherever one
          falls inside a still, that patch is blurred. A still is left alone when the
          thing is more than 40 m away, or when a building stands between it and the
          camera (--buildings: the footprints are read from garden.json), so that a gate
          on the far side of the house does not put a smudge on the house wall.

Stills are numbered in the order of the list, so adding walks at the end leaves the
earlier files exactly as they were.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey, frame_no  # noqa: E402


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


def _crosses(a, b, poly):
    """Does the plan segment a-b cross the polygon's outline?"""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    for k in range(len(poly)):
        c, d = poly[k], poly[(k + 1) % len(poly)]
        if side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0:
            return True
    return False


def blur_patches(sv, i, im, blurs, clip, screens=()):
    """Blur whatever in `blurs` shows in frame i (im is the still, already resized)."""
    W, _ = sv.frame_size(i)
    scale = im.width / W
    done = 0
    cam = sv.centre(i)
    for b in blurs:
        if b.get("clips") and clip not in b["clips"]:
            continue
        u, v, z = (float(a) for a in sv.project(i, np.asarray(b["at"], float)))
        if z < 0.4 or z > 40.0 or any(_crosses(cam[:2], b["at"][:2], poly) for poly in screens):
            continue
        r = max(10.0, 0.5 * b.get("size", 0.8) * float(sv.intr[i][0]) / z * scale)     # half-size in still pixels
        box = [int(u * scale - r), int(v * scale - r), int(u * scale + r), int(v * scale + r)]
        box = [max(0, box[0]), max(0, box[1]), min(im.width, box[2]), min(im.height, box[3])]
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        im.paste(im.crop(box).filter(ImageFilter.GaussianBlur(float(max(4.0, r / 3)))), tuple(box[:2]))
        done += 1
    return done


def main():
    args = sys.argv[1:]
    screens = []
    if "--buildings" in args:
        from terrain import building_outlines
        k = args.index("--buildings")
        screens = building_outlines(json.load(open(args[k + 1])))
        del args[k:k + 2]
    npz, viewer_dir, cfg_path = args[0], args[1], args[2]
    cfg = json.load(open(cfg_path))
    sv = Survey(npz, args[3:])
    out_dir = os.path.join(viewer_dir, "frames")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    frames, walks, n_blur = [], [], 0
    for walk in cfg["walks"]:
        start = len(frames)
        for c in walk["clips"]:
            ids = sv.clip(c["clip"])
            lo, hi = c.get("first", 0), c.get("last", 10 ** 9)
            ids = [i for i in ids if lo <= frame_no(sv.names[i]) <= hi
                   and not any(a <= frame_no(sv.names[i]) <= b for a, b in c.get("skip", []))]
            if not ids:
                continue
            for i in choose(sv, ids, c.get("stills", 4), c.get("spread", "time")):
                R = sv.R[i] @ sv.T.T                  # plan frame -> camera
                right, down, fwd = R[0], R[1], R[2]
                M = np.stack([to_three(right), to_three(-down), to_three(-fwd)], axis=1)
                name = f"{len(frames):02d}.jpg"
                f_px, cx, cy = (float(v) for v in sv.intr[i][:3])
                w_px, h_px = 2 * cx, 2 * cy           # this frame's own shape: some clips are filmed upright
                im = sv.image(i).resize((640, 480) if w_px >= h_px else (480, 640), Image.LANCZOS)
                n_blur += blur_patches(sv, i, im, cfg.get("blur", []), c["clip"], screens)
                im.save(os.path.join(out_dir, name), quality=72, optimize=True)
                num = frame_no(sv.names[i])
                frames.append({
                    "file": f"frames/{name}", "clip": len(walks), "seq": c["clip"], "src": num, "t": round(num / 30.0, 2),
                    "pos": [round(float(v), 4) for v in to_three(sv.centre(i))],
                    "quat": [round(v, 6) for v in mat_to_quat(M)],
                    "vfov": round(float(np.degrees(2 * np.arctan((h_px / 2) / f_px))), 3),
                    "aspect": round(w_px / h_px, 4),
                })
        if len(frames) > start:
            walks.append({"title": walk["title"], "start": start, "count": len(frames) - start})
    meta = {"aspect": frames[0]["aspect"], "vfov": frames[0]["vfov"], "clips": walks, "frames": frames}
    with open(os.path.join(viewer_dir, "frames.json"), "w") as fh:
        json.dump(meta, fh, separators=(",", ":"))
    size = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir))
    print(f"{len(frames)} stills in {len(walks)} walks, {size / 1e6:.2f} MB; {n_blur} patches blurred")


if __name__ == "__main__":
    main()
