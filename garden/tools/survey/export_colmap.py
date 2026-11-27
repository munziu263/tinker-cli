"""Write a survey out as a COLMAP dataset in the site grid, for tools that learn a scene from photographs.

    export_colmap.py <survey.npz> <out_dir> [--clips <clip,clip,...>] [--every 1] [--max-side 0]
                     [--points 300000] -- <frames_dir> [...]

Programs that turn photographs into a photoreal scene (Gaussian splat trainers such as
Brush, and most of their relatives) do not find the cameras themselves. They read a folder
laid out the way COLMAP leaves one:

    <out_dir>/images/<name>.jpg                       the pictures
    <out_dir>/sparse/0/cameras.txt, images.txt, points3D.txt

This writes that folder from a survey. Two things make it worth doing here and not with
the trainer's own tools:

  * the cameras are written in the site grid, in metres, Z up. Whatever is trained from
    them comes out on the same grid as the model, so a photoreal copy of the place can be
    laid over the plan model with no fitting at all;
  * the survey's corrections come with it: clips that were slid back onto the house or
    pinned by sightings (anchor.py) are where they ended up, not where they began.

The pictures are written with the lens distortion taken out and the cameras as plain
pinholes, which is the one camera every trainer understands. --max-side shrinks the
pictures (and the cameras with them) so that the longer side is at most that many pixels:
training time and memory go with the number of pixels. --every takes one frame in N of
each clip; --clips keeps only the clips named. --points is how many of the survey's points
to pass on as a starting cloud (the best-seen ones first).

Check a dataset by opening it in COLMAP's own viewer, or with
    python -c "import pycolmap; print(pycolmap.Reconstruction('<out_dir>/sparse/0').summary())"
"""
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey, clip_of  # noqa: E402


def main():
    a = sys.argv[1:]
    dirs = a[a.index("--") + 1:] if "--" in a else []
    a = a[:a.index("--")] if "--" in a else a
    opt = {}
    for key in ("--clips", "--every", "--max-side", "--points"):
        if key in a:
            i = a.index(key)
            opt[key] = a[i + 1]
            del a[i:i + 2]
    sv, out = Survey(a[0], dirs), a[1]
    clips = set(opt["--clips"].split(",")) if "--clips" in opt else None
    every, max_side, n_points = int(opt.get("--every", 1)), int(opt.get("--max-side", 0)), int(opt.get("--points", 300000))
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "sparse", "0"), exist_ok=True)

    count, chosen = {}, []
    for i, name in enumerate(sv.names):
        c = clip_of(name)
        if clips is not None and c not in clips:
            continue
        count[c] = count.get(c, 0) + 1
        if (count[c] - 1) % every == 0:
            chosen.append(i)
    cameras, lines, missing = {}, [], 0
    for n, i in enumerate(chosen):
        try:
            img = cv2.imread(sv.image_path(i), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        except Exception:
            img = None
        if img is None:
            missing += 1
            continue
        f, cx, cy, k1 = sv.intr[i]
        # COLMAP counts pixels from the corner of the first one, OpenCV from its middle
        K = np.array([[f, 0, cx - 0.5], [0, f, cy - 0.5], [0, 0, 1.0]])
        img = cv2.undistort(img, K, np.array([k1, 0, 0, 0.0]), None, K)
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(w, h)) if max_side else 1.0
        if s < 1.0:
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            sx, sy = img.shape[1] / w, img.shape[0] / h
            h, w = img.shape[:2]
        else:
            sx = sy = 1.0
        key = (w, h, round(f * sx, 3), round(f * sy, 3), round(cx * sx, 3), round(cy * sy, 3))
        cam = cameras.setdefault(key, len(cameras) + 1)
        name = sv.names[i]
        cv2.imwrite(os.path.join(out, "images", name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        R = sv.R[i]
        t = -R @ sv.C[i]
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        lines.append(f"{len(lines) + 1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {cam} {name}\n\n")
        if (n + 1) % 200 == 0:
            print(f"   {n + 1} of {len(chosen)} pictures", flush=True)
    with open(os.path.join(out, "sparse", "0", "cameras.txt"), "w") as fh:
        fh.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for (w, h, fx, fy, cx, cy), cam in sorted(cameras.items(), key=lambda kv: kv[1]):
            fh.write(f"{cam} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    with open(os.path.join(out, "sparse", "0", "images.txt"), "w") as fh:
        fh.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fh.writelines(lines)

    # a starting cloud: the survey's points that the chosen clips own, best-seen first
    keep = (sv.track >= 3) & (sv.err < 1.5)
    if clips is not None and sv.home is not None:
        keep &= sv.points_of(clips)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-sv.track[idx], kind="stable")][:n_points]
    with open(os.path.join(out, "sparse", "0", "points3D.txt"), "w") as fh:
        fh.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for k, j in enumerate(idx, 1):
            x, y, z = sv.P[j]
            r, g, b = sv.col[j]
            fh.write(f"{k} {x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)} {float(sv.err[j]):.3f}\n")
    with open(os.path.join(out, "README.txt"), "w") as fh:
        fh.write("COLMAP dataset written by tools/survey/export_colmap.py.\n"
                 "World frame: the site grid, in metres. +X south, +Y east, +Z up; origin at the outer south-east corner of the rear outshot, ground level.\n"
                 "Pictures have had the lens distortion taken out; cameras are pinholes.\n")
    print(f"{len(lines)} pictures, {len(cameras)} camera(s), {len(idx)} points written to {out}" + (f"; {missing} pictures were not found in the frame directories given" if missing else ""))


if __name__ == "__main__":
    main()
