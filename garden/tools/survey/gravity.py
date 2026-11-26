"""Find true vertical, and level the survey with it.

    gravity.py <survey.npz> <out.npz> --courses walls.json <frames_dir> [<frames_dir> ...]
    gravity.py <survey.npz> <out.npz> --edges <frames_dir> [...]
    gravity.py <survey.npz> <out.npz> --up x,y,z

Structure-from-motion has no idea which way is up. align.py guesses it from the lawn,
which is only right if the lawn is level, and here it is not: that guess was 1.5 degrees
out, enough to misplace anything 15 m away by a couple of metres when its position was
read off "the ground plane".

Two better plumb lines:

  --courses   Brick walls are laid in level courses, so the direction shared by all the
              mortar joints of a wall is a horizontal line. Two walls that are not
              parallel give two horizontal lines, and vertical is the direction square
              to both. This is the method used: three garden walls agree to 0.2 degrees.
              walls.json lists each wall as
              {"name", "frames": [first, last, step], "a": [x, y], "b": [x, y], "z": [lo, hi]}
              in the survey's plan frame (see "plan_rot" in the same file).

  --edges     Upright edges of buildings (corners, door frames, downpipes). Works on a
              new house; on a 300-year-old timber frame it is a degree or so out, which
              is why it is only a cross-check here.

The survey is then rotated about its origin so that Z is the vertical found. Ground
that looked flat may now slope: that slope is real.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey  # noqa: E402

_cache = {}


def segments(sv, i, refine=cv2.LSD_REFINE_ADV):
    if i not in _cache:
        img = cv2.imread(sv.image_path(i), cv2.IMREAD_GRAYSCALE)
        s = cv2.createLineSegmentDetector(refine).detect(img)[0]
        _cache[i] = np.zeros((0, 4)) if s is None else s.reshape(-1, 4)
    return _cache[i]


# ----------------------------------------------------------------------------- brick courses


def wall_edges(sv, frames, a, b, z0, z1, min_len=22, margin=0.06):
    """Edge planes (unit normals, plan frame) of the line segments that lie on the wall
    rectangle a->b, z0..z1 and run along it: the mortar joints."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w0 = np.r_[(b - a) / np.linalg.norm(b - a), 0.0]
    nrm, Lw = np.array([-w0[1], w0[0]]), np.linalg.norm(b - a)
    N, L = [], []
    for f in frames:
        try:
            i = sv.idx(f)
        except ValueError:
            continue
        seg = segments(sv, i)
        ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        seg, ln = seg[ln >= min_len], ln[ln >= min_len]
        if not len(seg):
            continue
        c, ok, rays = sv.centre(i), np.ones(len(seg), bool), []
        for k in (0, 2):
            r = sv.ray(i, seg[:, k], seg[:, k + 1])
            t = ((a - c[:2]) @ nrm) / (r[:, :2] @ nrm + 1e-12)
            p = c + r * t[:, None]
            s = (p[:, :2] - a) @ w0[:2]
            ok &= (t > 0) & (s > margin * Lw) & (s < (1 - margin) * Lw) & (p[:, 2] > z0) & (p[:, 2] < z1)
            rays.append(r)
        n = np.cross(rays[0], rays[1])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        ok &= (np.abs(n @ w0) < np.sin(np.radians(10))) & (np.abs(n[:, 2]) > 0.8)
        N.append(n[ok])
        L.append(ln[ok])
    return np.vstack(N), np.concatenate(L), w0


def course_direction(N, L, w0):
    """The one direction lying in every joint's plane: where the courses point."""
    w = w0.copy()
    for tol in (3.0, 1.5, 0.8, 0.5, 0.35):
        for _ in range(5):
            keep = np.abs(N @ w) < np.sin(np.radians(tol))
            M = (N[keep] * L[keep, None]).T @ N[keep]
            w = np.linalg.eigh(M)[1][:, 0]
            w *= np.sign(w @ w0)
    rms = np.degrees(np.sqrt(np.average((N[keep] @ w) ** 2, weights=L[keep])))
    return w, int(keep.sum()), rms


def up_from_courses(sv, walls):
    dirs, weights = [], []
    for wl in walls:
        f0, f1, step = wl["frames"]
        N, L, w0 = wall_edges(sv, range(f0, f1, step), wl["a"], wl["b"], wl["z"][0], wl["z"][1])
        if len(N) < 30:
            print(f"  {wl['name']:28s} only {len(N)} joints found, skipped")
            continue
        w, k, rms = course_direction(N, L, w0)
        slope = np.degrees(np.arcsin(w[2]))
        print(f"  {wl['name']:28s} {k:5d} joints, courses run at heading {np.degrees(np.arctan2(w[1], w[0])):7.1f} "
              f"and slope {slope:+.2f} deg in the current frame (rms {rms:.2f})")
        dirs.append(w)
        weights.append(k)
    M = sum(wt * np.outer(w, w) for w, wt in zip(dirs, weights))
    up = np.linalg.eigh(M)[1][:, 0]
    return up * np.sign(up[2])


# ----------------------------------------------------------------------------- upright edges


def up_from_edges(sv, step=2, min_len=70):
    z = np.array([0.0, 0.0, 1.0])
    Ns, Ws = [], []
    for i in range(0, len(sv.names), step):
        seg = segments(sv, i, cv2.LSD_REFINE_STD)
        ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        seg, ln = seg[ln >= min_len], ln[ln >= min_len]
        if not len(seg):
            continue
        a, b = sv.ray(i, seg[:, 0], seg[:, 1]), sv.ray(i, seg[:, 2], seg[:, 3])
        n = np.cross(a, b)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        mid = a + b
        ok = mid[:, 2] / np.linalg.norm(mid, axis=1) > np.sin(np.radians(-14))   # not looking down at the ground
        Ns.append(n[ok])
        Ws.append(ln[ok])
    N, W, up = np.vstack(Ns), np.concatenate(Ws), z
    for tol in (4.0, 2.5, 1.5, 1.0):
        for _ in range(4):
            keep = np.abs(N @ up) < np.sin(np.radians(tol))
            up = np.linalg.eigh((N[keep] * W[keep, None]).T @ N[keep])[1][:, 0]
            up *= np.sign(up[2])
    print(f"  {int(keep.sum())} upright edges in {len(Ns)} frames")
    return up


# ----------------------------------------------------------------------------- levelling


def level(npz, out, up_plan, plan_rot):
    """Rotate the survey about its origin so that `up_plan` (given in the plan frame)
    becomes +Z."""
    a = np.radians(plan_rot)
    T = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    up = np.asarray(up_plan, float) @ T            # plan frame -> survey frame
    up /= np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, z)
    s, c = np.linalg.norm(axis), float(up @ z)
    Q = np.eye(3)
    if s > 1e-9:
        k = axis / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Q = np.eye(3) + s * K + (1 - c) * (K @ K)
    d = dict(np.load(npz, allow_pickle=False))
    d["P"], d["C"], d["R"] = d["P"] @ Q.T, d["C"] @ Q.T, d["R"] @ Q.T
    if "A" in d:
        d["A"] = Q @ d["A"]
    d["pitch"] = np.degrees(np.arcsin(np.clip(d["R"][:, 2, 2], -1, 1)))
    d["roll"] = np.degrees(np.arctan2(-d["R"][:, 0, 2], -d["R"][:, 1, 2]))
    d["levelled_by_deg"] = np.array(np.degrees(np.arccos(np.clip(c, -1, 1))))
    np.savez(out, **d)
    print("saved", out)


def describe(up):
    return (f"{np.degrees(np.arccos(np.clip(up[2], -1, 1))):.2f} deg from the current Z, leaning towards "
            f"plan heading {np.degrees(np.arctan2(up[1], up[0])):.0f} deg")


def main():
    npz, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == "--up":
        plan_rot = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        up = np.array([float(v) for v in sys.argv[4].split(",")])
    elif mode == "--courses":
        spec = json.load(open(sys.argv[4]))
        plan_rot = spec.get("plan_rot", 0.0)
        sv = Survey(npz, sys.argv[5:])
        sv.set_plan_frame(plan_rot)
        up = up_from_courses(sv, spec["walls"])
        print("vertical from brick courses:", describe(up))
        if spec.get("compare_edges"):
            print("vertical from upright edges:", describe(up_from_edges(sv)), "(cross-check only)")
    else:
        plan_rot = 0.0
        sv = Survey(npz, sys.argv[4:])
        up = up_from_edges(sv)
        print("vertical from upright edges:", describe(up))
    level(npz, out, up, plan_rot)


if __name__ == "__main__":
    main()
