"""Pin the survey to things whose real position is known: survey control.

    calibrate.py <levelled.npz> <out.npz> <control.json> <frames_dir> [<frames_dir> ...]

A reconstruction from video has no scale of its own, and guessing it is not good enough:
the first version of this model sized itself on "standard" objects (brick courses,
paving flags, a door) and came out 27% too big. This step replaces guessing with control,
the way a surveyor would. Places that appear both in the footage and on a measured
drawing fix the scale, rotation and origin together, and each one reports how far off it
is afterwards, which is the honest measure of how good the model is.

control.json, in real metres in the site grid (see its "frame"):

  planes     wall faces, each with a recipe for finding its points in the reconstruction
             (a frame, a pixel box, a range and height gate). They only serve to turn a
             pixel into a position: a pick is the place where its ray meets the wall.
  points     {"name", "real": [x, y], "pick": {"frame", "pixel", "on"}, "sigma"}: a
             corner, jamb or post whose site-grid position is read off the measured
             survey.
  distances  tape measurements, {"name", "real_m", "between": [[frame, u, v, plane], ...]}.
  levels     spot levels from the measured survey, to put z = 0 on its datum.

The similarity found is baked into the output, so later tools read site-grid metres.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey  # noqa: E402


def gather(sv, find):
    """Reconstructed points (plan frame) on one wall face."""
    i = sv.idx(find["frame"])
    colour = tuple(find["colour"]) if find.get("colour") else None
    pts = sv.points_in(i, tuple(find["box"]), z=tuple(find.get("z", (0.2, 3.0))),
                       min_track=4, max_err=1.5, colour=colour)
    c = sv.centre(i)
    r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    lo, hi = find.get("range", (0.0, 1e9))
    return pts[(r > lo) & (r < hi)]


def pick_on(sv, frame, pixel, plane):
    """Where the ray through a pixel meets the vertical plane of a fitted wall face."""
    i = sv.idx(frame)
    c, r = sv.centre(i), sv.ray(i, pixel[0], pixel[1])
    n = np.array([-plane["dir"][1], plane["dir"][0]])
    t = ((plane["centre"] - c[:2]) @ n) / (r[:2] @ n)
    return c + r * t


def similarity(P, Q, sigma, extra=()):
    """site = s R p + t, weighted least squares. Unknowns a = s cos, b = s sin, tx, ty.
    extra: (pa, pb, length) tape distances, added as soft constraints on the scale."""
    w = np.repeat(1.0 / np.asarray(sigma), 2)
    A = np.zeros((2 * len(P), 4))
    A[0::2] = np.c_[P[:, 0], -P[:, 1], np.ones(len(P)), np.zeros(len(P))]
    A[1::2] = np.c_[P[:, 1], P[:, 0], np.zeros(len(P)), np.ones(len(P))]
    y = Q.reshape(-1).astype(float)
    x = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    for _ in range(3 if extra else 0):          # tapes are linear in the scale once the rotation is known
        th = np.arctan2(x[1], x[0])
        rows, ys, ws = [], [], []
        for pa, pb, L, sg in extra:
            d = np.linalg.norm(np.asarray(pa) - np.asarray(pb))
            rows.append([d * np.cos(th), d * np.sin(th), 0.0, 0.0])
            ys.append(L)
            ws.append(1.0 / sg)
        A2, y2, w2 = np.vstack([A, rows]), np.r_[y, ys], np.r_[w, ws]
        x = np.linalg.lstsq(A2 * w2[:, None], y2 * w2, rcond=None)[0]
    M = np.array([[x[0], -x[1]], [x[1], x[0]]])
    return M, x[2:4], P @ M.T + x[2:4] - Q


def main():
    npz, out, control_path, dirs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    ctl = json.load(open(control_path))
    plan_rot = ctl.get("survey_plan_rot", 0.0)
    sv = Survey(npz, dirs)
    sv.set_plan_frame(plan_rot)

    planes = {}
    for name, find in ctl["planes"].items():
        pts = gather(sv, find)
        fit = Survey.fit_wall(pts, floor=0.04) if len(pts) >= 8 else None
        if fit is None:
            print(f"  plane {name}: only {len(pts)} points found, skipped")
            continue
        planes[name] = fit
        print(f"  plane {name:22s} {fit['n']:4d} points, heading {fit['heading'] % 180:6.2f}, flat to {fit['rms'] * 100:4.1f} cm")
    names, P, Q, sg = [], [], [], []
    for q in ctl["points"]:
        if q["pick"]["on"] not in planes:
            continue
        names.append(q["name"])
        P.append(pick_on(sv, q["pick"]["frame"], q["pick"]["pixel"], planes[q["pick"]["on"]])[:2])
        Q.append(q["real"])
        sg.append(q.get("sigma", 0.05))
    P, Q = np.array(P), np.array(Q, float)
    tapes = []
    for d in ctl.get("distances", []):
        ends = [pick_on(sv, f, (u, v), planes[pl])[:2] for f, u, v, pl in d["between"]]
        tapes.append((ends[0], ends[1], d["real_m"], d.get("sigma", 0.02)))
    M, t, res = similarity(P, Q, sg, tapes)
    s, delta = float(np.hypot(M[0, 0], M[1, 0])), float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    print(f"scale {s:.4f} (one survey unit is {s:.4f} m), rotation {delta:+.2f} deg, origin shift ({t[0]:.2f}, {t[1]:.2f}) m")
    print("control points afterwards (reconstruction minus drawing):")
    for n, r in zip(names, res):
        print(f"  {n:28s} {r[0] * 100:+6.1f} cm along X, {r[1] * 100:+6.1f} cm along Y")
    print(f"  root mean square {np.sqrt((res ** 2).sum(1).mean()) * 100:.1f} cm over {len(names)} points")
    for (pa, pb, L, _), d in zip(tapes, ctl.get("distances", [])):
        m = np.linalg.norm((pa - pb) @ M.T)
        print(f"  tape: {d['name']:22s} measured {L:.3f} m, model {m:.3f} m ({(m / L - 1) * 100:+.1f}%)")

    a = np.radians(plan_rot)
    T = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])   # raw survey -> plan frame
    R2 = np.eye(3)
    R2[:2, :2] = M / s
    Qr = R2 @ T                                       # raw survey -> site grid (rotation only)
    d = dict(np.load(npz, allow_pickle=False))
    shift = np.array([t[0], t[1], 0.0])
    Pw = d["P"] @ Qr.T * s + shift
    Cw = d["C"] @ Qr.T * s + shift
    z0 = float(ctl.get("z_datum_level", 0.0))
    found = []
    ok = (d["track"] >= 4) & (d["err"] < 1.5)
    for lv in ctl.get("levels", []):
        m = ok & (np.hypot(Pw[:, 0] - lv["at"][0], Pw[:, 1] - lv["at"][1]) < lv.get("radius", 0.7)) & (Pw[:, 2] < np.median(Pw[ok, 2]) + 1.0)
        z = Pw[m, 2]
        if len(z) >= 25:
            h, e = np.histogram(z, bins=np.arange(z.min(), z.min() + 1.2, 0.03))
            found.append((lv, float(e[int(np.argmax(np.convolve(h, [1, 2, 1], "same")))] + 0.015)))
    dz = float(np.median([lv["level"] - z0 - z for lv, z in found])) if found else 0.0
    print(f"heights: shifted {dz:+.3f} m so that z = 0 is level {z0:.2f} on the measured survey")
    for lv, z in found:
        print(f"  {lv['name']:28s} drawn {lv['level'] - z0:+.2f}   reconstructed {z + dz:+.2f}   ({(z + dz - lv['level'] + z0) * 100:+5.1f} cm)")
    Pw[:, 2] += dz
    Cw[:, 2] += dz
    d["P"], d["C"], d["R"] = Pw, Cw, d["R"] @ Qr.T
    d["scale"] = np.array(float(d["scale"]) * s)
    d["cam_h"] = np.array(float(d["cam_h"]) * s)
    d["calibration"] = np.array([s, delta, t[0], t[1], dz])
    yaw = np.degrees(np.arctan2(d["R"][:, 2, 1], d["R"][:, 2, 0]))
    clip_names = np.array([str(n).rsplit("_", 1)[0] for n in d["names"]])
    for prefix in sorted(set(clip_names)):                          # continuous within each clip
        m = clip_names == prefix
        yaw[m] = np.degrees(np.unwrap(np.radians(yaw[m])))
    d["yaw"] = yaw
    np.savez(out, **d)
    print("saved", out)


if __name__ == "__main__":
    main()
