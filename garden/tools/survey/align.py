"""Turn a raw COLMAP reconstruction into a metric, gravity-aligned survey.

    align.py <sparse_model_dir> <out.npz> [camera_height_m]
    align.py <sparse_model_dir> <out.npz> --like <earlier_survey.npz> [--without <clip[:from-to],...>]
    align.py <sparse_model_dir> <out.npz> --markers <obs.json> [--size 0.160]

First form: world frame from scratch. Z up, ground at z = 0, origin on the ground below
the mean camera position, metres (scale set by the camera height given).

Second form: for a reconstruction that extends an earlier one (see sfm_add.py). The frames
the two share are used to put the new model in exactly the same world frame, so every
coordinate measured before stays valid.

Third form: for a set of photographs that shows printed survey markers (photos.py,
markers.py detect). A frame of the set's own: Z up from the way the phone was held, metres
from the printed size of the markers' black squares, z = 0 near the lowest things seen. It
is not the site grid (place.py lays the set on that); it is what `markers.py locate` needs
to check the set against taped distances, and enough to measure in.

--without leaves clips out: their frames, and every point that most of its sightings come
from them. For clips that the incremental registration got wrong (a walk that drifted),
which are then reconstructed properly as a piece and put back with place.py. A clip may
be given with a range of frame numbers, c3426:86-999, to leave out only that part of it:
for a clip that is sound up to the moment the camera swings round and cannot be trusted
after it. Frames with nothing to check them against are better out of a survey than in
it: their points go into the ground levels and the fit to the model like any others.
"""
import sys

import numpy as np
import pycolmap


def parse_without(text):
    """'c3411,c3426:86-999' -> {clip: None for all of it, or (first, last) frame numbers}."""
    out = {}
    for item in text.split(","):
        clip, _, span = item.partition(":")
        out[clip] = tuple(int(v) for v in span.split("-")) if span else None
    return out


def left_out(name, without):
    clip, _, num = str(name).rpartition("_")
    if clip not in without:
        return False
    span = without[clip]
    return span is None or span[0] <= int(num.split(".")[0]) <= span[1]


def load(model_dir, without=()):
    rec = pycolmap.Reconstruction(model_dir)
    cams = {cid: c for cid, c in rec.cameras.items()}
    frames, dropped = [], set()
    for iid, im in rec.images.items():
        if not im.has_pose:
            continue
        if without and left_out(im.name, without):
            dropped.add(iid)
            continue
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        frames.append((im.name, np.asarray(T.rotation.matrix()), np.asarray(T.translation), im.camera_id))
    frames.sort(key=lambda f: f[0])
    clip_of = {iid: str(im.name).rsplit("_", 1)[0] for iid, im in rec.images.items()}
    pts, cols, errs, tracks, homes = [], [], [], [], []
    for pid, p in rec.points3D.items():
        if dropped and 2 * sum(e.image_id in dropped for e in p.track.elements) > p.track.length():
            continue
        seen = {}
        for e in p.track.elements:
            seen[clip_of[e.image_id]] = seen.get(clip_of[e.image_id], 0) + 1
        homes.append(max(seen, key=seen.get))            # the clip that saw it most: where the point "lives"
        pts.append(p.xyz)
        cols.append(p.color)
        errs.append(p.error)
        tracks.append(p.track.length())
    return rec, cams, frames, np.array(pts), np.array(cols), np.array(errs), np.array(tracks), homes


def estimate_up(frames):
    """Up = direction most perpendicular to every camera's right-axis (level phone)."""
    X = np.array([f[1][0] for f in frames])
    w, v = np.linalg.eigh(X.T @ X)
    up = v[:, 0]
    down_mean = np.mean([f[1][1] for f in frames], axis=0)
    if np.dot(up, down_mean) > 0:
        up = -up
    return up / np.linalg.norm(up)


def ransac_ground(P, up, centre, iters=4000, tol=0.02, max_tilt_deg=12.0, seed=0):
    """Plane through the lawn: RANSAC over points below the cameras, normal near `up`."""
    rng = np.random.default_rng(seed)
    rel = P - centre
    hgt = rel @ up
    scale = np.median(np.linalg.norm(rel, axis=1))
    cand = P[hgt < -0.02 * scale]
    best, best_n = None, -1
    cos_lim = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            continue
        n /= ln
        if n @ up < 0:
            n = -n
        if n @ up < cos_lim:
            continue
        d = (cand - a) @ n
        cnt = int((np.abs(d) < tol * scale).sum())
        if cnt > best_n:
            best_n, best = cnt, (n, a)
    n, a = best
    inl = cand[np.abs((cand - a) @ n) < tol * scale]
    c0 = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c0, full_matrices=False)
    n2 = vt[2]
    if n2 @ up < 0:
        n2 = -n2
    return n2, c0, len(inl), len(cand)


def frame_from_scratch(frames, P, E, TL, cam_h):
    centres = np.array([-R.T @ t for _, R, t, _ in frames])
    cmean = centres.mean(0)
    up0 = estimate_up(frames)
    good = (TL >= 3) & (E < 2.0)
    n, c0, n_inl, n_cand = ransac_ground(P[good], up0, cmean)
    tilt = np.degrees(np.arccos(np.clip(n @ up0, -1, 1)))
    print(f"up from level-phone assumption vs ground-plane normal: {tilt:.2f} deg apart")
    print(f"ground plane: {n_inl} inliers of {n_cand} below-camera points")
    up = n
    h_sfm = (centres - c0) @ up
    print(f"camera height above ground (sfm units): mean {h_sfm.mean():.4f} "
          f"min {h_sfm.min():.4f} max {h_sfm.max():.4f}")
    s = cam_h / h_sfm.mean()
    z = up
    f0 = frames[0][1][2]
    x = f0 - (f0 @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    A = np.stack([x, y, z])                       # rows: world axes in sfm coordinates
    origin = cmean - ((cmean - c0) @ up) * up     # foot of the mean camera on the ground
    print(f"scale: {s:.4f} m per sfm unit (camera height set to {cam_h} m)")
    return A, origin, s


def frame_from_markers(frames, cams, P, obs_path, size):
    """A frame of its own for photographs that show printed markers: up from how the phone was held, metres from
    the printed size of the markers. Returns A, origin, s as the others do, and the cameras' mean height."""
    import json
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from markers import meet
    with open(obs_path) as fh:
        obs = json.load(fh)["frames"]
    by_name = {f[0]: f for f in frames}
    rays = {}                                                    # marker -> corner -> [(camera centre, direction, side in px)]
    for name, tags in obs.items():
        if name not in by_name:
            continue
        _, R, t, cid = by_name[name]
        par = cams[cid].params
        f, cx, cy, k1 = par[0], par[1], par[2], (par[3] if len(par) > 3 else 0.0)
        C = -R.T @ t
        for k, c in tags.items():
            c = np.array(c, float) + 0.5                         # the detector counts from the middle of the first pixel, COLMAP from its corner
            side = float(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean())
            for j, (u, v) in enumerate(c):
                xd, yd = (u - cx) / f, (v - cy) / f
                x, y = xd, yd
                for _ in range(8):
                    g = 1 + k1 * (x * x + y * y)
                    x, y = xd / g, yd / g
                d = R.T @ np.array([x, y, 1.0])
                rays.setdefault(int(k), {}).setdefault(j, []).append((C, d / np.linalg.norm(d), side))

    def corners(k, soft):
        out = []
        for j in range(4):
            r = rays[k].get(j, [])
            if len(r) < 2:
                return None
            C, D = np.array([x[0] for x in r]), np.array([x[1] for x in r])
            if np.degrees(np.arccos(np.clip((D @ D.T).min(), -1, 1))) < 4.0:      # seen from one direction only: no depth
                return None
            say = np.clip(np.array([x[2] for x in r]) / 50.0, 0.15, 1.0) ** 2
            out.append(meet(C, D, say, rounds=1 if soft is None else 4, soft=soft or 1.0)[0])
        return np.array(out)

    def side_of(c):
        return float(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1).mean())

    rough = [side_of(c) for c in (corners(k, None) for k in sorted(rays)) if c is not None]
    if not rough:
        raise SystemExit("no marker is seen from two directions in this reconstruction: there is nothing to size it by")
    unit = float(np.median(rough))                               # a first idea of a marker's size in the reconstruction's own units
    fine = {k: c for k, c in ((k, corners(k, 0.06 * unit)) for k in sorted(rays)) if c is not None}
    sides = {k: side_of(c) for k, c in fine.items()}
    s = size / float(np.median(list(sides.values())))
    print(f"sized by {len(sides)} marker(s), printed {size * 1000:.0f} mm: " + ", ".join(f"{k}: {v * s * 1000:.1f} mm" for k, v in sides.items()))
    if len(sides) > 1:
        v = np.array(list(sides.values())) * s
        print(f"   the markers agree on the size to {(v.max() - v.min()) / size * 100:.1f}% between the largest and the smallest. "
              "A square is a short ruler: taped distances between markers (markers.py locate --distances) are the real check.")
    centres = np.array([-R.T @ t for _, R, t, _ in frames])
    up = estimate_up(frames)
    f0 = frames[0][1][2]
    x = f0 - (f0 @ up) * up
    x /= np.linalg.norm(x)
    A = np.stack([x, np.cross(up, x), up])
    cmean = centres.mean(0)
    low = float(np.percentile((P - cmean) @ up, 2)) if len(P) else 0.0
    origin = cmean + low * up                                    # below the middle of the cameras, level with the lowest things seen
    cam_h = float(((centres - origin) @ up).mean() * s)
    print(f"scale: {s:.5f} m per sfm unit; up taken from how the phone was held; cameras {cam_h:.2f} m above z = 0 on average")
    return A, origin, s, cam_h


def frame_like(frames, earlier_npz):
    """Recover the earlier survey's world frame from the cameras both models share:
    world = s * A @ (sfm - origin). A similarity (Umeyama) fit of the shared camera
    positions gives scale, rotation and origin together, so this also works after a free
    bundle adjustment has let the reconstruction's own scale drift."""
    d = np.load(earlier_npz, allow_pickle=False)
    idx = {str(n): i for i, n in enumerate(d["names"])}
    shared = [(f, idx[f[0]]) for f in frames if f[0] in idx]
    if len(shared) < 3:
        raise SystemExit("fewer than three frames in common with the earlier survey")
    X = np.array([-f[1].T @ f[2] for f, _ in shared])          # camera centres, new sfm coordinates
    Y = np.array([d["C"][i] for _, i in shared])                # the same cameras in the earlier world
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    A = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (Xc ** 2).sum() * len(X))
    origin = mx - (A.T @ my) / s
    res = np.linalg.norm((X - origin) @ A.T * s - Y, axis=1)
    # the cameras' orientations are an independent check on the rotation
    dev = [np.degrees(np.arccos(np.clip((np.trace((d["R"][i].T @ f[1]) @ A.T) - 1) / 2, -1, 1))) for f, i in shared]
    print(f"frame taken from {len(shared)} shared cameras: positions agree to {res.mean() * 100:.1f} cm on average "
          f"(worst {res.max() * 100:.1f}), orientations to {np.mean(dev):.2f} deg; scale {s:.5f} per sfm unit")
    return A, origin, s


def main():
    without, markers, size = (), None, 0.160
    if "--without" in sys.argv:
        k = sys.argv.index("--without")
        without = parse_without(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    if "--markers" in sys.argv:
        k = sys.argv.index("--markers")
        markers = sys.argv[k + 1]
        del sys.argv[k:k + 2]
    if "--size" in sys.argv:
        k = sys.argv.index("--size")
        size = float(sys.argv[k + 1])
        del sys.argv[k:k + 2]
    model_dir, out = sys.argv[1], sys.argv[2]
    rec, cams, frames, P, C, E, TL, homes = load(model_dir, without)
    if without:
        print("left out:", ", ".join(c if v is None else f"{c} frames {v[0]} to {v[1]}" for c, v in sorted(without.items())))
    for cid, c in cams.items():
        print(f"camera {cid}:", c.model, c.width, c.height, "params", np.round(c.params, 4))
    print("registered frames:", len(frames), " points:", len(P))
    if markers:
        A, origin, s, cam_h = frame_from_markers(frames, cams, P, markers, size)
    elif len(sys.argv) > 4 and sys.argv[3] == "--like":
        A, origin, s = frame_like(frames, sys.argv[4])
        cam_h = float(np.load(sys.argv[4])["cam_h"])
    else:
        cam_h = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
        A, origin, s = frame_from_scratch(frames, P, E, TL, cam_h)

    centres = np.array([-R.T @ t for _, R, t, _ in frames])
    Pw = (P - origin) @ A.T * s
    Cw = (centres - origin) @ A.T * s
    Rw = np.array([R @ A.T for _, R, _, _ in frames])        # world -> camera
    names = np.array([f[0] for f in frames])
    intr = np.array([[cams[f[3]].params[0], cams[f[3]].params[1], cams[f[3]].params[2],
                      cams[f[3]].params[3] if len(cams[f[3]].params) > 3 else 0.0] for f in frames])
    yaw = np.degrees(np.arctan2(Rw[:, 2, 1], Rw[:, 2, 0]))
    pitch = np.degrees(np.arcsin(np.clip(Rw[:, 2, 2], -1, 1)))
    roll = np.degrees(np.arctan2(-Rw[:, 0, 2], -Rw[:, 1, 2]))
    clip_names = np.array([str(n).rsplit("_", 1)[0] for n in names])
    for prefix in sorted(set(clip_names)):
        m = clip_names == prefix
        c = Cw[m]
        path = float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())
        print(f"clip {prefix}: {int(m.sum())} frames, camera height {c[:, 2].mean():.2f} m "
              f"({c[:, 2].min():.2f}-{c[:, 2].max():.2f}), path length {path:.1f} m, "
              f"from ({c[0, 0]:.1f}, {c[0, 1]:.1f}) to ({c[-1, 0]:.1f}, {c[-1, 1]:.1f})")
        yaw[m] = np.degrees(np.unwrap(np.radians(yaw[m])))
    print("points z-range (m): 1%%..99%% = %.2f .. %.2f" % tuple(np.percentile(Pw[:, 2], [1, 99])))
    first = cams[frames[0][3]]
    home_names = np.array(sorted(set(homes)))
    home = np.searchsorted(home_names, np.array(homes)).astype(np.int16)
    np.savez(out, names=names, R=Rw, C=Cw, P=Pw, col=C, err=E, track=TL, intr=intr, home=home, home_names=home_names,
             K=np.array(first.params[:3]), k1=np.array(first.params[3] if len(first.params) > 3 else 0.0),
             size=np.array([first.width, first.height]), scale=s, cam_h=cam_h,
             yaw=yaw, pitch=pitch, roll=roll, A=A, origin=origin)
    print("saved", out)


if __name__ == "__main__":
    main()
