"""Put a separately reconstructed piece (see sfm_part.py) into the site grid and add it to a survey.

    place.py <part_model_dir> <survey.npz> <out.npz> --db <work_dir>/db.db
             [--clips <clip,clip,...>] [--scale <metres per unit of the piece>] [--control <control.json>]
             [--shared <clip,clip,...>] [--add <clip,clip,...>] [--frames <frames_dir> ...]

A piece reconstructed on its own has the right shape and an arbitrary size, position and
direction: seven numbers to find. Three kinds of evidence fix them, and whatever is
available is used together in one least-squares fit:

  sightings   A frame that is already in the survey (and not in the piece) has a known
              position, and where it shows a feature that the piece reconstructed in 3D,
              the piece's point must lie on that frame's line of sight. The matches are
              already in the database, so this needs no picking by hand. A standing pan
              is ideal for it: poor at distances, excellent at directions. Sightings from
              one spot cannot tell a small piece nearby from a big one far away, though,
              so the size must come from somewhere else:
  --scale     the piece's own size, in metres per unit of the piece. From something in it
              of known length: brick courses (brickscale.py), or a paving flag or a door
              measured between two points with `--probe` (below). Or by trial: placed from
              a pan's sightings, the piece slides along the line of sight as the scale
              changes, and up or down with it, so the right scale is the one at which the
              piece's own ground comes out level with the ground the survey has there.
  --shared    clips that are in the piece and also sound in the survey: their cameras
              should land where the survey has them (this alone is `align.py --like`).
  --control   points with known site coordinates picked in frames of the piece:
              {"points": [{"name": "...", "site": [x, y, z], "frame": "c3363_0150", "px": [u, v]}, ...]}

The fit is reported class by class, so a wrong assumption shows: sightings should agree
to a pixel or two. If they do for some survey frames and are hundreds of pixels out for
others, the piece is not rigid: it has a weak hinge somewhere (two clips that barely
overlap). Place each rigid length by itself with --clips.

The frames and points of the clips named by --add (default: every clip of the piece that
is not in the survey) are then written, with the survey's own, to <out.npz>.

    place.py <part_model_dir> --probe <frame> <u> <v> <frame> <u> <v>   [--frames <frames_dir> ...]

prints the 3D point of the piece seen at one pixel in two of its frames (in the piece's own
units), for measuring a known length. Give two such points to get a distance.
"""
import json
import os
import sqlite3
import sys

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey, clip_of  # noqa: E402

PAIR = 2147483647


class Piece:
    def __init__(self, model_dir, clips=None):
        """clips: use only these clips of the piece (a long walk can come out as rigid lengths
        joined by a weak hinge where two clips barely overlap: place each length by itself)."""
        self.rec = pycolmap.Reconstruction(model_dir)
        self.frames = {}
        for iid, im in self.rec.images.items():
            if not im.has_pose or (clips and clip_of(im.name) not in clips):
                continue
            T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R, t = np.asarray(T.rotation.matrix()), np.asarray(T.translation)
            cam = self.rec.cameras[im.camera_id]
            p = cam.params
            self.frames[im.name] = dict(id=iid, R=R, C=-R.T @ t, intr=np.array([p[0], p[1], p[2], p[3] if len(p) > 3 else 0.0]), im=im)
        self.xyz = {pid: np.asarray(p.xyz) for pid, p in self.rec.points3D.items()}

    def point_of_keypoint(self, name):
        """keypoint index -> 3D point id, for one frame"""
        out = {}
        for k, p2 in enumerate(self.frames[name]["im"].points2D):
            if p2.has_point3D():
                out[k] = p2.point3D_id
        return out

    def ray(self, name, u, v):
        f = self.frames[name]
        fl, cx, cy, k1 = f["intr"]
        xd, yd = (u - cx) / fl, (v - cy) / fl
        x, y = xd, yd
        for _ in range(8):
            k = 1 + k1 * (x * x + y * y)
            x, y = xd / k, yd / k
        d = np.array([x, y, 1.0])
        return f["C"], f["R"].T @ (d / np.linalg.norm(d))

    def triangulate(self, picks):
        A, b = np.zeros((3, 3)), np.zeros(3)
        for name, u, v in picks:
            c, r = self.ray(name, float(u), float(v))
            M = np.eye(3) - np.outer(r, r)
            A += M
            b += M @ c
        return np.linalg.solve(A, b)


def sightings(piece, sv, db_path, min_inliers=15):
    """(survey frame index, u, v, piece point id) for every verified match between a frame
    that is only in the survey and a keypoint of the piece that has a 3D point."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ids = dict((n, i) for i, n in con.execute("select image_id, name from images"))
    names = {i: n for n, i in ids.items()}
    in_piece = {ids[n] for n in piece.frames if n in ids}
    in_survey = {ids[n]: k for k, n in enumerate(sv.names) if n in ids and n not in piece.frames}
    out = []
    kp_cache, map_cache = {}, {}

    def keypoints(i):
        if i not in kp_cache:
            r, c, blob = con.execute("select rows, cols, data from keypoints where image_id=?", (int(i),)).fetchone()
            kp_cache[i] = np.frombuffer(blob, np.float32).reshape(r, c)[:, :2]
        return kp_cache[i]

    pairs = con.execute("select pair_id, rows from two_view_geometries where rows >= ?", (min_inliers,)).fetchall()
    used = 0
    for pid, _ in pairs:
        a, b = divmod(pid, PAIR)
        if a in in_piece and b in in_survey:
            p_id, s_id, col_p, col_s = a, b, 0, 1
        elif b in in_piece and a in in_survey:
            p_id, s_id, col_p, col_s = b, a, 1, 0
        else:
            continue
        rows, cols, blob = con.execute("select rows, cols, data from two_view_geometries where pair_id=?", (pid,)).fetchone()
        M = np.frombuffer(blob, np.uint32).reshape(rows, cols)
        if names[p_id] not in map_cache:
            map_cache[names[p_id]] = piece.point_of_keypoint(names[p_id])
        k2p = map_cache[names[p_id]]
        ks = keypoints(s_id)
        n = 0
        for row in M:
            pt = k2p.get(int(row[col_p]))
            if pt is not None:
                u, v = ks[int(row[col_s])]
                out.append((in_survey[s_id], float(u), float(v), pt))
                n += 1
        used += n > 0
    con.close()
    return out, used


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (Xc ** 2).sum() * len(X))
    return s, R, my - s * R @ mx


def project(sv, i, P):
    f, cx, cy, k1 = sv.intr[i]
    pc = (P - sv.C[i]) @ sv.R[i].T
    z = np.maximum(pc[:, 2], 1e-6)
    x, y = pc[:, 0] / z, pc[:, 1] / z
    k = 1 + k1 * (x * x + y * y)
    return np.stack([f * x * k + cx, f * y * k + cy], 1), pc[:, 2]


def first_guess(piece, sv, sight, scale):
    """Rotation and position from resecting survey frames against the piece's points (one
    frame fixes the rotation; the scale is given, or comes from how far apart the resected
    frames are, which is weak if they stand close together)."""
    import cv2
    by_frame = {}
    for i, u, v, pt in sight:
        by_frame.setdefault(i, []).append((u, v, pt))
    cands = []
    for i, obs in sorted(by_frame.items(), key=lambda kv: -len(kv[1]))[:12]:
        if len(obs) < 12:
            continue
        f, cx, cy, k1 = sv.intr[i]
        uv = np.array([(u, v) for u, v, _ in obs])
        xd, yd = (uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f
        x, y = xd, yd
        for _ in range(8):
            k = 1 + k1 * (x * x + y * y)
            x, y = xd / k, yd / k
        X = np.array([piece.xyz[pt] for _, _, pt in obs])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(X, np.stack([x, y], 1), np.eye(3), None, reprojectionError=6.0 / f, iterationsCount=2000,
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 10:
            continue
        ok, rvec, tvec = cv2.solvePnP(X[inl[:, 0]], np.stack([x, y], 1)[inl[:, 0]], np.eye(3), None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        Rcp = cv2.Rodrigues(rvec)[0]
        cands.append((len(inl), i, sv.R[i].T @ Rcp, sv.R[i].T @ tvec.ravel(), sv.C[i]))     # R, t = C + s * tt
    if not cands:
        raise SystemExit("no survey frame sees enough of the piece to start from: give --shared or --control")
    cands.sort(key=lambda c: -c[0])
    n, i, R, tt, C = cands[0]
    if scale is None:
        if len(cands) < 2:
            raise SystemExit("one sighting frame only: the size cannot be found from it, give --scale")
        num = den = 0.0
        for _, _, R2, tt2, C2 in cands[1:]:
            d = tt - tt2
            num += float(d @ (C2 - C))
            den += float(d @ d)
        scale = abs(num / den)
    return scale, R, C + scale * tt


def bearing(sv, i, u, v):
    """Unit direction, in the survey's frame, of the line of sight of pixel (u, v) of frame i."""
    f, cx, cy, k1 = sv.intr[i]
    xd, yd = (u - cx) / f, (v - cy) / f
    x, y = xd, yd
    for _ in range(8):
        k = 1 + k1 * (x * x + y * y)
        x, y = xd / k, yd / k
    d = np.array([x, y, 1.0])
    return sv.R[i].T @ (d / np.linalg.norm(d))


def pooled_guess(piece, sv, sight, scale, iters=6000, tol_px=5.0, seed=0):
    """Start from all the sightings at once, for sighting frames that stand in nearly the
    same place (a pan). Together they are one camera that sees all round, and three
    sightings fix where the piece is in relation to it. Tried on many random threes, the
    answer that the most sightings agree with wins, so wrong matches do no harm. Returns
    (scale, R, t, number agreeing) or None if the sighting frames are not close together."""
    import cv2
    C = np.array([sv.C[i] for i, *_ in sight])
    near = np.linalg.norm(C - np.median(C, axis=0), axis=1) < 1.2     # the frames that stand together; any others sit this out
    if near.sum() < 12 or near.mean() < 0.7:
        return None
    sight = [q for q, ok in zip(sight, near) if ok]
    C = C[near]
    A = C.mean(0)
    D = np.array([bearing(sv, i, u, v) for i, u, v, _ in sight])
    X = np.array([piece.xyz[pt] for *_, pt in sight]) * (scale or 1.0)
    f = float(np.median([sv.intr[i][0] for i, *_ in sight]))
    tol = tol_px / f
    rng = np.random.default_rng(seed)
    best = (0, None, None)
    for _ in range(iters):
        k = rng.choice(len(sight), 3, replace=False)
        m = D[k].mean(0)
        if np.linalg.norm(m) < 0.5:                                   # the three look in very different directions: skip
            continue
        z = m / np.linalg.norm(m)
        x = np.cross([0.0, 0.0, 1.0], z)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
        Q = np.stack([x, np.cross(z, x), z])                          # turns the survey's frame so the three look along +z
        d = D[k] @ Q.T
        if d[:, 2].min() < 0.2:
            continue
        try:
            n, rvecs, tvecs = cv2.solveP3P(X[k].astype(np.float64), (d[:, :2] / d[:, 2:3]).astype(np.float64), np.eye(3), None, flags=cv2.SOLVEPNP_P3P)
        except cv2.error:
            continue
        for rv, tv in zip(rvecs, tvecs):
            R = Q.T @ cv2.Rodrigues(rv)[0]
            t = Q.T @ tv.ravel() + A
            V = X @ R.T + t - C
            ang = np.arccos(np.clip((V * D).sum(1) / np.maximum(np.linalg.norm(V, axis=1), 1e-9), -1, 1))
            cnt = int((ang < tol).sum())
            if cnt > best[0]:
                best = (cnt, R, t)
    if best[1] is None:
        return None
    return (scale or 1.0), best[1], best[2], best[0]


def control_guess(piece, control, scale):
    """Start from control points: the frame holding the most of them (four or more) is
    resected against their site coordinates, and since that frame's place in the piece is
    known, so is the piece's place on the site. Needs the scale."""
    import cv2
    by = {}
    for c in control:
        by.setdefault(c["frame"], []).append(c)
    name, pts = max(by.items(), key=lambda kv: len(kv[1]))
    if len(pts) < 4 or scale is None:
        return None
    f = piece.frames[name]
    fl, cx, cy, k1 = f["intr"]
    K = np.array([[fl, 0, cx], [0, fl, cy], [0, 0, 1.0]])
    obj = np.array([c["site"] for c in pts], float)
    img = np.array([c["px"] for c in pts], float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, np.array([k1, 0, 0, 0.0]), flags=cv2.SOLVEPNP_EPNP if len(pts) >= 6 else cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, np.array([k1, 0, 0, 0.0]), rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    Rs = cv2.Rodrigues(rvec)[0]                       # site -> this camera
    Cs = -Rs.T @ tvec.ravel()                         # the camera's position on the site
    R = Rs.T @ f["R"]                                 # piece -> site
    return scale, R, Cs - scale * R @ f["C"], name, Cs


def solve(piece, sv, sight, shared, control, scale, guess):
    s0, R0, t0 = guess
    pts = sorted({pt for _, _, _, pt in sight})
    index = {pt: k for k, pt in enumerate(pts)}
    X = np.array([piece.xyz[pt] for pt in pts]) if pts else np.zeros((0, 3))
    by_frame = {}
    for i, u, v, pt in sight:
        by_frame.setdefault(i, []).append((u, v, index[pt]))
    groups = [(i, np.array([(u, v) for u, v, _ in o]), np.array([k for _, _, k in o])) for i, o in by_frame.items()]
    Xs = np.array([piece.frames[n]["C"] for n, _ in shared]) if shared else np.zeros((0, 3))
    Ys = np.array([sv.C[k] for _, k in shared]) if shared else np.zeros((0, 3))

    def unpack(p):
        return (scale if scale is not None else s0 * np.exp(p[6])), Rotation.from_rotvec(p[:3]).as_matrix() @ R0, t0 + p[3:6]

    def residuals(p, parts=False):
        s, R, t = unpack(p)
        out = {}
        if len(X):
            W = s * X @ R.T + t
            r = []
            for i, uv, k in groups:
                q, z = project(sv, i, W[k])
                r.append(np.where(z[:, None] > 0.2, q - uv, 300.0))
            out["sightings"] = np.concatenate(r).ravel()
        if len(Xs):
            out["shared"] = ((s * Xs @ R.T + t) - Ys).ravel() * 300.0          # 1 cm counts as 3 px
        if control:
            r = []
            for c in control:
                f = piece.frames[c["frame"]]
                Cw, Rw = s * R @ f["C"] + t, f["R"] @ R.T
                pc = Rw @ (np.asarray(c["site"], float) - Cw)
                fl, cx, cy, k1 = f["intr"]
                x, y = pc[0] / pc[2], pc[1] / pc[2]
                k = 1 + k1 * (x * x + y * y)
                r += [fl * x * k + cx - c["px"][0], fl * y * k + cy - c["px"][1]]
            out["control"] = np.array(r) * 3.0                               # hand-picked, so few: weigh them up
        return out if parts else np.concatenate(list(out.values()))

    fit = least_squares(residuals, np.zeros(7 if scale is None else 6), loss="soft_l1", f_scale=3.0, max_nfev=200)
    return unpack(fit.x), residuals(fit.x, parts=True)


def keep_agreeing(piece, sv, sight, s, R, t, tol_px=25.0):
    """Drop the sightings that are far out at the starting placement: wrong matches."""
    out = []
    for i, u, v, pt in sight:
        q, z = project(sv, i, (s * R @ piece.xyz[pt] + t)[None])
        if z[0] > 0.2 and np.hypot(q[0, 0] - u, q[0, 1] - v) < tol_px:
            out.append((i, u, v, pt))
    return out


def angles(Rw):
    yaw = np.degrees(np.arctan2(Rw[:, 2, 1], Rw[:, 2, 0]))
    pitch = np.degrees(np.arcsin(np.clip(Rw[:, 2, 2], -1, 1)))
    roll = np.degrees(np.arctan2(-Rw[:, 0, 2], -Rw[:, 1, 2]))
    return yaw, pitch, roll


def merge(sv_path, out_path, piece, s, R, t, add):
    d = dict(np.load(sv_path, allow_pickle=False))
    have = {str(n) for n in d["names"]}
    names = sorted(n for n in piece.frames if clip_of(n) in add and n not in have)
    Rw = np.array([piece.frames[n]["R"] @ R.T for n in names])
    Cw = np.array([s * R @ piece.frames[n]["C"] + t for n in names])
    intr = np.array([piece.frames[n]["intr"] for n in names])
    yaw, pitch, roll = angles(Rw)
    clips = np.array([clip_of(n) for n in names])
    for c in sorted(set(clips)):
        m = clips == c
        yaw[m] = np.degrees(np.unwrap(np.radians(yaw[m])))
    keep = {piece.frames[n]["id"] for n in names}
    P, col, err, trk = [], [], [], []
    for pid, p in piece.rec.points3D.items():
        if any(e.image_id in keep for e in p.track.elements):
            P.append(s * R @ np.asarray(p.xyz) + t)
            col.append(p.color)
            err.append(p.error)
            trk.append(p.track.length())
    for key, extra in (("names", np.array(names)), ("R", Rw), ("C", Cw), ("intr", intr), ("yaw", yaw), ("pitch", pitch), ("roll", roll),
                       ("P", np.array(P)), ("col", np.array(col, d["col"].dtype)), ("err", np.array(err)), ("track", np.array(trk))):
        d[key] = np.concatenate([d[key], extra.astype(d[key].dtype) if key != "names" else extra]) if len(extra) else d[key]
    order = np.argsort(d["names"], kind="stable")
    for key in ("names", "R", "C", "intr", "yaw", "pitch", "roll"):
        d[key] = d[key][order]
    np.savez(out_path, **d)
    for c in sorted(set(clips)):
        m = clips == c
        path = float(np.linalg.norm(np.diff(Cw[m], axis=0), axis=1).sum())
        print(f"clip {c}: {int(m.sum())} frames, camera height {Cw[m][:, 2].mean():.2f} m, path length {path:.1f} m, "
              f"from ({Cw[m][0, 0]:.1f}, {Cw[m][0, 1]:.1f}) to ({Cw[m][-1, 0]:.1f}, {Cw[m][-1, 1]:.1f})")
    print(f"added {len(names)} frames and {len(P)} points; {out_path} now has {len(d['names'])} frames and {len(d['P'])} points")


def main():
    a = sys.argv[1:]
    opt, flags = {}, ("--db", "--scale", "--control", "--shared", "--add", "--clips")
    frames_dirs = []
    if "--frames" in a:
        i = a.index("--frames")
        frames_dirs, a = a[i + 1:], a[:i]
    if "--probe" in a:
        piece = Piece(a[0])
        v = a[a.index("--probe") + 1:]
        picks = [(v[k] if v[k].endswith(".jpg") else v[k] + ".jpg", float(v[k + 1]), float(v[k + 2])) for k in range(0, len(v), 3)]
        p = piece.triangulate(picks)
        miss = max(np.linalg.norm((np.eye(3) - np.outer(r, r)) @ (p - c)) for c, r in (piece.ray(*q) for q in picks))
        print("point", " ".join(f"{x:.5f}" for x in p), " rays miss by", f"{miss:.5f}", "(units of the piece)")
        return
    for key in flags:
        if key in a:
            i = a.index(key)
            opt[key] = a[i + 1]
            del a[i:i + 2]
    piece = Piece(a[0], set(opt["--clips"].split(",")) if "--clips" in opt else None)
    sv = Survey(a[1], frames_dirs)
    scale = float(opt["--scale"]) if "--scale" in opt else None
    shared_clips = set(opt.get("--shared", "").split(",")) - {""}
    idx = {n: k for k, n in enumerate(sv.names)}
    shared = [(n, idx[n]) for n in piece.frames if n in idx and clip_of(n) in shared_clips]
    control = json.load(open(opt["--control"]))["points"] if "--control" in opt else []
    for c in control:
        c["frame"] = c["frame"] if c["frame"].endswith(".jpg") else c["frame"] + ".jpg"
    sight, n_pairs = sightings(piece, sv, opt["--db"]) if "--db" in opt else ([], 0)
    print(f"piece: {len(piece.frames)} frames, {len(piece.xyz)} points. Evidence: {len(sight)} sightings from {len({i for i, *_ in sight})} survey frames "
          f"({n_pairs} matched pairs), {len(shared)} shared cameras, {len(control)} control points" + (f", scale given as {scale}" if scale else ""))
    if len(shared) >= 3:
        s0, R0, t0 = umeyama(np.array([piece.frames[n]["C"] for n, _ in shared]), np.array([sv.C[k] for _, k in shared]))
        if scale is not None:
            s0 = scale
    elif control and control_guess(piece, control, scale) is not None:
        s0, R0, t0, name, Cs = control_guess(piece, control, scale)
        print(f"start: {name} resected from its control points, standing at ({Cs[0]:.2f}, {Cs[1]:.2f}, {Cs[2]:.2f})")
        n0 = len(sight)
        sight = keep_agreeing(piece, sv, sight, s0, R0, t0, tol_px=60.0)
        print(f"  {len(sight)} of {n0} sightings are within 60 px of that placing and are kept")
    else:
        pooled = pooled_guess(piece, sv, sight, scale) if scale is not None else None
        if pooled is not None and pooled[3] >= 12:
            s0, R0, t0, agree = pooled
            print(f"start: the sighting frames stand together (a pan); {agree} of {len(sight)} sightings agree on one placing to 5 px")
            sight = keep_agreeing(piece, sv, sight, s0, R0, t0)
        else:
            s0, R0, t0 = first_guess(piece, sv, sight, scale)
    (s, R, t), res = solve(piece, sv, sight, shared, control, scale, (s0, R0, t0))
    print(f"scale {s:.5f} m per unit of the piece" + ("" if scale is None else " (as given)"))
    if "sightings" in res:
        r = np.linalg.norm(res["sightings"].reshape(-1, 2), axis=1)
        print(f"sightings: median {np.median(r):.1f} px, 80% within {np.percentile(r, 80):.1f} px, {100 * (r < 8).mean():.0f}% within 8 px")
        by_clip, k = {}, 0
        order = {}
        for i, *_ in sight:
            order.setdefault(i, []).append(None)
        for i, obs in order.items():                    # the residuals are grouped by survey frame, in this order
            by_clip.setdefault(clip_of(sv.names[i]), []).append(r[k:k + len(obs)])
            k += len(obs)
        print("  by clip of the sighting frame: " + ", ".join(f"{c} {np.median(np.concatenate(v)):.1f} px ({sum(len(x) for x in v)})" for c, v in sorted(by_clip.items())))
    if "shared" in res:
        r = np.linalg.norm(res["shared"].reshape(-1, 3), axis=1) / 300.0
        print(f"shared cameras: mean {r.mean() * 100:.1f} cm apart, worst {r.max() * 100:.1f} cm")
    if "control" in res:
        for c, r in zip(control, res["control"].reshape(-1, 2) / 3.0):
            print(f"control {c.get('name', '?'):30s} {np.hypot(*r):6.1f} px off")
    add = set(opt["--add"].split(",")) if "--add" in opt else {clip_of(n) for n in piece.frames if n not in idx}
    merge(a[1], a[2], piece, s, R, t, add)


if __name__ == "__main__":
    main()
