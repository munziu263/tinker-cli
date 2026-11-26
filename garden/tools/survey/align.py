"""Turn a raw COLMAP reconstruction of the pan video into a metric, gravity-aligned survey.

    align.py <sparse_model_dir> <out.npz> [camera_height_m]

World frame after alignment: Z up, ground at z = 0, origin on the ground below the
mean camera position, metres. X/Y are rotated later (see layout notes) so the main
walls run along the axes.
"""
import sys

import numpy as np
import pycolmap


def load(model_dir):
    rec = pycolmap.Reconstruction(model_dir)
    cams = rec.cameras
    cam = next(iter(cams.values()))
    frames = []
    for iid, im in rec.images.items():
        if not im.has_pose:
            continue
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = np.asarray(T.rotation.matrix())
        t = np.asarray(T.translation)
        frames.append((im.name, R, t))
    frames.sort(key=lambda f: f[0])
    pts, cols, errs, tracks = [], [], [], []
    for pid, p in rec.points3D.items():
        pts.append(p.xyz)
        cols.append(p.color)
        errs.append(p.error)
        tracks.append(p.track.length())
    return rec, cam, frames, np.array(pts), np.array(cols), np.array(errs), np.array(tracks)


def estimate_up(frames):
    """Up = direction most perpendicular to every camera's right-axis (level phone)."""
    X = np.array([R[0] for _, R, _ in frames])
    w, v = np.linalg.eigh(X.T @ X)
    up = v[:, 0]
    down_mean = np.mean([R[1] for _, R, _ in frames], axis=0)
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
    # least-squares refit on the inliers
    c0 = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c0, full_matrices=False)
    n2 = vt[2]
    if n2 @ up < 0:
        n2 = -n2
    return n2, c0, len(inl), len(cand)


def main():
    model_dir, out = sys.argv[1], sys.argv[2]
    cam_h = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    rec, cam, frames, P, C, E, TL = load(model_dir)
    print("camera:", cam.model, cam.width, cam.height, "params", np.round(cam.params, 3))
    print("registered frames:", len(frames), " points:", len(P))
    centres = np.array([-R.T @ t for _, R, t in frames])
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

    # rotation taking sfm world -> Z-up world; X chosen as first camera's forward, flattened
    z = up
    f0 = frames[0][1][2]
    x = f0 - (f0 @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    A = np.stack([x, y, z])             # rows: new axes in old coords
    origin = cmean - ((cmean - c0) @ up) * up   # foot of the mean camera on the ground

    def to_world(p):
        return (np.asarray(p) - origin) @ A.T * s

    Pw = to_world(P)
    Cw = to_world(centres)
    Rw = np.array([R @ A.T for _, R, _ in frames])       # world(Z-up) -> camera
    names = np.array([f[0] for f in frames])
    span = Cw.max(0) - Cw.min(0)
    print(f"scale: {s:.4f} m per sfm unit (camera height set to {cam_h} m)")
    print("camera path extent (m):", np.round(span, 3))
    # radius of the circle the phone travelled on
    cc = Cw[:, :2].mean(0)
    rad = np.linalg.norm(Cw[:, :2] - cc, axis=1)
    print(f"camera path radius about its centroid: mean {rad.mean():.3f} m, sd {rad.std():.3f}")
    yaw = np.degrees(np.arctan2(Rw[:, 2, 1], Rw[:, 2, 0]))
    pitch = np.degrees(np.arcsin(np.clip(Rw[:, 2, 2], -1, 1)))
    roll = np.degrees(np.arctan2(-Rw[:, 0, 2], -Rw[:, 1, 2]))
    yaw_u = np.degrees(np.unwrap(np.radians(yaw)))
    print(f"yaw sweep: {yaw_u[0]:.1f} -> {yaw_u[-1]:.1f} deg (total {abs(yaw_u[-1]-yaw_u[0]):.1f})")
    print(f"pitch: min {pitch.min():.1f} max {pitch.max():.1f}; roll: mean {roll.mean():.2f} sd {roll.std():.2f}")
    print("points z-range (m): 1%%..99%% = %.2f .. %.2f" % tuple(np.percentile(Pw[:, 2], [1, 99])))
    np.savez(out, names=names, R=Rw, C=Cw, P=Pw, col=C, err=E, track=TL,
             K=np.array([cam.params[0], cam.params[1], cam.params[2]]),
             k1=np.array(cam.params[3] if len(cam.params) > 3 else 0.0),
             size=np.array([cam.width, cam.height]), scale=s, cam_h=cam_h,
             yaw=yaw_u, pitch=pitch, roll=roll)
    print("saved", out)


if __name__ == "__main__":
    main()
