"""Find printed survey markers in the frames, and fix them on the site.

    markers.py detect  <obs.json> [--ids 0-23] <frames_dir or image> [...]
    markers.py locate  <survey.npz> <obs.json> <tags.json> [--size 0.160] [--distances <distances.json>]
    markers.py control <obs.json> <tags.json> <control.json> [--per-corner 6]

The markers are the sheets that tools/capture/make_markers.py prints: AprilTags of the
36h11 family, fixed about the site and left there.

detect   reads every frame and writes where each marker's four corners are, to a fraction
         of a pixel. Nothing is picked by hand. Corners are in the pattern's own order:
         the corner the arrow on the sheet points at first, then clockwise.

locate   takes a survey that holds those frames and finds each corner in the site grid,
         as the point nearest to all the lines of sight to it (the pictures that show the
         marker large count for most). For each marker it prints where it is, how many
         frames saw it, how well their lines of sight meet, and the side of the black
         square as found. Blur rounds the corners of a marker that is small in the
         picture and draws them inward, evenly all round: in frames of video the square
         reads 2% small while its middle is right to a millimetre. So what is written to
         tags.json is a square of the printed size (--size: check a sheet with a ruler)
         laid on the four corners found, same middle, same turn.
         --distances compares taped distances with the survey's:
             {"corner": 0, "pairs": [[3, 7, 12.345], [3, 11, 8.02], ...]}
         each from the arrowed corner of one marker to the arrowed corner of another,
         in metres. Printed: each difference, and the one change of size that would fit
         them all best. A dozen distances across the site, long ones, are worth more
         than any number of short ones.

control  turns markers whose places are known (a tags.json from `locate`) and their
         detections in a new set of frames into control points for place.py --control:
         a set of photographs reconstructed by itself (sfm_part.py) is then laid on the
         site grid by the markers it shows. This is how a later visit is joined to an
         earlier one without the two sharing a single photograph.

A marker is read while its side covers about 30 pixels. Seen at a glancing angle it is
found less often and its corners are less sure, so photograph each one square-on from a
couple of metres at least once, and from two or three directions.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAMILY = cv2.aruco.DICT_APRILTAG_36h11


def detector():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementMaxIterations = 60
    p.cornerRefinementMinAccuracy = 0.01
    p.minMarkerPerimeterRate = 0.01                 # down to a few dozen pixels a side, whatever the size of the picture
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 5, 53, 8
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(FAMILY), p)


def find(gray, det, ids=None):
    """{marker number: 4 x 2 corners in pixels} for one picture."""
    corners, found, _ = det.detectMarkers(gray)
    out = {}
    if found is None:
        return out
    for c, k in zip(corners, found.ravel()):
        if ids is None or int(k) in ids:
            out[int(k)] = c.reshape(4, 2).astype(float)
    return out


def images(args):
    for a in args:
        if os.path.isdir(a):
            for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
                yield from sorted(glob.glob(os.path.join(a, f"*.{ext}")))
        else:
            yield a


def detect(a):
    ids = None
    if "--ids" in a:
        i = a.index("--ids")
        lo, hi = (int(v) for v in a[i + 1].split("-"))
        ids = set(range(lo, hi + 1))
        del a[i:i + 2]
    out, det, frames, seen, n = a[0], detector(), {}, {}, 0
    for path in images(a[1:]):
        # as the reconstruction sees them: the pixels as stored, whatever the file says about which way up it was held
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if gray is None:
            continue
        n += 1
        got = find(gray, det, ids)
        if got:
            frames[os.path.basename(path)] = {str(k): np.round(c, 2).tolist() for k, c in sorted(got.items())}
            for k, c in got.items():
                seen.setdefault(k, []).append(float(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean()))
    with open(out, "w") as fh:
        json.dump({"family": "apriltag_36h11", "frames": frames}, fh)
    print(f"{n} pictures read; markers found in {len(frames)} of them")
    for k in sorted(seen):
        s = np.array(seen[k])
        print(f"   marker {k:2d}: in {len(s):4d} pictures, side {s.min():.0f} to {s.max():.0f} px (median {np.median(s):.0f})")
    print("saved", out)


def meet(C, D, say=None, rounds=4, soft=0.01):
    """The point nearest to lines of sight (centres C, unit directions D), and how far each line passes from it.
    say: how much each line counts to begin with. Lines that pass wide are given less say each round."""
    say = np.ones(len(C)) if say is None else np.asarray(say, float)
    w = say.copy()
    for _ in range(rounds):
        M = np.eye(3)[None] - D[:, :, None] * D[:, None, :]
        A = (w[:, None, None] * M).sum(0)
        b = (w[:, None, None] * M @ C[:, :, None]).sum(0).ravel()
        p = np.linalg.solve(A, b)
        miss = np.linalg.norm(np.einsum("nij,nj->ni", M, p - C), axis=1)
        w = say / (1.0 + (miss / soft) ** 2)
    return p, miss


def square_on(c, size):
    """A square of the printed size laid on four found corners: the same middle, the same plane and turn."""
    h = size / 2
    T = np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0.0]])
    m = c.mean(0)
    U, _, Vt = np.linalg.svd(T.T @ (c - m))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return T @ R.T + m


def locate(a):
    from plan import Survey
    opt = {}
    for key in ("--size", "--distances"):
        if key in a:
            i = a.index(key)
            opt[key] = a[i + 1]
            del a[i:i + 2]
    size = float(opt.get("--size", 0.160))
    sv = Survey(a[0], [])
    with open(a[1]) as fh:
        obs = json.load(fh)["frames"]
    where = {n: i for i, n in enumerate(sv.names)}
    rays = {}                                                    # marker -> corner -> [(frame index, u, v, side of the square in px)]
    unknown = 0
    for name, tags in obs.items():
        name = name if name in where else os.path.splitext(name)[0] + ".jpg"
        if name not in where:
            unknown += 1
            continue
        for k, c in tags.items():
            side = float(np.linalg.norm(np.array(c) - np.roll(np.array(c), 1, axis=0), axis=1).mean())
            for j, (u, v) in enumerate(c):
                rays.setdefault(int(k), {}).setdefault(j, []).append((where[name], u, v, side))
    if unknown:
        print(f"{unknown} pictures with markers are not in this survey and were passed over")
    tags = {}
    print(f"{'marker':>6s} {'X':>8s} {'Y':>8s} {'z':>7s}   pictures  spread  lines meet   in the picture   side of the square")
    for k in sorted(rays):
        if len(rays[k]) < 4:
            continue
        corners, misses, px, n, spread = [], [], [], 0, 0.0
        for j in range(4):
            r = rays[k][j]
            C = np.array([sv.centre(i) for i, *_ in r])
            D = np.array([sv.ray(i, u, v) for i, u, v, _ in r])
            # a marker that is small in the picture has its corners rounded off by blur and read a little
            # inside where they are: the pictures that show it large have the say
            say = np.clip(np.array([side for *_, side in r]) / 50.0, 0.15, 1.0) ** 2
            if len(r) < 2:
                corners = []
                break
            p, miss = meet(C, D, say)
            keep = miss < max(0.03, 3 * np.median(miss))
            if keep.sum() >= 2 and not keep.all():
                p, m2 = meet(C[keep], D[keep], say[keep])
                miss = np.where(keep, np.linalg.norm(np.cross(p - C, D), axis=1), miss)
            corners.append(p)
            misses.append(np.median(miss[keep]) if keep.any() else np.median(miss))
            for (i, u, v, _), good in zip(r, keep):
                if good:
                    uu, vv, _ = sv.project(i, p)
                    px.append(np.hypot(uu - u, vv - v))
            n = max(n, int(keep.sum()))
            cosines = np.clip(D[keep] @ D[keep].T, -1, 1) if keep.sum() >= 2 else np.ones((1, 1))
            spread = max(spread, float(np.degrees(np.arccos(cosines.min()))))
        if len(corners) < 4:
            continue
        c = np.array(corners)
        sides = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
        centre = c.mean(0)
        flat = float(np.abs((c - centre) @ np.linalg.svd(c - centre)[2][2]).max())
        weak = spread < 4.0
        c = square_on(c, size)                      # blur draws a small marker's corners inward, evenly: its middle and turn are surer than its edges
        tags[k] = dict(corners=np.round(c, 4).tolist(), centre=np.round(centre, 4).tolist(), side=float(sides.mean()), pictures=n,
                       spread_deg=round(spread, 1), miss_m=float(np.mean(misses)), rms_px=float(np.sqrt(np.mean(np.square(px)))), weak=bool(weak))
        print(f"{k:6d} {centre[0]:8.3f} {centre[1]:8.3f} {centre[2]:7.3f}   {n:6d}   {spread:5.1f}°   {np.mean(misses) * 1000:6.1f} mm   {tags[k]['rms_px']:6.2f} px rms    "
              f"{sides.mean() * 1000:6.1f} mm ({(sides.mean() / size - 1) * 100:+.1f}%), corners {flat * 1000:.1f} mm off flat" + ("   WEAK: seen from too nearly one direction" if weak else ""))
    good = [t for t in tags.values() if not t["weak"]]
    if good:
        ratio = np.median([t["side"] for t in good]) / size
        print(f"{len(good)} markers fixed. Their squares come out {ratio:.4f} times the printed {size * 1000:.0f} mm (median): "
              "that is the survey's size against the markers', spot by spot. A square is too short a ruler to set the size by; taped distances between markers are.")
    if "--distances" in opt:
        with open(opt["--distances"]) as fh:
            dist = json.load(fh)
        j = int(dist.get("corner", 0))
        rows = [(p, q, m, float(np.linalg.norm(np.array(tags[p]["corners"][j]) - np.array(tags[q]["corners"][j]))))
                for p, q, m in dist["pairs"] if p in tags and q in tags]
        print(f"taped distances, from the arrowed corner of one marker to that of another ({len(rows)} of {len(dist['pairs'])} between markers that were fixed):")
        for p, q, m, s in rows:
            print(f"   {p:2d} to {q:2d}: tape {m:7.3f} m, survey {s:7.3f} m, difference {(s - m) * 1000:+6.0f} mm ({(s / m - 1) * 100:+.2f}%)")
        if rows:
            m, s = np.array([r[2] for r in rows]), np.array([r[3] for r in rows])
            k = float((m * s).sum() / (s * s).sum())
            print(f"   all together: the survey would fit the tape best at {k:.4f} times its size; "
                  f"differences {np.sqrt(np.mean((s - m) ** 2)) * 1000:.0f} mm rms as it stands, {np.sqrt(np.mean((k * s - m) ** 2)) * 1000:.0f} mm rms at that size")
    with open(a[2], "w") as fh:
        json.dump({"size": size, "tags": {str(k): t for k, t in tags.items()}}, fh, indent=1)
    print("saved", a[2])


def control(a):
    per = 6
    if "--per-corner" in a:
        i = a.index("--per-corner")
        per = int(a[i + 1])
        del a[i:i + 2]
    with open(a[0]) as fh:
        obs = json.load(fh)["frames"]
    with open(a[1]) as fh:
        tags = json.load(fh)["tags"]
    best = {}                                                     # marker -> [(side in px, frame, corners)]
    for name, found in obs.items():
        for k, c in found.items():
            if k in tags and not tags[k].get("weak"):
                c = np.array(c)
                best.setdefault(k, []).append((float(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1).mean()), name, c))
    points = []
    for k, seen in sorted(best.items(), key=lambda kv: int(kv[0])):
        for _, name, c in sorted(seen, key=lambda t: -t[0])[:per]:           # the pictures that show it largest
            for j in range(4):
                points.append({"name": f"marker {k} corner {j}", "site": tags[k]["corners"][j], "frame": name, "px": [float(c[j, 0]), float(c[j, 1])]})
    with open(a[2], "w") as fh:
        json.dump({"points": points}, fh)
    print(f"{len(points)} control points from {len(best)} markers of known place; saved {a[2]}")


if __name__ == "__main__":
    cmd = {"detect": detect, "locate": locate, "control": control}.get(sys.argv[1] if len(sys.argv) > 1 else "")
    if cmd is None:
        raise SystemExit(__doc__)
    cmd(sys.argv[2:])
