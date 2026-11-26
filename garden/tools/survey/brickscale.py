"""The size of a reconstructed piece from the brick courses of a wall in it.

    brickscale.py <part_model_dir> <clip> [--frames a:b] [--course 0.075] [--out wall.png] -- <frames_dir> [...]

A piece reconstructed on its own (sfm_part.py) has no size. A brick wall in it is a ruler:
courses are laid to a gauge, 75 mm for metric bricks (four courses to 300 mm), near
enough 3 inches for imperial ones. Counting courses by hand between two picked points
works but is fiddly and no better than the picking; this does it for every course at once:

  1. the wall is found as the largest flat surface among the points seen by the chosen
     frames that stands upright (its normal square to the direction the phone calls up);
  2. the frames are projected onto that plane to make a straightened picture of the wall
     in the piece's own units, as elevation.py does in the site grid;
  3. the picture's rows are averaged along the wall, which leaves the mortar beds as a
     regular ripple, and the ripple's wavelength is read off its spectrum.

The answer is metres per unit of the piece (for place.py --scale), with the strength of
the spectral peak as a guide to whether a ripple was really found: below about 4 it was
not. Always look at the straightened picture (--out): if it does not show a wall with
its courses running level across it, the plane found was something else (a clip that
only glimpses the wall through shrubs gives the lawn, and a meaningless number).

--frames a:b uses only frames numbered a to b of the clip (those that look at the wall).
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from place import Piece  # noqa: E402
from plan import clip_of, frame_no  # noqa: E402


def find_wall(P, up, rng, iters=3000, tol=0.012):
    """Largest plane among P whose normal is within 20 degrees of horizontal. tol in units of the piece."""
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        if abs(n @ up) > 0.34:
            continue
        cnt = int((np.abs((P - a) @ n) < tol).sum())
        if cnt > best_n:
            best_n, best = cnt, (n, a)
    n, a = best
    inl = P[np.abs((P - a) @ n) < tol]
    c0 = inl.mean(0)
    n = np.linalg.svd(inl - c0, full_matrices=False)[2][2]
    return n, c0, inl


def rectify(piece, names, find_image, n, c0, inl, up, res):
    """Straightened picture of the wall plane: columns along the wall, rows up it."""
    across = np.cross(up, n)
    across /= np.linalg.norm(across)
    upw = np.cross(n, across)
    a, b = (inl - c0) @ across, (inl - c0) @ upw
    a0, a1, b0, b1 = np.percentile(a, 2), np.percentile(a, 98), np.percentile(b, 2), np.percentile(b, 98)
    na, nb = int((a1 - a0) / res), int((b1 - b0) / res)
    A, B = np.meshgrid(a0 + (np.arange(na) + 0.5) * res, b1 - (np.arange(nb) + 0.5) * res)
    G = (c0 + A[..., None] * across + B[..., None] * upw).reshape(-1, 3)
    acc, wsum = np.zeros(len(G), np.float32), np.zeros(len(G), np.float32)
    for name in names:
        f = piece.frames[name]
        pc = (G - f["C"]) @ f["R"].T
        z = pc[:, 2]
        fl, cx, cy, k1 = f["intr"]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = pc[:, 0] / z, pc[:, 1] / z
        k = 1 + k1 * (x * x + y * y)
        u, v = fl * x * k + cx, fl * y * k + cy
        img = np.asarray(Image.open(find_image(name)).convert("L"), np.float32)
        H, W = img.shape
        ok = (z > 0.1) & (u >= 1) & (u < W - 2) & (v >= 1) & (v < H - 2)
        if not ok.any():
            continue
        iu, iv = u[ok].astype(int), v[ok].astype(int)
        fu, fv = u[ok] - iu, v[ok] - iv
        s = (img[iv, iu] * (1 - fu) * (1 - fv) + img[iv, iu + 1] * fu * (1 - fv) + img[iv + 1, iu] * (1 - fu) * fv + img[iv + 1, iu + 1] * fu * fv)
        w = 1.0 / np.maximum(z[ok], 0.3) ** 2
        acc[ok] += s * w
        wsum[ok] += w
    return np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), np.nan).reshape(nb, na)


def course_period(img, res, lo=0.04, hi=0.14):
    """Wavelength of the horizontal ripple, in units of the piece, and how far its spectral peak stands out."""
    cols = np.isfinite(img).mean(0) > 0.9
    rows = np.isfinite(img[:, cols]).mean(1) > 0.9
    prof = np.nanmean(img[rows][:, cols], axis=1)
    x = np.arange(len(prof))
    prof = prof - np.polyval(np.polyfit(x, prof, 3), x)                 # take out the shading
    n = 1 << int(np.ceil(np.log2(len(prof) * 16)))
    spec = np.abs(np.fft.rfft(prof * np.hanning(len(prof)), n))
    freq = np.fft.rfftfreq(n, res)
    band = (freq > 1 / hi) & (freq < 1 / lo)
    k = np.argmax(np.where(band, spec, 0))
    return 1.0 / freq[k], float(spec[k] / np.median(spec[band])), len(prof) * res


def main():
    a = sys.argv[1:]
    dirs = a[a.index("--") + 1:] if "--" in a else []
    a = a[:a.index("--")] if "--" in a else a
    opt = {}
    for key in ("--frames", "--course", "--out", "--res"):
        if key in a:
            i = a.index(key)
            opt[key] = a[i + 1]
            del a[i:i + 2]
    piece = Piece(a[0], {a[1]})
    lo, hi = (int(v) for v in opt.get("--frames", "0:1000000").split(":"))
    names = sorted(n for n in piece.frames if lo <= frame_no(n) <= hi)

    def find_image(name):
        for d in dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        raise SystemExit(f"{name} is in none of the frame directories given")

    ids = {piece.frames[n]["id"] for n in names}
    P = np.array([p.xyz for p in piece.rec.points3D.values() if p.track.length() >= 3 and sum(e.image_id in ids for e in p.track.elements) >= 2])
    up = -np.mean([piece.frames[n]["R"][1] for n in names], axis=0)
    up /= np.linalg.norm(up)
    cams = np.array([piece.frames[n]["C"] for n in names])
    unit = np.median(np.linalg.norm(P - cams.mean(0), axis=1))          # a typical distance: the piece's unit is near a metre, not exactly
    n, c0, inl = find_wall(P, up, np.random.default_rng(0), tol=0.006 * unit)
    res = float(opt.get("--res", 0.002))
    img = rectify(piece, names, find_image, n, c0, inl, up, res)
    period, peak, height = course_period(img, res)
    course = float(opt.get("--course", 0.075))
    print(f"{len(names)} frames, {len(P)} points, {len(inl)} of them on the wall; picture {img.shape[1]} x {img.shape[0]} px covering {height:.2f} units of height")
    print(f"course every {period:.4f} units of the piece (spectral peak {peak:.1f} times the background)")
    print(f"scale: {course / period:.4f} m per unit, taking a course as {course * 1000:.0f} mm")
    if "--out" in opt:
        v = np.nan_to_num(img, nan=0.0)
        Image.fromarray(np.clip(v, 0, 255).astype(np.uint8)).save(opt["--out"])


if __name__ == "__main__":
    main()
