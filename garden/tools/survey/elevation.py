"""A straightened photograph of a wall, to scale: windows and doors can be read off it.

    elevation.py <survey.npz> <out.png> <xa,ya> <xb,yb> [--z z0,z1] [--res 0.01] [--clips a,b,...]
                 [--range 8] [--grid 0.5] -- <frames_dir> [...]

The wall is the vertical plane through the two plan points given (site grid, metres): a
is the left-hand end and b the right-hand end as you look at the wall from the side the
camera was on. Every frame that faces the wall is projected onto that plane and the
results are averaged, weighted towards frames that look at it squarely and from close
by. Whatever lies in the plane comes out sharp and to scale; anything in front of it or
behind it (a shrub, a sill, a porch) is smeared and shifted, which is how to tell.

The picture is ruled every `grid` metres: across, the distance along the wall from a;
up, the height above the survey datum. With --points the survey's own points lying within
5 cm of the plane are dotted in, a check that the plane is where the wall is.

Fit the plane first: take the wall's two ends from a plan of the points (the points on a
rendered wall fall within a couple of centimetres of a line).

In code:  im, geo = facade(sv, a, b, z0, z1, res, frames)   ->   PIL image, (z1, res)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey, clip_of  # noqa: E402


def facade(sv, a, b, z0=0.0, z1=2.5, res=0.01, frames=None, max_range=8.0, min_cos=0.35):
    """Rectified picture of the vertical plane through plan points a and b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    d = (b - a) / length
    normal = np.array([-d[1], d[0], 0.0])                 # horizontal, pointing to the left of a->b ...
    normal = -normal                                       # ... so towards a viewer who has a on the left
    ns, nz = int(round(length / res)), int(round((z1 - z0) / res))
    s = (np.arange(ns) + 0.5) * res
    z = z1 - (np.arange(nz) + 0.5) * res
    S, Z = np.meshgrid(s, z)
    G = np.stack([a[0] + S * d[0], a[1] + S * d[1], Z], -1).reshape(-1, 3)
    acc = np.zeros((ns * nz, 3), np.float32)
    wsum = np.zeros(ns * nz, np.float32)
    used = 0
    for i in (frames if frames is not None else range(len(sv.names))):
        c = sv.centre(i)
        front = float((c - np.array([a[0], a[1], c[2]])) @ normal)
        if front < 0.3:                                    # behind the wall, or in it
            continue
        W, H = sv.frame_size(i)
        to_cam = c - G
        dist = np.linalg.norm(to_cam, axis=1)
        cosang = (to_cam @ normal) / np.maximum(dist, 1e-6)
        u, v, zc = sv.project(i, G)
        ok = (zc > 0.2) & (u >= 1) & (u < W - 2) & (v >= 1) & (v < H - 2) & (dist < max_range) & (cosang > min_cos)
        if not ok.any():
            continue
        used += 1
        img = np.asarray(sv.image(i), np.float32)
        uu, vv = u[ok], v[ok]
        iu, iv = uu.astype(int), vv.astype(int)
        fu, fv = (uu - iu)[:, None], (vv - iv)[:, None]
        px = (img[iv, iu] * (1 - fu) * (1 - fv) + img[iv, iu + 1] * fu * (1 - fv)
              + img[iv + 1, iu] * (1 - fu) * fv + img[iv + 1, iu + 1] * fu * fv)
        wgt = (1 - ((uu - W / 2) / (W / 2)) ** 2) * (1 - ((vv - H / 2) / (H / 2)) ** 2)
        wgt = np.clip(wgt, 0.02, 1) ** 2 * cosang[ok] ** 2 / np.maximum(dist[ok], 0.5) ** 2
        acc[ok] += px * wgt[:, None]
        wsum[ok] += wgt
    out = np.where(wsum[:, None] > 0, acc / np.maximum(wsum[:, None], 1e-9), 40)
    return Image.fromarray(out.reshape(nz, ns, 3).astype(np.uint8)), (z1, res, used)


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ruled(im, geo, grid=0.5, z0=0.0, scale=1.0):
    """Rule the picture: distance along the wall across the top, height up the side."""
    z1, res, _ = geo
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    k = scale / res
    d = ImageDraw.Draw(im, "RGBA")
    font = _font(13)
    n = int(im.width / k / grid) + 1
    for j in range(n + 1):
        x = j * grid * k
        major = abs(j * grid - round(j * grid)) < 1e-6
        d.line([(x, 0), (x, im.height)], fill=(255, 255, 0, 150 if major else 70), width=1)
        d.text((x + 2, 2), f"{j * grid:g}", fill=(255, 255, 0, 255), font=font)
    zz = np.ceil(z0 / grid) * grid
    while zz < z1:
        y = (z1 - zz) * k
        major = abs(zz - round(zz)) < 1e-6
        d.line([(0, y), (im.width, y)], fill=(0, 255, 255, 150 if major else 70), width=1)
        d.text((2, y - 15), f"{zz:g}", fill=(0, 255, 255, 255), font=font)
        zz += grid
    return im


def dot_points(im, sv, a, b, geo, z0, scale=1.0, tol=0.05):
    """Mark the survey points that lie within `tol` of the plane."""
    z1, res, _ = geo
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.linalg.norm(b - a)
    d = (b - a) / length
    n = np.array([-d[1], d[0]])
    P = sv.P[(sv.track >= 3) & (sv.err < 1.5)]
    rel = P[:, :2] - a
    m = (np.abs(rel @ n) < tol) & (rel @ d > 0) & (rel @ d < length) & (P[:, 2] > z0) & (P[:, 2] < z1)
    k = scale / res
    dr = ImageDraw.Draw(im)
    for s, z in zip(rel[m] @ d, P[m, 2]):
        dr.point((s * k, (z1 - z) * k), fill=(255, 0, 255))
    return int(m.sum())


def main():
    args = sys.argv[1:]
    dirs = args[args.index("--") + 1:] if "--" in args else []
    args = args[:args.index("--")] if "--" in args else args
    opt = {}
    for key in ("--z", "--res", "--clips", "--range", "--grid"):
        if key in args:
            i = args.index(key)
            opt[key] = args[i + 1]
            del args[i:i + 2]
    points = "--points" in args
    args = [x for x in args if x != "--points"]
    sv = Survey(args[0], dirs)
    a = [float(v) for v in args[2].split(",")]
    b = [float(v) for v in args[3].split(",")]
    z0, z1 = (float(v) for v in opt.get("--z", "0,2.5").split(","))
    res = float(opt.get("--res", 0.01))
    frames = None
    if "--clips" in opt:
        want = set(opt["--clips"].split(","))
        frames = [i for i, n in enumerate(sv.names) if clip_of(n) in want]
    im, geo = facade(sv, a, b, z0, z1, res, frames, float(opt.get("--range", 8.0)))
    im = ruled(im, geo, float(opt.get("--grid", 0.5)), z0)
    note = ""
    if points:
        note = f", {dot_points(im, sv, a, b, geo, z0)} survey points within 5 cm of the plane"
    im.save(args[1])
    print(f"{args[1]}: {im.width} x {im.height} px at {res * 100:g} cm per pixel from {geo[2]} frames{note}")


if __name__ == "__main__":
    main()
