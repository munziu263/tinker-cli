"""Ground levels from the survey's own points.

    terrain.py <survey.npz> <out.npz> [--levels control/levels.json] [--buildings garden.json] [--step 0.25]
               [--extent x0,x1,y0,y1] [--json levels.json] [--plot out.png]

Every reconstructed point has a height, and the lowest points in any patch of garden are
the ground. This turns them into a surface:

  1. in each cell of a grid (0.25 m), take a low percentile of the heights of the points that
     could be ground (within a metre of the datum): low rather than lowest, so that a stray
     point below the surface does not drag it down;
  2. a cell under a shrub or against a wall has no ground points, only things standing on
     it, so anything that sticks up over a width of less than 3 m is planed off (a grey
     opening) and the gaps are filled from their surroundings. Buildings are wider than
     that, so their footprints (read from garden.json) are cut out first;
  3. the result is smoothed: over a quarter of a metre where the cells are full of points
     (paths and lawn filmed from close by: a sunken path half a metre wide survives), over
     a metre where they are thin.

Where the measured survey gives spot levels (control/levels.json: [[x, y, z], ...] in the
site grid) the surface is compared with them, and the difference, spread smoothly between
the spots, is taken out, so near the house the surface is the measured one.

The surface is only as good as the points: on open lawn and gravel filmed on foot it
repeats to a few centimetres; under trees and behind buildings it is interpolated.

--json writes the surface as plain numbers for garden.json's ground "relief" (build.py
gives the ground mesh this shape and stands everything on it).

In code:  t = Terrain.load(path);  t(x, y) -> z  (arrays allowed).
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Terrain:
    def __init__(self, x0, y0, step, z, support=None):
        self.x0, self.y0, self.step = float(x0), float(y0), float(step)
        self.z = np.asarray(z, np.float32)                 # z[ix, iy]
        self.support = support                             # points per cell, where known

    def __call__(self, x, y):
        """Bilinear interpolation; outside the grid the nearest edge value is used."""
        fx = np.clip((np.asarray(x, float) - self.x0) / self.step, 0, self.z.shape[0] - 1.001)
        fy = np.clip((np.asarray(y, float) - self.y0) / self.step, 0, self.z.shape[1] - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        z = self.z
        return (z[ix, iy] * (1 - tx) * (1 - ty) + z[ix + 1, iy] * tx * (1 - ty)
                + z[ix, iy + 1] * (1 - tx) * ty + z[ix + 1, iy + 1] * tx * ty)

    def save(self, path):
        np.savez_compressed(path, x0=self.x0, y0=self.y0, step=self.step, z=self.z,
                            support=self.support if self.support is not None else np.zeros((0, 0)))

    @staticmethod
    def load(path):
        d = np.load(path)
        return Terrain(d["x0"], d["y0"], d["step"], d["z"], d["support"] if d["support"].size else None)

    def as_json(self, step=None, digits=2):
        """The surface as plain lists: {x0, y0, step, z} with z[i][j] the height at
        (x0 + i * step, y0 + j * step). step: resample to this spacing (default: as stored)."""
        step = step or self.step
        nx = int(np.floor((self.z.shape[0] - 1) * self.step / step + 1e-9)) + 1
        ny = int(np.floor((self.z.shape[1] - 1) * self.step / step + 1e-9)) + 1
        X, Y = np.meshgrid(self.x0 + np.arange(nx) * step, self.y0 + np.arange(ny) * step, indexing="ij")
        return {"x0": round(self.x0, 3), "y0": round(self.y0, 3), "step": step,
                "z": [[round(float(v), digits) for v in row] for row in self(X, Y)]}


def _fill(a):
    """Fill NaN cells from the nearest valid cell."""
    bad = np.isnan(a)
    if not bad.any():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def building_outlines(spec):
    """Footprints of the buildings in a garden.json, as polygons in the site grid."""
    out = []
    for f in spec["features"]:
        p, at, rot = f.get("params", {}), f.get("at", [0, 0]), np.radians(f.get("rot", 0))
        c, s = np.cos(rot), np.sin(rot)
        place = lambda q: (at[0] + c * q[0] - s * q[1], at[1] + s * q[0] + c * q[1])
        if f["type"] == "cottage":
            for key in ("main", "outshot"):
                xn, xs, yw, ye = p[key]
                out.append([(xn, yw), (xs, yw), (xs, ye), (xn, ye)])
            if p.get("leanto"):
                xn, xs, yw, ye = p["leanto"]["box"]
                out.append([(xn, yw), (xs, yw), (xs, ye), (xn, ye)])
        elif f["type"] == "outbuilding":
            w, d = p["width"], p["depth"]
            br, bl = p.get("back") or ((w, d), (0, d))
            out.append([place(q) for q in ((0, 0), (w, 0), tuple(br), tuple(bl))])
        elif f["type"] == "timber_shed":
            w, d = p["width"] / 2, p["depth"] / 2
            out.append([place(q) for q in ((-w, -d), (w, -d), (w, d), (-w, d))])
    return out


def _inside(poly, X, Y, grow=0.0):
    """Mask of grid points inside a convex-ish polygon grown outwards by `grow` metres."""
    P = np.asarray(poly, float)
    c = P.mean(0)
    P = c + (P - c) * (1 + grow / np.maximum(np.linalg.norm(P - c, axis=1, keepdims=True), 1e-6))
    inside = np.zeros(X.shape, bool)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        inside ^= ((a[1] > Y) != (b[1] > Y)) & (X < (b[0] - a[0]) * (Y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
    return inside


def from_points(P, track, err, extent, step=0.25, band=(-1.0, 0.8), pct=20, min_pts=6, plane_m=3.0, smooth_m=(0.25, 1.0),
                holes=()):
    """Ground surface over extent = (x0, x1, y0, y1) from points P (n, 3). holes: polygons to
    ignore points in (building footprints), filled afterwards from the ground around them."""
    x0, x1, y0, y1 = extent
    nx, ny = int(round((x1 - x0) / step)) + 1, int(round((y1 - y0) / step)) + 1
    m = (track >= 3) & (err < 1.5) & (P[:, 2] > band[0]) & (P[:, 2] < band[1]) \
        & (P[:, 0] >= x0 - step / 2) & (P[:, 0] < x1 + step / 2) & (P[:, 1] >= y0 - step / 2) & (P[:, 1] < y1 + step / 2)
    q = P[m]
    ix = np.clip(np.round((q[:, 0] - x0) / step).astype(int), 0, nx - 1)
    iy = np.clip(np.round((q[:, 1] - y0) / step).astype(int), 0, ny - 1)
    order = np.lexsort((q[:, 2], iy, ix))
    ix, iy, zz = ix[order], iy[order], q[order, 2]
    key = ix * ny + iy
    starts = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    ends = np.r_[starts[1:], len(key)]
    g = np.full((nx, ny), np.nan, np.float32)
    n = np.zeros((nx, ny), np.int32)
    for s, e in zip(starts, ends):
        n[ix[s], iy[s]] = e - s
        if e - s >= min_pts:
            g[ix[s], iy[s]] = zz[s + int((e - s - 1) * pct / 100.0)]
    if holes:
        X, Y = np.meshgrid(x0 + np.arange(nx) * step, y0 + np.arange(ny) * step, indexing="ij")
        for poly in holes:
            cut = _inside(poly, X, Y, grow=0.6)
            g[cut] = np.nan
            n[cut] = 0
    known = ~np.isnan(g)
    f = _fill(g)
    w = max(3, int(round(plane_m / step)) | 1)
    f = ndimage.grey_opening(f, size=(w, w))               # plane off shrubs, walls and heaps
    f = np.where(known, np.minimum(f, _fill(g)), f)
    f = ndimage.median_filter(f, size=3)
    # smooth, trusting cells in proportion to how many points they hold: lightly where the
    # neighbourhood is dense with points, broadly where it is not
    wgt = np.clip(n / 30.0, 0.05, 1.0).astype(np.float32)

    def blur(sigma_m):
        sg = sigma_m / step
        return ndimage.gaussian_filter(f * wgt, sg) / np.maximum(ndimage.gaussian_filter(wgt, sg), 1e-6)

    dense = ndimage.uniform_filter(np.minimum(n, 60).astype(np.float32), size=5) / 60.0      # 0..1 over about a metre
    mix = np.clip((dense - 0.15) / 0.35, 0.0, 1.0)
    mix = ndimage.gaussian_filter(mix, 2.0)
    f = blur(smooth_m[0]) * mix + blur(smooth_m[1]) * (1 - mix)
    return Terrain(x0, y0, step, f, n)


def tie_to_levels(t, levels, reach=6.0, power=2.0):
    """Take out the difference from measured spot levels, spread by inverse distance and
    fading to nothing `reach` metres from the nearest spot. Returns (terrain, residuals before)."""
    L = np.asarray(levels, float)
    res = L[:, 2] - t(L[:, 0], L[:, 1])
    xs = t.x0 + np.arange(t.z.shape[0]) * t.step
    ys = t.y0 + np.arange(t.z.shape[1]) * t.step
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    d = np.hypot(X[..., None] - L[:, 0], Y[..., None] - L[:, 1])
    w = 1.0 / np.maximum(d, 0.25) ** power
    corr = (w * res).sum(-1) / w.sum(-1)
    fade = np.clip(1.0 - (d.min(-1) - reach / 2) / (reach / 2), 0.0, 1.0)
    return Terrain(t.x0, t.y0, t.step, t.z + (corr * fade).astype(np.float32), t.support), res


def main():
    a = sys.argv[1:]
    opt = {}
    for k in ("--levels", "--step", "--plot", "--extent", "--buildings", "--json"):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    d = np.load(a[0])
    P = d["P"]
    extent = tuple(float(v) for v in opt["--extent"].split(",")) if "--extent" in opt else \
        (np.floor(np.percentile(P[:, 0], 1)), np.ceil(np.percentile(P[:, 0], 99)), np.floor(np.percentile(P[:, 1], 1)), np.ceil(np.percentile(P[:, 1], 99)))
    holes = building_outlines(json.load(open(opt["--buildings"]))) if "--buildings" in opt else ()
    t = from_points(P, d["track"], d["err"], extent, float(opt.get("--step", 0.25)), holes=holes)
    if "--levels" in opt:
        t, res = tie_to_levels(t, json.load(open(opt["--levels"]))["levels"])
        print(f"against {len(res)} measured levels before tying: mean {res.mean():+.3f} m, rms {np.sqrt((res ** 2).mean()):.3f} m, worst {np.abs(res).max():.3f} m")
    t.save(a[1])
    if "--json" in opt:
        with open(opt["--json"], "w") as fh:
            json.dump(t.as_json(), fh, separators=(",", ":"))
    print(a[1], "grid", t.z.shape, "step", t.step, "levels", float(t.z.min()).__round__(2), "to", float(t.z.max()).__round__(2))
    if "--plot" in opt:
        from PIL import Image
        z = np.clip((t.z.T[::-1] + 0.5) / 1.0, 0, 1)        # -0.5 .. +0.5 m, north-up like the other plans
        rgb = np.stack([z * 255, 255 - np.abs(z - 0.5) * 300, (1 - z) * 255], -1).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).resize((rgb.shape[1] * 8, rgb.shape[0] * 8), Image.NEAREST).save(opt["--plot"])


if __name__ == "__main__":
    main()
