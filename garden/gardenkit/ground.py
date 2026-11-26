"""The ground: one painted texture for the whole plot, plus the paving.

The lawn is not uniform, so instead of tiling a grass texture the surface is painted in
plan from the zones in garden.json (bare soil under the willow, leaf litter along the
walls, gravel beyond the gate ...) and laid over a single mesh.

The mesh has the shape of the ground. garden.json's ground may carry a "relief": heights
on a regular grid ({x0, y0, step, z}, made by tools/survey/terrain.py from the survey and
the measured spot levels), or the name of a JSON file holding one. Without it the ground
is flat at z = 0, the survey datum.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from . import textures as tx
from .core import TEX, Mesh


EDGE_GREEN = (106, 130, 70)   # the viewer's surrounding ground uses the same colour


def _zone_mask(z, X, Y, seed):
    """Soft 0..1 mask of a zone on the plan grid. Edges are roughened with noise."""
    soft = float(z.get("soft", 0.5))
    shape = z.get("shape", "ellipse")
    if shape == "ellipse":
        cx, cy = z["c"]
        rx, ry = z["r"]
        a = math.radians(z.get("rot", 0.0))
        dx, dy = X - cx, Y - cy
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        d = (np.sqrt((u / rx) ** 2 + (v / ry) ** 2) - 1.0) * min(rx, ry)   # ~ metres outside
    elif shape == "rect":
        x0, y0, x1, y1 = z["box"]
        d = np.maximum(np.maximum(x0 - X, X - x1), np.maximum(y0 - Y, Y - y1))
    elif shape == "band":     # corridor of half-width w around a polyline
        P = np.asarray(z["path"], float)
        d = np.full(X.shape, 1e9, np.float32)
        for a, b in zip(P[:-1], P[1:]):
            ab = b - a
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / (ab @ ab), 0, 1)
            d = np.minimum(d, np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1])))
        d = d - float(z["w"])
    elif shape == "poly":
        P = np.asarray(z["pts"], float)
        inside = np.zeros(X.shape, bool)
        d = np.full(X.shape, 1e9, np.float32)
        n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            cond = ((a[1] > Y) != (b[1] > Y)) & (X < (b[0] - a[0]) * (Y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
            inside ^= cond
            ab = b - a
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1]) / (ab @ ab), 0, 1)
            d = np.minimum(d, np.hypot(X - (a[0] + t * ab[0]), Y - (a[1] + t * ab[1])))
        d = np.where(inside, -d, d)
    else:
        raise ValueError(shape)
    rough = float(z.get("rough", 0.6))
    n1 = tx.fbm(X.shape, cells=10, octaves=5, seed=seed, tile=False) - 0.5
    d = d + n1 * rough * 2.2
    return np.clip(0.5 - d / max(soft, 1e-3), 0.0, 1.0) * float(z.get("strength", 1.0))


def paint_ground(path, extent, zones, px_per_m=64, seed=5):
    """Paint the plan texture. extent = (x0, y0, x1, y1) in metres. Returns (w_px, h_px)."""
    x0, y0, x1, y1 = extent
    W, H = int(round((x1 - x0) * px_per_m)), int(round((y1 - y0) * px_per_m))
    xs = x0 + (np.arange(W) + 0.5) / px_per_m
    ys = y1 - (np.arange(H) + 0.5) / px_per_m          # image row 0 = north edge
    X, Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    shp = (H, W)
    cells = max(4, int((x1 - x0) / 6))
    asp = (H / W, 1)

    def n(c, o, s):
        return tx.fbm(shp, cells=c, octaves=o, seed=seed + s, tile=False, aspect=asp)

    broad, mid, fine, speck = n(cells, 4, 1), n(cells * 6, 4, 2), n(cells * 40, 3, 3), n(cells * 110, 2, 4)

    def col(*c):
        return np.array(c, np.float32) / 255.0

    def mix(a, b, t):
        return a * (1 - t[..., None]) + b * t[..., None]

    # late-November lawn under a flat grey sky: tired green, mossy and thin in places
    g = mix(np.broadcast_to(col(96, 124, 66), shp + (3,)), np.broadcast_to(col(120, 144, 76), shp + (3,)),
            tx._smooth(0.3, 0.7, broad))
    g = mix(g, np.broadcast_to(col(82, 110, 60), shp + (3,)), tx._smooth(0.45, 0.75, mid) * 0.6)
    g = mix(g, np.broadcast_to(col(138, 144, 86), shp + (3,)), tx._smooth(0.6, 0.85, n(cells * 3, 4, 7)) * 0.4)
    blades = 0.80 + 0.28 * fine + 0.2 * (speck - 0.5)
    img = g * blades[..., None]
    # thin, muddy patches scattered through the grass
    wear = tx._smooth(0.66, 0.84, n(cells * 5, 5, 9)) * 0.4
    soil = np.broadcast_to(col(96, 80, 66), shp + (3,)) * (0.8 + 0.3 * fine + 0.25 * (speck - 0.5))[..., None]
    img = mix(img, soil, wear)

    palette = {
        "soil": (col(92, 76, 64), 0.4, 0.3),
        "litter": (col(110, 86, 60), 0.5, 0.5),
        "gravel": (col(156, 148, 130), 0.3, 0.8),
        "rough": (col(88, 108, 60), 0.45, 0.35),      # unmown weedy grass
        "field": (col(104, 136, 70), 0.3, 0.2),
        "mulch": (col(74, 60, 50), 0.45, 0.4),
        "tarmac": (col(84, 84, 86), 0.2, 0.4),
        "paving": (col(158, 156, 148), 0.18, 0.3),     # concrete slabs laid as a path
        "leaves": (col(118, 84, 52), 0.5, 0.6),        # a path or corner deep in fallen leaves
    }
    for k, z in enumerate(zones):
        base, a_fine, a_speck = palette[z["kind"]]
        m = _zone_mask(z, X, Y, seed + 100 + k)
        tex = np.broadcast_to(base, shp + (3,)) * (1 - a_fine / 2 + a_fine * fine + a_speck * (speck - 0.5))[..., None]
        if z["kind"] == "litter":   # a scatter of yellow and russet leaves over earth and thin grass
            leaf = tx._smooth(0.60, 0.66, n(cells * 150, 2, 30 + k))
            leafc = mix(np.broadcast_to(col(170, 136, 70), shp + (3,)), np.broadcast_to(col(128, 88, 54), shp + (3,)),
                        tx._smooth(0.4, 0.6, n(cells * 60, 2, 60 + k)))
            earth = mix(np.broadcast_to(col(88, 72, 58), shp + (3,)) * (0.85 + 0.3 * fine)[..., None], img,
                        tx._smooth(0.5, 0.75, n(cells * 9, 4, 90 + k)) * 0.6)
            tex = mix(earth, leafc, leaf * 0.85)
        if z["kind"] == "soil":     # a little green creeping back in
            weeds = tx._smooth(0.66, 0.8, n(cells * 14, 4, 40 + k)) * 0.5
            tex = mix(tex, img, weeds)
        img = mix(img, tex, m)
    # fade to one flat green towards the edges, so the surveyed patch melts into the land around it
    edge = np.minimum(np.minimum(X - x0, x1 - X), np.minimum(Y - y0, y1 - Y))
    fade = 1.0 - tx._smooth(0.0, 5.0, edge)
    img = mix(img, np.broadcast_to(col(*EDGE_GREEN), shp + (3,)), fade)
    out = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(out).save(path, quality=86, subsampling=1, optimize=True)
    return W, H


class Relief:
    """Ground heights on a grid, read with bilinear interpolation. Beyond the grid the
    nearest edge value holds, and everything eases to zero over the last `fade` metres
    inside `extent`, so that the modelled patch meets the flat land drawn round it."""

    def __init__(self, data=None, extent=None, fade=4.0):
        self.flat = not data
        self.extent, self.fade = extent, fade
        if not self.flat:
            self.x0, self.y0, self.step = float(data["x0"]), float(data["y0"]), float(data["step"])
            self.z = np.asarray(data["z"], np.float32)

    def __call__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.flat:
            return np.zeros(np.broadcast(x, y).shape) if x.ndim or y.ndim else 0.0
        fx = np.clip((x - self.x0) / self.step, 0, self.z.shape[0] - 1.001)
        fy = np.clip((y - self.y0) / self.step, 0, self.z.shape[1] - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        z = self.z
        h = (z[ix, iy] * (1 - tx) * (1 - ty) + z[ix + 1, iy] * tx * (1 - ty)
             + z[ix, iy + 1] * (1 - tx) * ty + z[ix + 1, iy + 1] * tx * ty)
        if self.extent is not None:
            ex0, ey0, ex1, ey1 = self.extent
            edge = np.minimum(np.minimum(x - ex0, ex1 - x), np.minimum(y - ey0, ey1 - y))
            t = np.clip(edge / self.fade, 0.0, 1.0)
            h = h * (t * t * (3 - 2 * t))
        return h if np.ndim(h) else float(h)


def _grid_lines(lo, hi, heights=None, coarse=0.5, fine=0.25, tol=0.08):
    """Mesh lines between lo and hi: every `coarse` metres, and every `fine` metres across
    any strip where the ground rises or falls by more than `tol` within the strip.
    heights(t) gives the ground along positions t of this axis, one row per test line of
    the other axis. A sunken path half a metre wide then gets the lines it needs for its
    whole length, and open lawn gets few."""
    out = [lo]
    n = int(math.ceil((hi - lo) / coarse - 1e-9))
    for k in range(n):
        a, b = lo + k * coarse, min(hi, lo + (k + 1) * coarse)
        sub = 1
        if heights is not None:
            h = heights(np.linspace(a, b, int(round((b - a) / fine)) + 1))
            if float((h.max(1) - h.min(1)).max()) > tol:
                sub = int(round((b - a) / fine))
        out += [a + (b - a) * (j + 1) / sub for j in range(sub)]
    return np.array(out)


def _inside(poly, X, Y, inset=0.0):
    """Mask of the points inside a polygon drawn in from its corners by about `inset` metres."""
    P = np.asarray(poly, float)
    if inset:
        c = P.mean(0)
        P = c + (P - c) * (1 - inset * 1.4 / np.maximum(np.linalg.norm(P - c, axis=1, keepdims=True), 1e-6))
    inside = np.zeros(np.shape(X), bool)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        inside ^= ((a[1] > Y) != (b[1] > Y)) & (X < (b[0] - a[0]) * (Y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
    return inside


def ground(extent, zones, texture_dir, px_per_m=64, seed=5, relief=None, cuts=()):
    """Ground mesh with the painted plan texture, shaped by the relief if there is one.
    Registers the 'ground' texture.

    cuts: (outline, level) pairs, the footprints of buildings that have floors of their
    own. The ground outside a building is often higher than the floor inside it (earth
    builds up against old walls), and a surface carried straight across would come up
    through the floor. Inside each outline the ground is therefore held down to `level`.
    The step is made within a hand's breadth of the outline, so it is hidden in the
    thickness of the wall: mesh lines are added along the walls to make that possible."""
    x0, y0, x1, y1 = extent
    path = os.path.join(texture_dir, "ground.jpg")
    paint_ground(path, extent, zones, px_per_m, seed)
    TEX["ground"] = {"path": path, "size": (x1 - x0, y1 - y0), "normal": None}
    relief = relief or Relief()
    m = Mesh()
    if relief.flat:
        xs = np.linspace(x0, x1, max(1, int((x1 - x0) / 4)) + 1)
        ys = np.linspace(y0, y1, max(1, int((y1 - y0) / 4)) + 1)
    else:
        probe_y, probe_x = np.arange(y0, y1 + 1e-6, 0.25), np.arange(x0, x1 + 1e-6, 0.25)
        xs = _grid_lines(x0, x1, lambda t: relief(*np.meshgrid(t, probe_y, indexing="xy")))
        ys = _grid_lines(y0, y1, lambda t: relief(*np.meshgrid(probe_x, t, indexing="ij")))
        ex, ey = [], []
        for poly, _ in cuts:                            # lines along each wall, a tenth of a metre apart across its run
            for k in range(len(poly)):
                (ax, ay), (bx, by) = poly[k], poly[(k + 1) % len(poly)]
                if abs(by - ay) >= abs(bx - ax):
                    ex += list(np.arange(min(ax, bx) - 0.1, max(ax, bx) + 0.1001, 0.1))
                else:
                    ey += list(np.arange(min(ay, by) - 0.1, max(ay, by) + 0.1001, 0.1))
        xs = np.unique(np.round(np.concatenate([xs, [v for v in ex if x0 < v < x1]]), 3))
        ys = np.unique(np.round(np.concatenate([ys, [v for v in ey if y0 < v < y1]]), 3))
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.array(relief(X, Y), float)
    for poly, level in cuts:
        Z = np.where(_inside(poly, X, Y, inset=0.06), np.minimum(Z, level), Z)
    nx, ny = len(xs) - 1, len(ys) - 1
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    U, V = (X - x0) / (x1 - x0), (Y - y0) / (y1 - y0)
    faces, uvs = [], []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            quad = (a, a + 1, a + nx + 2, a + nx + 1)
            faces.append(quad)
            uvs.append([(float(U.flat[q]), float(V.flat[q])) for q in quad])
    m.add(verts, faces, uvs, "ground", smooth=not relief.flat)
    return m


def flag_paving(x0, x1, y0, y1, clip_line=None, seed=3, rows=None, extra=()):
    """Riven concrete flags laid in courses running east-west: every course is one flag
    size (600 x 900, 750 x 750, 600 x 750 ...), as old council-flag patios usually are.
    clip_line: ((px, py), (dx, dy)) - slabs whose centre is left of this line are dropped
    (the house wall cuts across the grid at an angle). Returns a list of (x, y, w, d)."""
    rng = np.random.default_rng(seed)
    rows = rows or [(0.6, 0.6), (0.9, 0.6), (0.75, 0.75), (0.75, 0.6), (0.6, 0.6), (0.9, 0.6),
                    (0.6, 0.75), (0.75, 0.75), (0.6, 0.6), (0.9, 0.6)]
    slabs, y, r = [], y0, 0
    while y < y1 - 0.2:
        h, w = rows[r % len(rows)]
        h = min(h, y1 - y)
        x = x1 - (0.0 if r % 2 == 0 else w / 2)   # break the joints between courses
        while x > x0 - 0.05:
            xa = max(x - w, x0)
            xb = min(x, x1)
            if xb - xa > 0.12:
                cx, cy = (xa + xb) / 2, y + h / 2
                keep = True
                if clip_line is not None:
                    (px, py), (dx, dy) = clip_line
                    keep = (dx * (cy - py) - dy * (cx - px)) < 0   # right of the line
                if keep:
                    slabs.append((xa, y, xb - xa, h))
            x -= w
        y += h
        r += 1
    slabs += [tuple(e) for e in extra]
    return slabs
