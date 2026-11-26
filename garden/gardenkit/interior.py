"""Inside a building: walls, floors, stairs, beams and fixed fittings.

Everything is drawn the way a floor plan shows it once the storey above has been lifted
off: walls are solid, flat-topped blocks at their true thickness with a dark cut face on
top, so that when the viewer slices the house at ceiling height the rooms read like a model
you can look down into. Coordinates are metres in the feature's own frame, which is the
site grid when the feature sits at the origin with no rotation.
"""
from __future__ import annotations

import math

import numpy as np

from .core import Mesh, rot_z, translate


def _block(m, M, s0, s1, t, zb, zt0, zt1, mat, cap=None):
    """One piece of wall in the wall's own frame: from s0 to s1 along it, t thick (centred),
    from zb up to a top that may slope from zt0 (at s0) to zt1 (at s1)."""
    if s1 - s0 < 1e-4 or max(zt0, zt1) - zb < 1e-4:
        return
    h = t / 2
    b = [(s0, -h, zb), (s1, -h, zb), (s1, h, zb), (s0, h, zb)]
    tp = [(s0, -h, zt0), (s1, -h, zt1), (s1, h, zt1), (s0, h, zt0)]
    quads = [(b[0], b[1], tp[1], tp[0]), (b[1], b[2], tp[2], tp[1]), (b[2], b[3], tp[3], tp[2]), (b[3], b[0], tp[0], tp[3])]
    for q in quads:
        m.add(q, [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], mat, M=M)
    m.add(tp, [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], cap or mat, M=M)


def walls(segments, z0=0.0, z1=2.05, mat="plaster", cap="wall_cut", thickness=0.14):
    """Walls as solid blocks along centre lines.

    segments: dicts with
      a, b      ends of the centre line, [x, y]
      t         thickness (default `thickness`)
      z         [bottom, top] if this wall differs from z0, z1
      top       [top at a, top at b] for a wall whose head slopes (under a low roof)
      mat, cap  materials, if not the defaults
      openings  dicts(at=[from, to] in metres from a, z=[sill, head]); a door has no sill:
                z=[None, head]. Heights are in the same datum as z0, z1.
    """
    m = Mesh()
    for s in segments:
        a, b = np.asarray(s["a"], float), np.asarray(s["b"], float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-4:
            continue
        zb, zt = s.get("z", (z0, z1))
        ta, tb = s.get("top", (zt, zt))
        t = s.get("t", thickness)
        wm, cm = s.get("mat", mat), s.get("cap", cap)
        M = translate(a[0], a[1], 0) @ rot_z(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
        top = lambda q: ta + (tb - ta) * q / L                       # noqa: E731
        ops = sorted(s.get("openings", ()), key=lambda o: o["at"][0])
        cur = 0.0
        for o in ops:
            lo, hi = max(0.0, o["at"][0]), min(L, o["at"][1])
            if hi - lo < 1e-3:
                continue
            _block(m, M, cur, lo, t, zb, top(cur), top(lo), wm, cm)
            sill, head = o["z"]
            if sill is not None and sill > zb + 0.01:                # wall below a window
                _block(m, M, lo, hi, t, zb, sill, sill, wm, wm)
            if head < min(top(lo), top(hi)) - 0.01:                  # wall above the head
                _block(m, M, lo, hi, t, head, top(lo), top(hi), wm, cm)
            cur = hi
        _block(m, M, cur, L, t, zb, top(cur), top(L), wm, cm)
    return m


def _ccw(outline):
    P = [tuple(map(float, p)) for p in outline]
    area = sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))
    return P if area > 0 else P[::-1]


def floor(outline=None, rects=(), z=0.0, mat="floor_boards", thick=0.03):
    """A room's floor: a plan outline and/or rectangles [x0, x1, y0, y1], top surface at z."""
    m = Mesh()
    if outline:
        m.prism(_ccw(outline), z - thick, z, mat, bottom=True)
    for x0, x1, y0, y1 in rects:
        m.prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], z - thick, z, mat, bottom=True)
    return m


def stairs(flights, landings=(), mat="carpet_red", solid=True):
    """Straight flights and landings.

    flights:  dicts(risers=[positions along the going], across=[lo, hi], axis="x"|"y",
                    z0=level at the foot, rise=height of one riser, end=position where the
                    last tread stops). Each riser line starts a tread one rise higher.
    landings: dicts(rect=[x0, x1, y0, y1], z, thick)
    """
    m = Mesh()
    for f in flights:
        r = list(f["risers"]) + [f["end"]]
        lo, hi = f["across"]
        for i in range(len(r) - 1):
            zt = f["z0"] + (i + 1) * f["rise"]
            zb = f["z0"] if solid else zt - 0.05
            u0, u1 = sorted((r[i], r[i + 1]))
            rect = [(u0, lo), (u1, lo), (u1, hi), (u0, hi)] if f.get("axis", "x") == "x" else [(lo, u0), (hi, u0), (hi, u1), (lo, u1)]
            m.prism(_ccw(rect), zb, zt, mat, bottom=True)
    for ld in landings:
        x0, x1, y0, y1 = ld["rect"]
        m.prism([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], ld["z"] - ld.get("thick", 0.06), ld["z"], ld.get("mat", mat), bottom=True)
    return m


def beams(items, mat="timber_black"):
    """Ceiling timbers: dicts(a=[x, y], b=[x, y], w=width, z=[underside, top])."""
    m = Mesh()
    for it in items:
        a, b = np.asarray(it["a"], float), np.asarray(it["b"], float)
        L = float(np.linalg.norm(b - a))
        M = translate(a[0], a[1], 0) @ rot_z(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
        zb, zt = it["z"]
        _block(m, M, 0.0, L, it.get("w", 0.2), zb, zt, zt, it.get("mat", mat))
        m.add([(0, -it.get("w", 0.2) / 2, zb), (0, it.get("w", 0.2) / 2, zb), (L, it.get("w", 0.2) / 2, zb), (L, -it.get("w", 0.2) / 2, zb)],
              [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], it.get("mat", mat), M=M)
    return m


def fittings(items):
    """Fixed things drawn as simple blocks: dicts(box=[x0, x1, y0, y1], z=[z0, z1], mat)
    or dicts(at=[x, y], r, z=[z0, z1], mat) for something round."""
    m = Mesh()
    for it in items:
        z0, z1 = it["z"]
        if "box" in it:
            x0, x1, y0, y1 = it["box"]
            m.box((x1 - x0, y1 - y0, z1 - z0), it["mat"], M=translate(x0, y0, z0), origin="corner",
                  mats={"+z": it["top"]} if it.get("top") else None)
        else:
            m.cylinder(it["r"], z1 - z0, it["mat"], seg=20, M=translate(it["at"][0], it["at"][1], z0))
    return m
