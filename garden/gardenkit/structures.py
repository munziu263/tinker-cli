"""Boundaries and buildings: brick walls, fences, sheds, the outbuilding, the house wing.

All functions return a core.Mesh in local coordinates (metres, Z up).
Paths are lists of (x, y) points relative to the feature's origin.
"""
from __future__ import annotations

import math

import numpy as np

from .core import Mesh, rot_x, rot_y, rot_z, tessellate, tex_size, translate


# --------------------------------------------------------------------------- path helpers


def _offset_path(path, dist):
    """Offset a polyline to its left by dist (mitred joins)."""
    P = np.asarray(path, float)
    n = len(P)
    out = []
    for i in range(n):
        d_prev = P[i] - P[i - 1] if i > 0 else None
        d_next = P[i + 1] - P[i] if i < n - 1 else None
        dirs = [d / np.linalg.norm(d) for d in (d_prev, d_next) if d is not None]
        if len(dirs) == 1:
            nrm = np.array([-dirs[0][1], dirs[0][0]])
            out.append(P[i] + nrm * dist)
        else:
            n0 = np.array([-dirs[0][1], dirs[0][0]])
            n1 = np.array([-dirs[1][1], dirs[1][0]])
            m = n0 + n1
            m /= np.linalg.norm(m)
            out.append(P[i] + m * dist / max(0.25, float(np.dot(m, n0))))
    return np.array(out)


def path_length(path):
    P = np.asarray(path, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def point_along(path, s):
    """(point, unit tangent) at distance s along the polyline."""
    P = np.asarray(path, float)
    for i in range(len(P) - 1):
        d = P[i + 1] - P[i]
        L = float(np.linalg.norm(d))
        if s <= L or i == len(P) - 2:
            return P[i] + d * (s / L), d / L
        s -= L
    return P[-1], d / L


def _ribbon(mesh, left, right, z0, z1, mat, ts, v_top_aligned=True, closed_ends=True):
    """Wall solid between two offset polylines, textured at true scale."""
    n = len(left)
    H = z1 - z0
    v1 = 1.0 if v_top_aligned else H / ts[1]
    v0 = v1 - H / ts[1]
    for side, pts, flip in (("L", left, True), ("R", right, False)):
        run = 0.0
        for i in range(n - 1):
            a, b = pts[i], pts[i + 1]
            L = float(np.linalg.norm(b - a))
            u0, u1 = run / ts[0], (run + L) / ts[0]
            quad = [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
            uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            if flip:
                quad, uv = quad[::-1], uv[::-1]
            mesh.add(quad, [(0, 1, 2, 3)], [uv], mat)
            run += L
    for i in range(n - 1):  # top
        quad = [(right[i][0], right[i][1], z1), (right[i + 1][0], right[i + 1][1], z1),
                (left[i + 1][0], left[i + 1][1], z1), (left[i][0], left[i][1], z1)]
        w = float(np.linalg.norm(left[i] - right[i]))
        L = float(np.linalg.norm(right[i + 1] - right[i]))
        mesh.add(quad, [(0, 1, 2, 3)], [[(0, 0.9), (L / ts[0], 0.9), (L / ts[0], 0.9 + w / ts[1]),
                                         (0, 0.9 + w / ts[1])]], mat)
    if closed_ends:
        for i, rev in ((0, False), (n - 1, True)):
            a, b = left[i], right[i]
            quad = [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
            w = float(np.linalg.norm(b - a))
            uv = [(0, v0), (w / ts[0], v0), (w / ts[0], v1), (0, v1)]
            if rev:
                quad, uv = quad[::-1], uv[::-1]
            mesh.add(quad, [(0, 1, 2, 3)], [uv], mat)


def brick_wall(path, height=1.8, thickness=0.225, mat="brick_stock", coping=True,
               coping_mat=None, piers=(), pier_size=0.33, pier_side=1):
    """Free-standing brick wall along a polyline. piers: distances along the path."""
    m = Mesh()
    ts = tex_size(mat)
    left = _offset_path(path, thickness / 2)
    right = _offset_path(path, -thickness / 2)
    body_h = height - (0.075 if coping else 0.0)
    _ribbon(m, left, right, 0.0, body_h, mat, ts)
    if coping:  # brick-on-edge coping course, slightly proud of both faces
        cm = coping_mat or mat
        cl = _offset_path(path, thickness / 2 + 0.02)
        cr = _offset_path(path, -thickness / 2 - 0.02)
        _ribbon(m, cl, cr, body_h, height, cm, (ts[0] * 0.48, ts[1]))
    for s in piers:
        p, t = point_along(path, s)
        ang = math.degrees(math.atan2(t[1], t[0]))
        nrm = np.array([-t[1], t[0]]) * pier_side
        c = p + nrm * (pier_size - thickness) / 2
        m.box((pier_size, pier_size, height + 0.02), mat,
              M=translate(c[0], c[1], 0) @ rot_z(ang), ts=ts,
              uv_off=(0.0, 1.0 - (height + 0.02) / ts[1]))
    return m


def board_fence(path, height=1.5, mat="fence_boards", post_every=1.8, post=0.09):
    """Close-board timber fence: vertical boards on a rail, square posts."""
    m = Mesh()
    ts = tex_size(mat)
    left = _offset_path(path, 0.012)
    right = _offset_path(path, -0.012)
    _ribbon(m, left, right, 0.04, height, mat, ts, v_top_aligned=True)
    L = path_length(path)
    k = max(1, round(L / post_every))
    for i in range(k + 1):
        p, t = point_along(path, L * i / k)
        nrm = np.array([-t[1], t[0]])
        c = p + nrm * (post / 2 + 0.012)
        m.box((post, post, height + 0.06), "wood_grey",
              M=translate(c[0], c[1], 0) @ rot_z(math.degrees(math.atan2(t[1], t[0]))), grain=True)
    for z in (0.3, height - 0.3):  # arris rails on the back
        for i in range(len(path) - 1):
            a, b = np.asarray(path[i], float), np.asarray(path[i + 1], float)
            t = (b - a) / np.linalg.norm(b - a)
            nrm = np.array([-t[1], t[0]]) * 0.04
            m.beam((a[0] + nrm[0], a[1] + nrm[1], z), (b[0] + nrm[0], b[1] + nrm[1], z),
                   0.05, 0.07, "wood_grey")
    return m


def post_and_wire(path, height=1.1, post_every=2.5, post=0.08, wires=3):
    """Stock fence: round timber posts with line wires."""
    m = Mesh()
    L = path_length(path)
    k = max(1, round(L / post_every))
    tops = []
    for i in range(k + 1):
        p, _ = point_along(path, L * i / k)
        m.cylinder(post / 2, height, "wood_grey", seg=8, M=translate(p[0], p[1], 0))
        tops.append(p)
    for w in range(wires):
        z = height * (0.3 + 0.6 * w / max(1, wires - 1))
        for a, b in zip(tops[:-1], tops[1:]):
            m.tube([(a[0], a[1], z), (b[0], b[1], z)], 0.004, "galvanised", seg=4,
                   caps=(False, False))
    return m


# --------------------------------------------------------------------------- building parts


def _wall_with_openings(m, width, height, mat, openings=(), M=None, reveal=0.09, ts=None,
                        reveal_mat=None):
    """Vertical wall panel in the local XZ plane (facing -Y), from x=0..width, z=0..height,
    with rectangular holes. openings: dicts with x, w, z, h. Reveals go back along +Y."""
    ts = ts or tex_size(mat)
    xs = sorted({0.0, width, *[o["x"] for o in openings], *[o["x"] + o["w"] for o in openings]})
    zs = sorted({0.0, height, *[o["z"] for o in openings], *[o["z"] + o["h"] for o in openings]})

    def is_hole(xa, xb, za, zb):
        cx, cz = (xa + xb) / 2, (za + zb) / 2
        return any(o["x"] < cx < o["x"] + o["w"] and o["z"] < cz < o["z"] + o["h"] for o in openings)

    for xa, xb in zip(xs[:-1], xs[1:]):
        for za, zb in zip(zs[:-1], zs[1:]):
            if xb - xa < 1e-6 or zb - za < 1e-6 or is_hole(xa, xb, za, zb):
                continue
            uv = [(xa / ts[0], za / ts[1]), (xb / ts[0], za / ts[1]),
                  (xb / ts[0], zb / ts[1]), (xa / ts[0], zb / ts[1])]
            m.add([(xa, 0, za), (xb, 0, za), (xb, 0, zb), (xa, 0, zb)], [(0, 1, 2, 3)], [uv], mat, M=M)
    rm = reveal_mat or mat
    for o in openings:
        x0, x1, z0, z1 = o["x"], o["x"] + o["w"], o["z"], o["z"] + o["h"]
        r = o.get("reveal", reveal)
        m.quad((x0, 0, z0), (x0, r, z0), (x0, r, z1), (x0, 0, z1), rm, M=M)       # left jamb
        m.quad((x1, r, z0), (x1, 0, z0), (x1, 0, z1), (x1, r, z1), rm, M=M)       # right jamb
        m.quad((x0, r, z1), (x1, r, z1), (x1, 0, z1), (x0, 0, z1), rm, M=M)       # head
        m.quad((x0, 0, z0), (x1, 0, z0), (x1, r, z0), (x0, r, z0), o.get("sill_mat", rm), M=M)


def _glazing(m, x, z, w, h, M=None, depth=0.09, frame="paint_black", bars=(2, 2),
             frame_w=0.045, glass="glass_dark", bar_w=0.022, blind=None):
    """Glass pane set back in an opening, with a frame and glazing bars (cols, rows)."""
    y = depth
    m.quad((x, y, z), (x + w, y, z), (x + w, y, z + h), (x, y, z + h), blind or glass, M=M,
           uv=[(0, 0), (1, 0), (1, 1), (0, 1)])
    T = (M if M is not None else np.eye(4))
    yf = depth - 0.02
    for (bx, bz, bw, bh) in (
        (x, z, w, frame_w), (x, z + h - frame_w, w, frame_w),
        (x, z, frame_w, h), (x + w - frame_w, z, frame_w, h),
    ):
        m.box((bw, 0.04, bh), frame, M=T @ translate(bx, yf - 0.02, bz), origin="corner")
    cols, rows = bars
    for i in range(1, cols):
        bx = x + w * i / cols - bar_w / 2
        m.box((bar_w, 0.03, h), frame, M=T @ translate(bx, yf - 0.012, z), origin="corner")
    for j in range(1, rows):
        bz = z + h * j / rows - bar_w / 2
        m.box((w, 0.03, bar_w), frame, M=T @ translate(x, yf - 0.012, bz), origin="corner")


def gable_roof(m, width, depth, eaves_z, ridge_z, mat, overhang=0.15, verge=0.12,
               thickness=0.06, ridge_mat="terracotta", M=None, soffit="boards_dark"):
    """Duo-pitch roof over a width(X) x depth(Y) box centred on the origin; the ridge
    runs along Y, so the gables face +-Y. Tiles are laid at true scale down the slope."""
    T = M if M is not None else np.eye(4)
    ts = tex_size(mat)
    half = width / 2
    rise = ridge_z - eaves_z
    slope = math.hypot(half, rise)
    # extend the slope past the wall for the eaves overhang
    ex = overhang * half / slope
    ez = overhang * rise / slope
    y0, y1 = -depth / 2 - verge, depth / 2 + verge
    for sgn in (-1, 1):
        eave = (sgn * (half + ex), eaves_z - ez)
        ridge = (0.0, ridge_z)
        L = slope + overhang
        a = (eave[0], y0, eave[1])
        b = (eave[0], y1, eave[1])
        c = (ridge[0], y1, ridge[1])
        d = (ridge[0], y0, ridge[1])
        run = y1 - y0
        uv = [(0, 0), (run / ts[0], 0), (run / ts[0], L / ts[1]), (0, L / ts[1])]
        quad = [a, b, c, d] if sgn > 0 else [b, a, d, c]
        m.add(quad, [(0, 1, 2, 3)], [uv], mat, M=T)
        # underside
        t = thickness
        und = [(p[0], p[1], p[2] - t) for p in quad][::-1]
        m.add(und, [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], soffit, M=T)
        # eaves edge + verge edges
        m.add([quad[0], (quad[0][0], quad[0][1], quad[0][2] - t),
               (quad[1][0], quad[1][1], quad[1][2] - t), quad[1]][::(1 if sgn > 0 else -1)],
              [(0, 1, 2, 3)], [[(0, 0), (0, 0.05), (1, 0.05), (1, 0)]], soffit, M=T)
    for y in (y0, y1):  # barge boards closing the verge
        for sgn in (-1, 1):
            p0 = (sgn * (half + ex), y, eaves_z - ez)
            p1 = (0.0, y, ridge_z)
            q = [p0, p1, (p1[0], p1[1], p1[2] - 0.14), (p0[0], p0[1], p0[2] - 0.14)]
            if (sgn > 0) == (y < 0):
                q = q[::-1]
            m.add(q, [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 0.1), (0, 0.1)]], soffit, M=T)
    # half-round ridge tiles
    m.tube([(0, y0 - 0.02, ridge_z + 0.01), (0, y1 + 0.02, ridge_z + 0.01)], 0.085, ridge_mat,
           seg=8, M=M)
    return m


def gable_wall(m, width, eaves_z, ridge_z, mat, y, facing=-1, M=None, ts=None, openings=()):
    """Pentagonal gable-end wall in the XZ plane at Y=y, facing -Y (facing=-1) or +Y."""
    ts = ts or tex_size(mat)
    half = width / 2
    pts = [(-half, y, 0), (half, y, 0), (half, y, eaves_z), (0, y, ridge_z), (-half, y, eaves_z)]
    uv = [((p[0] + half) / ts[0], p[2] / ts[1]) for p in pts]
    if facing > 0:
        pts, uv = pts[::-1], uv[::-1]
    m.add(pts, [(0, 1, 2, 3, 4)], [uv], mat, M=M)


# --------------------------------------------------------------------------- buildings


def timber_shed(width=2.6, depth=3.4, eaves=1.75, ridge=2.75, wall="boards_dark",
                roof="mossy_roof", door_side="+x", birdhouse=True, birdhouse_x=None,
                birdhouse_z=None, door="side", door_at=0.45, windows=()):
    """The dark creosoted shed: gable ends face +-Y (local), -Y is the garden-facing gable.
    Origin: centre of the footprint on the ground.
    door: "side" (on the door_side wall, door_at metres from the -Y end), "gable" (in the
    -Y gable, standing open) or None. windows: dicts(wall="+x"|"-x"|"+y", at=[lo, hi]
    along that wall from its left end seen from outside, z=[sill, head])."""
    m = Mesh()
    ts = tex_size(wall)
    half_w, half_d = width / 2, depth / 2
    gable_wall(m, width, eaves, ridge, wall, -half_d, facing=-1, ts=ts)
    gable_wall(m, width, eaves, ridge, wall, half_d, facing=1, ts=ts)
    # side walls
    for sgn in (-1, 1):
        a = (sgn * half_w, -half_d * sgn, 0)
        b = (sgn * half_w, half_d * sgn, 0)
        m.quad(a, b, (b[0], b[1], eaves), (a[0], a[1], eaves), wall, ts=ts)
    gable_roof(m, width, depth, eaves, ridge, roof, overhang=0.22, verge=0.16, ridge_mat=roof, soffit="roof_under")
    # corner trims and a plinth board
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.box((0.07, 0.07, eaves), "boards_dark",
                  M=translate(sx * (half_w + 0.012), sy * (half_d + 0.012), 0), grain=True)
    dh = min(1.7, eaves - 0.06)
    if door == "side":                               # door on the side wall with a pale weathered frame
        sgn = 1 if door_side == "+x" else -1
        dM = translate(sgn * (half_w + 0.02), -half_d + door_at, 0) @ rot_z(90 if sgn > 0 else -90)
        m.box((0.8, 0.035, dh), wall, M=dM @ translate(0.4, 0, 0), ts=ts)
        for bx in (0.0, 0.8):
            m.box((0.06, 0.05, dh + 0.06), "wood_grey", M=dM @ translate(bx, 0, 0), grain=True)
        m.box((0.92, 0.05, 0.06), "wood_grey", M=dM @ translate(0.4, 0, dh + 0.02), grain=True)
    elif door == "gable":                            # doorway in the gable, the door standing open beside it
        dw = min(0.75, width - 0.5)
        m.box((dw, 0.03, dh), "paint_black", M=translate(0, -half_d - 0.012, 0.04))
        m.box((0.035, dw, dh), wall, M=translate(dw / 2 + 0.02, -half_d - dw / 2 - 0.02, 0.04), ts=ts)
    walls_M = {"+x": translate(half_w + 0.012, -half_d, 0) @ rot_z(90), "-x": translate(-half_w - 0.012, half_d, 0) @ rot_z(-90),
               "+y": translate(half_w, half_d + 0.012, 0) @ rot_z(180)}
    for w in windows:
        (lo, hi), (z0, z1) = w["at"], w["z"]
        wM = walls_M[w["wall"]]
        m.box((hi - lo, 0.02, z1 - z0), "glass_dark", M=wM @ translate(lo, -0.01, z0), origin="corner")
        for k in range(w.get("lights", 2) + 1):
            xx = lo + (hi - lo) * k / w.get("lights", 2)
            m.box((0.04, 0.035, z1 - z0 + 0.06), "wood_grey", M=wM @ translate(xx - 0.02, -0.025, z0 - 0.03), origin="corner")
        for zz in (z0 - 0.03, z1):
            m.box((hi - lo + 0.04, 0.035, 0.035), "wood_grey", M=wM @ translate(lo - 0.02, -0.025, zz), origin="corner")
    if birdhouse:
        bM = translate(width * 0.18 if birdhouse_x is None else birdhouse_x, -half_d - 0.07,
                       (eaves + 0.02) if birdhouse_z is None else birdhouse_z)
        m.box((0.16, 0.13, 0.17), "paint_white", M=bM)
        r = Mesh()
        r.box((0.15, 0.17, 0.02), "wood_grey")
        m.merge(r, bM @ translate(-0.05, 0, 0.19) @ rot_y(-40))
        m.merge(r, bM @ translate(0.05, 0, 0.19) @ rot_y(40))
        m.cylinder(0.022, 0.012, "paint_black", seg=8,
                   M=bM @ translate(0, -0.066, 0.1) @ rot_x(90))
    return m


def outbuilding(width=6.2, depth=3.6, front_h=2.15, back_h=2.15, openings=(), wall="render_white",
                roof="roof_felt", sill="terracotta", gutter_back=True, gutter_front=False, back=None):
    """White roughcast garden room under a low sheet roof with a single fall: from the
    front to the back if front_h is the greater, towards the front if back_h is (give the
    gutter to the low side). Origin: front-left corner on the ground; the front is the
    local -Y face, x runs 0..width along it and y runs back into the building.

    back: [[x, y], [x, y]], the back-right and back-left corners, for a building that is
          not a true rectangle (old outbuildings seldom are); default (width, depth) and
          (0, depth). The walls run corner to corner.
    openings: dicts(face="front"|"right"|"back"|"left", at=[lo, hi] along that wall measured
              from its left-hand end as you face it from outside, z=[sill, head],
              kind="window"|"door", bars=[cols, rows], frame=material)"""
    m = Mesh()
    ts = tex_size(wall)
    low = min(front_h, back_h)
    br, bl = (tuple(back[0]), tuple(back[1])) if back else ((width, depth), (0.0, depth))
    depth = (br[1] + bl[1]) / 2
    corners = [(0.0, 0.0), (width, 0.0), br, bl]

    def wall_top(y):
        return front_h + (back_h - front_h) * (y / depth)

    for k, face in enumerate(("front", "right", "back", "left")):
        p0, p1 = corners[k], corners[(k + 1) % 4]
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        M = translate(p0[0], p0[1], 0) @ rot_z(math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])))
        h0, h1 = wall_top(p0[1]), wall_top(p1[1])
        ops = [o for o in openings if o["face"] == face]
        holes = [dict(x=o["at"][0], w=o["at"][1] - o["at"][0], z=o["z"][0], h=o["z"][1] - o["z"][0],
                      reveal=0.07, sill_mat=sill if o.get("kind", "window") == "window" else None) for o in ops]
        for hl in holes:
            if hl["sill_mat"] is None:
                del hl["sill_mat"]
        _wall_with_openings(m, L, low, wall, holes, M=M, ts=ts)
        if max(h0, h1) > low + 1e-6:                    # the strip between the level wall-head and the sloping roof
            m.add([(0, 0, low), (L, 0, low), (L, 0, h1), (0, 0, h0)], [(0, 1, 2, 3)],
                  [[(0, low / ts[1]), (L / ts[0], low / ts[1]), (L / ts[0], h1 / ts[1]), (0, h0 / ts[1])]], wall, M=M)
        for o, hl in zip(ops, holes):
            door = o.get("kind", "window") == "door"
            if not door:
                m.box((hl["w"] + 0.12, 0.14, 0.05), sill, M=M @ translate(hl["x"] - 0.06, -0.05, hl["z"] - 0.05), origin="corner")
            _glazing(m, hl["x"], hl["z"], hl["w"], hl["h"], M=M, frame=o.get("frame", "paint_brown" if door else "paint_black"),
                     bars=tuple(o.get("bars", (2, 5) if door else (4, 3))), frame_w=0.09 if door else 0.045,
                     depth=0.06, bar_w=0.03 if door else 0.022)
    # roof: profiled sheet on a deep fascia, oversailing all round, ribs running down the fall
    oh, fas = 0.22, 0.17

    def top(y):
        return wall_top(y) + 0.11

    ring = [(-oh, -oh), (width + oh, -oh), (br[0] + oh, br[1] + oh), (bl[0] - oh, bl[1] + oh)]
    T = [(x, y, top(y)) for x, y in ring]
    m.add(T, [(0, 1, 2, 3)], [[(0, 0), (8, 0), (8, 4), (0, 4)]], roof)
    m.add([(q[0], q[1], q[2] - fas) for q in T][::-1], [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], "fascia_black")
    for a in range(4):
        q0, q1 = T[a], T[(a + 1) % 4]
        m.add([(q0[0], q0[1], q0[2] - fas), (q1[0], q1[1], q1[2] - fas), q1, q0], [(0, 1, 2, 3)],
              [[(0, 0), (1, 0), (1, 1), (0, 1)]], "fascia_black")
    n_rib = int((width + 2 * oh) / 0.25)
    for a in range(n_rib + 1):
        t = a / n_rib
        f = (T[0][0] + (T[1][0] - T[0][0]) * t, T[0][1] + (T[1][1] - T[0][1]) * t, T[0][2] + (T[1][2] - T[0][2]) * t + 0.005)
        k = (T[3][0] + (T[2][0] - T[3][0]) * t, T[3][1] + (T[2][1] - T[3][1]) * t, T[3][2] + (T[2][2] - T[3][2]) * t + 0.005)
        m.beam(f, k, 0.03, 0.025, roof)
    if gutter_back:
        m.tube([(T[3][0], T[3][1] + 0.05, T[3][2] - 0.1), (T[2][0], T[2][1] + 0.05, T[2][2] - 0.1)], 0.05, "fascia_black", seg=8)
    if gutter_front:                                    # with a downpipe at the left-hand front corner
        m.tube([(-oh, -oh - 0.05, top(-oh) - 0.1), (width + oh, -oh - 0.05, top(-oh) - 0.1)], 0.05, "fascia_black", seg=8)
        m.tube([(-0.05, -oh - 0.05, top(-oh) - 0.12), (-0.05, -0.06, top(-oh) - 0.3), (-0.05, -0.06, 0.12)], 0.034, "fascia_black", seg=8)
    return m


def weatherboard_block(width=3.2, depth=3.0, eaves=2.5, ridge=3.5, window=None, canopy=True,
                       chimney=True, wall="weatherboard_dark", roof="clay_tiles",
                       canopy_z=None, canopy_out=1.1, canopy_w=None, mono=False):
    """Dark weather-boarded part of the house. Front (garden side) is local -Y; origin at
    the front-left corner. The roof ridge runs along X (parallel to the front)."""
    m = Mesh()
    ts = tex_size(wall)
    window = window or dict(x=0.5, w=0.9, z=1.15, h=1.15)
    _wall_with_openings(m, width, eaves, wall, [window], ts=ts, reveal=0.06)
    _glazing(m, window["x"], window["z"], window["w"], window["h"], frame="wood_grey",
             bars=(1, 1), frame_w=0.07, depth=0.05, blind="sheet_white")
    m.quad((width, 0, 0), (width, depth, 0), (width, depth, eaves), (width, 0, eaves), wall, ts=ts)
    m.quad((width, depth, 0), (0, depth, 0), (0, depth, eaves), (width, depth, eaves), wall, ts=ts)
    m.quad((0, depth, 0), (0, 0, 0), (0, 0, eaves), (0, depth, eaves), wall, ts=ts)
    if mono:   # lean-to: one sheet roof rising from the front eaves to the back wall
        oh = 0.2
        m.add([(-oh, -oh, eaves + 0.06), (width + oh, -oh, eaves + 0.06), (width + oh, depth, ridge + 0.06),
               (-oh, depth, ridge + 0.06)], [(0, 1, 2, 3)], [[(0, 0), (4, 0), (4, 4), (0, 4)]], "roof_felt")
        m.add([(-oh, depth, ridge - 0.04), (width + oh, depth, ridge - 0.04), (width + oh, -oh, eaves - 0.04),
               (-oh, -oh, eaves - 0.04)], [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], "paint_black")
        m.box((width + 2 * oh, 0.03, 0.14), "wood_grey", M=translate(-oh, -oh - 0.03, eaves - 0.08), origin="corner")
        for x, flip in ((0.0, False), (width, True)):      # wedge of wall under the slope
            tri = [(x, 0, eaves), (x, depth, ridge), (x, depth, eaves)]
            uv = [(p[1] / ts[0], p[2] / ts[1]) for p in tri]
            if flip:
                tri, uv = tri[::-1], uv[::-1]
            m.add(tri, [(0, 1, 2)], [uv], wall)
        m.quad((width, depth, eaves), (0, depth, eaves), (0, depth, ridge), (width, depth, ridge), wall, ts=ts)
    else:
        # roof: build a gable roof with ridge along local Y, then turn it so the ridge runs along X
        r = Mesh()
        gable_wall(r, depth, eaves, ridge, wall, -width / 2, facing=-1, ts=ts)
        gable_wall(r, depth, eaves, ridge, wall, width / 2, facing=1, ts=ts)
        gable_roof(r, depth, width, eaves, ridge, roof, overhang=0.25, verge=0.1, soffit="paint_black")
        m.merge(r, translate(width / 2, depth / 2, 0) @ rot_z(90))
    if canopy:  # small lean-to roof against the wall, with a gutter and downpipe
        z_lo = (eaves - 0.75) if canopy_z is None else canopy_z
        z_hi, out = z_lo + 0.42, canopy_out
        cx0 = -0.05
        cx1 = (min(width, window["x"] + window["w"] + 1.0)) if canopy_w is None else canopy_w
        m.add([(cx0, -out, z_lo), (cx1, -out, z_lo), (cx1, 0, z_hi), (cx0, 0, z_hi)], [(0, 1, 2, 3)],
              [[(0, 0), ((cx1 - cx0) / 1.65, 0), ((cx1 - cx0) / 1.65, 0.85), (0, 0.85)]], "roof_felt")
        m.add([(cx0, 0, z_hi - 0.05), (cx1, 0, z_hi - 0.05), (cx1, -out, z_lo - 0.05),
               (cx0, -out, z_lo - 0.05)], [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], "wood_brown")
        for x in (cx0, cx1):   # boarded cheeks
            m.add([(x, -out, 0), (x, 0, 0), (x, 0, z_hi - 0.05), (x, -out, z_lo - 0.05)][::(1 if x == cx1 else -1)],
                  [(0, 1, 2, 3)], [[(0, 0), (out / ts[0], 0), (out / ts[0], z_hi / ts[1]), (0, z_lo / ts[1])]], wall)
        m.quad((cx0, -out, 0), (cx1, -out, 0), (cx1, -out, z_lo - 0.05), (cx0, -out, z_lo - 0.05), wall, ts=ts)
        m.tube([(cx0 - 0.05, -out - 0.04, z_lo - 0.02), (cx1 + 0.05, -out - 0.04, z_lo - 0.02)],
               0.05, "paint_brown", seg=8)
        m.tube([(cx0 - 0.02, -out - 0.04, z_lo - 0.04), (cx0 - 0.04, -out + 0.1, 0.05)], 0.035, "paint_brown", seg=6)
    if chimney:
        z0 = 1.7   # the brick stack shows above the lean-to
        cM = translate(-0.2, 0.34, z0)
        cts = tex_size("brick_red_old")
        m.box((0.45, 0.45, ridge + 1.3 - z0), "brick_red_old", M=cM, ts=cts)
        m.cylinder(0.11, 0.3, "terracotta", seg=10, M=cM @ translate(0, 0, ridge + 1.3 - z0))
    return m


def catslide_wing(length=7.5, eaves=1.05, depth=3.6, ridge=3.9, wall="render_white",
                  roof="clay_tiles", posts=(2.9,), gutter=True, end_beams=False,
                  back_depth=0.0, back_eaves=2.3):
    """The low white wing of the house: a long clay-tile roof sweeping down to low eaves
    over a white wall. Front (garden side) is local -Y, x runs 0..length; origin at the
    front-left corner. Only the garden-facing slope and end walls are modelled."""
    m = Mesh()
    ts = tex_size(wall)
    m.quad((0, 0, 0), (length, 0, 0), (length, 0, eaves), (0, 0, eaves), wall, ts=ts)
    oh = 0.28
    slope = math.hypot(depth, ridge - eaves)
    dy, dz = depth / slope, (ridge - eaves) / slope
    rts = tex_size(roof)
    a = (-0.1, -oh * dy, eaves - oh * dz + 0.08)
    b = (length + 0.1, a[1], a[2])
    c = (length + 0.1, depth, ridge + 0.08)
    d = (-0.1, depth, ridge + 0.08)
    L = slope + oh
    m.add([a, b, c, d], [(0, 1, 2, 3)],
          [[(0, 0), ((length + 0.2) / rts[0], 0), ((length + 0.2) / rts[0], L / rts[1]), (0, L / rts[1])]], roof)
    m.add([(p[0], p[1], p[2] - 0.07) for p in (d, c, b, a)], [(0, 1, 2, 3)],
          [[(0, 0), (1, 0), (1, 1), (0, 1)]], "paint_black")
    # end walls, and either a tall back wall or (back_depth > 0) a second slope behind the ridge
    bd = back_depth
    for x, flip in ((0.0, False), (length, True)):
        if bd > 0:
            poly = [(x, 0, 0), (x, 0, eaves), (x, depth, ridge), (x, depth + bd, back_eaves), (x, depth + bd, 0)]
        else:
            poly = [(x, 0, 0), (x, 0, eaves), (x, depth, ridge), (x, depth, 0)]
        uv = [(p[1] / ts[0], p[2] / ts[1]) for p in poly]
        if flip:
            poly, uv = poly[::-1], uv[::-1]
        m.add(poly, [tuple(range(len(poly)))], [uv], wall)
    if bd > 0:
        yb = depth + bd
        m.quad((length, yb, 0), (0, yb, 0), (0, yb, back_eaves), (length, yb, back_eaves), wall, ts=ts)
        bs = math.hypot(bd, ridge - back_eaves)
        e2 = (length + 0.1, yb + oh * bd / bs, back_eaves - oh * (ridge - back_eaves) / bs + 0.08)
        f2 = (-0.1, e2[1], e2[2])
        m.add([c, e2, f2, d], [(0, 1, 2, 3)],
              [[(0, 0), ((bs + oh) / rts[1], 0), ((bs + oh) / rts[1], (length + 0.2) / rts[0]), (0, (length + 0.2) / rts[0])]], roof)
        m.add([(p[0], p[1], p[2] - 0.07) for p in (d, f2, e2, c)], [(0, 1, 2, 3)],
              [[(0, 0), (1, 0), (1, 1), (0, 1)]], "paint_black")
    else:
        m.quad((length, depth, 0), (0, depth, 0), (0, depth, ridge), (length, depth, ridge), wall, ts=ts)
    for x in posts:  # black timber framing members showing on the white wall
        m.box((0.11, 0.03, eaves), "paint_black", M=translate(x, -0.015, 0))
    if gutter:
        gz = eaves - oh * dz + 0.03
        m.tube([(-0.12, -oh * dy - 0.05, gz), (length + 0.12, -oh * dy - 0.05, gz)], 0.055,
               "paint_black", seg=8)
        for x in np.arange(0.5, length, 1.1):
            m.box((0.025, oh * dy + 0.03, 0.025), "paint_black",
                  M=translate(x, -oh * dy - 0.03, gz - 0.02), origin="corner")
    m.tube([(0, depth, ridge + 0.1), (length, depth, ridge + 0.1)], 0.09, "terracotta", seg=8)
    if end_beams:  # exposed black timbers on the south gable: post, rail and a raking brace
        x = -0.02
        zt = lambda y: eaves + (ridge - eaves) * y / depth
        m.beam((x, 0.06, 0), (x, 0.06, eaves), 0.12, 0.03, "paint_black")
        m.beam((x, depth * 0.45, 0), (x, depth * 0.45, zt(depth * 0.45) - 0.05), 0.12, 0.03, "paint_black")
        m.beam((x, 0.06, eaves * 0.98), (x, depth * 0.9, eaves * 0.98), 0.1, 0.03, "paint_black", roll=90)
        m.beam((x, 0.1, eaves + 0.05), (x, depth * 0.45, zt(depth * 0.45) - 0.1), 0.1, 0.03, "paint_black", roll=90)
        m.tube([(-0.09, 0.9, 0.05), (-0.09, 0.9, zt(0.9) - 0.1)], 0.04, "paint_black", seg=6)
    return m


def cottage(main, outshot, leanto, ridge, profile, hip_z, north_eaves, chimneys=(), openings=(),
            rooflights=(), posts=(), wall="render_white", roof="clay_tiles", outshot_east_kink=None,
            gable_windows=(), frame=(), oriels=(), porch=None, north_roof=None):
    """The house, from the measured building survey. Everything is in site-grid metres
    relative to the feature's origin: X south, Y east, Z up.

      main      [x_north, x_south, y_west, y_east]  the two-storey range
      outshot   [x_north, x_south, y_west, y_east]  the rear room under the catslide
      leanto    dict(box=[x_n, x_s, y_w, y_e], z_low, z_high) timber-clad lean-to, roof
                rising from its west side to its east side
      ridge     dict(x, z, y_west, y_east)           ridge line (shorter than the house:
                the roof is half-hipped at both ends)
      profile   [(x, z), ...] the south roof slope from the ridge down to the low eaves
                over the outshot, as drawn on the section (it steepens towards the top)
      hip_z     height at which the half-hips start, at the gables
      north_eaves dict(x, z) edge of the north slope
      chimneys  dicts(box=[x_n, x_s, y_w, y_e], top, z0, pots)
      openings  dicts(face="south"|"north"|"east"|"west"|"outshot_south"|"outshot_east"|
                "outshot_west"|"leanto_south"|"leanto_west", at=[lo, hi] along the face,
                z=[sill, head], kind="window"|"door", bars)
      rooflights dicts(y=[west, east], z=[low, high]) on the south slope
      outshot_east_kink dict(x, y_north): the outshot's east wall runs straight from its
                south end to x, then turns in to reach y_north where it meets the main range
      gable_windows dicts(face="east"|"west", x=[north, south], z=[sill, head]) windows high
                in a gable, above the eaves
      frame     exposed timbers on the outshot's east end: dicts(a=[s, z], b=[s, z], w) with
                s measured north along the wall from its south end
      oriels    dicts(face, at=[lo, hi], z=[sill, head], out) bay windows carried on brackets
      porch     dict(face, at=[lo, hi], out, h) an open timber porch with a flat roof
      north_roof material of the north slope, if it differs from the south
    """
    m = Mesh()
    ts, rts = tex_size(wall), tex_size(roof)
    xn, xs, yw, ye = main
    oxn, oxs, oyw, oye = outshot
    prof = sorted([tuple(p) for p in profile], key=lambda p: -p[1])     # ridge first
    rx, rz, ryw, rye = ridge["x"], ridge["z"], ridge["y_west"], ridge["y_east"]

    def south_x(z):
        """X of the south roof slope at height z (interpolating the drawn profile)."""
        for (xa, za), (xb, zb) in zip(prof[:-1], prof[1:]):
            if zb <= z <= za:
                return xa + (xb - xa) * (za - z) / (za - zb)
        return prof[-1][0]

    def north_x(z):
        return rx + (north_eaves["x"] - rx) * (rz - z) / (rz - north_eaves["z"])

    main_eaves = max(z for x, z in prof if x >= xs - 0.25)       # where the slope crosses the south wall line
    for (xa, za), (xb, zb) in zip(prof[:-1], prof[1:]):
        if xa <= xs <= xb:
            main_eaves = za + (zb - za) * (xs - xa) / (xb - xa)

    def hole(face, o):
        return dict(x=o["at"][0], w=o["at"][1] - o["at"][0], z=o["z"][0], h=o["z"][1] - o["z"][0])

    def face_wall(a, b, height, face, mat=wall, frame="paint_slate"):
        """Wall from plan point a to b (outside is on the right of a->b), with openings."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        M = translate(a[0], a[1], 0) @ rot_z(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
        ops = [o for o in openings if o["face"] == face]
        _wall_with_openings(m, L, height, mat, [dict(hole(face, o), reveal=0.08) for o in ops], M=M, ts=tex_size(mat))
        for o in ops:
            h = hole(face, o)
            if o.get("kind") == "door":
                _glazing(m, h["x"], h["z"], h["w"], h["h"], M=M, frame=frame, bars=o.get("bars", (3, 6)),
                         frame_w=0.09, depth=0.07, bar_w=0.03)
                m.box((h["w"] - 0.16, 0.03, h["h"] * 0.48), frame, M=M @ translate(h["x"] + 0.08, 0.035, h["z"] + 0.05), origin="corner")
            else:
                _glazing(m, h["x"], h["z"], h["w"], h["h"], M=M, frame=frame, bars=o.get("bars", (2, 3)),
                         frame_w=0.06, depth=0.07, bar_w=0.025)
        return M

    # --- main range walls. A wall panel faces the right-hand side of its direction of travel.
    face_M = {}
    face_M["south"] = face_wall((xs, yw), (xs, ye), main_eaves, "south")               # south wall (east and west parts show)
    face_M["north"] = face_wall((xn, ye), (xn, yw), north_eaves["z"] - 0.05, "north")  # the front
    face_M["east"] = translate(xs, ye, 0) @ rot_z(math.degrees(math.atan2(0.0, xn - xs)))
    face_M["west"] = translate(xn, yw, 0) @ rot_z(math.degrees(math.atan2(0.0, xs - xn)))
    for y, a, b, face in ((ye, (xs, ye), (xn, ye), "east"), (yw, (xn, yw), (xs, yw), "west")):
        face_wall(a, b, min(main_eaves, north_eaves["z"]) - 0.05, face)
        # gable above the eaves, cut off at the half-hip
        zb = min(main_eaves, north_eaves["z"]) - 0.05
        pts = [(xs, y, zb), (xs, y, max(zb, main_eaves - 0.05)), (south_x(hip_z), y, hip_z), (north_x(hip_z), y, hip_z),
               (xn, y, max(zb, north_eaves["z"] - 0.1)), (xn, y, zb)]
        pts = [q for k, q in enumerate(pts) if k == 0 or math.dist(q, pts[k - 1]) > 1e-6]      # a level eaves gives a doubled corner
        if face == "east":
            pts = pts[::-1]
        uv = [(q[0] / ts[0], q[2] / ts[1]) for q in pts]
        # a fan of triangles: left as one many-sided face, the nearly doubled corner can make the fill drop a piece
        m.add(pts, [(0, k, k + 1) for k in range(1, len(pts) - 1)], [[uv[0], uv[k], uv[k + 1]] for k in range(1, len(pts) - 1)], wall)

    # --- outshot walls (low south wall, both ends up to the roof)
    face_wall((oxs, oyw), (oxs, oye), prof[-1][1] - 0.12, "outshot_south")
    def roof_under(x):
        zt = prof[-1][1]
        for (xa, za), (xb, zb) in zip(prof[:-1], prof[1:]):
            if xa <= x <= xb:
                zt = za + (zb - za) * (x - xa) / (xb - xa)
        return zt - 0.12

    kink = outshot_east_kink

    def east_y(x):
        if not kink or x >= kink["x"]:
            return oye
        return oye + (kink["y_north"] - oye) * (kink["x"] - x) / (kink["x"] - oxn)

    for y, flip, face in ((oye, False, "outshot_east"), (oyw, True, "outshot_west")):
        yf = east_y if face == "outshot_east" else (lambda x, y=y: y)
        runs = [(oxs, oxn)] if not (kink and face == "outshot_east") else [(oxs, kink["x"]), (kink["x"], oxn)]
        for xa_, xb_ in runs:                                       # each run is one flat piece of wall
            xs_ = np.linspace(xa_, xb_, 5)
            pts = [(x, yf(x), 0.0) for x in xs_] + [(x, yf(x), roof_under(x)) for x in xs_[::-1]]
            uv = [(q[0] / ts[0], q[2] / ts[1]) for q in pts]
            if flip:
                pts, uv = pts[::-1], uv[::-1]
            m.add(pts, [tuple(range(len(pts)))], [uv], wall)
        for o in openings:
            if o["face"] == face:
                x0, x1 = oxs - o["at"][1], oxs - o["at"][0]
                sgn = 1 if face == "outshot_east" else -1
                n_l = o.get("lights", 3)
                for k in range(n_l):                                # each light sits on its own bit of wall
                    xa_, xb_ = x0 + (x1 - x0) * k / n_l, x0 + (x1 - x0) * (k + 1) / n_l
                    ang = math.degrees(math.atan2(yf(xb_) - yf(xa_), xb_ - xa_))
                    Mw = translate(xa_, yf(xa_), 0) @ rot_z(ang)
                    Lw = math.hypot(xb_ - xa_, yf(xb_) - yf(xa_))
                    m.box((Lw, 0.05, o["z"][1] - o["z"][0]), "glass_dark", M=Mw @ translate(0, sgn * 0.01 - 0.025, o["z"][0]), origin="corner")
                    for xx in (0.0, Lw):
                        m.box((0.05, 0.07, o["z"][1] - o["z"][0] + 0.08), "paint_slate", M=Mw @ translate(xx - 0.025, sgn * 0.02 - 0.035, o["z"][0] - 0.04), origin="corner")
    def slab(pa, pb, n, w, t, mat):
        """A flat timber or pane lying on a wall: from pa to pb on the wall face, w wide in
        the plane of the wall, standing t proud of it along the wall's outward normal n."""
        pa, pb, n = np.asarray(pa, float), np.asarray(pb, float), np.asarray(n, float)
        d = pb - pa
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return
        d /= L
        q = np.cross(n, d)
        q /= np.linalg.norm(q)
        c = [pa - q * w / 2, pb - q * w / 2, pb + q * w / 2, pa + q * w / 2]
        V = c + [v + n * t for v in c]
        m.add(V, [(4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
              [[(0, 0), (1, 0), (1, 1), (0, 1)]] * 5, mat)

    def east_face(s):
        """Point on the outshot's east wall s metres north of its south end, and the wall's
        outward normal there."""
        x = oxs - s
        if kink and x < kink["x"]:
            dx, dy = oxn - kink["x"], kink["y_north"] - oye
            L = math.hypot(dx, dy)
            n = (-dy / L, dx / L, 0.0) if dx / L > 0 else (dy / L, -dx / L, 0.0)
        else:
            n = (0.0, 1.0, 0.0)
        return np.array([x, east_y(x), 0.0]), np.array(n)

    for fr in frame:                                                # exposed timber frame, east end
        (s0, z0), (s1, z1) = fr["a"], fr["b"]
        cuts = [0.0, 1.0]
        if kink and min(s0, s1) < oxs - kink["x"] < max(s0, s1):    # a timber that crosses the turn in the wall bends with it
            cuts.insert(1, (oxs - kink["x"] - s0) / (s1 - s0))
        for ta, tb in zip(cuts[:-1], cuts[1:]):
            tm = (ta + tb) / 2
            _, n = east_face(s0 + (s1 - s0) * tm)
            pa, _ = east_face(s0 + (s1 - s0) * ta)
            pb, _ = east_face(s0 + (s1 - s0) * tb)
            pa[2], pb[2] = z0 + (z1 - z0) * ta, z0 + (z1 - z0) * tb
            slab(pa, pb, n, fr.get("w", 0.12), fr.get("t", 0.035), fr.get("mat", "timber_black"))

    for gw in gable_windows:                                        # a window high in a gable, above the eaves
        east = gw["face"] == "east"
        y, n = (ye, (0.0, 1.0, 0.0)) if east else (yw, (0.0, -1.0, 0.0))
        (xa, xb), (za, zb) = sorted(gw["x"]), gw["z"]
        zc = (za + zb) / 2
        slab((xa, y, zc), (xb, y, zc), n, zb - za, 0.02, "glass_dark")
        fm, fw = gw.get("frame", "paint_slate"), 0.06
        for z_ in (za, zb):
            slab((xa - fw / 2, y, z_), (xb + fw / 2, y, z_), n, fw, 0.045, fm)
        for x_ in (xa, (xa + xb) / 2, xb):
            slab((x_, y, za), (x_, y, zb), n, fw if x_ in (xa, xb) else 0.04, 0.045, fm)

    for y in posts:                                                 # studs showing in the low wall
        m.box((0.03, 0.1, prof[-1][1] - 0.14), "paint_black", M=translate(oxs + 0.0, y - 0.05, 0), origin="corner")

    # --- roof. Each strip of the south slope spans the house (above the main eaves) or just
    #     the outshot (the catslide below them); the top narrows to the ridge between the hips.
    lift = 0.07

    def span(z, low, north=False):
        if z >= hip_z:
            t = (z - hip_z) / (rz - hip_z)
            return yw - 0.12 + (ryw - yw + 0.12) * t, ye + 0.12 + (rye - ye - 0.12) * t
        if north or (z >= main_eaves - 1e-6 and not low):      # the north slope keeps the full width down to its eaves
            return yw - 0.12, ye + 0.12
        return oyw - 0.1, oye + 0.1

    def strip(pa, pb, north=False):
        """One band of tiles between two (x, z) profile points, pa above pb."""
        (xa, za), (xb, zb) = pa, pb
        low = (zb < main_eaves - 0.02) and not north
        y0a, y1a = span(za, low and za <= main_eaves + 1e-6, north)
        y0b, y1b = span(zb, low, north)
        if not north and za > main_eaves and zb < main_eaves:      # split at the main eaves line
            xm = xa + (xb - xa) * (za - main_eaves) / (za - zb)
            strip(pa, (xm, main_eaves))
            strip((xm, main_eaves), pb)
            return
        if za > hip_z > zb:
            xm = xa + (xb - xa) * (za - hip_z) / (za - zb)
            strip(pa, (xm, hip_z), north)
            strip((xm, hip_z), pb, north)
            return
        L = math.hypot(xb - xa, za - zb)
        v0 = strip.run / rts[1]
        strip.run += L
        v1 = strip.run / rts[1]
        quad = [(xb, y0b, zb + lift), (xb, y1b, zb + lift), (xa, y1a, za + lift), (xa, y0a, za + lift)]
        uv = [(y0b / rts[0], v1), (y1b / rts[0], v1), (y1a / rts[0], v0), (y0a / rts[0], v0)]
        if north:           # as listed the quad faces up-and-south; the north slope needs the other winding
            quad, uv = quad[::-1], uv[::-1]
        m.add(quad, [(0, 1, 2, 3)], [uv], (north_roof or roof) if north else roof)
        m.add([(q[0], q[1], q[2] - 0.09) for q in quad][::-1], [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], "paint_black")

    strip.run = 0.0
    for pa, pb in zip(prof[:-1], prof[1:]):
        strip(pa, pb)
    strip.run = 0.0
    strip((rx, rz), (north_eaves["x"], north_eaves["z"]), north=True)
    for y, ry_, sgn in ((ye + 0.12, rye, 1), (yw - 0.12, ryw, -1)):      # the two half-hips
        tri = [(south_x(hip_z), y, hip_z + lift), (north_x(hip_z), y, hip_z + lift), (rx, ry_, rz + lift)]
        uv = [(q[0] / rts[0], q[2] / rts[1]) for q in tri]
        if sgn < 0:
            tri, uv = tri[::-1], uv[::-1]
        m.add(tri, [(0, 1, 2)], [uv], roof)
        m.tube([(south_x(hip_z), y, hip_z + lift + 0.03), (rx, ry_, rz + lift + 0.03)], 0.07, "terracotta", seg=6)
        m.tube([(north_x(hip_z), y, hip_z + lift + 0.03), (rx, ry_, rz + lift + 0.03)], 0.07, "terracotta", seg=6)
    m.tube([(rx, ryw, rz + lift + 0.04), (rx, rye, rz + lift + 0.04)], 0.1, "terracotta", seg=8)
    # verges of the catslide, gutters and downpipes
    for y in (oyw - 0.1, oye + 0.1):
        pts = [(x, y, z + lift - 0.05) for x, z in prof if z <= main_eaves + 0.3]
        for a, b in zip(pts[:-1], pts[1:]):
            m.beam(a, b, 0.03, 0.15, "paint_black")
    ex, ez = prof[-1]
    m.tube([(ex + 0.06, oyw - 0.12, ez + lift - 0.08), (ex + 0.06, oye + 0.12, ez + lift - 0.08)], 0.055, "paint_black", seg=8)
    m.tube([(ex + 0.02, oye - 0.15, ez), (oxs + 0.07, oye - 0.15, ez - 0.25), (oxs + 0.07, oye - 0.15, 0.05)], 0.04, "paint_black", seg=6)
    gx = south_x(main_eaves) + 0.1
    for ya, yb in ((oye + 0.1, ye + 0.14), (yw - 0.14, oyw - 0.1)):
        m.tube([(gx, ya, main_eaves + lift - 0.1), (gx, yb, main_eaves + lift - 0.1)], 0.055, "paint_black", seg=8)
    m.tube([(xs + 0.07, ye - 0.06, main_eaves - 0.1), (xs + 0.07, ye - 0.06, 0.05)], 0.04, "paint_black", seg=6)
    m.tube([(north_eaves["x"] - 0.06, yw - 0.14, north_eaves["z"] + lift - 0.1), (north_eaves["x"] - 0.06, ye + 0.14, north_eaves["z"] + lift - 0.1)],
           0.055, "paint_black", seg=8)
    for r in rooflights:
        (ya, yb), (za, zb) = r["y"], r["z"]
        q = [(south_x(za), ya, za + lift + 0.05), (south_x(za), yb, za + lift + 0.05), (south_x(zb), yb, zb + lift + 0.05), (south_x(zb), ya, zb + lift + 0.05)]
        m.add(q, [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], "rooflight_glass")
        for a, b in ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])):
            m.tube([a, b], 0.03, "paint_black", seg=4, caps=(False, False))

    # --- oriel windows: a glazed box standing proud of the wall on two brackets (outside is local -Y)
    for o in oriels:
        M = face_M[o["face"]]
        (lo, hi), (z0, z1), out = o["at"], o["z"], o.get("out", 0.32)
        fm = o.get("frame_mat", "paint_slate")
        m.box((hi - lo, out, z1 - z0), "glass_dark", M=M @ translate(lo, -out, z0), origin="corner", skip=("+y",))
        m.box((hi - lo + 0.1, out + 0.06, 0.07), fm, M=M @ translate(lo - 0.05, -out - 0.06, z1), origin="corner")       # head and lead top
        m.box((hi - lo + 0.1, out + 0.06, 0.09), fm, M=M @ translate(lo - 0.05, -out - 0.06, z0 - 0.09), origin="corner")  # sill board
        n_l = o.get("lights", 5)
        for k in range(n_l + 1):
            xx = lo + (hi - lo) * k / n_l
            m.box((0.05, 0.03, z1 - z0), fm, M=M @ translate(xx - 0.025, -out - 0.015, z0), origin="corner")
        for zz in [z0 + (z1 - z0) * j / o.get("rows", 3) for j in range(1, o.get("rows", 3))]:
            m.box((hi - lo, 0.02, 0.03), fm, M=M @ translate(lo, -out - 0.01, zz - 0.015), origin="corner")
        for xx in (lo, hi):                                          # the returns
            m.box((0.03, out, 0.05), fm, M=M @ translate(xx - 0.015, -out, (z0 + z1) / 2 - 0.025), origin="corner")
        for xx in (lo + 0.18, hi - 0.18):                            # brackets
            m.add([(xx - 0.03, 0, z0 - 0.09), (xx - 0.03, -out, z0 - 0.09), (xx - 0.03, 0, z0 - 0.42),
                   (xx + 0.03, 0, z0 - 0.09), (xx + 0.03, -out, z0 - 0.09), (xx + 0.03, 0, z0 - 0.42)],
                  [(0, 1, 2), (5, 4, 3), (1, 4, 5, 2), (0, 2, 5, 3)], [[(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)],
                                                                         [(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)]], fm, M=M)

    # --- porch: four posts, a flat roof, an open lattice under it and a bench either side
    if porch:
        M = face_M[porch["face"]]
        (lo, hi), out, ph = porch["at"], porch.get("out", 1.0), porch.get("h", 2.3)
        pm = porch.get("mat", "timber_black")
        for xx in (lo, hi):
            for yy in (-out, -0.06):
                m.box((0.1, 0.1, ph), pm, M=M @ translate(xx - 0.05, yy - 0.05, 0), origin="corner")
            m.box((0.08, out, 0.1), pm, M=M @ translate(xx - 0.04, -out, ph - 0.42), origin="corner")           # rail under the lattice
            for k in range(1, 4):                                                                                   # lattice uprights at the side
                m.box((0.04, 0.04, 0.32), pm, M=M @ translate(xx - 0.02, -out * k / 4 - 0.02, ph - 0.32), origin="corner")
            m.box((0.34, out - 0.2, 0.05), pm, M=M @ translate((lo + 0.04) if xx == lo else (hi - 0.38), -out + 0.1, 0.42), origin="corner")   # bench
            for yy in (-out + 0.14, -0.2):
                m.box((0.05, 0.05, 0.42), pm, M=M @ translate((lo + 0.1) if xx == lo else (hi - 0.15), yy, 0), origin="corner")
        m.box((hi - lo + 0.1, 0.08, 0.1), pm, M=M @ translate(lo - 0.05, -out - 0.04, ph - 0.42), origin="corner")
        for k in range(1, 5):
            m.box((0.04, 0.04, 0.32), pm, M=M @ translate(lo + (hi - lo) * k / 5 - 0.02, -out - 0.02, ph - 0.32), origin="corner")
        m.box((hi - lo + 0.3, out + 0.2, 0.09), pm, M=M @ translate(lo - 0.15, -out - 0.15, ph), origin="corner")  # roof
        m.box((hi - lo - 0.2, out - 0.1, 0.04), "brick_red_old", M=M @ translate(lo + 0.1, -out, 0.0), origin="corner")  # step

    # --- lean-to (timber-clad, clear roof sheet)
    if leanto:
        lxn, lxs, lyw, lye = leanto["box"]
        zl, zh = leanto["z_low"], leanto["z_high"]
        wts = tex_size("weatherboard_dark")
        face_wall((lxs, lyw), (lxs, lye), zl, "leanto_south", mat="weatherboard_dark", frame="wood_grey")
        m.add([(lxs, lyw, zl), (lxs, lye, zl), (lxs, lye, zh)], [(0, 1, 2)], [[(0, 0), (1, 0), (1, 1)]], "weatherboard_dark")
        face_wall((lxn, lyw), (lxs, lyw), zl, "leanto_west", mat="weatherboard_dark", frame="wood_grey")
        m.add([(lxn, lyw - 0.15, zl + 0.05), (lxs + 0.2, lyw - 0.15, zl + 0.05), (lxs + 0.2, lye, zh + 0.05), (lxn, lye, zh + 0.05)],
              [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], "polycarbonate")
        m.beam((lxs + 0.2, lyw - 0.15, zl), (lxs + 0.2, lye, zh), 0.04, 0.12, "wood_grey", roll=90)

    cts = tex_size("brick_red_old")
    for c in chimneys:
        cxn, cxs, cyw, cye = c["box"]
        z0 = c.get("z0", 0.0)
        cM = translate((cxn + cxs) / 2, (cyw + cye) / 2, z0)
        m.box((cxs - cxn, cye - cyw, c["top"] - z0), "brick_red_old", M=cM, ts=cts)
        if c.get("base"):                                           # wider foot of an external stack
            bxn, bxs, byw, bye = c["base"]["box"]
            m.box((bxs - bxn, bye - byw, c["base"]["top"]), "brick_red_old", M=translate((bxn + bxs) / 2, (byw + bye) / 2, 0), ts=cts)
        m.box((cxs - cxn + 0.1, cye - cyw + 0.1, 0.1), "brick_red_old", M=cM @ translate(0, 0, c["top"] - z0 - 0.28), ts=cts)
        for k in range(c.get("pots", 1)):
            off = (k - (c.get("pots", 1) - 1) / 2) * 0.32
            m.cylinder(0.1, 0.45, "terracotta", seg=10, M=cM @ translate(off if (cxs - cxn) > (cye - cyw) else 0,
                                                                     0 if (cxs - cxn) > (cye - cyw) else off, c["top"] - z0), r_top=0.085)
    return m


def ground_line(path, width=0.12, mat="marker", z=0.06, closed=False, dash=None):
    """A flat band on the ground along a path: a boundary or a buried pipe run.
    dash=(on, off) breaks it into dashes."""
    m = Mesh()
    P = [np.asarray(p, float) for p in path]
    if closed:
        P = P + [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        d = (b - a) / L
        n = np.array([-d[1], d[0]]) * width / 2
        segs = [(0.0, L)] if not dash else [(s0, min(L, s0 + dash[0])) for s0 in np.arange(0.0, L, dash[0] + dash[1])]
        for s0, s1 in segs:
            p0, p1 = a + d * s0, a + d * s1
            m.add([(p0[0] - n[0], p0[1] - n[1], z), (p1[0] - n[0], p1[1] - n[1], z), (p1[0] + n[0], p1[1] + n[1], z), (p0[0] + n[0], p0[1] + n[1], z)],
                  [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], mat)
    return m


def ground_patch(outline, mat="hatch", z=0.05):
    """A flat tinted area on the ground (for marking a zone)."""
    m = Mesh()
    V, T = tessellate(outline)
    m.add([(x, y, z) for x, y in V], T, [[(V[i][0] * 0.5, V[i][1] * 0.5) for i in t] for t in T], mat)
    return m


def marker_post(h=1.2, r=0.035, mat="marker", disc=0.22):
    """Survey marker: a thin post with a disc on the ground."""
    m = Mesh()
    m.cylinder(disc, 0.03, mat, seg=18)
    m.cylinder(r, h, mat, seg=8)
    m.cylinder(r * 2.6, 0.1, mat, seg=10, M=translate(0, 0, h))
    return m


def manhole(r=0.25, square=False, rot=0.0):
    """Inspection chamber cover, flush with the ground."""
    m = Mesh()
    if square:
        m.box((2 * r, 2 * r, 0.04), "cast_iron", M=rot_z(rot))
    else:
        m.cylinder(r, 0.04, "cast_iron", seg=20)
    return m


def patio(slabs, mat="concrete", thickness=0.03, gap=0.012, seed=3):
    """Paving from a list of slabs (x, y, w, d) in local metres (x,y = slab corner)."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    ts = tex_size(mat)
    for (x, y, w, d) in slabs:
        g = gap / 2
        z = thickness + float(rng.uniform(-0.006, 0.006))
        uo, vo = float(rng.uniform(0, 1)), float(rng.uniform(0, 1))
        m.box((w - gap, d - gap, z), mat, M=translate(x + g, y + g, 0), origin="corner",
              skip=("-z",), uv_off=(uo, vo), ts=ts)
    return m


def gate_posts(gap=1.1, height=1.85, post=0.1, mat="wood_new", rail=False):
    """A pair of new timber posts either side of a path. Origin midway between them."""
    m = Mesh()
    for sx in (-1, 1):
        m.box((post, post, height), mat, M=translate(sx * gap / 2, 0, 0), grain=True)
    if rail:
        m.beam((-gap / 2, 0, height - 0.1), (gap / 2, 0, height - 0.1), 0.07, 0.04, mat)
    return m


def wishing_well(r=0.5, wall_h=0.75, post_h=1.85, roof=(1.5, 1.2), ridge=0.45, brick="brick_red_old", tiles="clay_tiles"):
    """Ornamental garden well: a brick drum, two posts with a windlass, a small tiled roof.
    The posts stand on the local X axis; the roof ridge runs along X."""
    m = Mesh()
    m.lathe([(r, 0.0), (r, wall_h), (r - 0.11, wall_h)], brick, seg=20, cap_bottom=False)
    m.lathe([(r - 0.11, wall_h), (r - 0.11, 0.25)], "concrete", seg=20, cap_bottom=False)                # inside face
    m.cylinder(r - 0.11, 0.02, "soil", seg=20, M=translate(0, 0, 0.25))
    for sx in (-1, 1):
        m.box((0.09, 0.09, post_h), "wood_grey", M=translate(sx * (r + 0.05), 0, 0), grain=True)
        m.beam((sx * (r + 0.05), 0, post_h - 0.45), (sx * (r - 0.3), 0, post_h - 0.05), 0.05, 0.05, "wood_grey")   # brace
    m.tube([(-r - 0.05, 0, wall_h + 0.45), (r + 0.05, 0, wall_h + 0.45)], 0.045, "wood_grey", seg=8)       # windlass
    gable_roof(m, roof[1], roof[0], post_h, post_h + ridge, tiles, overhang=0.12, verge=0.08, M=rot_z(90))
    return m


def picket_gate(width=0.95, height=1.1, post_h=1.3, post=0.11, mat="paint_white", open_deg=0.0):
    """Painted picket gate between two posts. The posts stand on the local X axis, the
    gate hangs from the one at x = 0."""
    m = Mesh()
    for x in (-post / 2, width + post / 2):
        m.box((post, post, post_h), mat, M=translate(x, 0, 0))
        m.box((post + 0.04, post + 0.04, 0.03), mat, M=translate(x, 0, post_h))
    G = rot_z(open_deg)
    for z in (0.28, height - 0.25):
        m.box((width - 0.04, 0.03, 0.07), mat, M=G @ translate(0.02, 0.0, z), origin="corner")
    n = max(3, int(width / 0.105))
    for k in range(n):
        x = 0.03 + (width - 0.06 - 0.06) * k / (n - 1)
        h = height - 0.07 * math.cos(math.pi * (k / (n - 1) - 0.5)) + 0.05          # a dipped top line
        m.box((0.06, 0.02, h - 0.08), mat, M=G @ translate(x, -0.035, 0.08), origin="corner")
    return m


def log_store(width=1.9, depth=0.75, front_h=1.15, back_h=1.3, mat="wood_new", cover="sheet_white", fill=0.7):
    """Slatted timber log store with a sheeted top, open at the front (local -Y). The back
    stands against a wall on the local X axis; origin at the back-left corner."""
    m = Mesh()
    for x in (0.0, width - 0.07):
        m.box((0.07, 0.07, front_h), mat, M=translate(x, -depth, 0), origin="corner", grain=True)
        m.box((0.07, 0.07, back_h), mat, M=translate(x, -0.07, 0), origin="corner", grain=True)
        for z in np.arange(0.12, front_h - 0.05, 0.19):                                  # slatted ends
            m.box((0.02, depth - 0.14, 0.1), mat, M=translate(x + 0.025, -depth + 0.07, z), origin="corner")
    m.box((0.07, 0.07, front_h), mat, M=translate(width / 2 - 0.035, -depth, 0), origin="corner", grain=True)
    for z in np.arange(0.12, front_h - 0.05, 0.19):                                      # slatted front, with the logs showing through
        m.box((width, 0.02, 0.1), mat, M=translate(0, -depth - 0.02, z), origin="corner")
    m.box((width - 0.16, depth - 0.2, front_h * fill), "log_end", M=translate(0.08, -depth + 0.06, 0.08), origin="corner")
    top = [(-0.06, -depth - 0.1, front_h + 0.02), (width + 0.06, -depth - 0.1, front_h + 0.02), (width + 0.06, 0.0, back_h + 0.02), (-0.06, 0.0, back_h + 0.02)]
    m.add(top, [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], cover)
    m.add([(q[0], q[1], q[2] - 0.12) for q in top[:2]][::-1] + [top[0], top[1]][::-1], [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], cover)   # the sheet hanging over the front
    return m
