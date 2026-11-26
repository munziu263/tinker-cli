"""Boundaries and buildings: brick walls, fences, sheds, the outbuilding, the house wing.

All functions return a core.Mesh in local coordinates (metres, Z up).
Paths are lists of (x, y) points relative to the feature's origin.
"""
from __future__ import annotations

import math

import numpy as np

from .core import Mesh, rot_x, rot_y, rot_z, tex_size, translate


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
                birdhouse_z=None):
    """The dark creosoted shed: gable ends face +-Y (local), -Y is the garden-facing gable.
    Origin: centre of the footprint on the ground."""
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
    gable_roof(m, width, depth, eaves, ridge, roof, overhang=0.22, verge=0.16)
    # corner trims and a plinth board
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.box((0.07, 0.07, eaves), "boards_dark",
                  M=translate(sx * (half_w + 0.012), sy * (half_d + 0.012), 0), grain=True)
    # door on the side wall with a pale weathered frame
    sgn = 1 if door_side == "+x" else -1
    dM = translate(sgn * (half_w + 0.02), -half_d + 0.45, 0) @ rot_z(90 if sgn > 0 else -90)
    m.box((0.8, 0.035, 1.7), wall, M=dM @ translate(0.4, 0, 0), ts=ts)
    for bx in (0.0, 0.8):
        m.box((0.06, 0.05, 1.76), "wood_grey", M=dM @ translate(bx, 0, 0), grain=True)
    m.box((0.92, 0.05, 0.06), "wood_grey", M=dM @ translate(0.4, 0, 1.72), grain=True)
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


def outbuilding(width=6.2, depth=3.6, front_h=2.15, back_h=2.15, rise=0.0, window=None,
                door=None, extra_windows=(), wall="render_white"):
    """White roughcast garden room with a low sheet roof. The front (garden side) is the
    local -Y face; x runs 0..width from its left corner as seen from the garden.
    rise: how much higher the roof is at the x = width end (it drains to the x = 0 end).
    Origin: front-left corner on the ground."""
    m = Mesh()
    ts = tex_size(wall)
    window = window or dict(x=0.75, w=1.75, z=0.95, h=1.1)
    door = door or dict(x=3.25, w=0.85, z=0.02, h=2.0)
    ops = [dict(window, sill_mat="terracotta"), dict(door, reveal=0.07)] + [dict(w) for w in extra_windows]
    _wall_with_openings(m, width, front_h, wall, ops, ts=ts)
    hr_f, hr_b = front_h + rise, back_h + rise
    if rise > 1e-6:  # wedge of wall between the level wall-head and the sloping roof
        m.add([(0, 0, front_h), (width, 0, front_h), (width, 0, hr_f)], [(0, 1, 2)],
              [[(0, front_h / ts[1]), (width / ts[0], front_h / ts[1]), (width / ts[0], hr_f / ts[1])]], wall)
        m.add([(width, depth, back_h), (0, depth, back_h), (width, depth, hr_b)], [(0, 1, 2)],
              [[(0, back_h / ts[1]), (width / ts[0], back_h / ts[1]), (0, hr_b / ts[1])]], wall)
    m.quad((width, 0, 0), (width, depth, 0), (width, depth, hr_b), (width, 0, hr_f), wall, ts=ts,
           uv=[(0, 0), (depth / ts[0], 0), (depth / ts[0], hr_b / ts[1]), (0, hr_f / ts[1])])
    m.quad((width, depth, 0), (0, depth, 0), (0, depth, back_h), (width, depth, back_h), wall, ts=ts)
    m.quad((0, depth, 0), (0, 0, 0), (0, 0, front_h), (0, depth, back_h), wall, ts=ts,
           uv=[(0, 0), (depth / ts[0], 0), (depth / ts[0], front_h / ts[1]), (0, back_h / ts[1])])
    m.box((window["w"] + 0.12, 0.14, 0.05), "terracotta",
          M=translate(window["x"] - 0.06, -0.05, window["z"] - 0.05), origin="corner")
    _glazing(m, window["x"], window["z"], window["w"], window["h"], bars=(4, 3), depth=0.07)
    for w in extra_windows:
        _glazing(m, w["x"], w["z"], w["w"], w["h"], bars=w.get("bars", (2, 3)), depth=0.07)
    _glazing(m, door["x"], door["z"], door["w"], door["h"], frame="paint_brown", bars=(2, 5),
             frame_w=0.09, depth=0.06, bar_w=0.03)
    # roof: black profiled sheet with a deep fascia, oversailing all round
    oh, fas = 0.22, 0.17
    k = rise / width

    def top(x, y):
        base = front_h + (back_h - front_h) * min(1.0, max(0.0, y / depth))
        return base + 0.11 + k * x

    ring = [(-oh, -oh), (width + oh, -oh), (width + oh, depth + oh), (-oh, depth + oh)]
    T = [(x, y, top(x, y)) for x, y in ring]
    m.add(T, [(0, 1, 2, 3)], [[(0, 0), (8, 0), (8, 4), (0, 4)]], "roof_felt")
    m.add([(p[0], p[1], p[2] - fas) for p in T][::-1], [(0, 1, 2, 3)],
          [[(0, 0), (1, 0), (1, 1), (0, 1)]], "paint_black")
    for a in range(4):
        p, q = T[a], T[(a + 1) % 4]
        m.add([(p[0], p[1], p[2] - fas), (q[0], q[1], q[2] - fas), q, p], [(0, 1, 2, 3)],
              [[(0, 0), (1, 0), (1, 1), (0, 1)]], "paint_black")
    n_rib = int((width + 2 * oh) / 0.25)
    for a in range(n_rib + 1):
        x = -oh + (width + 2 * oh) * a / n_rib
        m.beam((x, -oh, top(x, -oh) + 0.005), (x, depth + oh, top(x, depth + oh) + 0.005), 0.03, 0.025, "roof_felt")
    return m


def weatherboard_block(width=3.2, depth=3.0, eaves=2.5, ridge=3.5, window=None, canopy=True,
                       chimney=True, wall="weatherboard_dark", roof="clay_tiles",
                       canopy_z=None, canopy_out=1.1, canopy_w=None):
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
