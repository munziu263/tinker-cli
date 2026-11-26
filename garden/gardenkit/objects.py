"""Furniture and loose objects. Each returns a core.Mesh with its origin on the ground
at the middle of the object, +X along its length."""
from __future__ import annotations

import math

import numpy as np

from .core import Mesh, frame_from_axis, rot_x, rot_y, rot_z, scale, translate
from .plants import _noise3, _unit, cards, shrub, spray


# --------------------------------------------------------------------------- seating & tables


def picnic_table(length=1.8, top_w=0.72, top_h=0.75, seat_h=0.45, seat_w=0.27, span=1.5,
                 mat="wood_new", plank=0.04):
    """A-frame picnic table: 5-plank top, two 2-plank benches, splayed legs."""
    m = Mesh()
    n_top = 5
    pw = top_w / n_top
    for i in range(n_top):
        y = -top_w / 2 + pw * (i + 0.5)
        m.box((length, pw - 0.008, plank), mat, M=translate(0, y, top_h - plank), grain=True)
    seat_y = span / 2 - seat_w / 2
    for sy in (-1, 1):
        for k in (0, 1):
            y = sy * (seat_y + (k - 0.5) * seat_w / 2)
            m.box((length, seat_w / 2 - 0.008, plank), mat, M=translate(0, y, seat_h - plank), grain=True)
    for sx in (-1, 1):
        x = sx * (length / 2 - 0.32)
        # top bearer, seat bearer
        m.box((0.045, top_w - 0.04, 0.09), mat, M=translate(x, 0, top_h - plank - 0.09), grain=True)
        m.box((0.045, span, 0.09), mat, M=translate(x + sx * 0.045, 0, seat_h - plank - 0.09), grain=True)
        for sy in (-1, 1):  # splayed legs
            m.beam((x - sx * 0.0, sy * (top_w / 2 - 0.1), top_h - plank - 0.02),
                   (x - sx * 0.0, sy * (span / 2 - 0.12), 0.0), 0.045, 0.09, mat)
        # diagonal brace up to the middle of the top
        m.beam((x, 0, seat_h - plank - 0.06), (sx * 0.12, 0, top_h - plank - 0.02), 0.04, 0.07, mat)
    return m


def garden_bench(length=1.5, seat_h=0.43, seat_d=0.5, back_h=0.92, arm_h=0.63, mat="wood_grey"):
    """Classic slatted hardwood bench with arms. The sitter faces local -Y."""
    m = Mesh()
    leg = 0.065
    hx = length / 2 - leg / 2
    for sx in (-1, 1):
        m.box((leg, leg, arm_h), mat, M=translate(sx * hx, -seat_d / 2 + leg / 2, 0), grain=True)   # front leg
        m.beam((sx * hx, seat_d / 2 - leg / 2, 0), (sx * hx, seat_d / 2 + 0.04, back_h), leg, leg, mat)  # back leg
        m.box((leg + 0.02, seat_d + 0.04, 0.035), mat, M=translate(sx * hx, 0.0, arm_h), grain=True)  # arm
        m.box((0.035, seat_d - leg, 0.07), mat, M=translate(sx * hx, 0, seat_h - 0.09), grain=True)   # side rail
        m.box((0.03, seat_d - leg, 0.04), mat, M=translate(sx * hx, 0, 0.14), grain=True)             # stretcher
    for i in range(5):  # seat slats
        y = -seat_d / 2 + 0.05 + i * (seat_d - 0.09) / 4
        m.box((length - leg * 0.2, 0.075, 0.025), mat, M=translate(0, y, seat_h - 0.025), grain=True)
    m.box((length - leg, 0.03, 0.07), mat, M=translate(0, -seat_d / 2 + 0.03, seat_h - 0.1), grain=True)
    # back: top + bottom rail and vertical slats (slightly reclined)
    y_lo, y_hi = seat_d / 2 + 0.005, seat_d / 2 + 0.04
    m.box((length - leg, 0.03, 0.08), mat, M=translate(0, y_hi, back_h - 0.08), grain=True)
    m.box((length - leg, 0.03, 0.05), mat, M=translate(0, y_lo, seat_h + 0.06), grain=True)
    n = max(7, int(length / 0.11))
    for i in range(n):
        x = -length / 2 + leg + (length - 2 * leg) * (i + 0.5) / n
        m.beam((x, y_lo, seat_h + 0.1), (x, y_hi, back_h - 0.07), 0.045, 0.016, mat)
    m.box((length - leg, 0.03, 0.04), mat, M=translate(0, seat_d / 2 - 0.03, 0.14), grain=True)
    return m


def bistro_table(w=0.55, d=0.54, h=0.70, slats=7, frame="steel_black", top="wood_grey"):
    """Folding bistro table: slatted wooden top on a crossed black steel frame."""
    m = Mesh()
    sw = w / slats
    for i in range(slats):
        x = -w / 2 + sw * (i + 0.5)
        m.box((sw - 0.008, d, 0.018), top, M=translate(x, 0, h - 0.018), grain=True)
    for sy in (-1, 1):
        y = sy * (d / 2 - 0.035)
        m.tube([(-w / 2 + 0.05, y, 0.0), (w / 2 - 0.05, y, h - 0.03)], 0.011, frame, seg=5)
        m.tube([(w / 2 - 0.05, y, 0.0), (-w / 2 + 0.05, y, h - 0.03)], 0.011, frame, seg=5)
    for x in (-w / 2 + 0.05, w / 2 - 0.05):
        m.tube([(x, -d / 2 + 0.035, 0.012), (x, d / 2 - 0.035, 0.012)], 0.011, frame, seg=5)
        m.tube([(x, -d / 2 + 0.035, h - 0.03), (x, d / 2 - 0.035, h - 0.03)], 0.009, frame, seg=5)
    return m


def folding_table_wood(w=0.6, d=0.6, h=0.62, top="wood_grey"):
    """Small weathered folding garden table with a slatted top and crossed wooden legs."""
    m = Mesh()
    n = 6
    for i in range(n):
        x = -w / 2 + w * (i + 0.5) / n
        m.box((w / n - 0.01, d, 0.02), top, M=translate(x, 0, h - 0.02), grain=True)
    for sy in (-1, 1):
        y = sy * (d / 2 - 0.05)
        m.beam((-w / 2 + 0.04, y, 0), (w / 2 - 0.06, y, h - 0.03), 0.045, 0.02, top)
        m.beam((w / 2 - 0.04, y - sy * 0.025, 0), (-w / 2 + 0.06, y - sy * 0.025, h - 0.03), 0.045, 0.02, top)
    m.box((0.04, d - 0.1, 0.02), top, M=translate(-w / 2 + 0.06, 0, h - 0.05), grain=True)
    m.box((0.04, d - 0.1, 0.02), top, M=translate(w / 2 - 0.06, 0, h - 0.05), grain=True)
    return m


def monobloc_chair(mat="plastic_teal", seat_h=0.43, w=0.56, d=0.52, back_h=0.80):
    """Moulded plastic stacking chair with arms. The sitter faces local -Y."""
    m = Mesh()
    sw = w * 0.8
    m.box((sw, d * 0.82, 0.025), mat, M=translate(0, 0, seat_h - 0.025))
    for sx in (-1, 1):
        for sy, splay in ((-1, 0.07), (1, 0.09)):
            top = (sx * (sw / 2 - 0.035), sy * (d * 0.36), seat_h - 0.02)
            bot = (sx * (w / 2 - 0.03), sy * (d * 0.36 + splay), 0.0)
            m.tube([bot, top], [0.02, 0.028], mat, seg=6)
        # arm: up from the front leg, back to the backrest
        a0 = (sx * (w / 2 - 0.04), -d * 0.36, seat_h)
        a1 = (sx * (w / 2 - 0.02), -d * 0.34, seat_h + 0.21)
        a2 = (sx * (w / 2 - 0.03), d * 0.36, seat_h + 0.23)
        m.tube([a0, a1, a2], [0.022, 0.024, 0.022], mat, seg=6, flatten=(1.0, 0.7))
    # reclined back: frame + vertical ribs with open slots
    y0, y1 = d * 0.38, d * 0.50
    z0, z1 = seat_h, back_h
    for x in np.linspace(-sw / 2 + 0.03, sw / 2 - 0.03, 6):
        m.beam((x, y0, z0), (x, y1, z1 - 0.04), 0.038, 0.014, mat)
    m.beam((-sw / 2, y1, z1 - 0.05), (sw / 2, y1, z1 - 0.05), 0.016, 0.09, mat, roll=90)
    m.beam((-sw / 2, y0 + 0.01, z0 + 0.03), (sw / 2, y0 + 0.01, z0 + 0.03), 0.016, 0.06, mat, roll=90)
    return m


def folding_chair(folded=False, slat="wood_grey", frame="steel_black", seat_h=0.45, w=0.4,
                  lean=14.0):
    """Slatted wood-and-steel folding garden chair. folded=True gives it flat, to lean
    against something (local Z is then the length of the folded chair)."""
    m = Mesh()
    if folded:   # stood on end, leaning back (+Y) against a wall
        f = Mesh()
        for sx in (-1, 1):
            f.tube([(sx * w / 2, 0, 0), (sx * w / 2, 0, 0.86)], 0.011, frame, seg=5)
            f.tube([(sx * (w / 2 - 0.03), 0.03, 0.05), (sx * (w / 2 - 0.03), 0.03, 0.62)], 0.01, frame, seg=5)
        for z in (0.2, 0.26, 0.32, 0.38, 0.44, 0.50, 0.68, 0.78):
            f.box((w - 0.02, 0.015, 0.045), slat, M=translate(0, 0.012, z), grain=True)
        return m.merge(f, rot_x(-lean))
    d = 0.38
    for sx in (-1, 1):
        x = sx * w / 2
        m.tube([(x, -d / 2, 0), (x, d / 2 + 0.06, 0.82)], 0.011, frame, seg=5)       # front foot -> back top
        m.tube([(x, d / 2 + 0.02, 0), (x, -d / 2 + 0.02, seat_h)], 0.011, frame, seg=5)  # back foot -> seat front
    for i in range(5):
        y = -d / 2 + 0.04 + i * (d - 0.06) / 4
        m.box((w - 0.02, 0.05, 0.016), slat, M=translate(0, y, seat_h), grain=True)
    for z in (0.62, 0.74):
        yb = d / 2 + 0.06 * (z / 0.82) + 0.01
        m.box((w - 0.02, 0.016, 0.06), slat, M=translate(0, yb, z), grain=True)
    return m


def step_stool(h=0.62, top=(0.42, 0.26), mat="wood_brown"):
    """Wooden workshop stool with splayed legs and rungs."""
    m = Mesh()
    m.box((top[0], top[1], 0.03), mat, M=translate(0, 0, h - 0.03), grain=True)
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.beam((sx * (top[0] / 2 + 0.07), sy * (top[1] / 2 + 0.05), 0),
                   (sx * (top[0] / 2 - 0.04), sy * (top[1] / 2 - 0.03), h - 0.03), 0.04, 0.04, mat)
        m.beam((sx * (top[0] / 2 + 0.03), -top[1] / 2 - 0.02, h * 0.42),
               (sx * (top[0] / 2 + 0.03), top[1] / 2 + 0.02, h * 0.42), 0.03, 0.02, mat)
    m.beam((-top[0] / 2 - 0.03, -top[1] / 2 - 0.02, h * 0.42), (top[0] / 2 + 0.03, -top[1] / 2 - 0.02, h * 0.42),
           0.03, 0.02, mat)
    return m


# --------------------------------------------------------------------------- wood & bags


def firewood_bag(w=0.55, d=0.42, h=0.85, lean=6.0, seed=1, mat="firewood_bag"):
    """Plump woven sack of logs with a gathered neck."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    prof = [(0.34, 0.0), (0.47, 0.06), (0.5, 0.25), (0.5, 0.5), (0.46, 0.7), (0.36, 0.84),
            (0.22, 0.93), (0.2, 1.0)]
    prof = [(r * (1 + rng.uniform(-0.04, 0.04)), z * h) for r, z in prof]
    M = rot_y(lean) @ scale(w, d, 1.0)
    m.lathe(prof, mat, seg=14, M=M, ts=(1.0 * 2 * math.pi * 0.5, h), cap_top="sheet_white")
    return m


def log_stack(length=2.2, height=1.0, depth=0.32, seed=2, bark="bark_willow", end="log_end"):
    """Neat stack of split logs, ends showing; heaped slightly higher in the middle.
    Local -Y is the visible face."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    z = 0.0
    row = 0
    while z < height:
        rh = rng.uniform(0.085, 0.115)
        frac = z / height
        half = (length / 2) * (1.0 - 0.28 * frac ** 2.2)   # sides batter in towards the top
        x = -half + (0.05 if row % 2 else 0.0)
        while x < half:
            wdt = rng.uniform(0.09, 0.15)
            cx = x + wdt / 2
            top_limit = height * (1 - 0.22 * (cx / (length / 2)) ** 2)
            if z + rh * 0.5 > top_limit:
                x += wdt
                continue
            L = depth * rng.uniform(0.85, 1.1)
            y0 = rng.uniform(-0.03, 0.03) - depth / 2
            k = rng.integers(0, 3)
            # cross-section of a split log: wedge, quarter or half-round
            if k == 0:
                sec = [(-wdt / 2, 0), (wdt / 2, 0), (rng.uniform(-0.2, 0.2) * wdt, rh)]
            elif k == 1:
                sec = [(-wdt / 2, 0), (wdt / 2, 0), (wdt / 2 * 0.9, rh * 0.75), (-wdt * 0.1, rh)]
            else:
                sec = [(-wdt / 2, 0), (wdt / 2, 0), (wdt * 0.35, rh * 0.8), (0, rh),
                       (-wdt * 0.35, rh * 0.8)]
            tilt = rng.uniform(-6, 6)
            ca, sa = math.cos(math.radians(tilt)), math.sin(math.radians(tilt))
            pts = [(cx + px * ca - pz * sa * 0.3, z + pz) for px, pz in sec]
            n = len(pts)
            front = [(p[0], y0, p[1]) for p in pts]
            back = [(p[0], y0 + L, p[1]) for p in pts]
            fuv = [(0.5 + (p[0] - cx) / wdt * 0.5, 0.2 + (p[1] - z) / rh * 0.6) for p in pts]
            m.add(front, [tuple(range(n))], [fuv], end)
            m.add(back[::-1], [tuple(range(n))], [fuv[::-1]], end)
            for i in range(n):
                j = (i + 1) % n
                m.add([front[j], front[i], back[i], back[j]], [(0, 1, 2, 3)],
                      [[(0, 0), (0.15, 0), (0.15, L / 2), (0, L / 2)]], bark)
            x += wdt + 0.004
        z += rh * 0.86
        row += 1
    return m


def log_rounds(n=14, area=(3.4, 1.3), seed=3, r=(0.16, 0.30), length=(0.28, 0.45),
               bark="bark_willow", end="log_end", debris=True):
    """Tumbled heap of sawn trunk rounds waiting to be split, with a litter of
    branch wood between them."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    for i in range(n):
        rad = rng.uniform(*r)
        L = rng.uniform(*length)
        x = rng.uniform(-area[0] / 2, area[0] / 2)
        y = rng.uniform(-area[1] / 2, area[1] / 2) * (0.5 + 0.5 * rng.random())
        c = Mesh()
        c.cylinder(rad, L, bark, seg=12, cap_mat=end, ts=(1.0, 1.0))
        mode = rng.random()
        if mode < 0.45:      # lying on its side
            M = translate(x, y, rad) @ rot_z(rng.uniform(0, 360)) @ rot_x(90) @ translate(0, 0, -L / 2)
        elif mode < 0.8:     # standing on end
            M = translate(x, y, 0) @ rot_z(rng.uniform(0, 360))
        else:                # tipped against a neighbour
            M = translate(x, y, rad * 0.5) @ rot_z(rng.uniform(0, 360)) @ rot_x(rng.uniform(40, 65)) \
                @ translate(0, 0, -L / 2)
        m.merge(c, M)
    if debris:
        for _ in range(16):
            x, y = rng.uniform(-area[0] / 2, area[0] / 2), rng.uniform(-area[1] / 2, area[1] / 2)
            a = rng.uniform(0, math.pi)
            L = rng.uniform(0.5, 1.4)
            dx, dy = math.cos(a) * L / 2, math.sin(a) * L / 2
            z0, z1 = rng.uniform(0.02, 0.25), rng.uniform(0.02, 0.4)
            m.tube([(x - dx, y - dy, z0), (x + dx, y + dy, z1)], rng.uniform(0.012, 0.03), "twig_dry", seg=5)
    return m


def plank_slab(length=0.95, width=0.26, thick=0.06, lean=14.0, mat="wood_new"):
    """Waney-edge slab of timber standing on end, leaning back against something (+Y)."""
    m = Mesh()
    m.box((width, thick, length), mat, M=rot_x(-lean), grain=True)
    return m


def sheet(w=0.61, h=1.22, thick=0.006, lean=12.0, mat="hardboard"):
    """Sheet of board leaning back (+Y) against a wall."""
    m = Mesh()
    m.box((w, thick, h), mat, M=rot_x(-lean))
    return m


def leaning_boards(length=1.95, height=0.72, lean=10.0, mat="wood_brown"):
    """Old door / pallet boards stood on their long edge against the shed (+Y side)."""
    m = Mesh()
    M = rot_x(-lean)
    m.box((length, 0.03, height), mat, M=M @ translate(0, 0, 0.0), grain=True)
    m.box((length + 0.12, 0.05, 0.11), "wood_grey", M=M @ translate(0.05, -0.04, height - 0.02), grain=True)
    m.box((length * 0.7, 0.04, 0.09), "wood_grey", M=M @ translate(-0.2, -0.035, height * 0.45), grain=True)
    return m


# --------------------------------------------------------------------------- pots & vessels


def pot(r=0.2, h=0.3, mat="terracotta", style="standard", soil=True, plant=None, seed=1,
        plant_kw=None):
    """Plant pot. style: standard (flared, rolled rim), bowl, belly, bucket, tall."""
    m = Mesh()
    profiles = {
        "standard": [(0.66, 0), (0.92, 0.82), (1.0, 0.84), (1.0, 1.0), (0.9, 1.0), (0.86, 0.9)],
        "bowl": [(0.5, 0), (0.85, 0.3), (1.0, 0.8), (1.0, 1.0), (0.9, 1.0), (0.88, 0.9)],
        "belly": [(0.55, 0), (0.95, 0.35), (1.0, 0.6), (0.82, 0.9), (0.9, 1.0), (0.8, 1.0), (0.74, 0.9)],
        "bucket": [(0.8, 0), (1.0, 0.95), (1.04, 0.96), (1.04, 1.0), (0.94, 1.0), (0.92, 0.9)],
        "tall": [(0.7, 0), (0.85, 0.5), (1.0, 0.95), (1.0, 1.0), (0.88, 1.0), (0.86, 0.92)],
    }
    prof = [(pr * r, pz * h) for pr, pz in profiles[style]]
    m.lathe(prof, mat, seg=14, ts=(1, 1))
    if soil:
        m.cylinder(prof[-1][0], 0.005, "soil", seg=14, M=translate(0, 0, prof[-1][1] - 0.005))
    if plant:
        kw = dict(seed=seed)
        kw.update(plant_kw or {})
        if plant == "shrub":
            kw.setdefault("rx", r * 1.2)
            kw.setdefault("h", r * 2.2)
            kw.setdefault("card", 0.16)
            m.merge(shrub(**kw), translate(0, 0, h * 0.85))
        elif plant == "grass":
            rng = np.random.default_rng(seed)
            for _ in range(10):
                a = rng.uniform(0, 2 * math.pi)
                d = _unit(np.array([math.cos(a) * 0.6, math.sin(a) * 0.6, 1.0]))
                spray(m, (0, 0, h * 0.85), d, kw.get("mat", "twigs_stems"), size=r * 2.4, rng=rng, crossed=1)
    return m


def trough(length=0.9, w=0.26, h=0.2, mat="glaze_teal", plant="leaf_green", seed=2):
    m = Mesh()
    m.box((length, w, h), mat)
    m.box((length - 0.03, w - 0.03, 0.01), "soil", M=translate(0, 0, h - 0.005))
    if plant:
        rng = np.random.default_rng(seed)
        P = [(rng.uniform(-length / 2, length / 2), rng.uniform(-w / 2, w / 2), h + rng.uniform(0.02, 0.2))
             for _ in range(40)]
        cards(m, P, [(0, 0, 1)] * len(P), plant, size=0.2, rng=rng, jitter=0.7)
    return m


def water_butt(r=0.3, h=0.92, mat="plastic_green", lid=True):
    m = Mesh()
    prof = [(r * 0.86, 0), (r, h * 0.12), (r, h * 0.88), (r * 0.95, h)]
    m.lathe(prof, mat, seg=16, ts=(1, 1), cap_top=mat)
    for z in (h * 0.3, h * 0.55, h * 0.8):
        m.lathe([(r * 1.0, z - 0.012), (r * 1.025, z), (r * 1.0, z + 0.012)], mat, seg=16, ts=(1, 1), cap_bottom=False)
    if lid:
        m.cylinder(r * 0.99, 0.035, mat, seg=16, M=translate(0, 0, h), r_top=r * 0.9)
    return m


def barrel_lying(r=0.16, length=0.45, mat="plastic_blue"):
    m = Mesh()
    c = Mesh()
    c.cylinder(r, length, mat, seg=14)
    m.merge(c, translate(0, 0, r) @ rot_y(90) @ translate(0, 0, -length / 2))
    return m


def metal_bin(r=0.24, h=0.62, mat="galvanised"):
    """Galvanised incinerator / dustbin."""
    m = Mesh()
    m.lathe([(r * 0.9, 0), (r, h), (r * 1.03, h)], mat, seg=14, ts=(1, 1))
    m.cylinder(r * 0.98, 0.01, "paint_black", seg=14, M=translate(0, 0, h - 0.06))
    return m


def bbq_covered(r=0.3, h=1.0, wreath=True, seed=3):
    """Kettle barbecue under a black cover, with the willow wreath left on top."""
    m = Mesh()
    prof = [(r * 1.05, 0.0), (r * 1.0, h * 0.25), (r * 1.08, h * 0.55), (r * 1.1, h * 0.72),
            (r * 0.85, h * 0.92), (r * 0.35, h), (0.01, h)]
    m.lathe(prof, "bbq_cover", seg=14, ts=(1, 1))
    if wreath:
        rng = np.random.default_rng(seed)
        R = r * 1.25
        M = translate(0.05, 0, h * 0.86) @ rot_x(12) @ rot_y(-8)
        for k in range(4):
            loop = [((R + rng.uniform(-0.03, 0.03)) * math.cos(t), (R + rng.uniform(-0.03, 0.03)) * math.sin(t),
                     rng.uniform(-0.025, 0.025)) for t in np.linspace(0, 2 * math.pi, 22)]
            loop[-1] = loop[0]
            m.tube(loop, 0.014, "twig", seg=5, caps=(False, False), M=M)
    return m


def tarp_heap(size=(1.1, 0.9, 0.85), mat="tarp_yellow", seed=4, skirt="sheet_white"):
    """Something lumpy under a tarpaulin, with a dust sheet bundled beside it."""
    m = Mesh()
    noise = _noise3(seed)
    from .plants import _icosphere
    V, F = _icosphere(2)
    P = []
    for p in V:
        k = 1 + 0.22 * float(noise(p * 1.8))
        q = np.array([p[0] * size[0] / 2 * k, p[1] * size[1] / 2 * k, max(0.0, (p[2] * 0.62 + 0.42) * size[2] * k)])
        P.append(q)
    m.add(P, F, [[(P[i][0], P[i][1]) for i in tri] for tri in F], mat, smooth=True)
    if skirt:
        Q = [np.array([p[0] * size[0] * 0.42 + size[0] * 0.38, p[1] * size[1] * 0.5 - size[1] * 0.25,
                       max(0.0, (p[2] * 0.6 + 0.35) * size[2] * 0.42)]) for p in V]
        m.add(Q, F, [[(Q[i][0], Q[i][1]) for i in tri] for tri in F], skirt, smooth=True)
    return m


def lump(size=(0.9, 0.9, 0.5), mat="sheet_white", seed=5, lumps=0.2):
    """Generic soft heap: bulk bag, crumpled tarp."""
    m = Mesh()
    noise = _noise3(seed)
    from .plants import _icosphere
    V, F = _icosphere(2)
    P = [np.array([p[0] * size[0] / 2, p[1] * size[1] / 2, max(0.0, (p[2] * 0.6 + 0.4) * size[2])])
         * (1 + lumps * float(noise(p * 2.0))) for p in V]
    m.add(P, F, [[(P[i][0], P[i][1]) for i in tri] for tri in F], mat, smooth=True)
    return m


# --------------------------------------------------------------------------- small things


def ball(r=0.11, mat="ball_white"):
    m = Mesh()
    from .plants import _icosphere
    V, F = _icosphere(2)
    P = V * r + (0, 0, r)
    m.add(P, F, [[(0.5, 0.5)] * 3 for _ in F], mat, smooth=True)
    # a few dark panels
    for i, p in enumerate(V[:12]):
        m.cylinder(r * 0.3, 0.004, "paint_black", seg=5, M=frame_from_axis(p * r * 0.985 + (0, 0, r), p))
    return m


def hose_coil(r=0.24, turns=4, z=1.55, mat="hose_white", tail=0.9):
    """Coil of white hose hung on a branch stub; a loose end trails down."""
    m = Mesh()
    pts = []
    for i in range(turns * 18 + 1):
        t = 2 * math.pi * i / 18
        rr = r * (1 + 0.06 * math.sin(i * 0.7))
        pts.append((rr * math.cos(t), 0.012 * (i / 18) - 0.02, z + rr * math.sin(t)))
    m.tube(pts, 0.009, mat, seg=5, caps=(False, False))
    m.tube([(r, 0.03, z), (r * 1.3, 0.05, z - tail * 0.5), (r * 0.6, 0.06, z - tail)], 0.009, mat, seg=5)
    return m


def pop_up_goal(w=1.3, h=0.85, d=0.7, mat="plastic_red"):
    """Child's pop-up football goal: a red hoop with a net sloping to the ground."""
    m = Mesh()
    arc = [(w / 2 * math.cos(t), 0, h * math.sin(t)) for t in np.linspace(0, math.pi, 15)]
    m.tube(arc, 0.022, mat, seg=6)
    m.tube([(-w / 2, 0, 0.02), (-w / 2, d, 0.02), (w / 2, d, 0.02), (w / 2, 0, 0.02)], 0.015, mat, seg=5)
    for a in arc[1:-1:2]:   # a few net cords from the hoop down to the back bar
        m.tube([a, (a[0] * 0.95, d, 0.02)], 0.004, "net_white", seg=3, caps=(False, False))
    return m


def stake(h=2.1, r=0.016, mat="wood_grey", lean=(0.02, 0.0)):
    m = Mesh()
    m.tube([(0, 0, 0), (lean[0] * h, lean[1] * h, h)], [r, r * 0.8], mat, seg=6)
    return m


def brick_edging(n=6, radius=1.6, mat="brick_red_old"):
    """A few half-buried bricks set in a curve as the edge of a bed."""
    m = Mesh()
    for i in range(n):
        a = math.radians(-12 + i * 9.0)
        x, y = radius * math.sin(a), radius * (1 - math.cos(a))
        m.box((0.215, 0.1025, 0.05), mat, M=translate(x, y, 0) @ rot_z(math.degrees(a)))
    return m


def telegraph_pole(h=8.5, r=0.13):
    m = Mesh()
    m.tube([(0, 0, 0), (0.05, 0, h)], [r, r * 0.7], "wood_grey", seg=8)
    m.box((1.4, 0.08, 0.1), "wood_grey", M=translate(0.04, 0, h - 0.5))
    return m


def sign_board(h=3.0, board=(0.55, 0.85), mat="plastic_red"):
    m = Mesh()
    m.box((0.07, 0.07, h), "paint_white", grain=False)
    m.box((board[0], 0.03, board[1]), mat, M=translate(board[0] / 2 + 0.03, 0, h - board[1]))
    return m


def goal_posts(w=3.6, h=1.9, r=0.04, mat="paint_white"):
    """Full-size-ish white goal out on the playing field beyond the fence."""
    m = Mesh()
    m.tube([(-w / 2, 0, 0), (-w / 2, 0, h), (w / 2, 0, h), (w / 2, 0, 0)], r, mat, seg=6)
    return m


def massing(size=(8.0, 7.0), eaves=2.6, ridge=5.2, wall="brick_red_old", roof="clay_tiles",
            chimney=True):
    """Neighbouring house as simple massing: brick box, gable roof (ridge along X)."""
    from .structures import gable_roof, gable_wall
    from .core import tex_size
    m = Mesh()
    ts = tex_size(wall)
    w, d = size
    r = Mesh()
    gable_wall(r, d, eaves, ridge, wall, -w / 2, facing=-1, ts=ts)
    gable_wall(r, d, eaves, ridge, wall, w / 2, facing=1, ts=ts)
    for sgn in (-1, 1):
        a = (sgn * d / 2, -w / 2 * sgn, 0)
        b = (sgn * d / 2, w / 2 * sgn, 0)
        r.quad(a, b, (b[0], b[1], eaves), (a[0], a[1], eaves), wall, ts=ts)
    gable_roof(r, d, w, eaves, ridge, roof, overhang=0.35, verge=0.25, soffit="paint_white")
    m.merge(r, rot_z(90))
    if chimney:
        m.box((0.7, 0.5, ridge + 0.9), wall, M=translate(w * 0.18, 0.3, 0), ts=ts, skip=("-z",))
    return m


def pot_row(length=4.4, seed=8, start=0.7, big=True):
    """The jumble of pots along the house wall: local +X runs along the wall, the pots
    stand on the -Y side of it. Mixed terracotta, black plastic and glazed pots, a low
    white table of jars, and a long blue-green trough of bedding. big: end the row with a
    large black tub of evergreen (leave it out where a separate feature stands there)."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    kinds = [("terracotta", "standard", 0.12, 0.2, None), ("plastic_black", "bucket", 0.1, 0.17, None),
             ("glaze_blue", "belly", 0.17, 0.26, "shrub"), ("terracotta", "standard", 0.16, 0.26, "shrub"),
             ("plastic_black", "bucket", 0.13, 0.2, None), ("terracotta", "bowl", 0.14, 0.12, "shrub"),
             ("terracotta_pale", "standard", 0.11, 0.18, None), ("terracotta", "standard", 0.13, 0.22, "grass")]
    x = start
    for i, (mat, style, r, h, plant) in enumerate(kinds):
        y = -rng.uniform(0.28, 0.62)
        kw = dict(leaf="leaf_green", rx=r * 1.3, h=r * 2.0, card=0.12, density=40) if plant == "shrub" else None
        m.merge(pot(r=r, h=h, mat=mat, style=style, plant=plant, seed=seed + i, plant_kw=kw), translate(x, y, 0))
        x += r * 2 + rng.uniform(0.02, 0.18)
        if x > length * 0.52:
            break
    # little white table with jars
    tx_ = start + 0.75
    m.box((0.5, 0.3, 0.03), "paint_white", M=translate(tx_, -0.25, 0.3))
    for sx in (-0.2, 0.2):
        for sy in (-0.1, 0.1):
            m.box((0.03, 0.03, 0.3), "paint_white", M=translate(tx_ + sx, -0.25 + sy, 0))
    for k in range(3):
        m.cylinder(0.045, 0.12, "glass_dark", seg=8, M=translate(tx_ - 0.15 + 0.15 * k, -0.25, 0.33))
    # trough and a couple of pots beyond it
    m.merge(trough(length=1.05, w=0.3, h=0.2, seed=seed + 20), translate(length * 0.57 + 0.3, -0.8, 0) @ rot_z(-2))
    m.merge(pot(r=0.1, h=0.16, mat="terracotta", seed=seed + 30), translate(length * 0.57 + 1.15, -0.6, 0))
    if big:
        m.merge(pot(r=0.3, h=0.4, mat="plastic_black", style="bucket", plant="shrub", seed=seed + 31,
                    plant_kw=dict(leaf="leaf_green", rx=0.42, h=0.5, card=0.2)), translate(length * 0.57 + 1.75, -0.75, 0))
    return m


def stump(r=0.75, h=0.5, flare=1.7, seed=11, roots=7):
    """Sawn-off stump of a big tree, with root flares spreading into the lawn."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    seg = 28
    lobes = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0)) for _ in range(roots)]

    def radius(a, t):       # t = 0 at the ground, 1 at the cut
        bump = sum(amp * max(0.0, math.cos(a - a0)) ** 6 for a0, amp in lobes)
        return r * (1 + (flare - 1) * (1 - t) ** 2 * (0.35 + 0.65 * min(1.0, bump)))
    rings = 5
    verts, faces, uvs = [], [], []
    for i in range(rings + 1):
        t = i / rings
        for k in range(seg):
            a = 2 * math.pi * k / seg
            rr = radius(a, t)
            verts.append((rr * math.cos(a), rr * math.sin(a), h * t ** 0.7))
    for i in range(rings):
        for k in range(seg):
            k2 = (k + 1) % seg
            faces.append((i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k))
            uvs.append([(k / seg * 3, i / rings), ((k + 1) / seg * 3, i / rings), ((k + 1) / seg * 3, (i + 1) / rings), (k / seg * 3, (i + 1) / rings)])
    m.add(verts, faces, uvs, "bark_willow", smooth=True)
    top = verts[rings * seg:]
    m.add(top, [tuple(range(seg))], [[(0.5 + 0.5 * math.cos(2 * math.pi * k / seg), 0.5 + 0.5 * math.sin(2 * math.pi * k / seg)) for k in range(seg)]], "stump_wood")
    return m


def compost_bin(r=0.4, h=0.95, mat="plastic_teal"):
    """Conical plastic compost bin with a lid."""
    m = Mesh()
    m.lathe([(r, 0.0), (r * 0.72, h * 0.93), (r * 0.74, h)], mat, seg=16, ts=(1, 1), cap_top=mat)
    m.cylinder(r * 0.5, 0.05, mat, seg=14, M=translate(0, 0, h))
    return m


def raised_bed(size=(3.0, 1.2), h=0.2, board=0.04, mat="wood_grey", planted=None, seed=5):
    """Timber-edged vegetable bed. planted: leaf material for a low crop, or None for bare soil."""
    m = Mesh()
    L, W = size
    for sy in (-1, 1):
        m.box((L, board, h), mat, M=translate(0, sy * (W / 2 - board / 2), 0), grain=True)
    for sx in (-1, 1):
        m.box((board, W - 2 * board, h), mat, M=translate(sx * (L / 2 - board / 2), 0, 0), grain=True)
    m.box((L - 2 * board, W - 2 * board, 0.01), "soil", M=translate(0, 0, h * 0.75))
    if planted:
        rng = np.random.default_rng(seed)
        P = [(rng.uniform(-L / 2 + 0.15, L / 2 - 0.15), rng.uniform(-W / 2 + 0.15, W / 2 - 0.15), h * 0.75 + rng.uniform(0.05, 0.22)) for _ in range(int(L * W * 14))]
        cards(m, P, [(0, 0, 1)] * len(P), planted, size=0.22, rng=rng, jitter=0.7)
    return m


def toys(seed=3):
    """A child's balance bike and ride-on left on the grass (small, bright shapes)."""
    m = Mesh()
    for (x, col) in ((0.0, "paint_white"), (0.55, "tarp_yellow")):
        m.tube([(x - 0.28, 0, 0.16), (x, 0.02, 0.36), (x + 0.3, 0, 0.16)], 0.02, col, seg=5)
        for dx in (-0.28, 0.3):
            c = Mesh()
            c.cylinder(0.15, 0.04, "plastic_black", seg=12)
            m.merge(c, translate(x + dx, -0.02, 0.15) @ rot_x(90))
    return m


def brick_row(path, mat="brick_red_old", seed=21, proud=0.045, gap=0.012):
    """Bricks laid end to end, half sunk, along a path: the edge of a bed."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    P = np.asarray(path, float)
    seg_len = np.linalg.norm(np.diff(P, axis=0), axis=1)
    total, s = float(seg_len.sum()), 0.0
    cum = np.r_[0.0, np.cumsum(seg_len)]
    while s + 0.215 <= total + 0.05:
        mid = min(s + 0.1075, total - 1e-6)
        k = int(np.searchsorted(cum, mid, side="right") - 1)
        k = min(k, len(seg_len) - 1)
        t = (mid - cum[k]) / seg_len[k]
        c = P[k] + (P[k + 1] - P[k]) * t
        ang = math.degrees(math.atan2(P[k + 1][1] - P[k][1], P[k + 1][0] - P[k][0])) + rng.normal(0, 4.0)
        M = translate(c[0] + rng.normal(0, 0.012), c[1] + rng.normal(0, 0.012), -0.065 + proud + rng.normal(0, 0.008)) @ rot_z(ang)
        m.box((0.215, 0.1025, 0.065), mat, M=M)
        s += 0.215 + gap + abs(rng.normal(0, 0.01))
    return m


def cane_wigwam(n=7, r=0.55, h=2.2, mat="twig", seed=31):
    """Bean canes pushed in round a circle and tied together near the top."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    tie = np.array([0.0, 0.0, h * 0.88])
    for i in range(n):
        a = 2 * math.pi * i / n + rng.normal(0, 0.12)
        foot = np.array([r * math.cos(a) * rng.uniform(0.85, 1.1), r * math.sin(a) * rng.uniform(0.85, 1.1), 0.0])
        top = foot + (tie - foot) * (h / tie[2]) * rng.uniform(0.97, 1.08)      # carries on past the tie
        m.tube([foot, top], [0.012, 0.007], mat, seg=5)
    m.cylinder(0.03, 0.05, "straw", seg=8, M=translate(0, 0, tie[2] - 0.025))
    return m


def leaning_poles(n=5, h=2.0, out=0.5, spread=0.45, mat="twig", seed=41):
    """A bundle of poles stood against a wall. The wall is the local X axis, facing -Y;
    the feet stand `out` from it and the tops rest on it at height h."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    for i in range(n):
        x0 = (i / max(1, n - 1) - 0.5) * spread + rng.normal(0, 0.03)
        foot = np.array([x0, -out * rng.uniform(0.75, 1.15), 0.0])
        top = np.array([x0 * 0.35 + rng.normal(0, 0.06), -0.02, h * rng.uniform(0.82, 1.0)])
        m.tube([foot, top], [0.014, 0.008], mat, seg=5)
    return m


def mushroom(h=0.5, cap_r=0.2, mat="wood_grey"):
    """Carved wooden mushroom, the garden ornament kind."""
    m = Mesh()
    m.lathe([(cap_r * 0.42, 0.0), (cap_r * 0.3, h * 0.55), (cap_r * 0.34, h * 0.6)], mat, seg=10)
    m.lathe([(cap_r, h * 0.56), (cap_r * 0.96, h * 0.68), (cap_r * 0.7, h * 0.88), (cap_r * 0.3, h * 0.99), (0.01, h)], mat, seg=14)
    return m


def tile_scatter(n=70, area=(1.9, 2.1), seed=17, tile=(0.265, 0.165, 0.014), mat="clay_tiles", ridges=2):
    """Old roof tiles dumped on the ground: plain tiles lying flat or propped on one another
    over an oval patch, with a ridge tile or two among them."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    for _ in range(n):
        while True:
            x, y = rng.uniform(-0.5, 0.5, 2)
            if x * x + y * y < 0.25:
                break
        t = Mesh()
        t.box(tile, mat, origin="base", ts=(0.5, 0.5))
        tilt = rng.uniform(0, 8) if rng.random() < 0.75 else rng.uniform(12, 35)
        m.merge(t, translate(x * area[0], y * area[1], rng.uniform(0.0, 0.05)) @ rot_z(rng.uniform(0, 360)) @ rot_x(tilt))
    for k in range(ridges):
        x, y = rng.uniform(-0.3, 0.3) * area[0], rng.uniform(-0.3, 0.3) * area[1]
        a = math.radians(rng.uniform(0, 180))
        L = 0.42
        dx, dy = math.cos(a) * L / 2, math.sin(a) * L / 2
        m.tube([(x - dx, y - dy, 0.02), (x + dx, y + dy, 0.02)], 0.11, "terracotta", seg=10, caps=(False, False))
    return m


def wheelie_bin(h=1.07, w=0.58, d=0.72, body="plastic_black", lid=None):
    """Council wheeled bin. The handle and wheels are at the back (+Y)."""
    m = Mesh()
    lid = lid or body
    bot = [(-w * 0.42, -d * 0.4), (w * 0.42, -d * 0.4), (w * 0.42, d * 0.42), (-w * 0.42, d * 0.42)]
    top = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]
    zb, zt = 0.06, h - 0.07
    V = [(x, y, zb) for x, y in bot] + [(x, y, zt) for x, y in top]
    m.add(V, [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (3, 2, 1, 0)], [[(0, 0), (1, 0), (1, 1), (0, 1)]] * 5, body)
    m.box((w + 0.04, d + 0.05, 0.07), lid, M=translate(0, -0.01, zt))
    m.tube([(-w * 0.4, d / 2 + 0.04, zt + 0.02), (w * 0.4, d / 2 + 0.04, zt + 0.02)], 0.018, body, seg=6)
    for sx in (-1, 1):
        m.cylinder(0.1, 0.05, "plastic_black", seg=12, M=translate(sx * (w * 0.42), d * 0.36, 0.1) @ rot_y(90) @ translate(0, 0, -0.025))
    return m


def round_log_stack(r=0.8, h=1.1, seed=7, end="log_end", bark="bark_willow"):
    """A round wood stack (holzhausen): split logs laid radially, ends outward, heaped in the middle."""
    m = Mesh()
    m.lathe([(r * 1.02, 0.0), (r, h * 0.55), (r * 0.94, h * 0.86)], end, seg=28, cap_bottom=False)
    m.lathe([(r * 0.94, h * 0.86), (r * 0.6, h * 0.96), (r * 0.2, h), (0.01, h)], bark, seg=28, cap_bottom=False)
    return m
