"""Procedural textures for the garden model (numpy + Pillow only).

Everything the model shows is generated from code, so the whole garden can be
rebuilt from scratch with no downloaded assets. Each function writes one image
and returns the real-world size (metres) that one repeat of the image covers,
so the mesh builders can lay out UVs at true scale.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------- noise


def _rng(seed):
    return np.random.default_rng(seed)


def _upsample(grid, shape, tile=True):
    """Bicubic-upsample a small random grid to (h, w); periodic when tile=True."""
    gh, gw = grid.shape
    pad = 2
    g = np.pad(grid, pad, mode="wrap" if tile else "edge").astype(np.float32)
    H = int(round(shape[0] * (gh + 2 * pad) / gh))
    W = int(round(shape[1] * (gw + 2 * pad) / gw))
    a = np.asarray(Image.fromarray(g).resize((W, H), Image.BICUBIC))
    y0 = int(round(pad * shape[0] / gh))
    x0 = int(round(pad * shape[1] / gw))
    return a[y0:y0 + shape[0], x0:x0 + shape[1]]


def fbm(shape, cells=4, octaves=5, seed=0, gain=0.5, tile=True, aspect=(1, 1)):
    """Fractal value noise, roughly 0..1 with mean 0.5. aspect=(cy, cx) stretches it."""
    r = _rng(seed)
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    cy, cx = cells * aspect[0], cells * aspect[1]
    for _ in range(octaves):
        gy, gx = max(1, int(round(cy))), max(1, int(round(cx)))
        if gy >= shape[0] or gx >= shape[1]:
            break
        out += amp * _upsample(r.random((gy, gx)), shape, tile)
        tot += amp
        amp *= gain
        cy *= 2
        cx *= 2
    return out / max(tot, 1e-6)


def _norm(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-6)


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _mix(a, b, t):
    """Blend a -> b by t. a/b may be HxWx3 images or plain RGB triples; t may be HxW."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if np.ndim(t) == 2 and (a.shape[-1:] == (3,) or b.shape[-1:] == (3,)):
        t = t[..., None]
    return a * (1 - t) + b * t


def _rgb(*c):
    return np.array(c, np.float32) / 255.0


def _save(arr, path, quality=88):
    """arr: float HxWx3 (or x4) in 0..1."""
    a = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.lower().endswith((".jpg", ".jpeg")):
        im.convert("RGB").save(path, quality=quality, subsampling=0, optimize=True)
    else:
        im.save(path, optimize=True)
    return path


def normal_from_height(height, strength=2.0, tile=True):
    """Tangent-space normal map (float RGB 0..1) from a height field 0..1."""
    h = height.astype(np.float32)
    if tile:
        dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
        dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    else:
        dy, dx = np.gradient(h)
    nx, ny = -dx * strength * 8, dy * strength * 8  # image y runs down, +V runs up
    nz = np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.dstack([nx / ln, ny / ln, nz / ln]) * 0.5 + 0.5


# --------------------------------------------------------------------------- brick

BRICK_L, BRICK_HDR, BRICK_T, BRICK_J = 0.215, 0.1025, 0.065, 0.010  # UK brick, metres
COURSE = BRICK_T + BRICK_J  # 75 mm per course


def brick(path, px=(1024, 1024), bond="flemish", courses=24, periods=6,
          palette=((186, 170, 132), (170, 156, 122), (198, 184, 146), (150, 138, 110)),
          mortar=(178, 174, 160), seed=1, algae=0.2, damp=0.35, soot=0.25,
          normal_path=None):
    """Brickwork at true UK brick size. Tiles horizontally; the full image height is
    `courses` courses, so top-of-wall algae and damp at the base are painted in.
    Returns (width_m, height_m)."""
    W, H = px
    L, HD, T, J = BRICK_L, BRICK_HDR, BRICK_T, BRICK_J
    P = (L + J + HD + J) if bond == "flemish" else (L + J)
    width_m, height_m = P * periods, COURSE * courses
    r = _rng(seed)

    xs = (np.arange(W) + 0.5) / W * width_m
    ys = (np.arange(H) + 0.5) / H * height_m
    X, Y = np.meshgrid(xs, ys)
    course = np.floor(Y / COURSE).astype(int)
    yin = Y - course * COURSE
    in_y = (yin > J / 2) & (yin < J / 2 + T)
    xo = X + np.where(course % 2 == 0, 0.0, P / 2)
    per = np.floor(xo / P).astype(int)
    xm = xo - per * P
    if bond == "flemish":
        is_str = xm < L
        is_hdr = (xm >= L + J) & (xm < L + J + HD)
        xin = np.where(is_str, xm, xm - (L + J))
        blen = np.where(is_str, L, HD)
        kind = np.where(is_str, 0, 1)
        in_x = is_str | is_hdr
    else:
        in_x = xm < L
        xin, blen, kind = xm, np.full_like(xm, L), np.zeros_like(per)
    is_brick = in_x & in_y

    tab = r.random((courses, periods, 2, 3)).astype(np.float32)
    rnd = tab[course % courses, per % periods, kind]  # HxWx3 per-brick randoms
    pal = np.array(palette, np.float32) / 255.0
    idx = np.minimum((rnd[..., 0] * len(pal)).astype(int), len(pal) - 1)
    col = pal[idx] * (0.84 + 0.30 * rnd[..., 1:2])
    # a few over-burnt dark bricks, as in real stock brickwork
    col = np.where((rnd[..., 2:3] > 0.92), col * 0.62, col)

    # bevel towards the joint and fine surface grain
    d = np.minimum(np.minimum(xin, blen - xin), np.minimum(yin - J / 2, J / 2 + T - yin))
    bevel = _smooth(0.0, 0.006, d)
    grain = fbm((H, W), cells=96, octaves=3, seed=seed + 11)
    col = col * (0.80 + 0.20 * bevel[..., None]) * (0.86 + 0.28 * grain[..., None])

    mort = (np.array(mortar, np.float32) / 255.0) * (
        0.82 + 0.3 * fbm((H, W), cells=64, octaves=3, seed=seed + 5)[..., None])
    img = np.where(is_brick[..., None], col, mort)

    # weathering: broad staining, soot, green algae under the coping, damp at the base
    v = Y / height_m  # 0 top .. 1 bottom
    broad = fbm((H, W), cells=3, octaves=4, seed=seed + 21)
    img *= (0.80 + 0.36 * broad[..., None])
    sootm = _smooth(0.52, 0.8, fbm((H, W), cells=5, octaves=4, seed=seed + 31)) * soot
    img = _mix(img, img * _rgb(120, 112, 104), sootm)
    alg = fbm((H, W), cells=6, octaves=4, seed=seed + 41)
    top = _smooth(0.42, 0.0, v) * 0.9 + _smooth(0.80, 1.0, v) * 0.7
    algm = np.clip(_smooth(0.35, 0.7, alg * 0.6 + top * 0.6) * algae * 1.6, 0, 0.75)
    img = _mix(img, img * _rgb(150, 190, 110) + _rgb(18, 30, 10) * 0.6, algm)
    dampm = _smooth(0.86, 1.0, v + 0.06 * (broad - 0.5)) * damp
    img = _mix(img, img * 0.55, dampm)
    _save(img, path)

    if normal_path:
        hgt = np.where(is_brick, 0.55 + 0.45 * bevel + 0.12 * (grain - 0.5), 0.15)
        _save(normal_from_height(hgt, strength=1.4), normal_path, quality=90)
    return width_m, height_m


# --------------------------------------------------------------------------- roofs


def clay_tiles(path, px=(1024, 1024), cols=10, rows=14, seed=3, moss=0.15,
               palette=((190, 98, 60), (204, 116, 70), (168, 84, 54), (140, 76, 54),
                        (212, 132, 86), (156, 96, 68)),
               normal_path=None):
    """Old hand-made clay peg tiles: 165 mm wide at 100 mm gauge, broken-bonded.
    Tiles in both directions. Returns (width_m, height_m)."""
    W, H = px
    tw, gauge = 0.165, 0.100
    width_m, height_m = cols * tw, rows * gauge
    r = _rng(seed)
    xs = (np.arange(W) + 0.5) / W * width_m
    ys = (np.arange(H) + 0.5) / H * height_m
    X, Y = np.meshgrid(xs, ys)
    row = np.floor(Y / gauge).astype(int)
    yin = Y - row * gauge  # 0 at the top of the exposed part (in shadow) .. gauge at the tail
    xo = X + np.where(row % 2 == 0, 0.0, tw / 2)
    c = np.floor(xo / tw).astype(int)
    xin = xo - c * tw
    tab = r.random((rows, cols, 3)).astype(np.float32)
    rnd = tab[row % rows, c % cols]
    pal = np.array(palette, np.float32) / 255.0
    idx = np.minimum((rnd[..., 0] * len(pal)).astype(int), len(pal) - 1)
    col = pal[idx] * (0.8 + 0.35 * rnd[..., 1:2])
    # uneven tails: each tile hangs slightly differently
    tail = gauge - 0.004 - 0.008 * rnd[..., 2]
    shadow = _smooth(0.0, 0.022, yin)  # under the tile above
    edge = _smooth(0.0, 0.004, np.minimum(xin, tw - xin))
    lip = _smooth(tail, tail - 0.006, yin)
    shade = (0.45 + 0.55 * shadow) * (0.7 + 0.3 * edge) * (0.55 + 0.45 * lip)
    grain = fbm((H, W), cells=80, octaves=3, seed=seed + 7)
    img = col * shade[..., None] * (0.85 + 0.3 * grain[..., None])
    lichen = _smooth(0.63, 0.75, fbm((H, W), cells=40, octaves=3, seed=seed + 9))
    img = _mix(img, _rgb(196, 190, 150) * (0.8 + 0.3 * grain[..., None]), lichen * 0.5)
    m = fbm((H, W), cells=7, octaves=5, seed=seed + 13)
    mossm = _smooth(1.0 - moss - 0.25, 1.0 - moss, m * 0.75 + 0.25 * (1 - shadow))
    mosscol = _mix(_rgb(92, 112, 40), _rgb(150, 160, 58),
                   fbm((H, W), cells=30, octaves=3, seed=seed + 15))
    img = _mix(img, mosscol * (0.75 + 0.5 * grain[..., None]), mossm)
    _save(img, path)
    if normal_path:
        hgt = (yin / gauge) * 0.7 * lip + 0.2 * edge + 0.5 * mossm + 0.1 * grain
        _save(normal_from_height(_norm(hgt), strength=1.6), normal_path, quality=90)
    return width_m, height_m


def mossy_roof(path, px=(768, 768), seed=5, normal_path=None):
    """The shed roof: old tiles almost buried under moss."""
    return clay_tiles(path, px=px, cols=8, rows=12, seed=seed, moss=0.72,
                      palette=((120, 84, 56), (104, 76, 54), (136, 96, 60), (96, 70, 50)),
                      normal_path=normal_path)


# --------------------------------------------------------------------------- timber


def boards(path, px=(768, 768), board_w=0.15, n=6, length_m=1.8, vertical=True,
           base=(58, 40, 30), seed=7, lap=False, gap=0.004, contrast=0.35,
           weather=0.2, normal_path=None):
    """Timber cladding / planking. vertical=True: boards run up the image.
    lap=True draws feather-edge/weatherboard shadow lines instead of open gaps.
    Returns (width_m, height_m) of one repeat."""
    W, H = px
    r = _rng(seed)
    across_m = board_w * n
    a = (np.arange(W if vertical else H) + 0.5) / (W if vertical else H) * across_m
    b = (np.arange(H if vertical else W) + 0.5) / (H if vertical else W) * length_m
    A, B = np.meshgrid(a, b) if vertical else np.meshgrid(a, b, indexing="ij")
    k = np.floor(A / board_w).astype(int)
    ain = A - k * board_w
    tab = r.random((n, 3)).astype(np.float32)
    rnd = tab[k % n]
    shape = (H, W)
    asp = (1, 14) if not vertical else (14, 1)
    # grain runs along the board; offset per board so neighbours differ
    g1 = fbm(shape, cells=3, octaves=5, seed=seed + 1, aspect=asp[::-1])
    g2 = fbm(shape, cells=22, octaves=3, seed=seed + 2, aspect=asp[::-1])
    grain = _norm(0.6 * g1 + 0.4 * g2)
    col = (np.array(base, np.float32) / 255.0) * (0.78 + 0.44 * rnd[..., 0:1])
    img = col * (1 - contrast / 2 + contrast * grain[..., None])
    if lap:
        # each board overlaps the one below: dark line under the lip, lighter top face
        shade = 0.55 + 0.45 * _smooth(0.0, 0.03, ain) if vertical else \
            0.5 + 0.5 * _smooth(0.0, 0.028, ain)
        img *= shade[..., None]
        hgt = ain / board_w
    else:
        e = np.minimum(ain, board_w - ain)
        gapm = _smooth(gap, gap + 0.004, e)
        img *= (0.18 + 0.82 * gapm)[..., None]
        hgt = gapm * (0.8 + 0.2 * grain)
    wm = _smooth(0.45, 0.8, fbm(shape, cells=5, octaves=4, seed=seed + 3)) * weather
    img = _mix(img, img * 0.6 + _rgb(120, 122, 112) * 0.4, wm)
    _save(img, path)
    if normal_path:
        _save(normal_from_height(_norm(hgt), strength=1.2), normal_path, quality=90)
    return (across_m, length_m) if vertical else (length_m, across_m)


def wood(path, px=(512, 512), base=(150, 146, 136), seed=9, contrast=0.4, knots=3,
         size_m=(0.4, 0.4), tint=None):
    """Plain sawn wood; grain runs along image X. Used for furniture slats."""
    W, H = px
    g1 = fbm((H, W), cells=2, octaves=5, seed=seed, aspect=(10, 1))
    g2 = fbm((H, W), cells=20, octaves=3, seed=seed + 1, aspect=(8, 1))
    grain = _norm(0.55 * g1 + 0.45 * g2)
    img = (np.array(base, np.float32) / 255.0) * (1 - contrast / 2 + contrast * grain[..., None])
    r = _rng(seed + 2)
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(knots):
        cx, cy, rad = r.integers(0, W), r.integers(0, H), r.uniform(5, 12)
        dd = np.sqrt(((xx - cx) / 1.8) ** 2 + (yy - cy) ** 2)
        img *= (1 - 0.45 * _smooth(rad, 0, dd))[..., None]
    if tint is not None:
        m = fbm((H, W), cells=4, octaves=4, seed=seed + 3)
        img = _mix(img, img * (np.array(tint, np.float32) / 255.0) * 1.6, _smooth(0.4, 0.75, m) * 0.6)
    _save(img, path)
    return size_m


# --------------------------------------------------------------------------- masonry


def render(path, px=(768, 768), base=(232, 232, 226), seed=11, size_m=(2.4, 2.4),
           stain=0.35, normal_path=None):
    """White roughcast / painted render with grime rising from the ground."""
    W, H = px
    speck = fbm((H, W), cells=120, octaves=2, seed=seed)
    broad = fbm((H, W), cells=4, octaves=4, seed=seed + 1)
    img = (np.array(base, np.float32) / 255.0) * (0.90 + 0.14 * speck[..., None]) \
        * (0.93 + 0.12 * broad[..., None])
    v = (np.arange(H) + 0.5)[:, None] / H
    streak = fbm((H, W), cells=28, octaves=3, seed=seed + 2, aspect=(0.15, 1))
    grime = np.clip(_smooth(0.62, 1.0, v + 0.25 * (streak - 0.5)) * stain
                    + _smooth(0.10, 0.0, v) * stain * 0.5 * streak, 0, 1)
    img = _mix(img, img * _rgb(150, 160, 120), grime)
    _save(img, path)
    if normal_path:
        _save(normal_from_height(speck, strength=0.9), normal_path, quality=90)
    return size_m


def concrete(path, px=(768, 768), base=(150, 148, 142), seed=13, size_m=(1.8, 1.8)):
    """Weathered concrete paving: speckle, lichen blooms, darker damp patches."""
    W, H = px
    speck = fbm((H, W), cells=140, octaves=2, seed=seed)
    broad = fbm((H, W), cells=3, octaves=5, seed=seed + 1)
    img = (np.array(base, np.float32) / 255.0) * (0.80 + 0.3 * speck[..., None]) \
        * (0.78 + 0.42 * broad[..., None])
    lich = _smooth(0.66, 0.74, fbm((H, W), cells=26, octaves=3, seed=seed + 2))
    img = _mix(img, _rgb(206, 204, 188), lich * 0.55)
    mossm = _smooth(0.68, 0.8, fbm((H, W), cells=9, octaves=4, seed=seed + 3))
    img = _mix(img, img * _rgb(140, 160, 110), mossm * 0.6)
    _save(img, path)
    return size_m


def flat(path, colour, px=(64, 64), seed=0, noise=0.08):
    W, H = px
    n = fbm((H, W), cells=8, octaves=3, seed=seed)
    img = (np.array(colour, np.float32) / 255.0) * (1 - noise / 2 + noise * n[..., None])
    _save(img, path)
    return (1.0, 1.0)


# --------------------------------------------------------------------------- bark


def bark(path, px=(512, 1024), kind="willow", seed=15, size_m=(1.0, 2.0), normal_path=None):
    """Bark that wraps a trunk: tiles around (X) and up (Y)."""
    W, H = px
    if kind == "willow":  # deep interlacing vertical ridges, grey-brown
        ridge = fbm((H, W), cells=7, octaves=4, seed=seed, aspect=(0.22, 1))
        warp = fbm((H, W), cells=3, octaves=3, seed=seed + 1)
        r2 = np.abs(np.sin((ridge * 5.0 + warp * 2.2) * np.pi))
        hgt = _norm(r2 ** 0.7 + 0.3 * fbm((H, W), cells=40, octaves=3, seed=seed + 2))
        dark, light = _rgb(46, 40, 34), _rgb(140, 126, 106)
        img = _mix(np.broadcast_to(dark, (H, W, 3)), np.broadcast_to(light, (H, W, 3)), hgt)
        mossm = _smooth(0.62, 0.8, fbm((H, W), cells=5, octaves=4, seed=seed + 3))
        img = _mix(img, img * _rgb(150, 175, 105), mossm * 0.55)
    elif kind == "pale":  # smooth silvery-grey trunk with lichen and algae bloom
        base = fbm((H, W), cells=5, octaves=5, seed=seed)
        fine = fbm((H, W), cells=70, octaves=2, seed=seed + 1)
        hgt = _norm(0.6 * base + 0.4 * fine)
        img = _mix(np.broadcast_to(_rgb(150, 150, 142), (H, W, 3)),
                   np.broadcast_to(_rgb(226, 226, 216), (H, W, 3)), hgt)
        crack = _smooth(0.04, 0.0, np.abs(fbm((H, W), cells=9, octaves=3, seed=seed + 2,
                                              aspect=(0.3, 1)) - 0.5))
        img = _mix(img, img * 0.55, crack * 0.6)
        alg = _smooth(0.55, 0.8, fbm((H, W), cells=4, octaves=4, seed=seed + 3))
        img = _mix(img, img * _rgb(190, 205, 160), alg * 0.5)
    else:  # smooth young bark with pale lenticels (hazel, apple, shrubs)
        base = fbm((H, W), cells=6, octaves=4, seed=seed, aspect=(0.5, 1))
        fine = fbm((H, W), cells=60, octaves=2, seed=seed + 1)
        hgt = _norm(0.6 * base + 0.4 * fine)
        c0, c1 = (_rgb(74, 62, 50), _rgb(150, 132, 108)) if kind == "hazel" else \
            (_rgb(96, 88, 76), _rgb(196, 186, 164))
        img = _mix(np.broadcast_to(c0, (H, W, 3)), np.broadcast_to(c1, (H, W, 3)), hgt)
        lent = _smooth(0.70, 0.76, fbm((H, W), cells=34, octaves=2, seed=seed + 2, aspect=(3, 0.6)))
        img = _mix(img, np.broadcast_to(_rgb(210, 200, 176), (H, W, 3)), lent * 0.45)
    _save(img, path)
    if normal_path:
        _save(normal_from_height(hgt, strength=2.2 if kind == "willow" else 1.0),
              normal_path, quality=90)
    return size_m


def log_end(path, px=(256, 256), seed=17):
    """Sawn end of a log: pale heartwood rings inside a bark rim."""
    W, H = px
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2 + 6, H / 2 - 4
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (W / 2)
    n = fbm((H, W), cells=6, octaves=3, seed=seed)
    rings = 0.5 + 0.5 * np.sin((d + 0.06 * n) * 46)
    img = _mix(np.broadcast_to(_rgb(168, 138, 100), (H, W, 3)),
               np.broadcast_to(_rgb(128, 98, 68), (H, W, 3)), rings * 0.6)
    img = img * (0.8 + 0.3 * fbm((H, W), cells=50, octaves=2, seed=seed + 1)[..., None])
    _save(img, path)
    return (1.0, 1.0)


# --------------------------------------------------------------------------- foliage


def _leaf_poly(kind):
    if kind == "ivy":
        half = [(0, -1.0), (0.22, -0.45), (0.78, -0.55), (0.5, -0.08), (0.95, 0.32),
                (0.38, 0.3), (0.2, 0.62), (0, 0.45)]
    elif kind == "round":  # eucalyptus juvenile / hazel
        t = np.linspace(-np.pi / 2, np.pi / 2, 12)
        half = [(0.72 * np.cos(a), 0.85 * np.sin(a) - 0.05) for a in t]
    elif kind == "narrow":  # willow, sage, olive
        t = np.linspace(-np.pi / 2, np.pi / 2, 10)
        half = [(0.22 * np.cos(a) ** 0.8, np.sin(a)) for a in t]
    else:  # ovate: beech, laurel, most shrubs
        t = np.linspace(-np.pi / 2, np.pi / 2, 12)
        half = [(0.46 * np.cos(a) ** 0.75 * (1 + 0.25 * np.sin(a)), np.sin(a)) for a in t]
    pts = list(half) + [(-x, y) for x, y in reversed(half) if abs(x) > 1e-6]
    return np.array(pts, np.float32)


def leaf_card(path, kind="ovate", colours=((60, 100, 40), (80, 120, 50), (44, 82, 34)),
              n=16, size=256, leaf_scale=0.24, seed=21, vein=0.75, spread=0.34):
    """A square RGBA card holding a spray of leaves on a transparent ground.
    Colour is bled into the transparent pixels so alpha-tested edges stay clean."""
    S = size * 4
    r = _rng(seed)
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    poly = _leaf_poly(kind)
    cols = [tuple(int(v) for v in c) for c in colours]
    for i in range(n):
        ang = r.uniform(0, 2 * np.pi)
        rad = spread * S * np.sqrt(r.uniform(0.0, 1.0))
        cx, cy = S / 2 + rad * np.cos(ang), S / 2 + rad * np.sin(ang)
        rot = ang + np.pi / 2 + r.uniform(-0.9, 0.9)  # leaves point outwards-ish
        sc = leaf_scale * S * 0.5 * r.uniform(0.7, 1.15)
        ca, sa = np.cos(rot), np.sin(rot)
        pts = [(cx + sc * (x * ca - y * sa), cy + sc * (x * sa + y * ca)) for x, y in poly]
        c = cols[int(r.integers(0, len(cols)))]
        k = r.uniform(0.82, 1.15)
        c = tuple(int(min(255, v * k)) for v in c)
        dr.polygon(pts, fill=c + (255,))
        tip = (cx + sc * (0 * ca - (-0.92) * sa), cy + sc * (0 * sa + (-0.92) * ca))
        base = (cx + sc * (0 * ca - 0.5 * sa), cy + sc * (0 * sa + 0.5 * ca))
        vc = tuple(int(min(255, v * vein + 255 * (1 - vein) * 0.5)) for v in c)
        dr.line([base, tip], fill=vc + (255,), width=max(2, S // 170))
    im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    alpha = a[..., 3]
    # bleed: fill transparent texels with blurred leaf colour (no dark fringes when mipmapped)
    rgb = a[..., :3] * alpha[..., None]
    blur_rgb = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(8))).astype(np.float32) / 255.0
    blur_a = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(8))).astype(np.float32) / 255.0
    mean = np.array(cols, np.float32).mean(0) / 255.0
    bleed = np.where(blur_a[..., None] > 0.02, blur_rgb / np.maximum(blur_a[..., None], 0.02), mean)
    out_rgb = np.where(alpha[..., None] > 0.5, a[..., :3], bleed)
    _save(np.dstack([out_rgb, alpha]), path)
    return (1.0, 1.0)


def twig_card(path, size=256, colour=(72, 54, 42), seed=41, depth=6, leaf=None,
              leaf_colours=((206, 180, 62), (188, 160, 54), (170, 150, 60)), leaf_chance=0.5,
              spread=0.55, droop=0.0):
    """A spray of bare twigs growing up from the bottom-centre of a transparent card.
    leaf: None, or a leaf kind to hang a few late leaves on the twig ends."""
    S = size * 4
    r = _rng(seed)
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(im)
    col = tuple(int(v) for v in colour) + (255,)
    poly = _leaf_poly(leaf) if leaf else None
    tips = []

    def grow(x, y, ang, length, width, d):
        x2, y2 = x + length * np.sin(ang), y - length * np.cos(ang)
        dr.line([(x, y), (x2, y2)], fill=col, width=max(4, int(width)))
        if d == 0 or length < S * 0.03 or not (0.02 * S < x2 < 0.98 * S and 0.02 * S < y2 < S):
            tips.append((x2, y2, ang))
            return
        n = 2 if r.random() < 0.7 else 3
        for _ in range(n):
            grow(x2, y2, ang + r.uniform(-spread, spread) + droop * np.sign(ang or r.normal()),
                 length * r.uniform(0.62, 0.9), width * 0.7, d - 1)

    for a0 in (-0.35, 0.0, 0.35):
        grow(S / 2, S * 0.995, a0 + r.uniform(-0.1, 0.1), S * 0.2, S * 0.02, depth)
    if poly is not None:
        cols = [tuple(int(v) for v in c) for c in leaf_colours]
        for (x, y, ang) in tips:
            if r.random() > leaf_chance:
                continue
            rot = ang + r.uniform(-1.2, 1.2)
            sc = S * 0.035 * r.uniform(0.7, 1.2)
            ca, sa = np.cos(rot), np.sin(rot)
            pts = [(x + sc * (px * ca - py * sa), y + sc * (px * sa + py * ca)) for px, py in poly]
            dr.polygon(pts, fill=cols[int(r.integers(0, len(cols)))] + (255,))
    im = im.resize((size, size), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    alpha = np.clip(a[..., 3] * 1.6, 0, 1)  # keep thin twigs above the alpha cutoff
    mean = np.array(colour, np.float32) / 255.0
    rgb = np.where(a[..., 3:4] > 0.3, a[..., :3], mean)
    _save(np.dstack([rgb, alpha]), path)
    return (1.0, 1.0)


def foliage_solid(path, px=(256, 256), colours=((52, 86, 36), (86, 120, 52), (34, 60, 28)), seed=23):
    """Opaque mottled leaf-mass texture for the inner 'body' of hedges and shrubs."""
    W, H = px
    a = fbm((H, W), cells=10, octaves=4, seed=seed)
    b = fbm((H, W), cells=34, octaves=3, seed=seed + 1)
    c0, c1, c2 = (np.array(c, np.float32) / 255.0 for c in colours)
    img = _mix(np.broadcast_to(c0, (H, W, 3)), np.broadcast_to(c1, (H, W, 3)), _smooth(0.35, 0.7, a))
    img = _mix(img, np.broadcast_to(c2, (H, W, 3)), _smooth(0.5, 0.8, b) * 0.7)
    _save(img, path)
    return (1.0, 1.0)


# --------------------------------------------------------------------------- printed things


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/opentype/inter/Inter-Bold.otf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


def firewood_bag(path, px=(512, 768), seed=25):
    """Woven white polypropylene sack with the dark printed panel of a kiln-dried
    firewood bag. Wraps once around the bag (U) and bottom-to-top (V)."""
    W, H = px
    yy, xx = np.mgrid[0:H, 0:W]
    weave = 0.5 + 0.25 * np.sin(xx * 1.3) * np.sin(yy * 1.3)
    creases = fbm((H, W), cells=6, octaves=4, seed=seed)
    img = np.broadcast_to(_rgb(226, 228, 228), (H, W, 3)) * (0.86 + 0.1 * weave[..., None]) \
        * (0.80 + 0.36 * creases[..., None])
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for x0 in (int(W * 0.06), int(W * 0.56)):  # printed panel on the front and back faces
        x1 = x0 + int(W * 0.38)
        dr.rectangle([x0, int(H * 0.30), x1, int(H * 0.86)], fill=(46, 48, 50))
        dr.rectangle([x0, int(H * 0.22), x1, int(H * 0.285)], fill=(60, 62, 64))
        f1, f2, f3 = _font(int(W * 0.050)), _font(int(W * 0.024)), _font(int(W * 0.019))
        cx = (x0 + x1) // 2
        dr.text((cx, int(H * 0.252)), "EXTRA LARGE BAG", font=f2, fill=(236, 236, 232), anchor="mm")
        dr.text((cx, int(H * 0.44)), "KILN DRIED", font=f1, fill=(240, 240, 236), anchor="mm")
        dr.text((cx, int(H * 0.50)), "FIREWOOD", font=f1, fill=(240, 240, 236), anchor="mm")
        dr.rectangle([x0 + 8, int(H * 0.555), x1 - 8, int(H * 0.60)], fill=(232, 232, 228))
        dr.text((cx, int(H * 0.578)), "RESPONSIBLY SOURCED", font=f3, fill=(40, 40, 40), anchor="mm")
        dr.ellipse([x0 + 14, int(H * 0.72), x0 + 14 + int(W * 0.09), int(H * 0.72) + int(W * 0.09)],
                   outline=(236, 236, 232), width=3)
    a = np.asarray(im).astype(np.float32) / 255.0
    a = a * (0.86 + 0.28 * creases[..., None])
    _save(a, path)
    return (1.0, 1.0)


# --------------------------------------------------------------------------- registry


def build_all(out_dir):
    """Generate every texture; returns {name: {"path","size","normal"}}."""
    T = {}

    def reg(name, fn, ext="jpg", normal=False, **kw):
        path = os.path.join(out_dir, f"{name}.{ext}")
        npath = os.path.join(out_dir, f"{name}_n.jpg") if normal else None
        size = fn(path, normal_path=npath, **kw) if normal else fn(path, **kw)
        if npath:
            im = Image.open(npath)
            if max(im.size) > 512:
                im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(npath, quality=82, optimize=True)
        T[name] = {"path": path, "size": size, "normal": npath}

    reg("brick_stock", brick, normal=True, seed=1)
    reg("brick_red_old", brick, normal=True, seed=2,
        palette=((176, 92, 62), (190, 108, 72), (160, 80, 56), (200, 126, 86), (146, 78, 60)),
        mortar=(196, 186, 164), algae=0.12, damp=0.25, soot=0.15)
    reg("brick_dark", brick, normal=True, seed=3, bond="stretcher", periods=9,
        palette=((112, 62, 50), (98, 56, 46), (124, 72, 56), (88, 54, 46)),
        mortar=(120, 112, 102), algae=0.2, damp=0.4, soot=0.3)
    reg("clay_tiles", clay_tiles, normal=True)
    reg("mossy_roof", mossy_roof, normal=True)
    reg("tiles_grey", clay_tiles, normal=True, px=(512, 512), cols=6, rows=6, seed=6, moss=0.08,
        palette=((112, 100, 90), (126, 112, 98), (98, 90, 84), (136, 120, 104)))
    reg("boards_dark", boards, normal=True, base=(52, 36, 28), seed=7)
    reg("weatherboard_dark", boards, normal=True, base=(50, 36, 30), seed=8, vertical=False,
        lap=True, board_w=0.14, n=8, length_m=2.4)
    reg("fence_boards", boards, normal=True, base=(112, 84, 60), seed=10, board_w=0.1, n=8,
        gap=0.006, weather=0.5)
    reg("wood_grey", wood, base=(148, 146, 138), seed=9, tint=(120, 140, 100))
    reg("wood_new", wood, base=(214, 180, 124), seed=12, contrast=0.22, knots=4)
    reg("wood_brown", wood, base=(118, 84, 56), seed=14, contrast=0.35)
    reg("render_white", render, normal=True)
    reg("concrete", concrete)
    reg("bark_willow", bark, normal=True, kind="willow")
    reg("bark_pale", bark, normal=True, kind="pale", seed=16)
    reg("bark_hazel", bark, kind="hazel", seed=18, px=(256, 512))
    reg("bark_grey", bark, kind="grey", seed=19, px=(256, 512))
    reg("log_end", log_end)
    reg("leaf_ivy", leaf_card, ext="png", kind="ivy", n=18, leaf_scale=0.3,
        colours=((46, 84, 42), (60, 102, 48), (36, 66, 36), (76, 114, 56)))
    reg("leaf_green", leaf_card, ext="png", kind="ovate", seed=22, n=22,
        colours=((78, 112, 56), (102, 136, 64), (62, 96, 50), (118, 146, 78)))
    reg("leaf_lime", leaf_card, ext="png", kind="ovate", seed=24,
        colours=((150, 170, 62), (176, 186, 70), (118, 146, 56)))
    reg("leaf_yellow", leaf_card, ext="png", kind="round", seed=26, n=9, leaf_scale=0.2,
        colours=((206, 180, 62), (188, 160, 54), (160, 150, 60), (214, 196, 96)))
    reg("leaf_beech", leaf_card, ext="png", kind="ovate", seed=28, n=20,
        colours=((128, 74, 44), (150, 92, 52), (108, 64, 42), (136, 84, 60)))
    reg("leaf_sage", leaf_card, ext="png", kind="narrow", seed=30, n=20, leaf_scale=0.34,
        colours=((128, 148, 112), (146, 164, 126), (108, 132, 98)))
    reg("leaf_silver", leaf_card, ext="png", kind="narrow", seed=32, n=22, leaf_scale=0.26,
        colours=((120, 136, 112), (150, 162, 140), (96, 116, 94)))
    reg("leaf_euc", leaf_card, ext="png", kind="round", seed=34, n=12, leaf_scale=0.26,
        colours=((132, 160, 148), (150, 176, 164), (112, 144, 132)))
    reg("leaf_dry", leaf_card, ext="png", kind="narrow", seed=36, n=26, leaf_scale=0.3,
        colours=((120, 74, 46), (144, 90, 54), (98, 62, 42), (160, 108, 66)))
    reg("twigs_brown", twig_card, ext="png", colour=(78, 58, 44), seed=41)
    reg("twigs_grey", twig_card, ext="png", colour=(120, 110, 98), seed=43)
    reg("twigs_hazel", twig_card, ext="png", colour=(84, 66, 50), seed=45, leaf="round",
        leaf_chance=0.45)
    reg("twigs_dry", twig_card, ext="png", colour=(126, 76, 48), seed=47, leaf="narrow",
        leaf_colours=((132, 80, 48), (150, 96, 56), (110, 66, 44)), leaf_chance=0.9, depth=5)
    reg("twigs_stems", twig_card, ext="png", colour=(104, 88, 62), seed=49, depth=3, spread=0.25)
    reg("foliage_ivy", foliage_solid, colours=((36, 62, 34), (50, 82, 42), (26, 46, 28)))
    reg("foliage_green", foliage_solid, seed=27, colours=((58, 86, 44), (80, 110, 54), (44, 68, 38)))
    reg("foliage_beech", foliage_solid, seed=29, colours=((92, 56, 38), (120, 74, 46), (70, 46, 34)))
    reg("firewood_bag", firewood_bag)
    return T


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "textures"
    for k, v in build_all(out).items():
        print(k, v["size"], os.path.getsize(v["path"]) // 1024, "KB")
