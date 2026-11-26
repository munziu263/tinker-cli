"""Trees, shrubs, hedges, ivy and other planting.

Woody plants are grown as tapering tube skeletons with sprays of fine twigs (alpha
cards) on the branch ends, which is what a garden looks like in late November.
Evergreens are a lumpy solid body dressed with leaf cards.
All functions return a core.Mesh with its origin at the base of the plant.
"""
from __future__ import annotations

import math

import numpy as np

from .core import Mesh, frame_from_axis, rot_z, tessellate, translate
from .structures import path_length, point_along

UP = np.array([0.0, 0.0, 1.0])


def _unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _rotate(v, axis, ang):
    axis = _unit(axis)
    return (v * math.cos(ang) + np.cross(axis, v) * math.sin(ang)
            + axis * np.dot(axis, v) * (1 - math.cos(ang)))


def _perp(v):
    a = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    return _unit(np.cross(v, a))


def _noise3(seed, octaves=3):
    """Cheap smooth 3-D noise: a sum of randomly oriented sine waves, about -1..1."""
    r = np.random.default_rng(seed)
    waves = [(r.normal(size=3) * (1.1 * 1.9 ** o), r.uniform(0, 6.28), 0.55 ** o)
             for o in range(octaves) for _ in range(3)]
    tot = sum(w[2] for w in waves)

    def f(p):
        p = np.asarray(p, float)
        return sum(a * np.sin(p @ k + ph) for k, ph, a in waves) / tot
    return f


# --------------------------------------------------------------------------- cards


def cards(m: Mesh, points, normals, mat, size=0.3, rng=None, jitter=0.5, lift=0.0,
          size_var=0.35, upright=0.0):
    """Scatter two-sided leaf quads: one per (point, normal), facing roughly along the
    normal with random roll and tilt. upright>0 biases the card plane to stand vertical."""
    rng = rng or np.random.default_rng(0)
    verts, faces, uvs = [], [], []
    for p, n in zip(points, normals):
        n = _unit(np.asarray(n, float) + rng.normal(size=3) * jitter)
        if upright:
            n = _unit(n * (1, 1, 1 - upright))
        t = _rotate(_perp(n), n, rng.uniform(0, 2 * math.pi))
        b = np.cross(n, t)
        s = size * (1 + rng.uniform(-size_var, size_var)) / 2
        c = np.asarray(p, float) + n * lift
        k = len(verts)
        verts += [c - t * s - b * s, c + t * s - b * s, c + t * s + b * s, c - t * s + b * s]
        faces.append((k, k + 1, k + 2, k + 3))
        uvs.append([(0, 0), (1, 0), (1, 1), (0, 1)])
    if verts:
        m.add(verts, faces, uvs, mat)
    return m


def spray(m: Mesh, base, direction, mat, size=0.8, rng=None, crossed=2, width=1.0):
    """Twig spray: `crossed` upright cards sharing an axis along `direction`,
    the texture growing from its bottom-centre at `base`."""
    rng = rng or np.random.default_rng(0)
    d = _unit(direction)
    side0 = _rotate(_perp(d), d, rng.uniform(0, math.pi))
    base = np.asarray(base, float)
    for i in range(crossed):
        side = _rotate(side0, d, math.pi * i / crossed)
        w = size * width / 2
        a, b = base - side * w, base + side * w
        m.add([a, b, b + d * size, a + d * size], [(0, 1, 2, 3)], [[(0, 0), (1, 0), (1, 1), (0, 1)]], mat)
    return m


# --------------------------------------------------------------------------- skeletons


class Skeleton:
    """Grows curved tapering branches into a Mesh and remembers the branch tips."""

    def __init__(self, bark, seed=0, sides=6):
        self.m = Mesh()
        self.bark = bark
        self.rng = np.random.default_rng(seed)
        self.sides = sides
        self.tips = []  # (point, direction, radius)

    def branch(self, p0, d0, length, r0, r1, bend=0.12, lift=0.0, n=None, sides=None,
               cap=False, v0=0.0):
        """One branch from p0 heading d0. bend: random wander per segment; lift: pull
        towards vertical (+) or droop (-). Returns (points, directions, radii)."""
        n = n or max(2, int(round(length / 0.4)))
        d = _unit(d0)
        pts, dirs = [np.asarray(p0, float)], [d]
        for _ in range(n):
            d = _unit(d + self.rng.normal(size=3) * bend + UP * lift)
            pts.append(pts[-1] + d * (length / n))
            dirs.append(d)
        radii = np.linspace(r0, r1, n + 1)
        s = sides or (self.sides if r0 > 0.03 else max(4, self.sides - 2))
        self.m.tube(pts, radii, self.bark, seg=s, caps=(False, cap), v0=v0)
        return pts, dirs, radii

    def grow(self, p0, d0, length, r0, levels, angle=38, len_ratio=0.62, rad_ratio=0.6,
             children=(3, 3, 2), start=0.35, bend=0.12, lift=0.03, taper=0.45, min_r=0.006):
        """Recursive branching. levels: how many generations still to spawn."""
        pts, dirs, radii = self.branch(p0, d0, length, r0, max(min_r, r0 * taper), bend, lift)
        if levels <= 0 or r0 * rad_ratio < min_r:
            self.tips.append((pts[-1], dirs[-1], radii[-1]))
            return
        k = children[0] if len(children) else 2
        az0 = self.rng.uniform(0, 2 * math.pi)
        for j in range(k):
            t = start + (1 - start) * (j + self.rng.uniform(0.2, 0.9)) / k
            i = min(len(pts) - 1, max(1, int(round(t * (len(pts) - 1)))))
            ang = math.radians(angle * self.rng.uniform(0.7, 1.3))
            side = _rotate(_perp(dirs[i]), dirs[i], az0 + j * 2.4)  # golden-angle spiral
            cd = _rotate(dirs[i], side, ang)
            self.grow(pts[i], cd, length * len_ratio * self.rng.uniform(0.75, 1.15),
                      radii[i] * rad_ratio, levels - 1, angle, len_ratio, rad_ratio,
                      children[1:], start, bend, lift, taper, min_r)
        self.tips.append((pts[-1], dirs[-1], radii[-1]))

    def dress(self, mat, size=0.8, every=1, crossed=2, lean=0.25):
        """Put a twig spray on the recorded tips."""
        for i, (p, d, _r) in enumerate(self.tips):
            if i % every:
                continue
            spray(self.m, p - d * size * 0.12, _unit(d + UP * lean), mat,
                  size=size * self.rng.uniform(0.75, 1.25), rng=self.rng, crossed=crossed)
        return self.m


# --------------------------------------------------------------------------- trees


def hazel(height=5.0, stems=14, spread=22, stool=0.35, seed=1, twigs="twigs_hazel",
          bark="bark_hazel"):
    """Coppiced hazel: a fan of slender stems leaning out from one stool."""
    sk = Skeleton(bark, seed, sides=6)
    r = sk.rng
    for i in range(stems):
        az = 2 * math.pi * i / stems + r.uniform(-0.25, 0.25)
        lean = math.radians(spread * r.uniform(0.25, 1.25))
        d = np.array([math.sin(lean) * math.cos(az), math.sin(lean) * math.sin(az), math.cos(lean)])
        rad = stool * math.sqrt(r.uniform(0.05, 1.0))
        p0 = np.array([rad * math.cos(az), rad * math.sin(az), 0.0])
        L = height * r.uniform(0.7, 1.08)
        sk.grow(p0, d, L, r.uniform(0.022, 0.05), levels=2, angle=30, len_ratio=0.42,
                rad_ratio=0.55, children=(3, 2), start=0.45, bend=0.05, lift=0.035, taper=0.3)
    return sk.dress(twigs, size=1.15, crossed=2)


def multistem_shrub(height=4.0, stems=9, spread=32, stool=0.45, seed=2, twigs="twigs_grey",
                    bark="bark_grey", stem_r=(0.035, 0.07), arch=-0.03, twig_size=1.2,
                    lean=(0.0, 0.0)):
    """Big old deciduous shrub (the one behind the bench): stout pale stems rising
    from the base and arching over into a wide dome of twigs."""
    sk = Skeleton(bark, seed, sides=7)
    r = sk.rng
    for i in range(stems):
        az = 2 * math.pi * i / stems + r.uniform(-0.3, 0.3)
        lean_a = math.radians(spread * r.uniform(0.3, 1.2))
        d = np.array([math.sin(lean_a) * math.cos(az), math.sin(lean_a) * math.sin(az), math.cos(lean_a)])
        d = _unit(d + np.array([lean[0], lean[1], 0.0]))   # the whole shrub reaches one way
        rad = stool * math.sqrt(r.uniform(0.05, 1.0))
        p0 = np.array([rad * math.cos(az), rad * math.sin(az), 0.0])
        sk.grow(p0, d, height * r.uniform(0.7, 1.05), r.uniform(*stem_r), levels=3, angle=40,
                len_ratio=0.5, rad_ratio=0.55, children=(3, 3, 2), start=0.4, bend=0.08,
                lift=arch, taper=0.35)
    return sk.dress(twigs, size=twig_size, crossed=2)


def broadleaf(height=8.0, trunk_r=0.25, trunk_h=2.2, limbs=4, seed=3, bark="bark_grey",
              twigs="twigs_brown", lean=(0.0, 0.0), limb_len=None, limb_angle=40,
              levels=3, twig_size=1.4, flare=1.35, trunk_sides=12, twist=0.0, pollard=None,
              sides=7, children=(3, 3, 2)):
    """General deciduous tree: flared trunk, a whorl of scaffold limbs, bare crown."""
    sk = Skeleton(bark, seed, sides=sides)
    r = sk.rng
    top = np.array([lean[0] * trunk_h, lean[1] * trunk_h, trunk_h])
    n = 6
    pts = []
    for i in range(n + 1):
        t = i / n
        wob = np.array([math.sin(t * 5.0 + 1.0), math.cos(t * 4.0), 0.0]) * twist * (t * (1 - t)) * 4
        pts.append(top * t + wob)
    radii = [trunk_r * (flare - (flare - 1) * min(1, t * 4)) * (1 - 0.18 * t)
             for t in np.linspace(0, 1, n + 1)]
    sk.m.tube(pts, radii, bark, seg=trunk_sides, caps=(False, False))
    limb_len = limb_len or (height - trunk_h) * 0.75
    if pollard:  # heavy limbs carried a short way, then a crown of younger growth
        limb_len = pollard
    az0 = r.uniform(0, 2 * math.pi)
    for i in range(limbs):
        az = az0 + 2 * math.pi * i / limbs + r.uniform(-0.3, 0.3)
        ang = math.radians(limb_angle * r.uniform(0.6, 1.25))
        d = np.array([math.sin(ang) * math.cos(az), math.sin(ang) * math.sin(az), math.cos(ang)])
        sk.grow(top - UP * r.uniform(0, 0.25 * trunk_h * 0.3), d, limb_len * r.uniform(0.8, 1.15),
                trunk_r * 0.62 * r.uniform(0.8, 1.0), levels=levels, angle=36, len_ratio=0.62,
                rad_ratio=0.55, children=children, start=0.35, bend=0.09, lift=0.05, taper=0.4)
    return sk.dress(twigs, size=twig_size, crossed=2)


def willow_pollard(trunk_r=0.30, trunk_h=3.4, seed=4, lean=(0.05, 0.02), stub_len=1.3,
                   leader_h=6.2, bark="bark_willow", ring=True):
    """The big willow as it stands now: a stout leaning trunk, one heavy limb sawn off
    short, the leader lopped higher up; a black strap and gym ring hang from the fork."""
    sk = Skeleton(bark, seed, sides=8)
    r = sk.rng
    top = np.array([lean[0] * trunk_h, lean[1] * trunk_h, trunk_h])
    n = 7
    ts = np.linspace(0, 1, n + 1)
    pts = [top * t + np.array([0.04 * math.sin(t * 6), 0.03 * math.cos(t * 5), 0]) for t in ts]
    radii = [trunk_r * (1.5 - 0.5 * min(1, t * 5)) * (1 - 0.14 * t) for t in ts]
    sk.m.tube(pts, radii, bark, seg=14, caps=(False, False))
    # sawn limb to one side
    d_stub = _unit(np.array([-0.82, 0.1, 0.55]))
    sp, sd, sr = sk.branch(top - UP * 0.15, d_stub, stub_len, trunk_r * 0.62, trunk_r * 0.5,
                           bend=0.04, sides=10, n=3)
    sk.m.cylinder(sr[-1] * 1.0, 0.012, "log_end", seg=10,
                  M=frame_from_axis(sp[-1] - sd[-1] * 0.004, sd[-1]), ts=(1, 1))
    # leader carries on up, then is lopped; two more short stubs near the top
    d_lead = _unit(np.array([0.22, 0.05, 1.0]))
    lp, ld, lr = sk.branch(top - UP * 0.1, d_lead, leader_h - trunk_h, trunk_r * 0.72,
                           trunk_r * 0.5, bend=0.035, sides=10, n=5)
    sk.m.cylinder(lr[-1], 0.012, "log_end", seg=10,
                  M=frame_from_axis(lp[-1] - ld[-1] * 0.004, ld[-1]), ts=(1, 1))
    for k, (az, up) in enumerate(((0.9, 0.7), (-2.3, 0.8))):
        i = 2 + k
        d = _unit(np.array([math.cos(az), math.sin(az), up]))
        bp, bd, br = sk.branch(lp[i], d, 0.9 + 0.3 * k, lr[i] * 0.6, lr[i] * 0.48, bend=0.04,
                               sides=8, n=3)
        sk.m.cylinder(br[-1], 0.01, "log_end", seg=8,
                      M=frame_from_axis(bp[-1] - bd[-1] * 0.004, bd[-1]), ts=(1, 1))
    # thin whippy regrowth around the cuts
    for (p, d) in ((sp[-1], sd[-1]), (lp[-1], ld[-1])):
        for _ in range(5):
            dd = _unit(d * 0.4 + UP + r.normal(size=3) * 0.35)
            sk.branch(p - d * 0.1, dd, r.uniform(0.6, 1.3), 0.012, 0.004, bend=0.08, sides=4, n=3)
    if ring:  # strap + wooden ring hanging from the fork
        a = sp[1] + np.array([0.05, -0.02, -0.12])
        ring_z = 1.95
        sk.m.tube([a, (a[0], a[1], ring_z + 0.12)], 0.018, "strap_black", seg=4,
                  flatten=(1.0, 0.25), caps=(False, False))
        c = np.array([a[0], a[1], ring_z])
        loop = [c + 0.105 * np.array([math.cos(t), 0, math.sin(t)])
                for t in np.linspace(0, 2 * math.pi, 17)]
        sk.m.tube(loop, 0.016, "wood_ring", seg=6, caps=(False, False))
    return sk.m


def small_tree(height=3.8, trunk_r=0.07, seed=5, bark="bark_grey", twigs="twigs_grey",
               twist=0.35, fork_h=1.25, twig_size=1.0, lean=(0.08, 0.0)):
    """Small gnarled tree with a twisting stem (the one in front of the outbuilding)."""
    return broadleaf(height=height, trunk_r=trunk_r, trunk_h=fork_h, limbs=3, seed=seed,
                     bark=bark, twigs=twigs, lean=lean, limb_len=(height - fork_h) * 0.8,
                     limb_angle=32, levels=2, twig_size=twig_size, flare=1.25, trunk_sides=8,
                     twist=twist)


def sapling(height=1.9, seed=6, leaf="leaf_euc", stem="stem_green", leaves=22, leaf_size=0.2,
            stake=None):
    """Whippy young tree in leaf (the eucalyptus), optionally tied to a stake."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    sk = Skeleton(stem, seed, sides=5)
    pts, dirs, _ = sk.branch((0, 0, 0), (0.05, 0.02, 1), height, 0.012, 0.004, bend=0.05, n=7)
    m.merge(sk.m)
    P, N = [], []
    for i in range(leaves):
        t = rng.uniform(0.3, 1.0)
        k = min(len(pts) - 1, int(t * (len(pts) - 1)))
        az = rng.uniform(0, 2 * math.pi)
        out = np.array([math.cos(az), math.sin(az), rng.uniform(-0.1, 0.5)])
        reach = rng.uniform(0.05, 0.3) * (1.1 - t * 0.5)
        P.append(pts[k] + out * reach)
        N.append(out)
    cards(m, P, N, leaf, size=leaf_size, rng=rng, jitter=0.6)
    if stake:
        m.cylinder(0.02, stake, "wood_grey", seg=6, M=translate(0.09, 0.03, 0))
    return m


# --------------------------------------------------------------------------- evergreen masses


def _icosphere(sub=2):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
         (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [tuple(_unit(p)) for p in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(sub):
        cache, nf = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                cache[key] = len(v)
                v.append(tuple(_unit((np.array(v[a]) + np.array(v[b])) / 2)))
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), f


def shrub(rx=0.8, ry=None, h=1.2, leaf="leaf_green", body="foliage_green", seed=7, lumps=0.36,
          density=30, card=0.26, base_lift=0.12, flat_bottom=True, sub=2, stems=None, twiggy=None):
    """Evergreen shrub or leafy mound: an irregular inner body with a loose shell of leaf
    cards around it so the outline is ragged, not a ball. rx, ry: half-widths; h: height.
    body=None gives an open, see-through shrub; twiggy=<material> adds bare sprays on top.
    Origin at the centre of its base."""
    ry = rx if ry is None else ry
    m = Mesh()
    rng = np.random.default_rng(seed)
    noise = _noise3(seed)
    rz = h / 2

    def surface(d):
        k = 1 + lumps * float(noise(d * 1.7)) + 0.5 * lumps * float(noise(d * 4.1 + 3.0))
        return k

    if body:
        V, F = _icosphere(sub)
        P = []
        for p in V:
            k = surface(p) * 0.8
            q = np.array([p[0] * rx * k, p[1] * ry * k, rz + p[2] * rz * k])
            if flat_bottom and q[2] < base_lift:
                q[2] = base_lift * 0.5
            P.append(q)
        P = np.array(P)
        uv = [[(math.atan2(V[i][1], V[i][0]) / math.pi * 2.0 * max(rx, ry), P[i][2] * 1.2) for i in tri] for tri in F]
        m.add(P, F, uv, body, smooth=True)
    area = 4 * math.pi * ((rx * ry) ** 0.8 + (rx * rz) ** 0.8 + (ry * rz) ** 0.8) ** 1.25 / 3 ** 1.25
    n = int(area * density * 1.3)
    pts, nrm = [], []
    for _ in range(n):
        d = _unit(rng.normal(size=3))
        if d[2] < -0.3:
            d[2] = -d[2]
        k = surface(d) * rng.uniform(0.78, 1.1)
        pts.append(np.array([d[0] * rx * k, d[1] * ry * k, max(base_lift, rz + d[2] * rz * k)]))
        nrm.append(_unit(np.array([d[0] / rx, d[1] / ry, d[2] / rz])))
    cards(m, pts, nrm, leaf, size=card, rng=rng, jitter=0.7, size_var=0.45)
    if twiggy:
        for _ in range(max(3, int(area * 0.9))):
            d = _unit(rng.normal(size=3) * (1, 1, 0.5) + (0, 0, 0.9))
            k = surface(d) * 0.75
            base = np.array([d[0] * rx * k, d[1] * ry * k, rz + d[2] * rz * k])
            spray(m, base, d, twiggy, size=max(rx, rz) * rng.uniform(0.5, 0.9), rng=rng, crossed=2)
    if stems:  # a few bare stems poking out of the top
        for _ in range(stems):
            a = rng.uniform(0, 2 * math.pi)
            p0 = np.array([rx * 0.4 * math.cos(a), ry * 0.4 * math.sin(a), h * 0.6])
            d = _unit(np.array([math.cos(a) * 0.4, math.sin(a) * 0.4, 1]))
            m.tube([p0, p0 + d * h * rng.uniform(0.4, 0.8)], [0.008, 0.003], "twig", seg=4)
    return m


def clump(radius=0.5, h=0.45, leaf="leaf_sage", seed=8, density=60, card=0.22, flowers=None,
          flower_h=0.12, n_flowers=14):
    """Low herbaceous mound made only of leaf cards (sage, sedum, foxglove rosettes).
    flowers: material for flat flower heads held above the leaves."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    n = int(math.pi * radius * radius * density)
    pts, nrm = [], []
    for _ in range(n):
        a, rr = rng.uniform(0, 2 * math.pi), radius * math.sqrt(rng.uniform(0, 1))
        dome = h * (1 - (rr / radius) ** 2) ** 0.6
        pts.append((rr * math.cos(a), rr * math.sin(a), 0.04 + dome * rng.uniform(0.35, 1.0)))
        nrm.append(_unit(np.array([math.cos(a) * rr / radius, math.sin(a) * rr / radius, 0.9])))
    cards(m, pts, nrm, leaf, size=card, rng=rng, jitter=0.5)
    if flowers:
        for _ in range(n_flowers):
            a, rr = rng.uniform(0, 2 * math.pi), radius * 0.8 * math.sqrt(rng.uniform(0, 1))
            x, y = rr * math.cos(a), rr * math.sin(a)
            z = h * (1 - (rr / radius) ** 2) ** 0.6 + flower_h * rng.uniform(0.6, 1.2)
            m.tube([(x, y, 0.05), (x, y, z)], 0.004, "stem_green", seg=4, caps=(False, False))
            m.cylinder(rng.uniform(0.035, 0.06), 0.02, flowers, seg=7, M=translate(x, y, z))
    return m


def sweep_foliage(path, profile, body="foliage_green", leaf="leaf_green", seed=9, lumps=0.12,
                  step=0.35, density=16, card=0.28, card_lift=0.03, closed=False, drape=None):
    """Sweep a lumpy cross-section along a path (hedges, ivy over a wall).
    profile: list of (offset, z); offset is measured to the LEFT of the path direction.
    The surface normal points to the right of the profile's direction of travel."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    noise = _noise3(seed)
    L = path_length(path)
    ns = max(2, int(round(L / step)) + 1)
    prof = np.asarray(profile, float)
    npf = len(prof)
    # outward normal of each profile vertex in the (offset, z) plane
    tang = np.gradient(prof, axis=0)
    pn = np.stack([tang[:, 1], -tang[:, 0]], 1)
    pn /= (np.linalg.norm(pn, axis=1, keepdims=True) + 1e-9)
    grid = np.zeros((ns, npf, 3))
    nrm = np.zeros((ns, npf, 3))
    for i in range(ns):
        s = L * i / (ns - 1)
        p, t = point_along(path, s)
        left = np.array([-t[1], t[0]])
        for j in range(npf):
            k = lumps * float(noise(np.array([s * 0.9, prof[j, 1] * 1.4, prof[j, 0] * 1.4])))
            off = prof[j, 0] + pn[j, 0] * k
            z = prof[j, 1] + pn[j, 1] * k
            if drape:  # ragged lower edge: cut the first/last row short here and there
                pass
            grid[i, j] = (p[0] + left[0] * off, p[1] + left[1] * off, max(0.02, z))
            nrm[i, j] = (left[0] * pn[j, 0], left[1] * pn[j, 0], pn[j, 1])
    verts = grid.reshape(-1, 3)
    faces, uvs = [], []
    run = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
    for i in range(ns - 1):
        for j in range(npf - 1):
            a, b, c, d = i * npf + j, (i + 1) * npf + j, (i + 1) * npf + j + 1, i * npf + j + 1
            faces.append((a, d, c, b))
            u0, u1 = L * i / (ns - 1), L * (i + 1) / (ns - 1)
            uvs.append([(u0, run[j]), (u0, run[j + 1]), (u1, run[j + 1]), (u1, run[j])])
    m.add(verts, faces, uvs, body, smooth=True)
    for i in (0, ns - 1):  # end caps
        idx = [i * npf + j for j in range(npf)]
        if i != 0:
            idx = idx[::-1]
        m.add([verts[k] for k in idx], [tuple(range(npf))],
              [[(verts[k][0], verts[k][2]) for k in idx]], body)
    area = L * run[-1]
    pts, ns_ = [], []
    for _ in range(int(area * density)):
        fi, fj = rng.uniform(0, ns - 1.001), rng.uniform(0, npf - 1.001)
        i, j = int(fi), int(fj)
        a, b = fi - i, fj - j
        p = (grid[i, j] * (1 - a) * (1 - b) + grid[i + 1, j] * a * (1 - b)
             + grid[i, j + 1] * (1 - a) * b + grid[i + 1, j + 1] * a * b)
        nn = nrm[i, j] * (1 - b) + nrm[i, j + 1] * b
        pts.append(p)
        ns_.append(nn)
    cards(m, pts, ns_, leaf, size=card, rng=rng, jitter=0.5, lift=card_lift)
    return m


def hedge(path, height=1.9, width=0.9, body="foliage_beech", leaf="leaf_beech", seed=10,
          density=14, card=0.3, lumps=0.1):
    """Clipped hedge: rounded-shoulder box section swept along the path."""
    w = width / 2
    prof = [(-w, 0.05), (-w * 1.02, height * 0.5), (-w * 0.92, height * 0.9), (-w * 0.55, height),
            (0, height * 1.02), (w * 0.55, height), (w * 0.92, height * 0.9), (w * 1.02, height * 0.5),
            (w, 0.05)]
    # profile runs right->over->left so that normals face outward
    return sweep_foliage(path, prof[::-1], body, leaf, seed, lumps, 0.4, density, card)


def ivy_on_wall(path, wall_h=1.8, thickness=0.28, z_min=0.25, over_top=0.3, back_drop=0.5,
                side=-1, seed=11, density=22, card=0.3, lumps=0.14):
    """Ivy smothering a wall: up the garden face, over the coping, a little way down the
    back. side: -1 = garden face is on the right of the path direction, +1 = left."""
    s = side
    prof = [(s * (0.12 + thickness * 0.5), z_min), (s * (0.12 + thickness), wall_h * 0.55),
            (s * (0.12 + thickness * 1.15), wall_h * 0.95), (s * 0.12, wall_h + over_top),
            (-s * (0.12 + thickness * 0.8), wall_h + over_top * 0.5),
            (-s * (0.12 + thickness * 0.6), wall_h - back_drop)]
    if s < 0:   # sweep_foliage wants the visible face on the right of the profile's travel
        prof = prof[::-1]
    return sweep_foliage(path, prof, "foliage_ivy", "leaf_ivy", seed, lumps, 0.3, density, card)


def brush_pile(radius=0.9, height=1.2, seed=12, mat="twigs_dry", n=70, sticks=14):
    """Heap of cut branches with dead brown leaves still on."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, rr = rng.uniform(0, 2 * math.pi), radius * 0.75 * math.sqrt(rng.uniform(0, 1))
        base = np.array([rr * math.cos(a), rr * math.sin(a), rng.uniform(0.0, height * 0.35)])
        d = _unit(np.array([math.cos(a) * 0.7, math.sin(a) * 0.7, rng.uniform(0.3, 1.4)])
                  + rng.normal(size=3) * 0.25)
        spray(m, base, d, mat, size=rng.uniform(0.6, 1.0) * height, rng=rng, crossed=2, width=1.1)
    for _ in range(sticks):
        a = rng.uniform(0, 2 * math.pi)
        p0 = np.array([radius * 0.9 * math.cos(a), radius * 0.9 * math.sin(a), 0.03])
        p1 = np.array([radius * 0.2 * math.cos(a + 2.5), radius * 0.2 * math.sin(a + 2.5),
                       height * rng.uniform(0.5, 1.0)])
        m.tube([p0, (p0 + p1) / 2 + rng.normal(size=3) * 0.08, p1], [0.02, 0.015, 0.008], "twig_dry", seg=5)
    return m


def border(outline, seed=13, plants=None, soil=True):
    """A planted bed: soil patch plus an informal mix of mounds inside the outline.
    plants: list of dicts {kind, x, y, ...} placed explicitly; anything else is filler."""
    m = Mesh()
    if soil:                                            # in small triangles, so that it can follow uneven ground
        V, T = tessellate(outline)
        m.add([(x, y, 0.025) for x, y in V], T, [[V[i] for i in t] for t in T], "soil", smooth=True)
    for i, p in enumerate(plants or []):
        kind = p.get("kind", "shrub")
        kw = {k: v for k, v in p.items() if k not in ("kind", "x", "y", "rot")}
        kw.setdefault("seed", seed + 17 * i)
        sub = {"shrub": shrub, "clump": clump, "sapling": sapling, "stems": bare_stems}[kind](**kw)
        m.merge(sub, translate(p["x"], p["y"], 0) @ rot_z(p.get("rot", 0)))
    return m


def bare_stems(height=1.3, n=9, radius=0.25, seed=14, mat="twig", twigs="twigs_brown",
               twig_size=0.5):
    """Leafless twiggy sub-shrub or cut-back perennial: a sheaf of thin upright stems."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a, rr = rng.uniform(0, 2 * math.pi), radius * math.sqrt(rng.uniform(0, 1))
        p0 = np.array([rr * math.cos(a), rr * math.sin(a), 0.0])
        d = _unit(np.array([math.cos(a) * 0.35, math.sin(a) * 0.35, 1.0]) + rng.normal(size=3) * 0.12)
        L = height * rng.uniform(0.6, 1.0)
        m.tube([p0, p0 + d * L * 0.5, p0 + d * L + rng.normal(size=3) * 0.05], [0.009, 0.007, 0.003],
               mat, seg=4, caps=(False, False))
        if twigs and rng.random() < 0.6:
            spray(m, p0 + d * L * 0.7, d, twigs, size=twig_size, rng=rng, crossed=2)
    return m


def olive_pot(height=2.6, pot_r=0.3, pot_h=0.42, seed=15):
    """Standard olive in a black tub: clear stem, loose silvery head."""
    from .objects import pot
    m = pot(r=pot_r, h=pot_h, style="bucket", mat="plastic_black")
    sk = Skeleton("bark_grey", seed, sides=6)
    stem_h = height * 0.5
    pts, dirs, radii = sk.branch((0, 0, pot_h * 0.8), (0.06, 0.02, 1), stem_h, 0.03, 0.022, bend=0.04, n=4)
    top = pts[-1]
    rng = sk.rng
    for i in range(5):
        az = 2 * math.pi * i / 5 + rng.uniform(-0.3, 0.3)
        d = _unit(np.array([math.cos(az) * 0.6, math.sin(az) * 0.6, 1.0]))
        sk.branch(top, d, height * 0.32, 0.014, 0.004, bend=0.08, n=3)
    m.merge(sk.m)
    head = shrub(rx=height * 0.27, ry=height * 0.25, h=height * 0.5, leaf="leaf_silver", body=None,
                 seed=seed, density=60, card=0.2, lumps=0.4, flat_bottom=False, base_lift=-9)
    inner = shrub(rx=height * 0.16, ry=height * 0.15, h=height * 0.32, leaf="leaf_silver", body=None,
                  seed=seed + 1, density=60, card=0.2, lumps=0.4, flat_bottom=False, base_lift=-9)
    m.merge(inner, translate(top[0], top[1], top[2] + height * 0.0))
    m.merge(head, translate(top[0], top[1], top[2] - height * 0.1))
    return m


def conifer(height=9.0, r=1.6, seed=31, leaf="leaf_green", body="foliage_ivy"):
    """Dark columnar evergreen (cypress / leylandii) for the skyline."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    noise = _noise3(seed)
    seg, rings = 9, 7
    verts = []
    for i in range(rings + 1):
        t = i / rings
        rr = r * (0.55 + 0.45 * math.sin(min(1.0, t * 2.2) * math.pi / 2)) * (1 - t ** 1.6) + 0.03
        for k in range(seg):
            a = 2 * math.pi * k / seg
            kk = 1 + 0.18 * float(noise(np.array([math.cos(a), math.sin(a), t * 3.0])))
            verts.append((rr * kk * math.cos(a), rr * kk * math.sin(a), 0.4 + t * (height - 0.4)))
    faces, uvs = [], []
    for i in range(rings):
        for k in range(seg):
            k2 = (k + 1) % seg
            faces.append((i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k))
            uvs.append([(k, i), (k + 1, i), (k + 1, i + 1), (k, i + 1)])
    m.add(verts, faces, uvs, body, smooth=True)
    m.cylinder(0.14, 0.5, "bark_grey", seg=6)
    return m


def tree_belt(path, n=5, height=10.0, seed=50, twigs=("twigs_brown", "twigs_grey"), bark="bark_grey",
              jitter=1.5, evergreens=0, spread=1.0):
    """A loose line of bare trees along a path: the borrowed landscape beyond the boundary.
    Built light (few sides, two branching levels) because there are many of them."""
    m = Mesh()
    rng = np.random.default_rng(seed)
    L = path_length(path)
    for i in range(n):
        p, t = point_along(path, L * (i + 0.5) / n)
        nrm = np.array([-t[1], t[0]])
        pos = p + nrm * rng.uniform(-jitter, jitter) + t * rng.uniform(-jitter, jitter)
        h = height * rng.uniform(0.75, 1.2)
        if i < evergreens:
            tree = conifer(height=h * 0.9, r=h * 0.17, seed=seed + i)
        else:
            tw = twigs[int(rng.integers(0, len(twigs)))] if isinstance(twigs, (list, tuple)) else twigs
            tree = broadleaf(height=h, trunk_r=0.022 * h + 0.06, trunk_h=h * rng.uniform(0.22, 0.34), limbs=4,
                             seed=seed + 7 * i, bark=bark, twigs=tw, limb_angle=34 * spread, levels=2,
                             twig_size=h * 0.3, trunk_sides=6, limb_len=h * 0.5, sides=5, children=(2, 3))
        m.merge(tree, translate(pos[0], pos[1], 0) @ rot_z(rng.uniform(0, 360)))
    return m
