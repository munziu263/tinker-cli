"""Blender (bpy) plumbing: materials, a small numpy mesh builder, object creation.

Builders elsewhere in the package return `Mesh` objects in local coordinates
(origin on the ground at the middle of the thing, +X along its length, +Z up, metres).
`build.py` places them from the positions in garden.json.
"""
from __future__ import annotations

import math

import bpy
import numpy as np

from . import textures as tx

TEX: dict = {}      # texture name -> {"path", "size": (w_m, h_m), "normal"}
_MATS: dict = {}

# Materials that are not just "texture X at its natural scale".
# tex: texture name, colour: flat RGB (sRGB 0..1), rough/metal, alpha: leaf cards.
MATERIAL_DEFS = {
    "brick_stock": dict(tex="brick_stock", rough=0.93),
    "brick_red_old": dict(tex="brick_red_old", rough=0.93),
    "brick_dark": dict(tex="brick_dark", rough=0.9),
    "clay_tiles": dict(tex="clay_tiles", rough=0.88, bump=0.6),
    "tiles_grey": dict(tex="tiles_grey", rough=0.9, bump=0.5),
    "mossy_roof": dict(tex="mossy_roof", rough=0.97),
    "boards_dark": dict(tex="boards_dark", rough=0.85),
    "weatherboard_dark": dict(tex="weatherboard_dark", rough=0.85),
    "fence_boards": dict(tex="fence_boards", rough=0.9),
    "wood_grey": dict(tex="wood_grey", rough=0.9),
    "wood_new": dict(tex="wood_new", rough=0.75),
    "wood_brown": dict(tex="wood_brown", rough=0.8),
    "render_white": dict(tex="render_white", rough=0.95),
    "concrete": dict(tex="concrete", rough=0.95),
    "bark_willow": dict(tex="bark_willow", rough=0.97),
    "bark_pale": dict(tex="bark_pale", rough=0.9),
    "bark_hazel": dict(tex="bark_hazel", rough=0.85),
    "bark_grey": dict(tex="bark_grey", rough=0.85),
    "log_end": dict(tex="log_end", rough=0.85),
    "firewood_bag": dict(tex="firewood_bag", rough=0.6),
    "foliage_ivy": dict(tex="foliage_ivy", rough=0.7),
    "foliage_green": dict(tex="foliage_green", rough=0.8),
    "foliage_beech": dict(tex="foliage_beech", rough=0.9),
    "ground": dict(tex="ground", rough=1.0),
    # leaf cards (alpha-tested, two-sided)
    **{k: dict(tex=k, rough=0.65, alpha=True) for k in (
        "leaf_ivy", "leaf_green", "leaf_lime", "leaf_yellow", "leaf_beech",
        "leaf_sage", "leaf_silver", "leaf_euc", "leaf_dry",
        "twigs_brown", "twigs_grey", "twigs_hazel", "twigs_dry", "twigs_stems")},
    # flat colours
    "glass_dark": dict(colour=(0.16, 0.20, 0.23), rough=0.08),
    "paint_black": dict(colour=(0.03, 0.03, 0.032), rough=0.55),
    "paint_white": dict(colour=(0.86, 0.86, 0.84), rough=0.6),
    "paint_brown": dict(colour=(0.22, 0.13, 0.09), rough=0.6),
    "paint_slate": dict(colour=(0.17, 0.19, 0.21), rough=0.5),
    "steel_black": dict(colour=(0.04, 0.04, 0.045), rough=0.45, metal=0.7),
    "galvanised": dict(colour=(0.55, 0.57, 0.58), rough=0.45, metal=0.25),     # more metal than this and it goes black without a sky to reflect
    "roof_felt": dict(colour=(0.07, 0.07, 0.075), rough=0.9),
    "plastic_teal": dict(colour=(0.07, 0.24, 0.24), rough=0.45),
    "plastic_green": dict(colour=(0.10, 0.19, 0.12), rough=0.5),
    "plastic_blue": dict(colour=(0.05, 0.22, 0.62), rough=0.45),
    "plastic_black": dict(colour=(0.03, 0.03, 0.035), rough=0.5),
    "plastic_red": dict(colour=(0.70, 0.07, 0.10), rough=0.5),
    "terracotta": dict(colour=(0.66, 0.31, 0.18), rough=0.9),
    "terracotta_pale": dict(colour=(0.72, 0.50, 0.36), rough=0.9),
    "glaze_black": dict(colour=(0.03, 0.035, 0.03), rough=0.15),
    "glaze_blue": dict(colour=(0.16, 0.34, 0.52), rough=0.2),
    "glaze_teal": dict(colour=(0.50, 0.66, 0.64), rough=0.35),
    "soil": dict(colour=(0.23, 0.18, 0.14), rough=1.0),
    "tarp_yellow": dict(colour=(0.78, 0.62, 0.08), rough=0.55),
    "tarp_blue": dict(colour=(0.06, 0.25, 0.70), rough=0.5),
    "sheet_white": dict(colour=(0.78, 0.76, 0.70), rough=0.8),
    "hardboard": dict(colour=(0.60, 0.42, 0.27), rough=0.7),
    "hose_white": dict(colour=(0.82, 0.82, 0.78), rough=0.5),
    "strap_black": dict(colour=(0.02, 0.02, 0.02), rough=0.8),
    "wood_ring": dict(colour=(0.62, 0.46, 0.28), rough=0.6),
    "bbq_cover": dict(colour=(0.035, 0.035, 0.04), rough=0.7),
    "twig": dict(colour=(0.30, 0.20, 0.13), rough=0.9),
    "twig_dry": dict(colour=(0.42, 0.24, 0.15), rough=0.95),
    "straw": dict(colour=(0.70, 0.60, 0.38), rough=0.95),
    "stem_green": dict(colour=(0.25, 0.33, 0.14), rough=0.8),
    "sedum": dict(colour=(0.42, 0.10, 0.16), rough=0.8),
    "ball_white": dict(colour=(0.9, 0.9, 0.9), rough=0.4),
    "net_white": dict(colour=(0.85, 0.85, 0.85), rough=0.7),
    "marker": dict(colour=(0.95, 0.35, 0.10), rough=0.5),
    "marker_blue": dict(colour=(0.10, 0.45, 0.85), rough=0.5),
    "marker_boundary": dict(colour=(0.85, 0.10, 0.12), rough=0.6),
    "marker_drain": dict(colour=(0.55, 0.30, 0.10), rough=0.6),
    "hatch": dict(colour=(0.93, 0.78, 0.25), rough=0.9),
    "cast_iron": dict(colour=(0.16, 0.16, 0.17), rough=0.55, metal=0.6),
    "polycarbonate": dict(colour=(0.72, 0.76, 0.78), rough=0.25),
    "gravel_grey": dict(colour=(0.60, 0.57, 0.52), rough=1.0),
    "stump_wood": dict(colour=(0.62, 0.58, 0.52), rough=0.9),
    # inside the house: flat colours, read like a painted model
    "plaster": dict(colour=(0.90, 0.89, 0.84), rough=0.9),
    "wall_cut": dict(colour=(0.17, 0.18, 0.19), rough=0.8),
    "timber_black": dict(colour=(0.06, 0.055, 0.05), rough=0.75),
    "quarry_tiles": dict(colour=(0.50, 0.21, 0.15), rough=0.6),
    "floor_red": dict(colour=(0.55, 0.11, 0.12), rough=0.55),
    "floor_boards": dict(colour=(0.40, 0.28, 0.19), rough=0.6),
    "carpet_red": dict(colour=(0.60, 0.07, 0.09), rough=1.0),
    "terracotta_tiles": dict(colour=(0.76, 0.45, 0.32), rough=0.6),
    "sanitary": dict(colour=(0.94, 0.94, 0.93), rough=0.25),
    "aga_cream": dict(colour=(0.88, 0.84, 0.72), rough=0.35),
    "appliance_white": dict(colour=(0.90, 0.90, 0.90), rough=0.4),
    "worktop": dict(colour=(0.70, 0.57, 0.40), rough=0.6),
    "limewash": dict(colour=(0.86, 0.84, 0.78), rough=0.95),
    "rooflight_glass": dict(colour=(0.16, 0.20, 0.23), rough=0.08),
    "roof_sheet_green": dict(colour=(0.33, 0.45, 0.42), rough=0.7),
    "tarp_cream": dict(colour=(0.80, 0.76, 0.66), rough=0.7),
    "fascia_black": dict(colour=(0.03, 0.03, 0.032), rough=0.55),      # as paint_black, but its own name: roofs can be lifted off
    "roof_under": dict(colour=(0.10, 0.08, 0.07), rough=0.9),
    "plastic_brown": dict(colour=(0.36, 0.20, 0.14), rough=0.5),
}


def init(texture_dir):
    """Empty scene + (re)generate all textures."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    TEX.clear()
    TEX.update(tx.build_all(texture_dir))
    _MATS.clear()


def tex_size(mat_name):
    d = MATERIAL_DEFS.get(mat_name, {})
    t = TEX.get(d.get("tex", mat_name))
    return t["size"] if t else (1.0, 1.0)


def material(name):
    if name in _MATS:
        return _MATS[name]
    d = MATERIAL_DEFS.get(name)
    if d is None:
        raise KeyError(f"unknown material {name!r}")
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = d.get("rough", 0.8)
    bsdf.inputs["Metallic"].default_value = d.get("metal", 0.0)
    if "colour" in d:
        c = [((v / 12.92) if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) for v in d["colour"]]
        bsdf.inputs["Base Color"].default_value = (*c, 1.0)
        mat.diffuse_color = (*c, 1.0)
    t = TEX.get(d.get("tex", ""))
    if t:
        img = bpy.data.images.load(t["path"], check_existing=True)
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        node.location = (-420, 300)
        nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        if d.get("alpha"):
            nt.links.new(node.outputs["Alpha"], bsdf.inputs["Alpha"])
            mat.use_backface_culling = False
        else:
            mat.use_backface_culling = True
        if t.get("normal"):
            nimg = bpy.data.images.load(t["normal"], check_existing=True)
            nimg.colorspace_settings.name = "Non-Color"
            nnode = nt.nodes.new("ShaderNodeTexImage")
            nnode.image = nimg
            nnode.location = (-720, -120)
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.location = (-380, -120)
            nmap.inputs["Strength"].default_value = d.get("bump", 1.0)
            nt.links.new(nnode.outputs["Color"], nmap.inputs["Color"])
            nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    _MATS[name] = mat
    return mat


# --------------------------------------------------------------------------- transforms


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], float)


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], float)


def translate(x=0.0, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def scale(x=1.0, y=None, z=None):
    y = x if y is None else y
    z = x if z is None else z
    return np.diag([x, y, z, 1.0])


def frame_from_axis(origin, direction, up=(0, 0, 1)):
    """4x4 matrix whose local +Z points along `direction`, placed at origin."""
    z = np.asarray(direction, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    u = np.asarray(up, float)
    if abs(np.dot(z, u)) > 0.98:
        u = np.array([1.0, 0, 0])
    x = np.cross(u, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


# --------------------------------------------------------------------------- mesh builder


def tessellate(outline, max_edge=0.35):
    """Cut a plan outline (any simple polygon, convex or not) into small triangles: ear
    clipping, then every triangle split in four until no edge is longer than max_edge.
    Returns (points, triangles). A flat patch built this way can be laid on uneven ground
    (Mesh.drape) and follow it; one big polygon would bridge the dips and bury the bumps."""
    V = [(float(p[0]), float(p[1])) for p in outline]
    n = len(V)
    idx = list(range(n))
    if sum(V[i][0] * V[(i + 1) % n][1] - V[(i + 1) % n][0] * V[i][1] for i in range(n)) < 0:
        idx.reverse()                                   # work counter-clockwise

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    T = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = V[i0], V[i1], V[i2]
            if cross(a, b, c) <= 1e-12:                 # a reflex corner is not an ear
                continue
            if any(cross(a, b, V[j]) >= 0 and cross(b, c, V[j]) >= 0 and cross(c, a, V[j]) >= 0
                   for j in idx if j not in (i0, i1, i2)):
                continue
            T.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break                                       # degenerate outline: keep what there is
    if len(idx) == 3:
        T.append(tuple(idx))
    while T and len(T) < 40000:
        if max(math.dist(V[a], V[b]) for t in T for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))) <= max_edge:
            break
        mid = {}

        def m(i, j):
            key = (min(i, j), max(i, j))
            if key not in mid:
                V.append(((V[i][0] + V[j][0]) / 2, (V[i][1] + V[j][1]) / 2))
                mid[key] = len(V) - 1
            return mid[key]

        T = [t for a, b, c in T for t in ((a, m(a, b), m(c, a)), (m(a, b), b, m(b, c)), (m(c, a), m(b, c), c), (m(a, b), m(b, c), m(c, a)))]
    return V, T


class Mesh:
    """Accumulates polygons with per-corner UVs and per-face material slots."""

    def __init__(self):
        self.v: list = []
        self.f: list = []
        self.uv: list = []
        self.m: list = []
        self.sm: list = []
        self.slots: list = []

    # -- bookkeeping
    def slot(self, mat):
        if mat not in self.slots:
            self.slots.append(mat)
        return self.slots.index(mat)

    def add(self, verts, faces, uvs, mat, smooth=False, M=None):
        verts = np.asarray(verts, float).reshape(-1, 3)
        if M is not None:
            verts = verts @ np.asarray(M)[:3, :3].T + np.asarray(M)[:3, 3]
        base = len(self.v)
        self.v.extend(map(tuple, verts))
        s = self.slot(mat)
        for face, fuv in zip(faces, uvs):
            self.f.append(tuple(base + i for i in face))
            self.uv.append([tuple(map(float, t)) for t in fuv])
            self.m.append(s)
            self.sm.append(bool(smooth))
        return self

    def merge(self, other, M=None):
        v = np.asarray(other.v, float).reshape(-1, 3)
        if M is not None and len(v):
            v = v @ np.asarray(M)[:3, :3].T + np.asarray(M)[:3, 3]
        base = len(self.v)
        self.v.extend(map(tuple, v))
        remap = [self.slot(s) for s in other.slots]
        for face, fuv, mi, sm in zip(other.f, other.uv, other.m, other.sm):
            self.f.append(tuple(base + i for i in face))
            self.uv.append(fuv)
            self.m.append(remap[mi])
            self.sm.append(sm)
        return self

    def transformed(self, M):
        return Mesh().merge(self, M)

    def drape(self, height, at=(0.0, 0.0), rot_deg=0.0):
        """Stand the mesh on a ground surface: every vertex is raised by the height of the
        ground under it, so a wall follows a slope and a heap lies on it. height(x, y) -> z
        takes site coordinates; at and rot_deg say where this mesh will be placed."""
        v = np.asarray(self.v, float).reshape(-1, 3)
        if len(v):
            a = math.radians(rot_deg)
            c, s = math.cos(a), math.sin(a)
            v[:, 2] += height(at[0] + c * v[:, 0] - s * v[:, 1], at[1] + s * v[:, 0] + c * v[:, 1])
            self.v = list(map(tuple, v))
        return self

    @property
    def tri_count(self):
        return sum(len(f) - 2 for f in self.f)

    def bounds(self):
        v = np.asarray(self.v, float).reshape(-1, 3)
        return v.min(0), v.max(0)

    # -- primitives
    def quad(self, a, b, c, d, mat, ts=None, uv=None, M=None, grain=False):
        """Quad a-b-c-d (counter-clockwise seen from the front). Without explicit uv the
        texture is laid at true scale: U along a->b, V along a->d. grain=True turns the
        texture so its X axis follows the longer side."""
        P = np.array([a, b, c, d], float)
        if uv is None:
            ts = ts or tex_size(mat)
            lu, lv = np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[3] - P[0])
            if grain and lv > lu:
                uv = [(0, 0), (0, lu / ts[1]), (lv / ts[0], lu / ts[1]), (lv / ts[0], 0)]
            else:
                uv = [(0, 0), (lu / ts[0], 0), (lu / ts[0], lv / ts[1]), (0, lv / ts[1])]
        return self.add(P, [(0, 1, 2, 3)], [uv], mat, M=M)

    def box(self, size, mat, M=None, origin="base", ts=None, grain=False, skip=(),
            mats=None, uv_off=(0.0, 0.0)):
        """Axis-aligned box in local space, then transformed by M.
        origin: 'base' (centred in x,y, sitting on z=0), 'centre', or 'corner'.
        skip / mats: per-face control keyed by '+x','-x','+y','-y','+z','-z'."""
        sx, sy, sz = size
        if origin == "base":
            lo = np.array([-sx / 2, -sy / 2, 0.0])
        elif origin == "centre":
            lo = np.array([-sx / 2, -sy / 2, -sz / 2])
        else:
            lo = np.zeros(3)
        hi = lo + (sx, sy, sz)
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        faces = {
            "-y": [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            "+y": [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
            "+x": [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
            "-x": [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
            "+z": [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            "-z": [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)],
        }
        for key, pts in faces.items():
            if key in skip:
                continue
            fm = (mats or {}).get(key, mat)
            fts = ts or tex_size(fm)
            P = np.array(pts, float)
            lu, lv = np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[3] - P[0])
            uo, vo = uv_off
            if grain and lv > lu:
                uv = [(uo, vo), (uo, vo + lu / fts[1]), (uo + lv / fts[0], vo + lu / fts[1]),
                      (uo + lv / fts[0], vo)]
            else:
                uv = [(uo, vo), (uo + lu / fts[0], vo), (uo + lu / fts[0], vo + lv / fts[1]),
                      (uo, vo + lv / fts[1])]
            self.add(P, [(0, 1, 2, 3)], [uv], fm, M=M)
        return self

    def beam(self, p0, p1, w, h, mat, ts=None, roll=0.0):
        """Rectangular timber from p0 to p1 (w across, h deep); grain along its length."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-6:
            return self
        F = frame_from_axis(p0, p1 - p0)
        if roll:
            F = F @ rot_z(roll)
        # local box: x across (w), y deep (h), z along
        return self.box((w, h, L), mat, M=F, origin="base", ts=ts, grain=True)

    def tube(self, pts, radii, mat, seg=8, ts=None, caps=(True, True), smooth=True,
             v0=0.0, flatten=None, M=None):
        """Swept circle along a polyline. radii: scalar or per-point. Bark-style UVs:
        U wraps the circumference a whole number of times, V runs along at true scale.
        flatten=(sx, sy) squashes the section (for slabs, straps)."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        if M is not None:  # rigid transforms only
            pts = pts @ np.asarray(M)[:3, :3].T + np.asarray(M)[:3, 3]
        n = len(pts)
        radii = np.full(n, radii, float) if np.isscalar(radii) else np.asarray(radii, float)
        ts = ts or tex_size(mat)
        # parallel-transport frames
        tang = np.gradient(pts, axis=0)
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
        ref = np.array([0.0, 0, 1]) if abs(tang[0][2]) < 0.9 else np.array([1.0, 0, 0])
        nrm = np.cross(tang[0], ref)
        nrm /= np.linalg.norm(nrm)
        rings, vs = [], [v0]
        for i in range(n):
            if i > 0:
                ax = np.cross(tang[i - 1], tang[i])
                s = np.linalg.norm(ax)
                if s > 1e-8:
                    ax /= s
                    ang = math.atan2(s, float(np.dot(tang[i - 1], tang[i])))
                    nrm = (nrm * math.cos(ang) + np.cross(ax, nrm) * math.sin(ang)
                           + ax * np.dot(ax, nrm) * (1 - math.cos(ang)))
                vs.append(vs[-1] + np.linalg.norm(pts[i] - pts[i - 1]) / ts[1])
            bi = np.cross(tang[i], nrm)
            ring = []
            for k in range(seg):
                a = 2 * math.pi * k / seg
                ca, sa = math.cos(a), math.sin(a)
                if flatten:
                    ca, sa = ca * flatten[0], sa * flatten[1]
                ring.append(pts[i] + radii[i] * (ca * nrm + sa * bi))
            rings.append(ring)
        verts = [p for ring in rings for p in ring]
        wraps = max(1, round(2 * math.pi * float(radii.max()) / ts[0]))
        faces, uvs = [], []
        for i in range(n - 1):
            for k in range(seg):
                k2 = (k + 1) % seg
                faces.append((i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k))
                u0, u1 = wraps * k / seg, wraps * (k + 1) / seg
                uvs.append([(u0, vs[i]), (u1, vs[i]), (u1, vs[i + 1]), (u0, vs[i + 1])])
        self.add(verts, faces, uvs, mat, smooth=smooth)
        for end, on in zip((0, n - 1), caps):
            if not on or radii[end] < 1e-4:
                continue
            idx = list(range(end * seg, end * seg + seg))
            if end == 0:
                idx = idx[::-1]
            cuv = [(0.5 + 0.5 * math.cos(2 * math.pi * k / seg),
                    0.5 + 0.5 * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
            if end == 0:
                cuv = cuv[::-1]
            self.add([verts[i] for i in idx], [tuple(range(seg))], [cuv], mat)
        return self

    def cylinder(self, r, h, mat, seg=16, M=None, cap_mat=None, r_top=None, ts=None,
                 caps=(True, True)):
        """Upright cylinder / cone frustum on z=0 (local), optional different cap material."""
        r_top = r if r_top is None else r_top
        ts = ts or tex_size(mat)
        wraps = max(1, round(2 * math.pi * max(r, r_top) / ts[0]))
        verts, faces, uvs = [], [], []
        for k in range(seg):
            a = 2 * math.pi * k / seg
            verts.append((r * math.cos(a), r * math.sin(a), 0.0))
        for k in range(seg):
            a = 2 * math.pi * k / seg
            verts.append((r_top * math.cos(a), r_top * math.sin(a), h))
        for k in range(seg):
            k2 = (k + 1) % seg
            faces.append((k, k2, seg + k2, seg + k))
            u0, u1 = wraps * k / seg, wraps * (k + 1) / seg
            uvs.append([(u0, 0), (u1, 0), (u1, h / ts[1]), (u0, h / ts[1])])
        self.add(verts, faces, uvs, mat, smooth=True, M=M)
        cm = cap_mat or mat
        cuv = [(0.5 + 0.5 * math.cos(2 * math.pi * k / seg),
                0.5 + 0.5 * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
        if caps[1] and r_top > 1e-4:
            self.add(verts[seg:], [tuple(range(seg))], [cuv], cm, M=M)
        if caps[0]:
            self.add(verts[:seg][::-1], [tuple(range(seg))], [cuv[::-1]], cm, M=M)
        return self

    def lathe(self, profile, mat, seg=16, M=None, ts=None, smooth=True, cap_top=None,
              cap_bottom=True):
        """Revolve a (radius, z) profile about Z. cap_top: material for a lid at the last ring."""
        ts = ts or tex_size(mat)
        prof = np.asarray(profile, float)
        n = len(prof)
        verts = []
        for r, z in prof:
            for k in range(seg):
                a = 2 * math.pi * k / seg
                verts.append((r * math.cos(a), r * math.sin(a), z))
        run = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))])
        wraps = max(1, round(2 * math.pi * prof[:, 0].max() / ts[0]))
        faces, uvs = [], []
        for i in range(n - 1):
            for k in range(seg):
                k2 = (k + 1) % seg
                faces.append((i * seg + k, i * seg + k2, (i + 1) * seg + k2, (i + 1) * seg + k))
                u0, u1 = wraps * k / seg, wraps * (k + 1) / seg
                uvs.append([(u0, run[i] / ts[1]), (u1, run[i] / ts[1]),
                            (u1, run[i + 1] / ts[1]), (u0, run[i + 1] / ts[1])])
        self.add(verts, faces, uvs, mat, smooth=smooth, M=M)
        cuv = [(0.5 + 0.5 * math.cos(2 * math.pi * k / seg),
                0.5 + 0.5 * math.sin(2 * math.pi * k / seg)) for k in range(seg)]
        if cap_top and prof[-1][0] > 1e-4:
            self.add(verts[(n - 1) * seg:], [tuple(range(seg))], [cuv], cap_top, M=M)
        if cap_bottom and prof[0][0] > 1e-4:
            self.add(verts[:seg][::-1], [tuple(range(seg))], [cuv[::-1]], mat, M=M)
        return self

    def prism(self, outline, z0, z1, mat, ts=None, top_mat=None, M=None, bottom=False,
              side_v0=0.0):
        """Extrude a counter-clockwise XY outline from z0 to z1 (walls + flat top)."""
        ts = ts or tex_size(mat)
        P = np.asarray(outline, float)
        n = len(P)
        run = 0.0
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            L = float(np.linalg.norm(b - a))
            uv = [(run / ts[0], side_v0), ((run + L) / ts[0], side_v0),
                  ((run + L) / ts[0], side_v0 + (z1 - z0) / ts[1]), (run / ts[0], side_v0 + (z1 - z0) / ts[1])]
            self.add([(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)],
                     [(0, 1, 2, 3)], [uv], mat, M=M)
            run += L
        tm = top_mat or mat
        tts = tex_size(tm) if top_mat else ts
        self.add([(p[0], p[1], z1) for p in P], [tuple(range(n))],
                 [[(p[0] / tts[0], p[1] / tts[1]) for p in P]], tm, M=M)
        if bottom:
            self.add([(p[0], p[1], z0) for p in P[::-1]], [tuple(range(n))],
                     [[(p[0] / tts[0], p[1] / tts[1]) for p in P[::-1]]], tm, M=M)
        return self

    # -- output
    def build(self, name, smooth_angle=None):
        me = bpy.data.meshes.new(name)
        if not self.f:
            return me
        me.from_pydata([tuple(map(float, p)) for p in self.v], [], self.f)
        uvl = me.uv_layers.new(name="UVMap")
        flat = np.array([c for fuv in self.uv for t in fuv for c in t], np.float32)
        uvl.data.foreach_set("uv", flat)
        me.polygons.foreach_set("material_index", np.array(self.m, np.int32))
        me.polygons.foreach_set("use_smooth", np.array(self.sm, bool))
        for s in self.slots:
            me.materials.append(material(s))
        me.update()
        return me


# --------------------------------------------------------------------------- scene objects


def get_collection(name, parent=None):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(col)
    return col


def add_empty(name, collection, parent=None, location=(0, 0, 0), size=0.5):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = size
    e.location = location
    collection.objects.link(e)
    if parent is not None:
        e.parent = parent
    return e


def add_object(name, mesh: Mesh, collection, parent=None, location=(0, 0, 0), rot_deg=0.0,
               props=None):
    obj = bpy.data.objects.new(name, mesh.build(name))
    obj.location = location
    obj.rotation_euler = (0.0, 0.0, math.radians(rot_deg))
    collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    for k, v in (props or {}).items():
        obj[k] = v
    return obj
