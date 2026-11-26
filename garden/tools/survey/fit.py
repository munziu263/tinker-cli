"""How well does the model fit the footage? Every survey point against every model surface.

    fit.py <survey.npz> <viewer/model dir> <out_dir> [--garden garden.json] [--sheets N -- <frames_dir> ...]

The survey is some hundreds of thousands of points that the camera saw from several
places, each good to a few centimetres where the footage was walked. The model is a set of
surfaces. Lay one on the other and two kinds of misfit show:

  points with no surface   Something is there in the footage that the model lacks, or the
                           model has it somewhere else. Points further than 0.3 m from
                           every hard surface (ground, walls, buildings, objects) and
                           0.6 m from every soft one (plants, which the model only
                           sketches) are gathered into clusters and listed by size, each
                           with its position, extent, colour and the nearest model
                           feature. A cluster the size of a bench lying 0.7 m from the
                           model's bench says the bench wants moving 0.7 m.
  surfaces with no points  Something is in the model where the footage shows nothing. For
                           every feature, the share of its surface that has survey points
                           on it, counting only the parts that at least three frames
                           looked at from within 7 m. A low share on a thing that was
                           well in view means it is a phantom, or misplaced. (What stands
                           in front is not allowed for, so a thing hidden behind a shrub
                           scores low too: look before deleting.)

Written to <out_dir>: fit.json (the numbers), fit_plan.png (a plan: unexplained clusters in
red, features coloured by support) and, with --sheets N, crops of the footage for the N
biggest clusters with each cluster's box drawn in, to see what it is.

One figure sums it up: the share of survey points within 0.2 m of the model. It is not a
grade (leaves and long grass will never sit on a surface) but it should rise as the model
improves, and a part of the garden where it is low is a part to look at.

The model is read from the web copy that build.py writes (garden.gltf.json and
geometry.b64.txt), so what is checked is what the viewer shows.
"""
import base64
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey, clip_of  # noqa: E402

SOFT_LAYERS = ("planting", "context")
SOFT_MATS = ("leaf_", "twigs_", "foliage_")
SKIP_LAYERS = ("survey", "gaps")
CTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


# ----------------------------------------------------------------------------- the model

def _trs(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in node:
        M = np.diag(list(node["scale"]) + [1.0]) @ M
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4)
        T[:3, :3] = R
        M = T @ M
    if "translation" in node:
        T = np.eye(4)
        T[:3, 3] = node["translation"]
        M = T @ M
    return M


def load_model(model_dir):
    """Triangles of every feature, in site coordinates: a list of dicts(id, label, layer, kind,
    tri (n, 3, 3), soft (n,) bool)."""
    js = json.load(open(os.path.join(model_dir, "garden.gltf.json")))
    buf = base64.b64decode(open(os.path.join(model_dir, "geometry.b64.txt"), "rb").read())

    def accessor(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        dt, nc = CTYPE[a["componentType"]], NCOMP[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0)
        if stride and stride != np.dtype(dt).itemsize * nc:
            raw = np.frombuffer(buf, np.uint8, count=stride * a["count"], offset=off).reshape(a["count"], stride)
            return raw[:, :np.dtype(dt).itemsize * nc].copy().view(dt).reshape(a["count"], nc)
        return np.frombuffer(buf, dt, count=a["count"] * nc, offset=off).reshape(a["count"], nc)

    mats = [m.get("name", "") for m in js.get("materials", [])]
    feats = []

    def walk(i, M, inherited):
        node = js["nodes"][i]
        M = M @ _trs(node)
        ex = node.get("extras", {})
        info = inherited
        if ex.get("label"):
            info = dict(id=node.get("name", ""), label=ex["label"], layer=ex.get("layer", ""), kind=ex.get("kind", ""), tri=[], soft=[])
            feats.append(info)
        if "mesh" in node and info is not None:
            for prim in js["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                P = accessor(prim["attributes"]["POSITION"]).astype(float)
                P = P @ M[:3, :3].T + M[:3, 3]
                P = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)              # glTF is Y up: back to the site grid
                idx = accessor(prim["indices"]).ravel().astype(int) if "indices" in prim else np.arange(len(P))
                tri = P[idx].reshape(-1, 3, 3)
                name = mats[prim["material"]] if "material" in prim else ""
                soft = info["layer"] in SOFT_LAYERS or name.startswith(SOFT_MATS)
                info["tri"].append(tri)
                info["soft"].append(np.full(len(tri), soft))
        for c in node.get("children", []):
            walk(c, M, info)

    for root in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(root, np.eye(4), None)
    out = []
    for f in feats:
        if f["layer"] in SKIP_LAYERS or not f["tri"]:
            continue
        f["tri"], f["soft"] = np.concatenate(f["tri"]), np.concatenate(f["soft"])
        out.append(f)
    return out


def sample(tri, spacing, rng):
    """Points scattered over triangles, about one per spacing squared."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.ceil(area / spacing ** 2)).astype(int)
    k = np.repeat(np.arange(len(tri)), n)
    u, v = rng.random(len(k)), rng.random(len(k))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[k] + (b[k] - a[k]) * u[:, None] + (c[k] - a[k]) * v[:, None]


# ----------------------------------------------------------------------------- the comparison

def inside(poly, X, Y):
    P = np.asarray(poly, float)
    m = np.zeros(X.shape, bool)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        m ^= ((a[1] > Y) != (b[1] > Y)) & (X < (b[0] - a[0]) * (Y - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
    return m


def grown(poly, by):
    P = np.asarray(poly, float)
    c = P.mean(0)
    return c + (P - c) * (1 + by / np.maximum(np.linalg.norm(P - c, axis=1, keepdims=True), 1e-6))


def seen_by(sv, pts, reach=7.0, every=2, need=3):
    """How many frames look at each point from within reach (occlusion is not considered)."""
    count = np.zeros(len(pts), np.int32)
    for i in range(0, len(sv.names), every):
        c = sv.centre(i)
        near = np.flatnonzero(np.linalg.norm(pts - c, axis=1) < reach)
        if not len(near):
            continue
        W, H = sv.frame_size(i)
        u, v, z = sv.project(i, pts[near])
        ok = (z > 0.3) & (u > 0) & (u < W) & (v > 0) & (v < H)
        count[near[ok]] += 1
    return count >= need


def main():
    a = sys.argv[1:]
    dirs = a[a.index("--") + 1:] if "--" in a else []
    a = a[:a.index("--")] if "--" in a else a
    opt = {}
    for key in ("--garden", "--sheets"):
        if key in a:
            i = a.index(key)
            opt[key] = a[i + 1]
            del a[i:i + 2]
    sv = Survey(a[0], dirs)
    feats = load_model(a[1])
    out_dir = a[2]
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    # the model as two clouds of surface points: hard things and soft things
    hard_pts, hard_id, soft_pts = [], [], []
    for k, f in enumerate(feats):
        fine = 0.05 if f["id"] != "lawn" else 0.08
        h = sample(f["tri"][~f["soft"]], fine, rng) if (~f["soft"]).any() else np.zeros((0, 3))
        s = sample(f["tri"][f["soft"]], 0.08, rng) if f["soft"].any() else np.zeros((0, 3))
        f["hard"] = h
        hard_pts.append(h)
        hard_id.append(np.full(len(h), k))
        soft_pts.append(s)
    hard_pts, hard_id, soft_pts = np.concatenate(hard_pts), np.concatenate(hard_id), np.concatenate(soft_pts)
    hard_tree, soft_tree = cKDTree(hard_pts), cKDTree(soft_pts)

    # the survey points that matter: well observed, on or near the plot
    P, C = sv.P, sv.col
    keep = (sv.track >= 3) & (sv.err < 1.5)
    if "--garden" in opt:
        spec = json.load(open(opt["--garden"]))
        plot = next((f["params"]["path"] for f in spec["features"] if f["id"] == "plot_boundary"), None)
        if plot:
            keep &= inside(grown(plot, 3.0), P[:, 0], P[:, 1])
    lo, hi = hard_pts.min(0) - 0.5, hard_pts.max(0) + 0.5
    keep &= np.all((P > lo) & (P < hi), axis=1)
    Q, QC = P[keep], C[keep].astype(float)
    dh, ih = hard_tree.query(Q, distance_upper_bound=3.0)
    ds, _ = soft_tree.query(Q, distance_upper_bound=3.0)
    explained = (dh < 0.3) | (ds < 0.6)
    summary = {"points": int(len(Q)), "within_0.1": round(float((dh < 0.1).mean()), 3), "within_0.2": round(float((dh < 0.2).mean()), 3),
               "within_0.3_or_by_a_plant": round(float(explained.mean()), 3)}
    print(f"{len(Q)} survey points on the plot; {100 * summary['within_0.1']:.0f}% within 0.1 m of a hard surface, "
          f"{100 * summary['within_0.2']:.0f}% within 0.2 m; {100 * summary['within_0.3_or_by_a_plant']:.0f}% within 0.3 m or beside a plant")

    # points with no surface, gathered into clusters
    U, UC = Q[~explained], QC[~explained]
    vox = 0.25
    g0 = U.min(0) if len(U) else np.zeros(3)
    ijk = np.floor((U - g0) / vox).astype(int)
    shape = ijk.max(0) + 1 if len(U) else (1, 1, 1)
    occ = np.zeros(shape, np.int32)
    np.add.at(occ, tuple(ijk.T), 1)
    lab, n = ndimage.label(occ >= 2, structure=np.ones((3, 3, 3)))
    pl = lab[tuple(ijk.T)]
    centres = np.array([f["hard"].mean(0) if len(f["hard"]) else f["tri"].reshape(-1, 3).mean(0) for f in feats])
    clusters = []
    for k in range(1, n + 1):
        sel = pl == k
        if sel.sum() < 80:
            continue
        q = U[sel]
        lo3, hi3 = np.percentile(q, 3, axis=0), np.percentile(q, 97, axis=0)
        c = np.median(q, axis=0)
        near = [j for j in np.argsort(np.linalg.norm(centres[:, :2] - c[:2], axis=1))[:6] if feats[j]["id"] != "lawn"][:3]
        clusters.append({"points": int(sel.sum()), "at": [round(float(v), 2) for v in c],
                         "x": [round(float(lo3[0]), 2), round(float(hi3[0]), 2)], "y": [round(float(lo3[1]), 2), round(float(hi3[1]), 2)],
                         "z": [round(float(lo3[2]), 2), round(float(hi3[2]), 2)], "colour": [int(v) for v in UC[sel].mean(0)],
                         "nearest": [{"id": feats[j]["id"], "m": round(float(np.linalg.norm(centres[j, :2] - c[:2])), 2)} for j in near]})
    clusters.sort(key=lambda r: -r["points"])
    print(f"{int((~explained).sum())} points have no surface; {len(clusters)} clusters of 80 or more")
    for r in clusters[:40]:
        near = ", ".join(f"{m['id']} {m['m']} m" for m in r["nearest"])
        print(f"  {r['points']:6d} pts at ({r['at'][0]:6.2f},{r['at'][1]:6.2f})  X {r['x'][0]:6.2f}..{r['x'][1]:6.2f}  Y {r['y'][0]:6.2f}..{r['y'][1]:6.2f}  "
              f"z {r['z'][0]:5.2f}..{r['z'][1]:5.2f}  rgb {r['colour']}  near: {near}")

    # surfaces with no points
    pt_tree = cKDTree(Q)
    rows = []
    for k, f in enumerate(feats):
        if f["id"] == "lawn" or len(f["hard"]) < 30 or f["layer"].startswith("inside"):
            continue
        h = f["hard"][::max(1, len(f["hard"]) // 4000)]
        vis = seen_by(sv, h)
        if vis.sum() < 20:
            rows.append({"id": f["id"], "label": f["label"], "layer": f["layer"], "seen": round(float(vis.mean()), 2), "support": None})
            continue
        d, _ = pt_tree.query(h[vis], distance_upper_bound=0.2)
        rows.append({"id": f["id"], "label": f["label"], "layer": f["layer"], "seen": round(float(vis.mean()), 2),
                     "support": round(float(np.isfinite(d).mean()), 2), "at": [round(float(v), 2) for v in centres[k]]})
    rated = sorted((r for r in rows if r["support"] is not None), key=lambda r: r["support"])
    print("features with the least of their surface confirmed by survey points (of the parts three or more frames looked at):")
    for r in rated[:40]:
        print(f"  {100 * r['support']:3.0f}% confirmed, {100 * r['seen']:3.0f}% looked at   {r['id']:24s} {r['label'][:50]}")
    unseen = [r["id"] for r in rows if r["support"] is None]
    print(f"not looked at by any three frames from within 7 m ({len(unseen)}): " + ", ".join(unseen))
    json.dump({"summary": summary, "clusters": clusters, "features": rows}, open(os.path.join(out_dir, "fit.json"), "w"), indent=1)

    # a plan
    from PIL import Image, ImageDraw
    x0, x1, y0, y1, res = float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), 0.04
    img = np.full((int((y1 - y0) / res) + 1, int((x1 - x0) / res) + 1, 3), 18, np.uint8)

    def put(pts, colour):
        ix, iy = ((pts[:, 0] - x0) / res).astype(int), ((y1 - pts[:, 1]) / res).astype(int)
        ok = (ix >= 0) & (ix < img.shape[1]) & (iy >= 0) & (iy < img.shape[0])
        img[iy[ok], ix[ok]] = colour

    put(Q[explained], (70, 78, 70))
    put(U, (255, 70, 60))
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for r in rated:
        if "at" in r:
            col = (255, 200, 0) if r["support"] < 0.25 else (90, 200, 255) if r["support"] < 0.6 else (90, 255, 120)
            px, py = (r["at"][0] - x0) / res, (y1 - r["at"][1]) / res
            d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=col, width=2)
            if r["support"] < 0.25:
                d.text((px + 6, py - 5), r["id"], fill=col)
    for n_, r in enumerate(clusters[:40], 1):
        d.rectangle([(r["x"][0] - x0) / res, (y1 - r["y"][1]) / res, (r["x"][1] - x0) / res, (y1 - r["y"][0]) / res], outline=(255, 255, 255))
        d.text(((r["x"][0] - x0) / res, (y1 - r["y"][1]) / res - 11), str(n_), fill=(255, 255, 255))
    im.save(os.path.join(out_dir, "fit_plan.png"))

    # what are the clusters? crops of the footage
    for n_, r in enumerate(clusters[:int(opt.get("--sheets", 0))], 1):
        c = np.array(r["at"])
        best = None
        for i in range(len(sv.names)):
            W, H = sv.frame_size(i)
            u, v, z = (float(t) for t in sv.project(i, c))
            if 0.8 < z < 9 and 250 < u < W - 250 and 250 < v < H - 250 and (best is None or z < best[0]):
                best = (z, i, u, v)
        if best is None:
            continue
        z, i, u, v = best
        im = sv.image(i)
        d = ImageDraw.Draw(im)
        corners = np.array([[x, y, zz] for x in r["x"] for y in r["y"] for zz in r["z"]])
        uu, vv, zz = sv.project(i, corners)
        if (zz > 0.2).all():
            d.rectangle([float(uu.min()), float(vv.min()), float(uu.max()), float(vv.max())], outline=(255, 0, 255), width=5)
        half = int(np.clip(1.2 * max(float(uu.max() - uu.min()), float(vv.max() - vv.min())), 260, 620))
        box = (int(max(0, u - half)), int(max(0, v - half)), int(min(im.width, u + half)), int(min(im.height, v + half)))
        im.crop(box).resize((520, int(520 * (box[3] - box[1]) / (box[2] - box[0])))).save(os.path.join(out_dir, f"cluster_{n_:02d}_{sv.names[i][:-4]}.jpg"), quality=85)


if __name__ == "__main__":
    main()
