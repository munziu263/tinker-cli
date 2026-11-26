"""Draw the model into a frame of the footage, as lines: a check that needs no rendering.

    overlay.py <survey.npz> <garden.json> <frame> <out.jpg> [--ids] -- <frames_dir> [...]

The house is drawn as a wireframe from its measured dimensions (walls, eaves, ridge and
hips, the long rear roof, chimneys, doors and windows), and so are the outbuilding (walls,
roof edge, openings) and the sheds (footprint, eaves, ridge). Walls, hedges and edgings are
drawn along their paths at ground level and at their height; everything else that has a
position gets a cross on the ground (with its id, if --ids is given).

Where the lines sit on the picture, that frame's camera and the model agree. A house
that is drawn off its picture means the camera is wrong (or the survey has drifted on
its way round to that side); a wall that is drawn off its picture while the house fits
means the wall is in the wrong place in the model.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey  # noqa: E402


def house_lines(p):
    """3D polylines of a 'cottage' feature (see gardenkit/structures.py for the parameters)."""
    xn, xs, yw, ye = p["main"]
    oxn, oxs, oyw, oye = p["outshot"]
    prof = sorted([tuple(q) for q in p["profile"]], key=lambda q: -q[1])
    rx, rz, ryw, rye = p["ridge"]["x"], p["ridge"]["z"], p["ridge"]["y_west"], p["ridge"]["y_east"]
    hz, ne = p["hip_z"], p["north_eaves"]

    def south_x(z):
        for (xa, za), (xb, zb) in zip(prof[:-1], prof[1:]):
            if zb <= z <= za:
                return xa + (xb - xa) * (za - z) / (za - zb)
        return prof[-1][0]

    def north_x(z):
        return rx + (ne["x"] - rx) * (rz - z) / (rz - ne["z"])

    se = prof[0][1]
    for (xa, za), (xb, zb) in zip(prof[:-1], prof[1:]):
        if xa <= xs <= xb:
            se = za + (zb - za) * (xs - xa) / (xb - xa)          # eaves height where the slope crosses the south wall
    L = []
    for y in (yw, ye):                                            # the two gable ends
        L.append([(xs, y, 0), (xs, y, se), (south_x(hz), y, hz), (north_x(hz), y, hz), (xn, y, ne["z"]), (xn, y, 0), (xs, y, 0)])
    L.append([(xs, yw, se), (xs, ye, se)])                        # south eaves of the main range
    L.append([(ne["x"], yw, ne["z"]), (ne["x"], ye, ne["z"])])    # north eaves
    L.append([(xn, yw, 0), (xn, ye, 0)])
    L.append([(rx, ryw, rz), (rx, rye, rz)])                      # ridge and the four hips
    for y, ry in ((yw, ryw), (ye, rye)):
        L.append([(south_x(hz), y, hz), (rx, ry, rz), (north_x(hz), y, hz)])
    for y in (oyw, oye):                                          # the rear wing under its long roof
        L.append([(x, y, z) for x, z in prof if z <= se + 0.3] + [(oxs, y, 0)])
    L.append([(prof[-1][0], oyw, prof[-1][1]), (prof[-1][0], oye, prof[-1][1])])
    L.append([(oxs, oyw, 0), (oxs, oye, 0)])
    for c in p.get("chimneys", []):
        cxn, cxs, cyw, cye = c["box"]
        z0, z1 = c.get("z0", 0.0), c["top"]
        ring = [(cxn, cyw), (cxs, cyw), (cxs, cye), (cxn, cye), (cxn, cyw)]
        L.append([(x, y, z1) for x, y in ring])
        for x, y in ring[:4]:
            L.append([(x, y, z0), (x, y, z1)])
    faces = {"south": lambda a: (xs, yw + a), "north": lambda a: (xn, ye - a), "east": lambda a: (xs - a, ye), "west": lambda a: (xn + a, yw)}
    for o in list(p.get("openings", [])) + list(p.get("oriels", [])):
        if o["face"] in faces:
            (a0, a1), (z0, z1) = o["at"], o["z"]
            (x0, y0), (x1, y1) = faces[o["face"]](a0), faces[o["face"]](a1)
            L.append([(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1), (x0, y0, z0)])
    return L


def outbuilding_lines(p):
    """Walls, roof edge and openings of an 'outbuilding' feature, in its own frame."""
    w, d, hf, hb = p["width"], p["depth"], p["front_h"], p["back_h"]
    br, bl = p.get("back") or ((w, d), (0, d))
    c = [(0.0, 0.0), (w, 0.0), tuple(br), tuple(bl)]
    L = [[(x, y, 0) for x, y in c + c[:1]],
         [(x, y, h) for (x, y), h in zip(c + c[:1], (hf, hf, hb, hb, hf))]]
    L += [[(x, y, 0), (x, y, h)] for (x, y), h in zip(c, (hf, hf, hb, hb))]

    def along(k):
        (x0, y0), (x1, y1) = c[k], c[(k + 1) % 4]
        n = np.hypot(x1 - x0, y1 - y0)
        return lambda a: (x0 + (x1 - x0) * a / n, y0 + (y1 - y0) * a / n)

    face = {"front": along(0), "right": along(1), "back": along(2), "left": along(3)}
    for o in p.get("openings", []):
        (a0, a1), (z0, z1) = o["at"], o["z"]
        (x0, y0), (x1, y1) = face[o["face"]](a0), face[o["face"]](a1)
        L.append([(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1), (x0, y0, z0)])
    return L


def shed_lines(p):
    """Footprint, eaves and ridge of a 'timber_shed' feature (the ridge runs along its depth)."""
    w, d, e, r = p["width"] / 2, p["depth"] / 2, p["eaves"], p["ridge"]
    L = [[(-w, -d, z), (w, -d, z), (w, d, z), (-w, d, z), (-w, -d, z)] for z in (0, e)]
    L += [[(x, y, 0), (x, y, e)] for x in (-w, w) for y in (-d, d)]
    L += [[(0, -d, r), (0, d, r)]] + [[(-w, y, e), (0, y, r), (w, y, e)] for y in (-d, d)]
    return L


def placed(lines, at, rot):
    a = np.radians(rot)
    c, s = np.cos(a), np.sin(a)
    return [[(at[0] + c * x - s * y, at[1] + s * x + c * y, z + (at[2] if len(at) > 2 else 0.0)) for x, y, z in ln] for ln in lines]


def model_lines(spec):
    """(polylines, colour) groups and ground marks for everything in the model."""
    groups, marks = [], []
    for f in spec["features"]:
        if f["layer"] in ("survey", "gaps", "context") or f["layer"].startswith("inside"):
            continue
        p, at, rot = f.get("params", {}), f.get("at", [0, 0]), f.get("rot", 0)
        if f["type"] == "cottage":
            groups.append((placed(house_lines(p), at, rot), (255, 0, 255)))
        elif f["type"] == "outbuilding":
            groups.append((placed(outbuilding_lines(p), at, rot), (255, 0, 255)))
        elif f["type"] == "timber_shed":
            groups.append((placed(shed_lines(p), at, rot), (255, 0, 255)))
        elif isinstance(p.get("path"), list) and p["path"] and isinstance(p["path"][0], list):
            h = p.get("height", p.get("wall_h", 0.0))
            lines = [[(x, y, 0.05) for x, y in p["path"]]]
            if h:
                lines.append([(x, y, h) for x, y in p["path"]])
            groups.append((placed(lines, at, rot), (255, 220, 0)))
        elif isinstance(p.get("outline"), list):
            groups.append((placed([[(x, y, 0.05) for x, y in p["outline"] + p["outline"][:1]]], at, rot), (0, 255, 255)))
        elif "at" in f and f["at"][:2] != [0, 0]:
            marks.append((f["id"], (at[0], at[1], at[2] if len(at) > 2 else 0.05)))
    return groups, marks


def draw(sv, frame, spec, ids=False, scale=0.6):
    i = sv.idx(frame)
    im = sv.image(i)
    W, H = im.size
    d = ImageDraw.Draw(im)
    groups, marks = model_lines(spec)
    for lines, colour in groups:
        for ln in lines:
            run = []
            for a, b in zip(ln[:-1], ln[1:]):
                for t in np.linspace(0, 1, 16):
                    q = np.array(a, float) + t * (np.array(b, float) - np.array(a, float))
                    u, v, depth = sv.project(i, q)
                    if depth > 0.3 and -W < u < 2 * W and -H < v < 2 * H:
                        run.append((float(u), float(v)))
                    else:
                        if len(run) > 1:
                            d.line(run, fill=colour, width=4)
                        run = []
            if len(run) > 1:
                d.line(run, fill=colour, width=4)
    for name, q in marks:
        u, v, depth = sv.project(i, np.array(q, float))
        if depth > 0.3 and 0 < u < W and 0 < v < H:
            r = max(6, int(40 / max(depth, 1.0)))
            d.line([(u - r, v), (u + r, v)], fill=(0, 255, 0), width=3)
            d.line([(u, v - r), (u, v + r)], fill=(0, 255, 0), width=3)
            if ids:
                d.text((u + r + 2, v - 6), name, fill=(0, 255, 0))
    return im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)


def main():
    a = sys.argv[1:]
    dirs = a[a.index("--") + 1:] if "--" in a else []
    a = a[:a.index("--")] if "--" in a else a
    ids = "--ids" in a
    a = [x for x in a if x != "--ids"]
    sv = Survey(a[0], dirs)
    spec = json.load(open(a[1]))
    frame = int(a[2]) if a[2].isdigit() else a[2]
    draw(sv, frame, spec, ids).save(a[3], quality=88)
    print(a[3])


if __name__ == "__main__":
    main()
