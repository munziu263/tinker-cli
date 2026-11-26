"""Build the garden model from garden.json.

    python build.py                 # writes out/garden.blend, out/garden.glb and viewer/model/

Runs with Blender as a Python module (pip install bpy) or inside Blender itself:
    blender --background --python build.py
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from gardenkit import core, ground, interior, objects, plants, structures  # noqa: E402

LAYERS = {
    "ground": "Ground",
    "boundaries": "Boundaries",
    "buildings": "Buildings",
    "planting": "Planting",
    "objects": "Furniture and objects",
    "context": "Surroundings",
    "survey": "Survey data",
    "gaps": "Still to survey",
    "inside_gf": "Inside: ground floor",
    "inside_ff": "Inside: first floor",
}

BUILDERS = {
    # boundaries
    "brick_wall": structures.brick_wall,
    "board_fence": structures.board_fence,
    "post_and_wire": structures.post_and_wire,
    "gate_posts": structures.gate_posts,
    "hedge": plants.hedge,
    "ivy": plants.ivy_on_wall,
    # buildings
    "outbuilding": structures.outbuilding,
    "timber_shed": structures.timber_shed,
    "weatherboard_block": structures.weatherboard_block,
    "catslide_wing": structures.catslide_wing,
    "cottage": structures.cottage,
    "ground_line": structures.ground_line,
    "ground_patch": structures.ground_patch,
    "marker_post": structures.marker_post,
    "manhole": structures.manhole,
    "stump": objects.stump,
    "compost_bin": objects.compost_bin,
    "raised_bed": objects.raised_bed,
    "toys": objects.toys,
    "flag_patio": lambda **kw: structures.patio(ground.flag_paving(**kw)),
    # planting
    "willow_pollard": plants.willow_pollard,
    "broadleaf": plants.broadleaf,
    "hazel": plants.hazel,
    "multistem_shrub": plants.multistem_shrub,
    "small_tree": plants.small_tree,
    "sapling": plants.sapling,
    "shrub": plants.shrub,
    "clump": plants.clump,
    "bare_stems": plants.bare_stems,
    "border": plants.border,
    "olive_pot": plants.olive_pot,
    "tree_belt": plants.tree_belt,
    "conifer": plants.conifer,
    "brush_pile": plants.brush_pile,
    # furniture and objects
    "picnic_table": objects.picnic_table,
    "garden_bench": objects.garden_bench,
    "bistro_table": objects.bistro_table,
    "folding_table_wood": objects.folding_table_wood,
    "monobloc_chair": objects.monobloc_chair,
    "folding_chair": objects.folding_chair,
    "step_stool": objects.step_stool,
    "firewood_bag": objects.firewood_bag,
    "log_stack": objects.log_stack,
    "log_rounds": objects.log_rounds,
    "plank_slab": objects.plank_slab,
    "sheet": objects.sheet,
    "leaning_boards": objects.leaning_boards,
    "pot": objects.pot,
    "pot_row": objects.pot_row,
    "trough": objects.trough,
    "water_butt": objects.water_butt,
    "barrel_lying": objects.barrel_lying,
    "metal_bin": objects.metal_bin,
    "bbq_covered": objects.bbq_covered,
    "tarp_heap": objects.tarp_heap,
    "lump": objects.lump,
    "ball": objects.ball,
    "hose_coil": objects.hose_coil,
    "pop_up_goal": objects.pop_up_goal,
    "stake": objects.stake,
    "brick_edging": objects.brick_edging,
    # surroundings
    "telegraph_pole": objects.telegraph_pole,
    "sign_board": objects.sign_board,
    "goal_posts": objects.goal_posts,
    "massing": objects.massing,
    # inside a building
    "walls": interior.walls,
    "floor": interior.floor,
    "stairs": interior.stairs,
    "beams": interior.beams,
    "fittings": interior.fittings,
}


def _tuplify(v):
    """JSON lists -> tuples where builders expect them (dict keys stay as they are)."""
    if isinstance(v, list):
        return [_tuplify(x) for x in v]
    if isinstance(v, dict):
        return {k: _tuplify(x) for k, x in v.items()}
    return v


def build_scene(spec, tex_dir):
    core.init(tex_dir)
    scene = bpy.context.scene
    scene.name = "Garden"
    scene.unit_settings.system = "METRIC"
    root = core.get_collection("Garden")
    cols, parents = {}, {}
    for key, label in LAYERS.items():
        cols[key] = core.get_collection(label, root)
        parents[key] = core.add_empty(label, cols[key])
        parents[key]["layer"] = key

    stats = {k: [0, 0] for k in LAYERS}
    g = spec["ground"]
    gm = ground.ground(tuple(g["extent"]), g["zones"], tex_dir, g.get("px_per_m", 64))
    core.add_object("lawn", gm, cols["ground"], parents["ground"],
                    props={"label": "Lawn and ground", "layer": "ground", "kind": "ground", "confidence": "measured",
                           "notes": "Painted from the survey: grass, bare soil under the pollarded tree, leaf litter along the walls, gravel, the road. Drawn flat: the measured levels put the whole garden within 0.3 m of level."})
    stats["ground"][0] += 1
    stats["ground"][1] += gm.tri_count

    for f in spec["features"]:
        t0 = time.time()
        fn = BUILDERS[f["type"]]
        mesh = fn(**_tuplify(f.get("params", {})))
        at = f.get("at", [0.0, 0.0])
        lo, hi = mesh.bounds()
        props = {
            "label": f.get("label", f["id"]),
            "layer": f["layer"],
            "kind": f["type"],
            "confidence": f.get("confidence", "estimated"),
            "notes": f.get("notes", ""),
            "size_w": round(float(hi[0] - lo[0]), 2),
            "size_d": round(float(hi[1] - lo[1]), 2),
            "size_h": round(float(hi[2]), 2),
        }
        core.add_object(f["id"], mesh, cols[f["layer"]], parents[f["layer"]],
                        location=(at[0], at[1], at[2] if len(at) > 2 else 0.0),
                        rot_deg=f.get("rot", 0.0), props=props)
        stats[f["layer"]][0] += 1
        stats[f["layer"]][1] += mesh.tri_count
        dt = time.time() - t0
        if dt > 1.5:
            print(f"  {f['id']}: {mesh.tri_count} tris, {dt:.1f}s", flush=True)

    # where the pan was filmed from, as a camera you can look through in Blender
    vp = spec["meta"]["viewpoint"]
    cam_data = bpy.data.cameras.new("Video viewpoint")
    cam_data.sensor_width = 36.0
    cam_data.lens = 36.0 / (2 * math.tan(math.radians(62.7) / 2))   # 62.7 deg across, as filmed
    cam = bpy.data.objects.new("Video viewpoint", cam_data)
    cam.location = (vp["x"], vp["y"], vp["eye"])
    cam.rotation_euler = (math.radians(90 - 12), 0.0, math.radians(vp.get("heading_deg", -90.0) - 90))  # heading: plan angle from +X
    root.objects.link(cam)
    scene.camera = cam
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(8)
    sun = bpy.data.objects.new("Sun", sun_data)
    sun.rotation_euler = (math.radians(58), 0.0, math.radians(35))
    root.objects.link(sun)
    world = bpy.data.worlds.new("Overcast sky")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.62, 0.70, 0.80, 1.0)
        bg.inputs[1].default_value = 1.0
    scene.world = world
    scene["garden_meta"] = json.dumps(spec["meta"])
    return stats


def _fix_materials(js):
    """Leaf and twig cards: alpha-tested and two-sided (the exporter writes them as BLEND)."""
    n = 0
    for m in js.get("materials", []):
        if m["name"].startswith(("leaf_", "twigs_")):
            m["alphaMode"] = "MASK"
            m["alphaCutoff"] = 0.4
            m["doubleSided"] = True
            n += 1
        else:
            m.pop("alphaMode", None)
    return n


def fix_glb(path):
    with open(path, "rb") as fh:
        data = fh.read()
    magic, version, length = struct.unpack("<4sII", data[:12])
    assert magic == b"glTF"
    off, chunks = 12, []
    while off < length:
        clen, ctype = struct.unpack("<I4s", data[off:off + 8])
        chunks.append((ctype, data[off + 8:off + 8 + clen]))
        off += 8 + clen
    js = json.loads(chunks[0][1].decode("utf8"))
    n = _fix_materials(js)
    out_json = json.dumps(js, separators=(",", ":")).encode("utf8")
    out_json += b" " * ((4 - len(out_json) % 4) % 4)
    body = struct.pack("<I4s", len(out_json), b"JSON") + out_json
    for ctype, payload in chunks[1:]:
        payload += b"\x00" * ((4 - len(payload) % 4) % 4)
        body += struct.pack("<I4s", len(payload), ctype) + payload
    with open(path, "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return n, js


def export_web_model(model_dir):
    """The viewer's copy of the model, as plain web file types: scene description as JSON,
    textures as ordinary image files, and the geometry buffer base64-encoded in a text file
    (static hosts that refuse .glb/.bin still serve these)."""
    import base64
    import shutil
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    gltf = os.path.join(model_dir, "garden.gltf")
    bpy.ops.export_scene.gltf(
        filepath=gltf, export_format="GLTF_SEPARATE", export_texture_dir="tex", export_extras=True,
        export_yup=True, export_apply=False, export_cameras=False, export_lights=False,
        export_image_format="AUTO", export_jpeg_quality=85, export_tangents=False,
        export_materials="EXPORT", export_texcoords=True, export_normals=True,
    )
    js = json.load(open(gltf))
    _fix_materials(js)
    bin_path = os.path.join(model_dir, js["buffers"][0]["uri"])
    raw = open(bin_path, "rb").read()
    js["buffers"] = [{"byteLength": len(raw)}]
    with open(os.path.join(model_dir, "garden.gltf.json"), "w") as fh:
        json.dump(js, fh, separators=(",", ":"))
    with open(os.path.join(model_dir, "geometry.b64.txt"), "wb") as fh:
        fh.write(base64.b64encode(raw))
    os.remove(gltf)
    os.remove(bin_path)
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(model_dir) for f in fs]
    return len(files), sum(os.path.getsize(f) for f in files), js


def main():
    t0 = time.time()
    spec = json.load(open(os.path.join(HERE, "garden.json")))
    inside = os.path.join(HERE, "interior.json")      # optional and kept out of git: the inside of the house
    if os.path.exists(inside):
        extra = json.load(open(inside))["features"]
        spec["features"] += extra
        print(f"interior.json: {len(extra)} features")
    out_dir = os.path.join(HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    stats = build_scene(spec, os.path.join(HERE, "textures"))
    total = 0
    for k, (n, tris) in stats.items():
        print(f"{LAYERS[k]:24s} {n:3d} objects {tris:8d} triangles")
        total += tris
    print(f"{'total':24s} {sum(v[0] for v in stats.values()):3d} objects {total:8d} triangles")

    glb = os.path.join(out_dir, "garden.glb")     # one-file copy for other 3D tools
    bpy.ops.export_scene.gltf(
        filepath=glb, export_format="GLB", export_extras=True, export_yup=True, export_apply=False,
        export_cameras=False, export_lights=False, export_image_format="AUTO", export_jpeg_quality=85,
        export_tangents=False, export_materials="EXPORT", export_texcoords=True, export_normals=True,
    )
    n_alpha, js = fix_glb(glb)
    print(f"glb: {os.path.getsize(glb) / 1e6:.2f} MB, {len(js.get('nodes', []))} nodes, "
          f"{len(js.get('materials', []))} materials ({n_alpha} alpha-tested), {len(js.get('images', []))} images")
    n_files, n_bytes, _ = export_web_model(os.path.join(HERE, "viewer", "model"))
    print(f"viewer/model: {n_files} files, {n_bytes / 1e6:.2f} MB")

    blend = os.path.join(out_dir, "garden.blend")
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
    print(f"blend: {os.path.getsize(blend) / 1e6:.2f} MB")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
