"""From a folder of phone photographs to a reconstruction the survey tools can read.

    photos.py intake <photos_dir> <work_dir> [--name <shoot>] [--max-side 0] [--quality 92]
    photos.py sfm    <work_dir> [--max-size 3200] [--features 10000] [--pairs all|<n>] [--lens <lens.json>] [--threads <n>]

The first survey was made from frames of video. Photographs are the better material (see
CAPTURE.md) and come differently: each file by itself, in the phone's own format, held
any way up, in no order that can be relied on. These two steps turn a folder of them into
what the rest of the chain expects.

intake  reads every photograph in <photos_dir> (HEIC, JPEG, PNG; nothing in it is changed)
        and writes <work_dir>/images/<shoot>_<number>.jpg: the picture the right way up,
        numbered in the order taken, with nothing of the file's own record left in it (a
        phone writes where it was into every photograph). What the reconstruction needs
        from that record goes into <work_dir>/intake.json instead: which file each picture
        came from, when it was taken, how the phone was held, and the lens. --name is the
        shoot's name (default: the folder's), which then does duty as the "clip" name in
        every other tool. Running it again adds photographs that have appeared since and
        leaves the numbers already given alone; another folder goes into the same
        <work_dir> under another name. --max-side shrinks the pictures written (the longer
        side, in pixels) when disk space matters more than the last of the detail.

        It also says what looks wrong with the set: pictures that are copies shrunk by a
        messaging app, a zoom or a second lens mixed in, long exposures, and the pictures
        that are much less sharp than the rest.

sfm     reconstructs the pictures in <work_dir>/images by themselves: features, matching,
        cameras and points, in <work_dir>/sparse/<n> (0 is not always the biggest: the
        printout says which is). Pictures of one shape taken with one lens share a camera,
        started from the focal length the phone recorded and then refined along with
        everything else.
        --pairs  which pictures are compared. "all" compares every pair and misses
                 nothing, at a cost that grows with the square of the number (139
                 pictures: 13 minutes on four slow cores). A number n compares each
                 picture with those taken just before and after it and with the n that
                 look most like it (the retrieval sfm_add.py uses): three times quicker
                 on that set, and it missed the one weak link between two halves of it,
                 which came out as two models. Default: all up to 500 pictures, 40
                 beyond. If a run with a number gives more than one model, the cure is
                 "all", overnight if need be.
        --lens   takes the lens from an earlier run's lens.json and holds it. A few dozen
                 pictures of one flat wall cannot calibrate a lens, and a set that mixes
                 upright and level pictures calibrates each shape separately; in both
                 cases a second run with the lens from the largest good set is better.
        --max-size, --features  how finely each picture is searched for things to
                 recognise: the longer side it is shrunk to first, and how many it may
                 keep. --threads: how many pictures are searched at once. Each takes
                 memory in proportion to its size, so the default is as many as the
                 machine's memory allows, not as many as it has cores.
        The result has the right shape and no size or direction of its own. Next:

            markers.py detect <work_dir>/markers.json <work_dir>/images
            align.py <work_dir>/sparse/0 <out.npz> --markers <work_dir>/markers.json
                 a frame of its own, upright, sized by the printed markers; then
                 `markers.py locate` checks it against taped distances
            place.py <work_dir>/sparse/0 <survey.npz> <out.npz> --control <control.json>
                 onto the site grid, by markers whose places are known or picked points
"""
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time

import numpy as np
from PIL import Image, ImageOps

try:                                                # the format phones use by default
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:                                 # JPEGs can still be taken in
    pillow_heif = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KINDS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
FILM_DIAGONAL = 43.2666                             # mm: what "35 mm equivalent" focal lengths are measured against
PER_PICTURE_GB = 2.0                                # memory to search one picture of 3200 px for features (measured: see threads_for)
Image.MAX_IMAGE_PIXELS = 200_000_000


def options(a, flags):
    """Take --key value pairs out of an argument list."""
    opt = {}
    for key in flags:
        if key in a:
            i = a.index(key)
            opt[key] = a[i + 1]
            del a[i:i + 2]
    return opt


def record_of(path):
    """What a photograph says about how it was taken. Never where: the location block is not read."""
    with Image.open(path) as im:
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        num = lambda v: None if v is None else float(v)
        taken = str(sub.get(0x9003) or ex.get(0x0132) or "").strip()
        if taken and sub.get(0x9291):
            taken += "." + str(sub.get(0x9291)).strip()
        # how the phone was held: 1 level, 6 or 8 upright, 3 upside down (a HEIC's reader has already turned it and says how)
        return dict(taken=taken, turn=int(im.info.get("original_orientation") or ex.get(0x0112) or 1),
                    camera=" ".join(str(ex.get(t) or "").strip() for t in (0x010F, 0x0110)).strip(),
                    lens=str(sub.get(0xA434) or "").strip(), focal35=num(sub.get(0xA405)), focal_mm=num(sub.get(0x920A)),
                    exposure_s=num(sub.get(0x829A)), iso=num(sub.get(0x8827)), zoom=num(sub.get(0xA404)))


def convert(job):
    """One photograph -> an upright JPEG with no record in it. Returns what there is to say about it."""
    src, dst, max_side, quality = job
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)        # the right way up (a HEIC is already turned by its reader)
            icc = im.info.get("icc_profile")
            im = im.convert("RGB")
            if max_side and max(im.size) > max_side:
                k = max_side / max(im.size)
                im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
            im.save(dst, "JPEG", quality=quality, icc_profile=icc)
            size = list(im.size)
            # how sharp: the spread of the second difference of a small grey copy
            g = np.asarray(im.convert("L").resize((1200, round(1200 * im.height / im.width)), Image.BILINEAR), np.float32)
    except Exception as e:                          # a file cut short in the copying, say: say so and carry on with the rest
        if os.path.exists(dst):
            os.remove(dst)
        return dict(failed=f"{type(e).__name__}: {str(e)[:60]}")
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return dict(size=size, sharp=round(float(lap.var()), 1))


def intake(a):
    opt = options(a, ("--name", "--max-side", "--quality"))
    src_dir, work = a[0], a[1]
    shoot = opt.get("--name") or os.path.basename(os.path.abspath(src_dir))
    shoot = re.sub(r"[^a-z0-9]+", "-", shoot.lower()).strip("-")       # safe in every tool's lists of clips
    if not shoot:
        raise SystemExit("give the shoot a name with --name (letters, digits, hyphens)")
    max_side, quality = int(opt.get("--max-side", 0)), int(opt.get("--quality", 92))
    img_dir = os.path.join(work, "images")
    os.makedirs(img_dir, exist_ok=True)
    book_path = os.path.join(work, "intake.json")
    book = json.load(open(book_path)) if os.path.exists(book_path) else {"shoots": {}}
    mine = book["shoots"].setdefault(shoot, {"photos": []})
    known = {p["source"]: p for p in mine["photos"]}

    files = sorted(f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in KINDS and not f.startswith("."))
    other = sorted({os.path.splitext(f)[1].lower() for f in os.listdir(src_dir)} - KINDS - {""})
    if any(os.path.splitext(f)[1].lower() in (".heic", ".heif") for f in files) and pillow_heif is None:
        raise SystemExit("these are HEIC photographs and pillow-heif is not installed: pip install pillow-heif")
    new, unread, seen_hash = [], [], {p.get("sha1"): p["source"] for p in mine["photos"]}
    for f in files:
        if f in known and os.path.exists(os.path.join(img_dir, known[f]["name"])):
            continue
        path = os.path.join(src_dir, f)
        with open(path, "rb") as fh:
            digest = hashlib.sha1(fh.read()).hexdigest()
        if digest in seen_hash and seen_hash[digest] != f:
            print(f"   {f} is the same file as {seen_hash[digest]}: passed over")
            continue
        seen_hash[digest] = f
        try:
            rec = record_of(path)
        except Exception as e:                      # not a picture after all, or one the reader cannot open
            unread.append(f"{f} ({type(e).__name__})")
            continue
        new.append(dict(rec, source=f, sha1=digest))
    # in the order taken; files with no time of their own after those, by name
    new.sort(key=lambda p: (p["taken"] == "", p["taken"], p["source"]))
    last = max([int(p["name"].rsplit("_", 1)[1].split(".")[0]) for p in mine["photos"]] or [0])
    jobs = []
    for p in new:
        if p["source"] in known:                    # written before and since lost: the same name again
            p["name"] = known[p["source"]]["name"]
        else:
            last += 1
            p["name"] = f"{shoot}_{last:04d}.jpg"
        jobs.append((os.path.join(src_dir, p["source"]), os.path.join(img_dir, p["name"]), max_side, quality))
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        for k, (p, got) in enumerate(zip(new, pool.map(convert, jobs))):
            p.update(got)
            if (k + 1) % 100 == 0:
                print(f"   {k + 1} of {len(new)} photographs", flush=True)
    unread += [f"{p['source']} ({p['failed']})" for p in new if "failed" in p]
    new = [p for p in new if "failed" not in p]
    again = {p["source"] for p in new}
    mine["photos"] = [p for p in mine["photos"] if p["source"] not in again] + new
    mine["photos"].sort(key=lambda p: p["name"])
    mine["from"] = os.path.basename(os.path.abspath(src_dir))
    with open(book_path, "w") as fh:
        json.dump(book, fh, indent=1)

    ph = mine["photos"]
    print(f"shoot '{shoot}': {len(new)} photographs taken in ({time.time() - t0:.0f}s), {len(ph)} in all, as {img_dir}/{shoot}_NNNN.jpg")
    if not ph:
        raise SystemExit(f"no photographs found in {src_dir}")
    if unread:
        print(f"   could not be read: {', '.join(unread[:8])}" + (" ..." if len(unread) > 8 else ""))
    if other:
        print(f"   files of other kinds were left alone: {', '.join(other)}")
    times = sorted(p["taken"] for p in ph if p["taken"])
    if times:
        print(f"   taken from {times[0]} to {times[-1]}")
    shapes = {}
    for p in ph:
        shapes[tuple(p["size"])] = shapes.get(tuple(p["size"]), 0) + 1
    print("   " + "; ".join(f"{n} of {w} x {h} ({w * h / 1e6:.0f} megapixels, {'upright' if h > w else 'level'})" for (w, h), n in sorted(shapes.items(), key=lambda kv: -kv[1])))
    lenses = {}
    for p in ph:
        key = (p["camera"], p["lens"], p["focal35"])
        lenses[key] = lenses.get(key, 0) + 1
    for (camera, lens, f35), n in sorted(lenses.items(), key=lambda kv: -kv[1]):
        print(f"   {n} with {camera or 'a camera that does not say what it is'}" + (f", {lens}" if lens else "") + (f", {f35:.0f} mm equivalent" if f35 else ", focal length not recorded"))
    # what looks wrong
    notes = []
    if len(lenses) > 1:
        notes.append("more than one lens or zoom setting: each gets a camera of its own in the reconstruction. Keep to the main lens at 1x.")
    small = [p for p in ph if p["size"][0] * p["size"][1] < 11e6 and not max_side]
    if small:
        notes.append(f"{len(small)} pictures are under 11 megapixels (first: {small[0]['source']}): shrunk copies, or the camera set low. The originals at 24 megapixels are what is wanted.")
    blank = [p for p in ph if not p["focal35"]]
    if blank:
        notes.append(f"{len(blank)} pictures carry no record of the lens (first: {blank[0]['source']}): they have been through something that strips it, which usually shrinks them too. Their focal length will be a guess to start from.")
    slow = [p for p in ph if p["exposure_s"] and p["exposure_s"] > 1 / 50]
    if slow:
        notes.append(f"{len(slow)} pictures were exposed for longer than 1/50 s (dim light): the hand moves in that time. Brighter light, or stand very still.")
    zoomed = [p for p in ph if p["zoom"] and p["zoom"] > 1.01]
    if zoomed:
        notes.append(f"{len(zoomed)} pictures were taken with digital zoom (first: {zoomed[0]['source']}): a crop, with a different field of view from the rest.")
    sharp = np.array([p["sharp"] for p in ph])
    soft = [p for p in ph if p["sharp"] < 0.3 * np.median(sharp)]
    if soft and len(ph) >= 8:
        notes.append(f"{len(soft)} pictures are much less sharp than the rest (may be blurred, or simply of something plain): " + ", ".join(f"{p['name']} ({p['source']})" for p in sorted(soft, key=lambda p: p["sharp"])[:6]))
    for n in notes:
        print("   NOTE: " + n)
    print(f"next: python tools/survey/markers.py detect {work}/markers.json {img_dir}      and      python tools/survey/photos.py sfm {work}")


# ----------------------------------------------------------------------------- reconstruction


def threads_for(max_size, asked=None):
    """How many pictures to search for features at once. COLMAP's search works on a copy of the picture at twice
    its size, in several layers: about 2 GB for a picture shrunk to 3200 pixels, and in proportion to its area."""
    cores = os.cpu_count() or 2
    if asked:
        return max(1, min(int(asked), cores))
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (ValueError, OSError, AttributeError):
        return max(1, cores // 2)
    each = PER_PICTURE_GB * (max_size / 3200) ** 2
    return max(1, min(cores, int(0.6 * ram / each)))


def first_focal(size, focal35):
    """Where to start the focal length (pixels): from the phone's record, else a wide phone lens."""
    w, h = size
    if focal35:
        return focal35 * float(np.hypot(w, h)) / FILM_DIAGONAL
    return 0.75 * max(w, h)


def sfm(a):
    import pycolmap
    opt = options(a, ("--max-size", "--features", "--pairs", "--lens", "--threads"))
    work = a[0]
    img_dir, db = os.path.join(work, "images"), os.path.join(work, "db.db")
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    if len(names) < 3:
        raise SystemExit(f"{len(names)} pictures in {img_dir}: nothing to reconstruct")
    book_path = os.path.join(work, "intake.json")
    book = {p["name"]: p for s in json.load(open(book_path))["shoots"].values() for p in s["photos"]} if os.path.exists(book_path) else {}
    lens = {}
    if "--lens" in opt:
        with open(opt["--lens"]) as fh:
            for c in sorted(json.load(fh)["cameras"], key=lambda c: c.get("pictures", 0)):       # of two for one lens, the one more pictures made
                lens[(round(max(c["size"]) / min(c["size"]), 2), c.get("focal35"))] = c

    # one camera for each shape of picture taken with one lens
    groups = {}
    for n in names:
        p = book.get(n)
        size = tuple(p["size"]) if p else Image.open(os.path.join(img_dir, n)).size
        groups.setdefault((size, p["focal35"] if p else None, p["lens"] if p else ""), []).append(n)
    for path in (db, db + "-wal", db + "-shm"):
        if os.path.exists(path):
            os.remove(path)
    out = os.path.join(work, "sparse")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    t0 = time.time()
    ext = pycolmap.FeatureExtractionOptions()
    ext.use_gpu = False
    ext.max_image_size = int(opt.get("--max-size", 3200))
    widest = max(max(Image.open(os.path.join(img_dir, g[0])).size) for g in groups.values())
    ext.num_threads = threads_for(min(ext.max_image_size, widest), opt.get("--threads"))
    ext.sift.max_num_features = int(opt.get("--features", 10000))
    held, started = bool(lens), {}
    for (size, f35, lens_name), group in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        w, h = size
        known = lens.get((round(max(size) / min(size), 2), f35))
        if known:
            f, k1 = known["f"] * max(size) / max(known["size"]), known["k1"]
            how = f"lens taken from {opt['--lens']} and held"
        else:
            f, k1 = first_focal(size, f35), 0.0
            how = "focal length started from the phone's record" if f35 else "focal length started from a guess (no record of the lens)"
            if lens:
                held = False
                how += "; NOT in the lens file, so nothing is held in this run"
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        reader.camera_params = f"{f},{w / 2},{h / 2},{k1}"
        pycolmap.extract_features(db, img_dir, image_names=group, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=ext)
        con = sqlite3.connect(db)
        (cid,), = con.execute("select camera_id from images where name = ?", (group[0],))
        con.close()
        started[int(cid)] = (f35, lens_name, f)
        print(f"   {len(group)} pictures of {w} x {h}: {how} ({f:.0f} px)", flush=True)
    print(f"features found in {len(names)} pictures ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    match = pycolmap.FeatureMatchingOptions()
    match.use_gpu = False
    pairs = opt.get("--pairs", "all" if len(names) <= 500 else "40")
    if pairs == "all":
        print(f"comparing every picture with every other: {len(names) * (len(names) - 1) // 2} pairs", flush=True)
        pycolmap.match_exhaustive(db, matching_options=match)
        print(f"every picture compared with every other ({time.time() - t0:.0f}s)", flush=True)
    else:
        from sfm_add import similar_frames
        seq = pycolmap.SequentialPairingOptions()
        seq.overlap = 12
        seq.quadratic_overlap = True
        pycolmap.match_sequential(db, matching_options=match, pairing_options=seq)
        print(f"each picture compared with those taken around it ({time.time() - t0:.0f}s)", flush=True)
        t0 = time.time()
        chosen = similar_frames(db, names, k=int(pairs), per_clip=int(pairs))
        pairs_path = os.path.join(work, "pairs.txt")
        with open(pairs_path, "w") as fh:
            for x, y in sorted(chosen):
                fh.write(f"{x} {y}\n")
        imp = pycolmap.ImportedPairingOptions()
        imp.match_list_path = pairs_path
        pycolmap.match_image_pairs(db, matching_options=match, pairing_options=imp)
        print(f"and with the {pairs} that look most like it: {len(chosen)} more pairs ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    o = pycolmap.IncrementalPipelineOptions()
    o.multiple_models = True                        # if the set does not all join up, say so
    o.max_num_models = 5
    o.min_model_size = max(3, min(10, len(names) // 3))
    o.ba_refine_focal_length = not held
    o.ba_refine_extra_params = not held
    o.ba_refine_principal_point = False
    recs = pycolmap.incremental_mapping(db, img_dir, out, options=o)
    print(f"reconstructed ({time.time() - t0:.0f}s)", flush=True)
    if not recs:
        raise SystemExit("no reconstruction came out: the pictures do not overlap enough to start from. See CAPTURE.md, 'How to shoot'.")
    order = sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images())
    for k, rec in order:
        placed = sorted(im.name for im in rec.images.values() if im.has_pose)
        print(f"model {k}: {len(placed)} of {len(names)} pictures, {rec.num_points3D()} points, mean reprojection error {rec.compute_mean_reprojection_error():.2f} px")
    best_k, best = order[0]
    placed = {im.name for im in best.images.values() if im.has_pose}
    lost = [n for n in names if n not in placed]
    if lost:
        print(f"   not in model {best_k}: " + ", ".join(lost[:20]) + (f" ... and {len(lost) - 20} more" if len(lost) > 20 else ""))
        print("   pictures left out share too little with the rest: too big a step or turn between them and their neighbours."
              + ("" if pairs == "all" else f" Or the link was there and was not looked for: --pairs {pairs} compares only some pairs. Run again with --pairs all to be sure."))
    count = {}
    for im in best.images.values():
        if im.has_pose:
            count[im.camera_id] = count.get(im.camera_id, 0) + 1
    cams = []
    for cid, n in sorted(count.items(), key=lambda kv: -kv[1]):
        c = best.cameras[cid]
        size = (c.width, c.height)
        f35, lens_name, f_first = started[int(cid)]
        f, k1 = float(c.params[0]), float(c.params[3])
        cams.append(dict(size=list(size), focal35=f35, lens=lens_name, f=f, cx=float(c.params[1]), cy=float(c.params[2]), k1=k1, pictures=n,
                         held=bool(held), reprojection_px=float(best.compute_mean_reprojection_error())))
        print(f"   camera for {size[0]} x {size[1]} ({n} pictures): focal length {f:.1f} px = {f / max(size):.4f} of the longer side"
              + (" (held)" if held else f", {(f / f_first - 1) * 100:+.1f}% on where it started") + f"; k1 {k1:+.4f}")
    if not held:
        rel = [c["f"] / max(c["size"]) for c in cams if c["pictures"] >= 5]
        if len(rel) > 1 and max(rel) / min(rel) > 1.005:
            print("   NOTE: the shapes disagree about one and the same lens. Run again with --lens from the larger group to hold them to one calibration.")
        if len(placed) < 40:
            print("   NOTE: this few pictures cannot be trusted to calibrate a lens, least of all on a flat wall. Sizes measured in the plane of a wall "
                  "are sound all the same; for anything more, run again with --lens from a large set that goes round things.")
    with open(os.path.join(work, "lens.json"), "w") as fh:
        json.dump({"from": f"{len(placed)} pictures, model {best_k}", "cameras": cams}, fh, indent=1)
    print(f"the biggest model is {out}/{best_k}; lens written to {work}/lens.json")
    print(f"next: python tools/survey/align.py {out}/{best_k} <out.npz> --markers {work}/markers.json")


if __name__ == "__main__":
    cmd = {"intake": intake, "sfm": sfm}.get(sys.argv[1] if len(sys.argv) > 1 else "")
    if cmd is None:
        raise SystemExit(__doc__)
    cmd(sys.argv[2:])
