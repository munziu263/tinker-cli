"""Register extra clips into an existing COLMAP survey without moving it.

    sfm_add.py <old_work_dir> <new_work_dir> [--standing <clip>] [--link A:B ...] [--resume N]
               <frames_dir>[:every] ... [--then <frames_dir> ...]

The existing reconstruction defines the coordinate frame the whole model is drawn in, so
its cameras are held fixed. Frames from the new clips (the sharpest of every `every`,
3 unless given) are matched to their neighbours and to frames of other clips that show
the same place, and registered into the same reconstruction; all 3D points are then
re-triangulated with the extra baseline the new viewpoints give.

Each directory holds the frames of one clip, named <clip>_<number>.jpg, as ffmpeg writes
them. Clips may be landscape or upright (ffmpeg applies the phone's rotation flag). A clip
whose frames have the same shape as a camera already in the survey is taken to be the same
phone and uses that camera's calibration unchanged; any other shape gets a camera of its
own, to be calibrated along the way.

Which frames of different clips to compare:
  up to four clips    a regular sample of every clip against every other (thorough);
  more than that      for each new frame, the frames elsewhere that look most like it,
                      found by bag-of-visual-words retrieval over the features already
                      extracted (see `similar_frames`). Clips that follow one another in
                      name order are also linked end to start, since a walk filmed in
                      pieces carries on where it stopped.

--link A:B compares two clips in full (every second chosen frame of A against every second
of B, old or new). Retrieval can miss a pair of clips that show the same place from far
apart or in different light; if you know that B was filmed where A stopped, or that both
look at the same corner, say so. May be given many times.

A clip may be given again later with a smaller `every`: frames already in the survey are
kept and the extra ones fill the gaps. That is the cure for a walk that broke off at a
corner turned too quickly for the first sampling.

--then splits the clips into stages. Each stage is registered with everything before it
held fixed, which keeps the adjustment small and the run time roughly linear in the
amount of footage; frames that could not be placed are tried again in later stages.

--standing names a clip in the existing survey that was filmed standing in one place (a
pan). Such a clip fixes directions well and distances badly: here its far points sat 10
to 30% too close. A new frame is placed by the points it recognises, so walking into
that part of the garden it would be placed against positions that are wrong. Points
seen only in a standing clip, and further from it than 3.5 times the distance its camera
moved, are therefore dropped first; the new clips put them back properly.

--resume N carries on a run that was stopped: features and matches in <new_work_dir> are
kept, stages before the Nth are taken as done (their result is <new_work_dir>/stage<N-1>)
and registration starts again at stage N. Give the same clips as before; the stages from N
on may be split differently.
"""
import os
import shutil
import sqlite3
import sys
import time

import numpy as np
import pycolmap
from PIL import Image

FOCAL, RADIAL = 1588.1, 0.017        # the phone's main camera at 1920 x 1440, from the first three clips


def sharpness(path):
    im = Image.open(path).convert("L")
    a = np.asarray(im.resize((480, 360) if im.width >= im.height else (360, 480)), np.float32)
    lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(lap.var())


def pick_sharp(frames_dir, group=3):
    names = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    out = []
    for i in range(0, len(names), group):
        chunk = names[i:i + group]
        out.append(max(chunk, key=lambda n: sharpness(os.path.join(frames_dir, n))))
    return out


def clip_of(name):
    return name.rsplit("_", 1)[0]


# ----------------------------------------------------------------------------- retrieval


def _rootsift(blob, rows, cols, rng, limit):
    D = np.frombuffer(blob, np.uint8).reshape(rows, cols)
    if rows > limit:
        D = D[rng.choice(rows, limit, replace=False)]
    D = D.astype(np.float32)
    return np.sqrt(D / (D.sum(1, keepdims=True) + 1e-9))


def _nearest(D, words):
    """Index of the nearest word for every descriptor (both unit length: use the dot product)."""
    out = np.empty(len(D), np.int32)
    for s in range(0, len(D), 20000):
        out[s:s + 20000] = np.argmax(D[s:s + 20000] @ words.T, axis=1)
    return out


def similar_frames(db_path, queries, k=20, per_clip=6, n_words=2048, per_image=2500, skip_same_clip=12, seed=0):
    """For each query frame, the k frames in the database that look most like it.

    A small visual vocabulary is trained on the spot from the SIFT descriptors in the
    database; every frame becomes a tf-idf histogram of words, and frames are compared by
    the cosine of those. It is crude next to a trained vocabulary tree, but the frames all
    show the same garden, which is the easy case, and it needs nothing downloaded.
    Returns a set of (name_a, name_b) pairs."""
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(db_path)
    names = dict(con.execute("select image_id, name from images"))
    ids = sorted(names)

    def desc(i, limit):
        rows, cols, blob = con.execute("select rows, cols, data from descriptors where image_id=?", (int(i),)).fetchone()
        return _rootsift(blob, rows, cols, rng, limit)

    sample = np.vstack([desc(i, 400) for i in rng.choice(ids, min(len(ids), 400), replace=False)])
    words = sample[rng.choice(len(sample), n_words, replace=False)].copy()
    for _ in range(6):                                   # a few rounds of k-means are enough
        a = _nearest(sample, words)
        for w in np.unique(a):
            words[w] = sample[a == w].mean(0)
        words /= np.linalg.norm(words, axis=1, keepdims=True) + 1e-9
    H = np.zeros((len(ids), n_words), np.float32)
    for r, i in enumerate(ids):
        H[r] = np.bincount(_nearest(desc(i, per_image), words), minlength=n_words)
    con.close()
    idf = np.log(len(ids) / (1.0 + (H > 0).sum(0)))
    V = H / (H.sum(1, keepdims=True) + 1e-9) * idf
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9

    row = {names[i]: r for r, i in enumerate(ids)}
    all_names = [names[i] for i in ids]
    clips = np.array([clip_of(n) for n in all_names])
    order = {n: j for j, n in enumerate(sorted(all_names))}
    pairs = set()
    for q in queries:
        sim = V @ V[row[q]]
        near = np.array([clips[j] == clip_of(q) and abs(order[all_names[j]] - order[q]) <= skip_same_clip for j in range(len(all_names))])
        sim[near] = -1                                   # the sequential pass has those
        taken, used = 0, {}
        for j in np.argsort(-sim):
            if taken >= k or sim[j] <= 0:
                break
            c = clips[j]
            if used.get(c, 0) >= per_clip:               # spread the picks over several clips
                continue
            used[c] = used.get(c, 0) + 1
            taken += 1
            pairs.add(tuple(sorted((q, all_names[j]))))
    return pairs


# ----------------------------------------------------------------------------- weak points


def drop_standing_points(rec, clip, ratio=3.5):
    """Delete the points that only a standing clip saw and that are too far from it to have
    a trustworthy distance. Returns how many were dropped."""
    centres, ids = [], set()
    for iid, im in rec.images.items():
        if im.has_pose and clip_of(im.name) == clip:
            T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            centres.append(-np.asarray(T.rotation.matrix()).T @ np.asarray(T.translation))
            ids.add(iid)
    if len(centres) < 2:
        return 0
    C = np.array(centres)
    moved = float(np.linalg.norm(C.max(0) - C.min(0)))            # how far the camera travelled, near enough
    mid = C.mean(0)
    doomed = [pid for pid, p in rec.points3D.items()
              if np.linalg.norm(np.asarray(p.xyz) - mid) > ratio * moved and all(e.image_id in ids for e in p.track.elements)]
    for pid in doomed:
        rec.delete_point3D(pid)
    return len(doomed)


# ----------------------------------------------------------------------------- main


def parse(args):
    stages, cur, standing, resume, links = [], [], [], 1, []
    args = list(args)
    while args:
        a = args.pop(0)
        if a == "--standing":
            standing.append(args.pop(0))
        elif a == "--resume":
            resume = int(args.pop(0))
        elif a == "--link":
            links.append(tuple(args.pop(0).split(":")))
        elif a == "--then":
            stages.append(cur)
            cur = []
        else:
            d, _, every = a.partition(":")
            cur.append((d, int(every) if every else 3))
    stages.append(cur)
    return [s for s in stages if s], standing, resume, links


def main():
    old, work = sys.argv[1], sys.argv[2]
    stages, standing, resume, links = parse(sys.argv[3:])
    os.makedirs(work, exist_ok=True)
    img_dir = os.path.join(work, "images")
    old_names = sorted(os.listdir(os.path.join(old, "images")))
    old_model = os.path.join(old, "sparse", "0")
    if not os.path.isdir(old_model):                # a refined or extended model sits directly in sparse/
        old_model = os.path.join(old, "sparse")
    if resume == 1:
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir)
        for n in old_names:
            os.symlink(os.path.realpath(os.path.join(old, "images", n)), os.path.join(img_dir, n))
        old_rec = pycolmap.Reconstruction(old_model)
        if standing:                                # start from a copy without the points nobody should lean on
            for clip in standing:
                print(f"standing clip {clip}: {drop_standing_points(old_rec, clip)} far points dropped", flush=True)
            old_model = os.path.join(work, "start")
            os.makedirs(old_model, exist_ok=True)
            old_rec.write(old_model)

    t0 = time.time()
    stage_names, new_by_clip = [], []
    for stage in stages:
        names_here = []
        for c, every in stage:
            names = pick_sharp(c, every)
            for n in names:
                if not os.path.lexists(os.path.join(img_dir, n)):
                    os.symlink(os.path.join(os.path.abspath(c), n), os.path.join(img_dir, n))
            new_by_clip.append(names)
            names_here += names
        stage_names.append(names_here)
    new_names = [n for names in new_by_clip for n in names]
    print(f"{len(new_by_clip)} clips in {len(stages)} stage(s): {len(new_names)} frames chosen ({time.time() - t0:.0f}s)", flush=True)

    db = os.path.join(work, "db.db")
    if resume == 1:
        shutil.copyfile(os.path.join(old, "db.db"), db)

        t0 = time.time()
        ext = pycolmap.FeatureExtractionOptions()
        ext.num_threads = 4
        ext.max_image_size = 1920
        ext.sift.max_num_features = 7000
        by_shape = {}
        for names in new_by_clip:                       # one camera per frame shape
            by_shape.setdefault(Image.open(os.path.join(img_dir, names[0])).size, []).extend(names)
        known = {(c.width, c.height): cid for cid, c in sorted(old_rec.cameras.items())}
        for (w, h), names in by_shape.items():
            reader = pycolmap.ImageReaderOptions()
            if (w, h) in known:                         # the same phone held the same way: its calibration stands
                reader.existing_camera_id = known[(w, h)]
                note = f"camera {known[(w, h)]} of the survey"
            else:
                reader.camera_model = "SIMPLE_RADIAL"
                reader.camera_params = f"{FOCAL},{w / 2},{h / 2},{RADIAL}"
                note = "a new camera"
            pycolmap.extract_features(db, img_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                      reader_options=reader, extraction_options=ext)
            print(f"  {len(names)} frames of {w} x {h}: {note}", flush=True)
        print(f"extract done {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        match = pycolmap.FeatureMatchingOptions()
        match.num_threads = 4
        pair = pycolmap.SequentialPairingOptions()
        pair.overlap = 8
        pair.quadratic_overlap = True
        pycolmap.match_sequential(db, matching_options=match, pairing_options=pair)   # skips pairs already matched
        print(f"sequential matching done {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        if len(new_by_clip) <= 4:
            subsets = [old_names[::4]] + [names[::3] for names in new_by_clip]
            pairs = {(a, b) for i in range(len(subsets)) for j in range(i + 1, len(subsets)) for a in subsets[i] for b in subsets[j]}
            how = "every clip against every other"
        else:
            fresh = [n for n in new_names if n not in set(old_names)]      # frames already in the survey have had their turn
            pairs = similar_frames(db, fresh)
            how = f"by retrieval for {len(fresh)} new frames"
        if links:
            con = sqlite3.connect(db)
            by_clip = {}
            for (n,) in con.execute("select name from images order by name"):
                by_clip.setdefault(clip_of(n), []).append(n)
            con.close()
            before = len(pairs)
            for a, b in links:
                pairs |= {tuple(sorted((x, y))) for x in by_clip.get(a, [])[::2] for y in by_clip.get(b, [])[::2]}
            how += f", plus {len(pairs) - before} from {len(links)} named pair(s) of clips"
        pairs_path = os.path.join(work, "cross_pairs.txt")
        with open(pairs_path, "w") as fh:
            for a, b in sorted(pairs):
                fh.write(f"{a} {b}\n")
        print(f"cross-clip pairs chosen {how}: {len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
        imp = pycolmap.ImportedPairingOptions()
        imp.match_list_path = pairs_path
        pycolmap.match_image_pairs(db, matching_options=match, pairing_options=imp)
        print(f"cross-clip matching done {time.time() - t0:.0f}s", flush=True)

    model, considered = old_model, list(old_names)
    for k, names in enumerate(stage_names):
        t0 = time.time()
        considered += names
        if k + 1 < resume:                                   # done in an earlier run
            model = os.path.join(work, f"stage{k + 1}")
            continue
        opts = pycolmap.IncrementalPipelineOptions()
        opts.num_threads = 4
        opts.multiple_models = False
        opts.fix_existing_frames = True
        opts.constant_cameras = set(pycolmap.Reconstruction(model).cameras.keys())
        opts.image_names = list(considered)                  # later stages wait their turn
        opts.ba_global_frames_ratio = 1.3                    # the fixed frames carry the model: adjust less often
        opts.ba_global_points_ratio = 1.3
        opts.ba_global_max_refinements = 2
        opts.ba_global_max_num_iterations = 30
        opts.ba_global_function_tolerance = 1e-6
        opts.mapper.ba_global_ignore_redundant_points3D = True   # solve with a thinned set of points, then fit the rest
        opts.mapper.filter_min_tri_angle = 0.5
        opts.triangulation.min_angle = 0.5
        out = os.path.join(work, f"stage{k + 1}")
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        recs = pycolmap.incremental_mapping(db, img_dir, out, options=opts, input_path=model)
        rec = recs[0] if 0 in recs else next(iter(recs.values()))
        placed = {im.name for im in rec.images.values() if im.has_pose}
        got = sum(n in placed for n in names)
        print(f"stage {k + 1}: {got} of {len(names)} frames placed, {rec.num_reg_images()} in all, "
              f"{rec.num_points3D()} points, mean reprojection error {rec.compute_mean_reprojection_error():.2f} px "
              f"({time.time() - t0:.0f}s)", flush=True)
        missing = {}
        for n in considered:
            if n not in placed:
                missing[clip_of(n)] = missing.get(clip_of(n), 0) + 1
        if missing:
            print("  not placed so far:", ", ".join(f"{c} ({v})" for c, v in sorted(missing.items())), flush=True)
        model = out if os.path.exists(os.path.join(out, "images.bin")) else os.path.join(out, "0")
    final = os.path.join(work, "sparse")
    if os.path.exists(final):
        shutil.rmtree(final)
    shutil.copytree(model, final)
    print("saved", final, flush=True)


if __name__ == "__main__":
    main()
