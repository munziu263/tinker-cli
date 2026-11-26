"""Register extra clips into an existing COLMAP survey without moving it.

    sfm_add.py <old_work_dir> <new_work_dir> <frames_dir> [<frames_dir> ...]

The first pan defines the coordinate frame the whole garden model is drawn in, so its
cameras are held fixed. Frames from the new clips (the sharpest of every three) get
their own camera model, are matched to their neighbours and across clips, and are
registered into the same reconstruction; all 3D points are then re-triangulated with
the extra baseline the new viewpoints give.
"""
import os
import shutil
import sys
import time

import numpy as np
import pycolmap
from PIL import Image


def sharpness(path):
    a = np.asarray(Image.open(path).convert("L").resize((480, 360)), np.float32)
    lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    return float(lap.var())


def pick_sharp(frames_dir, group=3):
    names = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    out = []
    for i in range(0, len(names), group):
        chunk = names[i:i + group]
        out.append(max(chunk, key=lambda n: sharpness(os.path.join(frames_dir, n))))
    return out


def main():
    old, work, clips = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(work, exist_ok=True)
    img_dir = os.path.join(work, "images")
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)
    old_names = sorted(os.listdir(os.path.join(old, "images")))
    for n in old_names:
        os.symlink(os.path.realpath(os.path.join(old, "images", n)), os.path.join(img_dir, n))
    new_by_clip = []
    for c in clips:
        t0 = time.time()
        names = pick_sharp(c)
        for n in names:
            os.symlink(os.path.join(os.path.abspath(c), n), os.path.join(img_dir, n))
        new_by_clip.append(names)
        print(f"{c}: {len(names)} frames chosen ({time.time() - t0:.0f}s)", flush=True)
    new_names = [n for names in new_by_clip for n in names]

    db = os.path.join(work, "db.db")
    shutil.copyfile(os.path.join(old, "db.db"), db)

    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = "1575.4,960,720,0.011"     # start from the first pan's calibration
    ext = pycolmap.FeatureExtractionOptions()
    ext.num_threads = 4
    ext.max_image_size = 1920
    ext.sift.max_num_features = 7000
    pycolmap.extract_features(db, img_dir, image_names=new_names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext)
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
    subsets = [old_names[::4]] + [names[::3] for names in new_by_clip]
    pairs_path = os.path.join(work, "cross_pairs.txt")
    n_pairs = 0
    with open(pairs_path, "w") as fh:
        for i in range(len(subsets)):
            for j in range(i + 1, len(subsets)):
                for a in subsets[i]:
                    for b in subsets[j]:
                        fh.write(f"{a} {b}\n")
                        n_pairs += 1
    imp = pycolmap.ImportedPairingOptions()
    imp.match_list_path = pairs_path
    pycolmap.match_image_pairs(db, matching_options=match, pairing_options=imp)
    print(f"cross-clip matching done: {n_pairs} pairs, {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = 4
    opts.multiple_models = False
    opts.fix_existing_frames = True
    opts.constant_cameras = {1}
    opts.mapper.filter_min_tri_angle = 0.5
    opts.triangulation.min_angle = 0.5
    out = os.path.join(work, "sparse")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    recs = pycolmap.incremental_mapping(db, img_dir, out, options=opts,
                                        input_path=os.path.join(old, "sparse", "0"))
    print(f"mapping done {time.time() - t0:.0f}s", flush=True)
    for k, r in recs.items():
        print("model", k, r.summary(), flush=True)


if __name__ == "__main__":
    main()
