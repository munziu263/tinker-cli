"""Reconstruct some of the clips on their own, from features and matches already computed.

    sfm_part.py <work_dir> <out_dir> <clip> [<clip> ...]

sfm_add.py holds the existing survey fixed and hangs new frames on it one at a time. That
is right when the new clips overlap what is already there. It goes wrong in two cases,
both met here:

  * a walk that leaves the surveyed part and comes back (round the back of a building).
    Each new frame is placed from points made by the frames just before it, so the walk's
    scale is free to creep, and with the old frames immovable the adjustment cannot spread
    the error. Here a walk along a wall came out 40% of its true length.
  * clips that share nothing with the survey but a standing pan whose far points are not
    trusted. They cannot be hung on anything.

In both cases reconstruct the clips by themselves, free, and bring the result into the
site grid afterwards: with `align.py --like` on the cameras it shares with the survey if
some of its clips are sound there (give those clips too, so the piece overlaps), or with
`place.py` on control points if it shares none.

<work_dir> is a directory made by sfm_add.py (db.db and images/). Nothing in it is
changed. The phone's calibration is taken as it stands in the database and not refined:
a piece of a survey is too small to calibrate a lens on.
"""
import os
import shutil
import sqlite3
import sys
import time

import pycolmap


def clip_of(name):
    return str(name).rsplit("_", 1)[0]


def main():
    work, out = sys.argv[1], sys.argv[2]
    clips = set(sys.argv[3:])
    db = os.path.join(work, "db.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    names = [n for (n,) in con.execute("select name from images order by name") if clip_of(n) in clips]
    con.close()
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    print(f"{len(names)} frames of {len(clips)} clips", flush=True)
    t0 = time.time()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = 4
    opts.multiple_models = True                    # if the clips do not all join up, say so rather than drop some
    opts.max_num_models = 4
    opts.min_model_size = 15
    opts.image_names = names
    opts.ba_refine_focal_length = False
    opts.ba_refine_extra_params = False
    opts.ba_refine_principal_point = False
    opts.ba_global_frames_ratio = 1.25
    opts.ba_global_points_ratio = 1.25
    opts.ba_global_max_num_iterations = 40
    opts.ba_global_function_tolerance = 1e-6
    opts.mapper.filter_min_tri_angle = 1.0
    recs = pycolmap.incremental_mapping(db, os.path.join(work, "images"), out, options=opts)
    for k, rec in sorted(recs.items(), key=lambda kv: -kv[1].num_reg_images()):
        placed = {}
        for im in rec.images.values():
            if im.has_pose:
                placed[clip_of(im.name)] = placed.get(clip_of(im.name), 0) + 1
        print(f"model {k}: {rec.num_reg_images()} frames, {rec.num_points3D()} points, "
              f"mean reprojection error {rec.compute_mean_reprojection_error():.2f} px; "
              + ", ".join(f"{c} {n}" for c, n in sorted(placed.items())), flush=True)
    print(f"done in {time.time() - t0:.0f}s; the models are in {out}/0, {out}/1 ...", flush=True)


if __name__ == "__main__":
    main()
