"""Free bundle adjustment of a multi-clip survey.

    sfm_refine.py <work_dir_in> <work_dir_out>

sfm_add.py holds the first clip's cameras fixed so the new clips land in the existing
frame. That keeps coordinates stable, but it also keeps whatever was wrong with the first
clip: a pan from one spot has almost no baseline, so its distances are weak, and the
clips that actually walk about are not allowed to correct them. Once there is outside
control to pin the frame down (control/control.json, see calibrate.py) there is no reason
to keep that constraint, so this lets every camera and every point move, re-triangulates,
and adjusts again.
"""
import os
import shutil
import sys
import time

import pycolmap


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(dst, "sparse"), exist_ok=True)
    for name in ("db.db", "images"):
        target = os.path.join(dst, name)
        if not os.path.lexists(target):
            os.symlink(os.path.realpath(os.path.join(src, name)), target)
    rec = pycolmap.Reconstruction(os.path.join(src, "sparse"))
    print("before:", rec.summary().replace("\n", " "), flush=True)

    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = True
    opts.refine_extra_params = True
    opts.refine_principal_point = False
    for rnd in range(3):
        t0 = time.time()
        pycolmap.bundle_adjustment(rec, opts)
        n0 = rec.num_points3D()
        try:    # drop points that no longer fit, so they cannot drag the next round
            obs = pycolmap.ObservationManager(rec)
            obs.filter_all_points3D(4.0, 0.75)
        except Exception as e:  # API differences between versions: filtering is an optional nicety
            print("  (filter skipped:", type(e).__name__, ")")
        print(f"round {rnd + 1}: {time.time() - t0:.0f}s, points {n0} -> {rec.num_points3D()}, "
              f"mean reprojection error {rec.compute_mean_reprojection_error():.3f} px", flush=True)
    for cid, cam in rec.cameras.items():
        print("camera", cid, cam.model, [round(float(p), 4) for p in cam.params])
    out = os.path.join(dst, "sparse")
    rec.write(out)
    print("saved", out)


if __name__ == "__main__":
    main()
