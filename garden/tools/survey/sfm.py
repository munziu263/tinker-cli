"""Sparse structure-from-motion on the garden pan video with COLMAP (CPU).

usage: sfm.py <frames_dir> <work_dir> [step] [overlap]
"""
import os
import shutil
import sys
import time

import pycolmap

frames_dir = sys.argv[1]
work = sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
overlap = int(sys.argv[4]) if len(sys.argv) > 4 else 10

os.makedirs(work, exist_ok=True)
img_dir = os.path.join(work, "images")
if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.makedirs(img_dir)
all_frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
names = all_frames[::step]
for n in names:
    os.symlink(os.path.join(frames_dir, n), os.path.join(img_dir, n))
print("frames:", len(names), flush=True)

db = os.path.join(work, "db.db")
if os.path.exists(db):
    os.remove(db)

t0 = time.time()
reader = pycolmap.ImageReaderOptions()
reader.camera_model = "SIMPLE_RADIAL"
# iPhone 16e main camera, 26 mm-equiv; QuickTake 1920x1440 video (slight EIS crop)
reader.camera_params = "1550,960,720,0"
ext = pycolmap.FeatureExtractionOptions()
ext.num_threads = 4
ext.max_image_size = 1920
ext.sift.max_num_features = 7000
pycolmap.extract_features(
    db, img_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
    reader_options=reader, extraction_options=ext,
)
print("extract done %.1fs" % (time.time() - t0), flush=True)

t0 = time.time()
pair = pycolmap.SequentialPairingOptions()
pair.overlap = overlap
pair.quadratic_overlap = True
match = pycolmap.FeatureMatchingOptions()
match.num_threads = 4
pycolmap.match_sequential(db, matching_options=match, pairing_options=pair)
print("match done %.1fs" % (time.time() - t0), flush=True)

t0 = time.time()
opts = pycolmap.IncrementalPipelineOptions()
opts.num_threads = 4
opts.multiple_models = True
opts.min_model_size = 5
# rotation-dominant capture: accept small triangulation angles
opts.mapper.init_min_tri_angle = 3.0
opts.mapper.filter_min_tri_angle = 0.5
opts.mapper.ba_local_min_tri_angle = 2.0
opts.triangulation.min_angle = 0.5
opts.ba_refine_focal_length = True
opts.ba_refine_extra_params = True
out = os.path.join(work, "sparse")
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(out)
recs = pycolmap.incremental_mapping(db, img_dir, out, options=opts)
print("mapping done %.1fs" % (time.time() - t0), flush=True)
for k, r in recs.items():
    print("model", k, r.summary(), flush=True)
