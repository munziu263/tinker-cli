"""Fix a position by sighting the same thing in two or more frames.

    triangulate.py <survey.npz> <sightings.json> <frames_dir> [<frames_dir> ...]

A pan from one spot gives good directions and poor distances. One more sighting of the
same corner from a few metres to the side turns that into a proper fix, the way a
surveyor intersects two bearings. sightings.json lists, for each thing,

    "name": [[frame, u, v], [frame, u, v], ...]

with pixel positions in the full-size frames (gridcrop.py makes them easy to read off).
For each one this prints the site-grid position, how far apart the rays pass (a blunder
shows up as a large miss) and the angle between them (under about 4 degrees the distance
is weak again: move further to the side for the second sighting).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey  # noqa: E402


def intersect(sv, picks):
    """Least-squares meeting point of the rays through the picks."""
    A, b, rays = np.zeros((3, 3)), np.zeros(3), []
    for frame, u, v in picks:
        i = sv.idx(frame)
        c, r = sv.centre(i), sv.ray(i, u, v)
        r = r / np.linalg.norm(r)
        M = np.eye(3) - np.outer(r, r)
        A += M
        b += M @ c
        rays.append((c, r))
    p = np.linalg.solve(A, b)
    miss = max(float(np.linalg.norm((np.eye(3) - np.outer(r, r)) @ (p - c))) for c, r in rays)
    spread = max(float(np.degrees(np.arccos(np.clip(r1 @ r2, -1, 1)))) for _, r1 in rays for _, r2 in rays)
    return p, miss, spread


def main():
    npz, picks_path, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
    sv = Survey(npz, dirs)
    for name, picks in json.load(open(picks_path)).items():
        if name.startswith("_"):
            continue
        p, miss, spread = intersect(sv, [tuple(q) for q in picks])
        flag = "" if spread >= 4 else "   (rays nearly parallel: distance is weak)"
        print(f"{name:40s} X {p[0]:7.2f}  Y {p[1]:7.2f}  z {p[2]:5.2f}   rays miss by {miss * 100:5.1f} cm, {spread:4.1f} deg apart{flag}")


if __name__ == "__main__":
    main()
