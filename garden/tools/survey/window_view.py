"""Fix things seen through a window, from a single frame filmed inside.

    window_view.py <frame.jpg> <view.json>

A frame taken through a window cannot be joined to the outdoor reconstruction: there is
glass in the way and nothing to match. But the window itself is as good as a theodolite
mount. Its position is known from the drawings, and its joinery gives the direction the
camera points:

  - every upright edge of the frame (jambs, mullion, glazing bars) points at the same
    vanishing point, which is straight down;
  - every level edge (head, sill, transoms) points along the wall the window is in, and
    the drawings say which way that wall runs.

Two directions fix the camera's orientation with no assumption about the size of the
window. If the phone's focal length is right the two come out at right angles, which is
the check printed first. The corners of the window, whose site coordinates are known,
then give where the camera was.

After that each pixel is a ray in the site grid. A ray alone has no distance, so every
sighting in view.json says what else is known about the thing:

    {"name": ..., "pixel": [u, v], "z": 1.93}                    it is at this height
    {"name": ..., "top": [u, v], "bottom": [u, v], "height": 1.17}   an upright edge this tall

The second kind gives a distance without knowing any level, and the height it returns
for the top of the edge is a check on the first kind.

view.json also holds
    camera   {"f", "cx", "cy", "k1"}  the phone's calibration (as found by the survey)
    window   {"wall_axis": [x, y, z], the direction the wall runs, in the site grid;
              "outward":   [x, y, z], the way the camera looks out, square to the wall;
              "corners_px": four pixels, "corners_site": the same corners in the grid}
"""
import json
import sys

import cv2
import numpy as np


class View:
    def __init__(self, frame_path, spec):
        c = spec["camera"]
        self.K = np.array([[c["f"], 0, c["cx"]], [0, c["f"], c["cy"]], [0, 0, 1.0]])
        self.dist = np.array([c.get("k1", 0.0), 0, 0, 0])
        w = spec["window"]
        self.R, self.report = self._orientation(frame_path, np.array(w["wall_axis"], float), np.array(w["outward"], float),
                                                spec.get("min_edge_px", 140))
        self.C, self.miss = self._centre(w["corners_px"], w["corners_site"])

    def _norm(self, pts):
        """pixels -> normalised image coordinates, lens distortion removed"""
        return cv2.undistortPoints(np.asarray(pts, float).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)

    def _shared_direction(self, seg, length, tol_deg=1.2, iters=3000, seed=0):
        """The direction (in camera coordinates) that most of the segments point along.
        Each segment and the lens centre span a plane; the direction lies in all of them."""
        a = np.c_[self._norm(seg[:, :2]), np.ones(len(seg))]
        b = np.c_[self._norm(seg[:, 2:]), np.ones(len(seg))]
        n = np.cross(a, b)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        rng, best, tol = np.random.default_rng(seed), (None, -1.0), np.sin(np.radians(tol_deg))
        for _ in range(iters):
            i, j = rng.choice(len(n), 2, replace=False)
            d = np.cross(n[i], n[j])
            if np.linalg.norm(d) < 1e-9:
                continue
            d /= np.linalg.norm(d)
            score = length[np.abs(n @ d) < tol].sum()
            if score > best[1]:
                best = (d, score)
        d = best[0]
        for _ in range(4):
            ok = np.abs(n @ d) < tol
            d = np.linalg.eigh((n[ok] * length[ok, None]).T @ n[ok])[1][:, 0]
        rms = np.degrees(np.sqrt(np.average(np.arcsin(np.clip(n[ok] @ d, -1, 1)) ** 2, weights=length[ok])))
        return d, int(ok.sum()), len(seg), float(rms)

    def _orientation(self, frame_path, wall_axis, outward, min_len):
        grey = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        seg = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(grey)[0].reshape(-1, 4)
        length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        seg, length = seg[length >= min_len], length[length >= min_len]
        ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180
        upright, level = (ang > 55) & (ang < 125), (ang < 22) | (ang > 158)
        v, nv, tv, rv = self._shared_direction(seg[upright], length[upright])
        h, nh, th, rh = self._shared_direction(seg[level], length[level])
        if v[1] < 0:                                   # image y runs down the picture: make v point down
            v = -v
        square = float(np.degrees(np.arccos(abs(v @ h))))
        h = h - (h @ v) * v
        h /= np.linalg.norm(h)
        up_w = np.array([0, 0, 1.0])
        s = float(np.sign(np.cross(wall_axis, outward) @ up_w))        # handedness of (wall axis, outward, up)
        z_c = -v
        a_c = h if (s * np.cross(z_c, h))[2] > 0 else -h               # the camera looks out, not back into the room
        n_c = s * np.cross(z_c, a_c)
        R = np.outer(a_c, wall_axis) + np.outer(n_c, outward) + np.outer(z_c, up_w)     # site grid -> camera
        f = R.T @ np.array([0, 0, 1.0])
        report = (f"{nv} of {tv} upright edges agree to {rv:.2f} deg, {nh} of {th} level edges to {rh:.2f} deg; "
                  f"the two directions are {square:.1f} deg apart (90 if the focal length is right)\n"
                  f"camera looks along site bearing {np.degrees(np.arctan2(f[1], f[0])):.1f} deg, "
                  f"{-np.degrees(np.arcsin(f[2])):.1f} deg below level")
        return R, report

    def ray(self, u, v):
        q = self._norm([(u, v)])[0]
        d = self.R.T @ np.array([q[0], q[1], 1.0])
        return d / np.linalg.norm(d)

    def _centre(self, pix, world):
        """Where the camera was: the point closest to the rays back from the window's corners."""
        A, b, rays = np.zeros((3, 3)), np.zeros(3), []
        for (u, v), P in zip(pix, world):
            d = self.ray(u, v)
            M = np.eye(3) - np.outer(d, d)
            A += M
            b += M @ np.asarray(P, float)
            rays.append((d, np.asarray(P, float)))
        C = np.linalg.solve(A, b)
        miss = [float(np.linalg.norm((P - C) - ((P - C) @ d) * d)) for d, P in rays]
        return C, miss

    def at_height(self, pixel, z):
        d = self.ray(*pixel)
        return self.C + d * (z - self.C[2]) / d[2]

    def upright(self, top, bottom, height):
        """Top of an upright edge of known height: its two ends are one above the other."""
        d1, d2 = self.ray(*top), self.ray(*bottom)
        r = np.hypot(d1[0], d1[1]) / np.hypot(d2[0], d2[1])
        return self.C + d1 * height / (d1[2] - r * d2[2])


def main():
    frame, spec = sys.argv[1], json.load(open(sys.argv[2]))
    vw = View(frame, spec)
    print(vw.report)
    print(f"camera at X {vw.C[0]:.2f}  Y {vw.C[1]:.2f}  z {vw.C[2]:.2f}; rays back from the window's corners miss them by "
          + ", ".join(f"{m * 100:.0f}" for m in vw.miss) + " cm")
    for s in spec.get("sightings", []):
        if "pixel" in s:
            p = vw.at_height(s["pixel"], s["z"])
            print(f"  {s['name']:44s} X {p[0]:7.2f}  Y {p[1]:7.2f}   (taking its height as {s['z']:.2f} m)")
        else:
            p = vw.upright(s["top"], s["bottom"], s["height"])
            print(f"  {s['name']:44s} X {p[0]:7.2f}  Y {p[1]:7.2f}   top at {p[2]:.2f} m, foot at {p[2] - s['height']:.2f} m "
                  f"(taking it as {s['height']:.2f} m tall)")


if __name__ == "__main__":
    main()
