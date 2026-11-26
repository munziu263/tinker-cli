"""Plan-view measurement tools on top of an aligned survey (see align.py).

Single-view metrology: every frame has a known pose and the lawn is the plane z = 0,
so any pixel that shows the ground maps to a plan position, and anything standing on
a known ground point can be given a height.
"""
import numpy as np
from PIL import Image, ImageDraw


class Survey:
    def __init__(self, npz_path, frames_dir=None):
        d = np.load(npz_path, allow_pickle=False)
        self.names = [str(n) for n in d["names"]]
        self.R, self.C, self.P, self.col = d["R"], d["C"], d["P"], d["col"]
        self.err, self.track = d["err"], d["track"]
        self.f, self.cx, self.cy = (float(v) for v in d["K"])
        self.k1 = float(d["k1"])
        self.W, self.H = (int(v) for v in d["size"])
        self.yaw, self.pitch, self.roll = d["yaw"], d["pitch"], d["roll"]
        self.frames_dir = frames_dir
        self.T = np.eye(3)           # extra plan transform (rotation about Z + shift), see set_plan_frame
        self.shift = np.zeros(3)

    # -- optional re-framing so that walls run along the axes
    def set_plan_frame(self, rot_deg=0.0, shift=(0.0, 0.0)):
        a = np.radians(rot_deg)
        c, s = np.cos(a), np.sin(a)
        self.T = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.shift = np.array([shift[0], shift[1], 0.0])

    def w(self, p):
        """survey coords -> plan-frame coords"""
        return np.asarray(p) @ self.T.T + self.shift

    def w_inv(self, p):
        return (np.asarray(p) - self.shift) @ self.T

    def idx(self, name_or_num):
        if isinstance(name_or_num, (int, np.integer)):
            name_or_num = f"f_{int(name_or_num):04d}.jpg"
        return self.names.index(name_or_num)

    def nearest(self, frame_num):
        nums = np.array([int(n[2:6]) for n in self.names])
        return int(np.argmin(np.abs(nums - frame_num)))

    # -- camera model (COLMAP SIMPLE_RADIAL)
    def project(self, i, Pw):
        """plan-frame points -> pixels (u, v), depth"""
        Ps = self.w_inv(Pw)
        pc = (Ps - self.C[i]) @ self.R[i].T
        z = pc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = pc[..., 0] / z, pc[..., 1] / z
        r2 = x * x + y * y
        k = 1 + self.k1 * r2
        return self.f * x * k + self.cx, self.f * y * k + self.cy, z

    def ray(self, i, u, v):
        """pixel -> unit direction in the plan frame"""
        xd, yd = (np.asarray(u, float) - self.cx) / self.f, (np.asarray(v, float) - self.cy) / self.f
        x, y = xd, yd
        for _ in range(8):
            k = 1 + self.k1 * (x * x + y * y)
            x, y = xd / k, yd / k
        d = np.stack([x, y, np.ones_like(x)], -1)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return (d @ self.R[i]) @ self.T.T

    def centre(self, i):
        return self.w(self.C[i])

    def ground(self, i, u, v, z=0.0):
        """Where the ray through pixel (u, v) of frame i meets the plane at height z."""
        d = self.ray(i, u, v)
        c = self.centre(i)
        t = (z - c[2]) / d[..., 2]
        return c + d * t[..., None] if np.ndim(t) else c + d * t

    def height_above(self, i, u, v, xy):
        """Height of the point seen at pixel (u, v), assuming it is vertically above xy."""
        d = self.ray(i, u, v)
        c = self.centre(i)
        dh = np.hypot(d[0], d[1])
        dist = np.hypot(xy[0] - c[0], xy[1] - c[1])
        return float(c[2] + d[2] / dh * dist)

    def bearing(self, i, u, v):
        d = self.ray(i, u, v)
        return float(np.degrees(np.arctan2(d[1], d[0])))

    # -- images
    def image(self, i):
        return Image.open(f"{self.frames_dir}/{self.names[i]}").convert("RGB")

    def gridded(self, i, step=100, scale=1.0, box=None):
        """Frame with a labelled pixel grid, for reading off coordinates by eye."""
        im = self.image(i)
        dr = ImageDraw.Draw(im)
        for x in range(0, self.W, step):
            dr.line([(x, 0), (x, self.H)], fill=(255, 255, 0) if x % 500 == 0 else (255, 255, 255), width=1)
            dr.text((x + 3, 3), str(x), fill=(255, 255, 0))
            dr.text((x + 3, self.H // 2 + 3), str(x), fill=(255, 255, 0))
        for y in range(0, self.H, step):
            dr.line([(0, y), (self.W, y)], fill=(255, 255, 0) if y % 500 == 0 else (255, 255, 255), width=1)
            dr.text((3, y + 3), str(y), fill=(255, 255, 0))
            dr.text((self.W // 2 + 3, y + 3), str(y), fill=(255, 255, 0))
        # horizon line of this frame
        us = np.linspace(0, self.W, 50)
        hv = []
        for u in us:
            lo, hi = 0.0, float(self.H)
            for _ in range(30):
                mid = (lo + hi) / 2
                if self.ray(i, u, mid)[2] > 0:
                    lo = mid
                else:
                    hi = mid
            hv.append((u, lo))
        dr.line(hv, fill=(255, 0, 255), width=2)
        if box:
            im = im.crop(box)
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        return im

    # -- ortho mosaic of the ground plane
    def mosaic(self, extent=(-20, 20, -20, 20), res=0.02, every=3, z=0.0, max_range=40.0,
               min_depress_deg=2.0):
        """Project the frames onto the plane at height z: an 'aerial photo' of the lawn.
        Things standing up smear radially away from the camera; their feet are true."""
        x0, x1, y0, y1 = extent
        nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        xs = x0 + (np.arange(nx) + 0.5) * res
        ys = y1 - (np.arange(ny) + 0.5) * res      # image row 0 = max y (north-up style)
        X, Y = np.meshgrid(xs, ys)
        G = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
        acc = np.zeros((nx * ny, 3), np.float32)
        wsum = np.zeros(nx * ny, np.float32)
        for i in range(0, len(self.names), every):
            c = self.centre(i)
            dist = np.hypot(G[:, 0] - c[0], G[:, 1] - c[1])
            u, v, zc = self.project(i, G)
            dep = np.degrees(np.arctan2(c[2] - z, np.maximum(dist, 1e-6)))
            ok = (zc > 0.1) & (u >= 1) & (u < self.W - 2) & (v >= 1) & (v < self.H - 2) \
                & (dist < max_range) & (dep > min_depress_deg)
            if not ok.any():
                continue
            img = np.asarray(self.image(i), np.float32)
            uu, vv = u[ok], v[ok]
            iu, iv = uu.astype(int), vv.astype(int)
            fu, fv = (uu - iu)[:, None], (vv - iv)[:, None]
            s = (img[iv, iu] * (1 - fu) * (1 - fv) + img[iv, iu + 1] * fu * (1 - fv)
                 + img[iv + 1, iu] * (1 - fu) * fv + img[iv + 1, iu + 1] * fu * fv)
            wgt = (1 - ((uu - self.cx) / (self.W / 2)) ** 2) * (1 - ((vv - self.cy) / (self.H / 2)) ** 2)
            wgt = np.clip(wgt, 0.02, 1) ** 2
            acc[ok] += s * wgt[:, None]
            wsum[ok] += wgt
        out = np.where(wsum[:, None] > 0, acc / np.maximum(wsum[:, None], 1e-6), 40)
        im = Image.fromarray(out.reshape(ny, nx, 3).astype(np.uint8))
        return im, (x0, y1, res)

    @staticmethod
    def draw_grid(im, geo, major=5.0, minor=1.0, label=True, marks=None):
        """Overlay a metre grid on a plan image. geo = (x0, y_top, res)."""
        x0, y1, res = geo
        im = im.copy()
        dr = ImageDraw.Draw(im, "RGBA")
        nx, ny = im.size
        xa, xb = x0, x0 + nx * res
        ya, yb = y1 - ny * res, y1
        for gx in np.arange(np.ceil(xa / minor) * minor, xb, minor):
            px = (gx - x0) / res
            maj = abs(gx / major - round(gx / major)) < 1e-6
            dr.line([(px, 0), (px, ny)], fill=(255, 255, 255, 150 if maj else 45), width=2 if maj else 1)
            if label and maj:
                dr.text((px + 3, 3), f"{gx:.0f}", fill=(255, 255, 0, 255))
                dr.text((px + 3, ny - 14), f"{gx:.0f}", fill=(255, 255, 0, 255))
        for gy in np.arange(np.ceil(ya / minor) * minor, yb, minor):
            py = (y1 - gy) / res
            maj = abs(gy / major - round(gy / major)) < 1e-6
            dr.line([(0, py), (nx, py)], fill=(255, 255, 255, 150 if maj else 45), width=2 if maj else 1)
            if label and maj:
                dr.text((3, py + 3), f"{gy:.0f}", fill=(255, 255, 0, 255))
                dr.text((nx - 26, py + 3), f"{gy:.0f}", fill=(255, 255, 0, 255))
        for mk in marks or []:
            px, py = (mk[0] - x0) / res, (y1 - mk[1]) / res
            colr = mk[3] if len(mk) > 3 else (255, 60, 60, 255)
            dr.ellipse([px - 5, py - 5, px + 5, py + 5], outline=colr, width=2)
            if len(mk) > 2 and mk[2]:
                dr.text((px + 7, py - 6), str(mk[2]), fill=colr)
        return im

    def points_plan(self, extent=(-20, 20, -20, 20), res=0.02, zmin=-0.5, zmax=8.0, min_track=3,
                    max_err=2.0, size=2):
        """Plan plot of the sparse point cloud, coloured by the points' own colours."""
        x0, x1, y0, y1 = extent
        nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
        im = Image.new("RGB", (nx, ny), (18, 18, 22))
        px = im.load()
        P = self.w(self.P)
        ok = (self.track >= min_track) & (self.err < max_err) & (P[:, 2] > zmin) & (P[:, 2] < zmax)
        order = np.argsort(P[ok][:, 2])
        for p, c in zip(P[ok][order], self.col[ok][order]):
            ix, iy = int((p[0] - x0) / res), int((y1 - p[1]) / res)
            for dx in range(size):
                for dy in range(size):
                    if 0 <= ix + dx < nx and 0 <= iy + dy < ny:
                        px[ix + dx, iy + dy] = tuple(int(v) for v in c)
        return im, (x0, y1, res)
