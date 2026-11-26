"""Draw the site grid on the ground in a frame, to read positions off by eye.

    groundgrid.py <survey.npz> <frame> <out.jpg> [z | terrain.npz] [step] [reach] -- <frames_dir> [...]

Once a frame's camera is known, every grid line of the site can be drawn into the picture
where it would lie on the ground. Anything standing on the ground can then be placed to
the nearest quarter of a grid square just by looking: brick edgings, paths, the edge of a
bed, the foot of a post. It only holds for things at the height the grid is drawn at
(z, default 0.08 m: the lawn near the house), so pick a frame that looks down on the spot.
The ground is not level everywhere (the drive lies a quarter of a metre below the lawn),
and at 5 m a grid drawn 0.25 m too high reads nearly a metre out. Give the ground surface
made by terrain.py in place of z and the grid follows the ground.

<frame> is a number for the pan (126) or a name (c3353_0120). Lines run every `step`
metres out to `reach` metres from the camera; every crossing has a dot on it and its
"X,Y" label below and to the right of the dot.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey  # noqa: E402


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw(sv, frame, z=0.08, step=1.0, reach=9.0, scale=0.75):
    i = sv.idx(frame)
    im = sv.image(i)
    W, H = im.size
    d = ImageDraw.Draw(im, "RGBA")
    c = sv.centre(i)
    x0, x1 = np.floor((c[0] - reach) / step) * step, np.ceil((c[0] + reach) / step) * step
    y0, y1 = np.floor((c[1] - reach) / step) * step, np.ceil((c[1] + reach) / step) * step

    zf = z if callable(z) else (lambda x, y: z)

    def line(a, b, colour, width):
        run = []
        for t in np.linspace(0, 1, 60):
            p = np.array([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0.0])
            p[2] = float(zf(p[0], p[1]))
            u, v, depth = sv.project(i, p)
            ok = depth > 0.2 and np.hypot(p[0] - c[0], p[1] - c[1]) < reach and -200 < u < W + 200 and -200 < v < H + 200
            if ok:
                run.append((float(u), float(v)))
            else:
                if len(run) > 1:
                    d.line(run, fill=colour, width=width)
                run = []
        if len(run) > 1:
            d.line(run, fill=colour, width=width)

    for x in np.arange(x0, x1 + 1e-6, step):
        line((x, y0), (x, y1), (255, 255, 0, 200), 3 if abs(x / 5 - round(x / 5)) < 1e-6 else 2)
    for y in np.arange(y0, y1 + 1e-6, step):
        line((x0, y), (x1, y), (0, 255, 255, 200), 3 if abs(y / 5 - round(y / 5)) < 1e-6 else 2)
    font = _font(30)
    for x in np.arange(x0, x1 + 1e-6, step):
        for y in np.arange(y0, y1 + 1e-6, step):
            u, v, depth = sv.project(i, np.array([x, y, float(zf(x, y))]))
            if depth > 0.2 and 0 < u < W - 110 and 0 < v < H - 40 and np.hypot(x - c[0], y - c[1]) < reach:
                label = f"{x:g},{y:g}"
                d.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(255, 0, 255, 255))       # the crossing itself: the label hangs below right of it
                box = d.textbbox((u + 5, v + 3), label, font=font)
                d.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(0, 0, 0, 150))
                d.text((u + 5, v + 3), label, fill=(255, 255, 255, 255), font=font)
    return im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)


def main():
    a = sys.argv[1:]
    dirs = a[a.index("--") + 1:] if "--" in a else []
    a = a[:a.index("--")] if "--" in a else a
    sv = Survey(a[0], dirs)
    frame = int(a[1]) if a[1].isdigit() else a[1]
    z = 0.08
    if len(a) > 3:
        if a[3].endswith(".npz"):
            from terrain import Terrain
            z = Terrain.load(a[3])
        else:
            z = float(a[3])
    step = float(a[4]) if len(a) > 4 else 1.0
    reach = float(a[5]) if len(a) > 5 else 9.0
    draw(sv, frame, z, step, reach).save(a[2], quality=88)
    i = sv.idx(frame)
    print(a[2], "camera at", np.round(sv.centre(i), 2), "looking along bearing", round(float(sv.yaw[i]), 0))


if __name__ == "__main__":
    main()
