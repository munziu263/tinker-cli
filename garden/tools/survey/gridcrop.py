"""Cut a piece out of a frame at full size and rule a labelled pixel grid over it.

    gridcrop.py <survey.npz> <frame> <x0> <y0> <x1> <y1> <out.jpg> [grid_step] [zoom] -- <frames_dir> [...]

For reading pixel positions off the footage by eye: the numbers along the grid are the
pixel coordinates in the whole frame, which is what control.json and sightings.json want.
<frame> is a number for the pan (126) or a file name (h_0040.jpg).
"""
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import Survey  # noqa: E402


def crop(sv, frame, x0, y0, x1, y1, out, step=50, zoom=1.0):
    im = sv.image(sv.idx(frame)).crop((x0, y0, x1, y1))
    if zoom != 1.0:
        im = im.resize((int(im.width * zoom), int(im.height * zoom)), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")
    for x in range((x0 // step + 1) * step, x1, step):
        px, major = (x - x0) * zoom, x % (step * 2) == 0
        d.line([(px, 0), (px, im.height)], fill=(255, 255, 0, 150 if major else 70), width=1)
        if major:
            for yy in (2, im.height // 2, im.height - 12):
                d.text((px + 2, yy), str(x), fill=(255, 255, 0, 255))
    for y in range((y0 // step + 1) * step, y1, step):
        py, major = (y - y0) * zoom, y % (step * 2) == 0
        d.line([(0, py), (im.width, py)], fill=(0, 255, 255, 150 if major else 70), width=1)
        if major:
            for xx in (2, im.width // 2, im.width - 30):
                d.text((xx, py + 1), str(y), fill=(0, 255, 255, 255))
    im.save(out, quality=90)
    print(out, im.size)


def main():
    a = sys.argv[1:]
    dirs = a[a.index("--") + 1:] if "--" in a else []
    a = a[:a.index("--")] if "--" in a else a
    sv = Survey(a[0], dirs)
    frame = int(a[1]) if a[1].isdigit() else a[1]
    crop(sv, frame, int(a[2]), int(a[3]), int(a[4]), int(a[5]), a[6], int(a[7]) if len(a) > 7 else 50, float(a[8]) if len(a) > 8 else 1.0)


if __name__ == "__main__":
    main()
