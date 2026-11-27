"""Printable survey markers: one coded target to a sheet of A4.

    make_markers.py <out.pdf> [--first 0] [--count 24] [--size 160] [--png-dir <dir>]

A marker is a square black-and-white pattern that software can find in a photograph by
itself, tell from every other marker, and pin down to a fraction of a pixel at each of
its four corners. Fixed about a site and left there, markers do for a photo survey what
pegs in the ground do for a land survey:

  * every photograph that shows one is tied to the same fixed points, so a walk cannot
    drift and two sets of photographs taken a week apart land on each other;
  * a few distances taped between markers give the whole survey its size, checked as
    many times over as there are distances;
  * and they are found automatically (tools/survey/markers.py), with nothing to pick by
    hand.

The pattern is an AprilTag of the 36h11 family, which OpenCV reads. Each sheet carries
the marker's number in figures, a line that must measure exactly 100 mm when the sheet
has been printed at full size, and an arrow at the corner to measure from.

--size is the side of the black square in millimetres (160 fills a sheet of A4 and
leaves the white margin the pattern needs; it must be a multiple of 8). A marker is read
reliably while its side covers about 30 pixels of the picture: for a 160 mm marker that
is out to 20 m in a 24-megapixel phone photograph and 8 m in a frame of phone video.

--png-dir also writes each sheet as a PNG, 10 pixels to the millimetre.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PX = 10                                   # pixels per millimetre: 254 dpi, so every millimetre is a whole number of pixels
A4 = (210, 297)
FAMILY = cv2.aruco.DICT_APRILTAG_36h11
TAG_TOP = 50                              # millimetres from the top of the sheet to the black square


def font(mm):
    try:
        return ImageFont.load_default(size=int(mm * PX))
    except TypeError:                     # an old Pillow: a fixed small font is all it has
        return ImageFont.load_default()


def tag_box(size=160):
    """Where the black square sits on the sheet: (left, top, right, bottom) in millimetres."""
    left = (A4[0] - size) / 2
    return left, TAG_TOP, left + size, TAG_TOP + size


def sheet(tag_id, size=160):
    """One sheet as a black-and-white PIL image, 10 px to the millimetre."""
    if size % 8:
        raise SystemExit("--size must be a multiple of 8 mm, so that each square of the pattern is a whole number of millimetres")
    page = Image.new("L", (A4[0] * PX, A4[1] * PX), 255)
    d = ImageDraw.Draw(page)
    left, top, right, bottom = tag_box(size)
    pattern = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(FAMILY), tag_id, int(size * PX), borderBits=1)
    page.paste(Image.fromarray(pattern), (int(left * PX), int(top * PX)))

    def mm(*v):
        return [int(round(x * PX)) for x in v]

    d.text(mm(left, 9), f"SURVEY MARKER {tag_id:02d}", fill=0, font=font(7))
    d.text(mm(left, 19), "Leave where it is. Please do not move or cover.", fill=0, font=font(4))
    # the corner to measure from: an arrow in the margin, stopping well short of the pattern
    ax, ay = left - 4.6, top - 4.6
    d.line(mm(left - 17, top - 17, ax - 2, ay - 2), fill=0, width=int(1.4 * PX))
    d.polygon(mm(ax, ay, ax - 6, ay - 1.6, ax - 1.6, ay - 6), fill=0)
    d.text(mm(left - 17, top - 26), "measure from this corner", fill=0, font=font(3.4))
    # the number, large, and the check on the print's size
    d.text(mm(left, bottom + 16), f"{tag_id:02d}", fill=0, font=font(44))
    x0, y0 = left + 62, bottom + 44
    d.line(mm(x0, y0, x0 + 100, y0), fill=0, width=int(0.5 * PX))
    for x in (x0, x0 + 100):
        d.line(mm(x, y0 - 4, x, y0 + 4), fill=0, width=int(0.5 * PX))
    d.text(mm(x0, y0 - 14), "This line is 100 mm long.", fill=0, font=font(4))
    d.text(mm(x0, y0 + 6), "If it is not, print again at 100% (actual size).", fill=0, font=font(3.4))
    d.text(mm(x0, y0 + 12), f"Black square: {size} mm. AprilTag 36h11, number {tag_id}.", fill=0, font=font(3.4))
    return page.point(lambda v: 255 if v > 127 else 0).convert("1")


def main():
    a = sys.argv[1:]
    opt = {}
    for key in ("--first", "--count", "--size", "--png-dir"):
        if key in a:
            i = a.index(key)
            opt[key] = a[i + 1]
            del a[i:i + 2]
    first, count, size = int(opt.get("--first", 0)), int(opt.get("--count", 24)), int(opt.get("--size", 160))
    pages = [sheet(k, size) for k in range(first, first + count)]
    pages[0].save(a[0], save_all=True, append_images=pages[1:], resolution=25.4 * PX)
    if "--png-dir" in opt:
        os.makedirs(opt["--png-dir"], exist_ok=True)
        for k, p in zip(range(first, first + count), pages):
            p.save(os.path.join(opt["--png-dir"], f"marker_{k:02d}.png"))
    print(f"{count} sheets, markers {first} to {first + count - 1}, black square {size} mm: {a[0]} ({os.path.getsize(a[0]) / 1e3:.0f} kB)")
    print(np.round(tag_box(size), 1), "mm: where the square sits on the sheet (left, top, right, bottom)")


if __name__ == "__main__":
    main()
