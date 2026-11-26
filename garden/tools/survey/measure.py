"""Pixel picks -> plan coordinates.

Every entry is (frame, u, v) read off a gridded video frame at the point where the
thing meets the ground; Survey.ground() turns it into metres on the lawn plane.
Run:  measure.py <survey.npz> <frames_dir>
Prints coordinates in the raw survey frame (origin under the phone, +X = first frame's
view direction). build-time coordinates are these rotated into the plan frame.
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plan import Survey  # noqa: E402

PICKS = {
    # --- east wall (modern dark brick), base of the red plinth
    "east_wall_base": [(22, 600, 668), (22, 700, 668), (22, 800, 668)],
    # --- south wall: old red brick, then ivy
    "south_wall_red_base": [(162, 750, 590), (162, 1045, 596)],
    "south_wall_ivy_base": [(162, 1200, 600), (218, 250, 560)],
    "hardboard": [(162, 1068, 600), (162, 1128, 600), (218, 100, 565), (218, 185, 565)],
    # --- willow corner
    "willow_trunk": [(162, 470, 645), (162, 580, 645), (126, 835, 665), (126, 945, 665)],
    "log_stack": [(162, 110, 632), (162, 425, 632), (126, 530, 640), (126, 800, 640)],
    "ball": [(126, 507, 629), (162, 82, 618)],
    "wall_base_by_ball": [(126, 430, 630), (94, 880, 632)],
    "rounds_pile": [(94, 180, 700), (94, 480, 640), (94, 780, 660), (94, 620, 700), (94, 300, 740)],
    "folded_chair": [(126, 355, 630), (94, 800, 630)],
    "picnic_table_near_feet": [(162, 765, 675), (162, 985, 693), (126, 1120, 700), (126, 1350, 728)],
    "picnic_table_far_feet": [(162, 808, 655), (162, 1005, 668)],
    "firewood_bags_table": [(162, 690, 680), (126, 1050, 700)],
    # --- seating group by the east wall
    "bench_feet": [(22, 1585, 968), (22, 1447, 925), (22, 1838, 878)],
    "chair_feet": [(22, 690, 1235), (22, 885, 1205), (22, 655, 1150)],
    "bistro_tables_feet": [(22, 980, 1170), (22, 1255, 1020), (22, 858, 1100), (22, 1110, 1095)],
    "multistem_shrub": [(22, 1150, 760), (22, 1000, 740), (22, 1260, 800)],
    "evergreen_shrub_e": [(22, 380, 880), (22, 150, 800)],
    "bush_near_camera": [(22, 150, 1200), (22, 60, 1050)],
    # --- hazel / brush pile area
    "eucalyptus": [(162, 1320, 820), (218, 440, 770)],
    "hazel": [(162, 1570, 610), (218, 650, 565)],
    "small_table": [(162, 1610, 785), (162, 1700, 770), (218, 715, 700), (218, 790, 690),
                    (250, 130, 660), (250, 255, 650)],
    "brush_pile": [(218, 760, 690), (218, 950, 680), (250, 170, 640), (250, 430, 630)],
    "cut_stems": [(162, 1580, 880), (218, 700, 800)],
    "fence": [(218, 910, 560), (218, 1040, 565), (250, 360, 470), (250, 520, 480)],
    "water_butt": [(218, 1028, 590), (250, 510, 530)],
    "folding_chair_w": [(218, 995, 600), (250, 470, 560)],
    # --- outbuilding
    "outbuilding_left_corner": [(218, 1045, 598), (250, 523, 545)],
    "outbuilding_front_base": [(250, 700, 565), (250, 880, 585), (250, 965, 590), (282, 380, 458), (282, 250, 450)],
    "twisted_tree": [(218, 1250, 635), (250, 745, 570), (282, 195, 440)],
    "stake_tall": [(218, 1065, 740), (250, 562, 695)],
    "firewood_bag_shed": [(250, 915, 642), (218, 1420, 715), (282, 395, 525)],
    "step_stool": [(250, 940, 680), (250, 1005, 690), (250, 1065, 655), (282, 430, 560), (282, 510, 570)],
    "bbq": [(250, 1010, 600), (282, 490, 450)],
    # --- shed, pale tree, gate
    "shed_front_left": [(250, 1095, 665), (282, 560, 505)],
    "shed_front_base": [(282, 1020, 500), (282, 800, 505)],
    "shed_side_door": [(282, 1320, 480), (314, 850, 110)],
    "pale_tree": [(282, 1105, 590), (282, 1260, 590), (314, 650, 285), (314, 800, 285)],
    "gate_posts": [(282, 1490, 490), (282, 1527, 490), (314, 1020, 190), (314, 1055, 190), (314, 1085, 190)],
    "house_block_base": [(314, 1110, 275), (314, 1560, 290), (282, 1560, 520)],
    "green_bin": [(314, 1215, 280), (282, 1685, 590)],
    "tarp_heap": [(314, 1450, 450), (314, 1340, 430), (314, 1600, 420)],
    "metal_bin": [(314, 1640, 390), (364, 115, 520)],
    "timber_slabs": [(314, 1690, 445), (314, 1780, 450), (364, 205, 540), (364, 275, 530)],
    "white_corner": [(314, 1830, 450), (364, 320, 525)],
    # --- white wing + patio
    "wing_wall_base": [(364, 420, 520), (364, 850, 470), (364, 1180, 450), (386, 320, 470), (386, 640, 470)],
    "wing_post": [(364, 857, 425), (386, 310, 470)],
    "big_pot_patio": [(364, 560, 780), (386, 50, 885)],
    "bowl_pot_bed": [(314, 770, 775), (282, 1125, 1085), (250, 1660, 1310)],
    "olive_pot": [(386, 700, 550), (364, 1225, 500)],
    "blue_trough": [(364, 840, 520), (364, 1030, 505), (386, 310, 590), (386, 500, 550)],
    "black_pot_slab": [(386, 1420, 530)],
    "patio_corner_near": [(386, 590, 985), (364, 1250, 990)],
    "patio_corner_e": [(386, 1130, 720)],
    "patio_corner_ne": [(386, 1440, 565)],
    # --- far lawn
    "white_bulk_bag": [(386, 785, 345), (364, 1320, 285)],
    "blue_tarp": [(386, 1055, 305), (364, 1618, 255)],
    "red_goal": [(386, 1360, 342), (386, 1495, 342)],
    "lawn_shrub": [(386, 1790, 445)],
    "twiggy_bush": [(386, 1550, 440)],
    "hedge_base": [(386, 1000, 290), (386, 1300, 280), (386, 800, 285)],
    "telegraph_pole": [(386, 967, 290)],
}


def main():
    sv = Survey(sys.argv[1], sys.argv[2])
    out = {}
    for name, picks in PICKS.items():
        pts = []
        for (f, u, v) in picks:
            i = sv.idx(f)
            g = sv.ground(i, u, v)
            pts.append(g[:2])
        pts = np.array(pts)
        out[name] = pts
        rng = np.hypot(pts[:, 0], pts[:, 1])
        print(f"{name:26s} " + "  ".join(f"({p[0]:6.2f},{p[1]:7.2f})" for p in pts)
              + f"   | range {rng.min():.1f}-{rng.max():.1f}")
    return sv, out


if __name__ == "__main__":
    main()
