// Bead-curtain motor-skills toy  --  parametric OpenSCAD model
// Sized for a Bambu Lab A1 mini (180 x 180 x 180 mm bed), grey PLA.
//
// Parts (set `part` below or pass -D part="..." on the command line):
//   "frame"   U-shaped frame with comb pockets in the top bar (prints flat)
//   "cover"   strip that closes the comb pockets (3x M3 screws or glue)
//   "base"    hollow weighted base, printed upside-down
//   "plate"   bottom plate that closes the base after filling it with rice
//   "ring"    optional grab ring
//   "assembly" everything in place, for a preview only
//
// Hardware:  ~2.5 m of #10 ball chain (4.5 mm balls, ideally stainless),
//            3x M3x8 screws (cover), 4x M3x8 screws (base plate),
//            ~250 g dry rice or coins as ballast.

part = "assembly";

// ---------- chain ----------
// Pick the ball chain you bought: 10 = #10 (4.5 mm balls, recommended, ~25 kg
// break strength) or 6 = #6 (3.2 mm balls, the common lamp pull-chain size).
chain_size   = 10;
ball_d       = chain_size == 10 ? 4.5  : 3.2;   // ball diameter
bar_d        = chain_size == 10 ? 1.5  : 1.1;   // diameter of the bar between balls
link_pitch   = chain_size == 10 ? 5.64 : 3.96;  // ball-to-ball pitch
chan_w       = ball_d + 0.6;            // vertical channel the hanging chain sits in
neck_w       = bar_d + 1.1;             // wider than the bar, narrower than a ball
neck_h       = 1.0;                     // shelf that traps the last ball
pocket_h     = ball_d + 2;              // room for the last ball plus its cut stub
strands_side = chain_size == 10 ? 8    : 10;   // strands per side of the centre screw
pitch        = chain_size == 10 ? 7.0  : 5.6;  // strand spacing
group_gap    = chain_size == 10 ? 10.4 : 9.6;  // gap between the two groups (centre screw)

// sanity: the neck must fit between two balls where it touches them
neck_clear   = link_pitch - 2*sqrt(pow(ball_d/2,2) - pow(neck_w/2,2));
echo(str("neck clearance between balls = ", neck_clear, " mm (neck_h = ", neck_h, ")"));

// ---------- frame ----------
W   = 150;   // outer width
H   = 150;   // height above the base
T   = 12;    // thickness (print Z when printed flat)
uw  = 16;    // upright width
bh  = 14;    // top bar height
cut_depth = 6;       // how deep the comb pockets go in from the front face
tenon_w   = 12;
tenon_h   = 15;
corner_r  = 6;

// ---------- cover ----------
cover_t = 4;

// ---------- base ----------
BL = 160; BW = 110; BH = 26;
base_r   = 16;      // corner radius
wall     = 3;
skin     = 4;       // top skin thickness
plate_t  = 2.4;
socket_d = 16;      // socket depth
fit      = 0.25;    // clearance on the tenon
plate_fit = 0.3;

// ---------- ring ----------
ring_R = 27; ring_r = 8;

$fn = 48;

// ======================================================================
//  helpers
// ======================================================================
module rrect(x, y, r) {
    hull() for (sx=[-1,1], sy=[-1,1])
        translate([sx*(x/2-r), sy*(y/2-r)]) circle(r);
}

function strand_xs() = concat(
    [for (i=[0:strands_side-1]) -(group_gap/2 + i*pitch)],
    [for (i=[0:strands_side-1])  (group_gap/2 + i*pitch)]);

// ======================================================================
//  frame  (modelled in PRINT orientation: X = width, Y = toy height,
//          Z = toy front/back.  Front face of the toy is the print top.)
// ======================================================================
module frame_profile() {
    difference() {
        union() {
            // outer body with rounded top corners
            hull() {
                translate([-W/2, 0]) square([W, 1]);
                translate([-W/2+corner_r, H-corner_r]) circle(corner_r);
                translate([ W/2-corner_r, H-corner_r]) circle(corner_r);
            }
            // tenons
            for (sx=[-1,1])
                translate([sx*(W/2-uw/2) - tenon_w/2, -tenon_h])
                    square([tenon_w, tenon_h+1]);
        }
        // inner window with filleted corners
        translate([-(W/2-uw), -1]) 
            offset(r=4) offset(delta=-4)
                square([W-2*uw, H-bh+1]);
        // tenon chamfer
        for (sx=[-1,1], s2=[-1,1])
            translate([sx*(W/2-uw/2) + s2*tenon_w/2, -tenon_h])
                rotate(45) square(2, center=true);
    }
}

module comb_cuts() {
    bar_bottom = H - bh;
    for (x = strand_xs()) {
        // vertical channel from bar underside up to the neck
        translate([x - chan_w/2, bar_bottom - 1])
            square([chan_w, bh - pocket_h - 1.5 - neck_h + 1]);
        // neck
        translate([x - neck_w/2, bar_bottom + bh - pocket_h - 1.5 - neck_h - 0.01])
            square([neck_w, neck_h + 0.02]);
        // pocket for the end ball (closed at the top so it can't pull up)
        translate([x - chan_w/2, bar_bottom + bh - pocket_h - 1.5])
            square([chan_w, pocket_h]);
    }
}

module frame() {
    difference() {
        linear_extrude(T) frame_profile();
        // comb cuts from the front face
        translate([0, 0, T - cut_depth]) linear_extrude(cut_depth + 1) comb_cuts();
        // cover screw holes (M3 self-tapping)
        for (x = [-(W/2-uw/2), 0, (W/2-uw/2)])
            translate([x, H - bh/2, T - 8]) cylinder(d=2.6, h=9);
    }
}

// ======================================================================
//  cover strip (prints flat)
// ======================================================================
module cover() {
    difference() {
        linear_extrude(cover_t) rrect(W, bh, 2);
        for (x = [-(W/2-uw/2), 0, (W/2-uw/2)]) {
            translate([x, 0, -1]) cylinder(d=3.4, h=cover_t+2);
            translate([x, 0, cover_t-2]) cylinder(d=6.5, h=3);   // counterbore
        }
    }
}

// ======================================================================
//  base  (modelled upright; exported upside-down for printing)
// ======================================================================
socket_x = W/2 - uw/2;
boss_xy  = [[-(BL/2-12), -(BW/2-12)], [(BL/2-12), -(BW/2-12)],
            [-(BL/2-12),  (BW/2-12)], [(BL/2-12),  (BW/2-12)]];

module base() {
    difference() {
        union() {
            difference() {
                // body with a small chamfer on the top edge
                hull() {
                    linear_extrude(BH-2) rrect(BL, BW, base_r);
                    linear_extrude(BH)   rrect(BL-4, BW-4, base_r-2);
                }
                // plate recess
                translate([0,0,-1]) linear_extrude(plate_t+1) rrect(BL-2*wall, BW-2*wall, base_r-wall);
                // ballast cavity
                translate([0,0,plate_t-0.01]) linear_extrude(BH-skin-plate_t) rrect(BL-4*wall, BW-4*wall, base_r-2*wall);
            }
            // socket bosses
            for (sx=[-1,1])
                translate([sx*socket_x, 0, plate_t]) linear_extrude(BH-skin-plate_t+0.01)
                    rrect(tenon_w+2*fit+8, T+2*fit+8, 3);
            // screw bosses
            for (p = boss_xy)
                translate([p[0], p[1], plate_t]) cylinder(d=9, h=BH-skin-plate_t+0.01);
        }
        // tenon sockets
        for (sx=[-1,1])
            translate([sx*socket_x, 0, BH-socket_d]) linear_extrude(socket_d+1)
                square([tenon_w+2*fit, T+2*fit], center=true);
        // plate screw holes
        for (p = boss_xy)
            translate([p[0], p[1], plate_t-1]) cylinder(d=2.6, h=11);
    }
}

module plate() {
    difference() {
        linear_extrude(plate_t) rrect(BL-2*wall-2*plate_fit, BW-2*wall-2*plate_fit, base_r-wall);
        for (p = boss_xy) {
            translate([p[0], p[1], -1]) cylinder(d=3.4, h=plate_t+2);
        }
    }
}

// ======================================================================
//  ring
// ======================================================================
module ring() {
    translate([0,0,ring_r]) rotate_extrude() translate([ring_R, 0]) circle(ring_r);
}

// ======================================================================
//  assembly preview (toy orientation)
// ======================================================================
module assembly() {
    color("dimgray") base();
    // frame: print X -> toy X, print Y -> toy Z, print Z -> toy -Y (front)
    translate([0, T/2, BH]) rotate([90,0,0]) {
        color("silver") frame();
        color("gray") translate([0, H-bh/2, T]) cover();
    }
    color("gray") translate([0, -32, BH]) ring();
    // chains
    color("goldenrod") for (x = strand_xs())
        translate([x, -T/2 + T - cut_depth + chan_w/2 - 0.3, BH + 8]) cylinder(d=ball_d, h=H - bh + 0.5);
}

if (part == "frame")    frame();
if (part == "cover")    cover();
if (part == "base")     translate([0,0,BH]) rotate([180,0,0]) base();   // cavity up for printing
if (part == "plate")    plate();
if (part == "ring")     ring();
if (part == "assembly") assembly();
