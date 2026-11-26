"""Slide and turn a group of clips until their points lie on a building whose shape is known.

    anchor.py <survey.npz> <viewer/model dir> <out.npz> --to <feature id>[,<feature id>] --clips <clip,clip,...>
              [--carry <clip,clip,...>] [--plane x0,y0,x1,y1,z0,z1 ...] [--sights <sights.json>]
              [--reach 0.7] [--no-turn] [--lift] [--size] [--tilt] [--hold <other_survey.npz>]

Clips are joined to the survey one frame at a time, each placed from points made by the
frames before it. Along a walk with a weak stretch in it (the camera pointed at the ground,
a corner turned fast) the placing slips, and everything after the slip is carried along:
here the clips walked across the front garden came out 0.3 to 0.5 m too far north. Nothing
inside the survey can show it, because the clips still agree with one another.

A building drawn from a measured survey can. The clips look at the house, and their points
on its walls should lie on the walls. This takes the points that live in the named clips
(align.py records which clip each point belongs to) and lie within `reach` of the building
as the model has it, and finds the turn about the vertical and the shift that bring them
onto its surfaces: each point is pulled square on to the nearest face (never along it, so
a blank wall decides only what a blank wall can), the pull is recomputed as the fit
closes in, and points that stay off are given less and less say (a shrub against the wall
is not the wall).

The group is moved on plan only, unless told otherwise. Walls say nothing about height,
and what little pull there is up or down comes from sills, eaves and roof slopes, which
are the parts of a model least likely to be exact: left free, the height followed them and
came out 0.18 m wrong against measured spot levels. --lift frees the height, for when
something does pin it (held frames, a roof seen from a ladder). --no-turn keeps the
group's direction, for a group with too short a run of wall to judge a turn by.
--size also lets the group grow or shrink about its own middle: a walk
that slips usually slips in size as well as in place, and a building with returns, reveals
and an eaves line holds size well. --tilt lets it lean, for clips whose sense of upright
has gone too.

--carry names clips that are moved with the group but have no say in where it goes: clips
filmed indoors on the same walk, whose points lie on the inside of the walls.

--plane adds an upright rectangular face to pull on: a wall whose line is known from
somewhere sound (fitted to the points of clips that are not in doubt, say) but which the
model does not have, or has from the very clips being corrected. From (x0, y0) to
(x1, y1) on plan, between heights z0 and z1. May be given several times; --to may then be
left out.

--sights is for a clip that hangs on the survey by one neighbour and has swung: here a
walk along the east wall, joined only to the clip before it, was turned 4.3 degrees, and
showed the well across the lawn 1.3 m from where the clips that walked round it had it.
It looks at few walls, so a building cannot hold it; but it does see things whose places
are known. The file lists them (see control/east_sights.json):

    {"sights": [{"what": "axis of the well", "at": [x, y], "picks": [["c3426_0030", u, v], ...]}, ...]}

`at` is where the thing stands on plan, from sound clips or from the measured house (a
corner, the middle of a round thing, the foot of a trunk); each pick is where it shows in
one of the group's own frames, in pixels. The group is turned and slid until every such
thing lies on the line of sight through its pick. Only the direction is used, never the
distance: a short clip judges direction well and distance badly. Three things are enough
if they are well apart and at least one is near; more, and in frames from both ends of
the clip, show whether the clip is sound in itself (the misses are printed for each
pick: here eleven picks of three things came to within 4 cm, 2 cm rms). Sights say
nothing about height or lean. They may be given with --to, and then have half the say
the faces have, or alone.

(Laying such a clip's points on the points of the other clips, cloud on cloud, was tried
first and is not offered: it gave three different answers for this clip. Rows of kerbs a
brick apart catch it one row over, and a clip that looks mostly at grass has few faces of
its own to lay on anything.)

--hold names another survey in which some frames of the group are known to be right (a
few frames at one end of the walk that a free reconstruction, sfm_part.py and place.py,
ties to sound clips). Those frames are pulled towards their places there, with as much
say between them as half the building has. That pins the direction a flat wall leaves
free: here the house fixed the walk square to its front and in size and turn, and nine
held frames fixed it along the front.

The frames and points of those clips are then moved together; everything else in the
survey stays where it is. Printed: how far the points stood from the building before and
after, the correction found, and how much of the pull came from faces looking each way,
which is what tells you whether the answer is pinned in every direction. A group that
sees one flat wall is pinned square to that wall and free to slide along it: it will be
left where it was in that direction, and the report says so.

The model is read from the web copy that build.py writes, like fit.py.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit import load_model  # noqa: E402
from plan import Survey, clip_of  # noqa: E402


def surface(tri, spacing, rng):
    """Points scattered over triangles, with each point's face normal."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    ok = area > 1e-8
    a, b, c, n, area = a[ok], b[ok], c[ok], n[ok], area[ok]
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    k = np.repeat(np.arange(len(a)), np.maximum(1, np.ceil(area / spacing ** 2)).astype(int))
    u, v = rng.random(len(k)), rng.random(len(k))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[k] + (b[k] - a[k]) * u[:, None] + (c[k] - a[k]) * v[:, None], n[k]


def rot(w):
    """Rotation matrix for a small rotation vector."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def fit(P, S, N, reach=0.7, tilt=False, size=False, rounds=30, held=None, turn=True, lift=False, sights=None):
    """The motion x -> c + k R (x - c) + t that lays points P on the surface (samples S, normals N).
    held: (positions now, positions wanted) of cameras that are to end up in known places.
    sights: (places on plan, camera centres, directions of the lines of sight) of known things picked in frames.
    With no surface (S empty) the held cameras and the sights decide alone; P then only gives the middle."""
    faces = len(S) > 0
    tree = cKDTree(S) if faces else None
    c = P.mean(0)
    R, t, k = np.eye(3), np.zeros(3), 1.0
    nr = 3 if tilt else 1
    ncol = nr + 3 + (1 if size else 0)
    report = {}

    def sight_rows(R, t, k):
        """For each pick: how far the known place lies to one side of the line of sight, and how that changes."""
        T, C, D = sights
        Cn = k * (C - c) @ R.T + c + t
        e = (D @ R.T)[:, :2]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        n = np.stack([-e[:, 1], e[:, 0]], axis=1)                            # square to the line of sight, on plan
        to = T - Cn[:, :2]
        L = Cn[:, :2] - c[:2]
        J = np.zeros((len(T), ncol))
        J[:, nr - 1] = -((e * to).sum(1) + n[:, 1] * L[:, 0] - n[:, 0] * L[:, 1])   # a turn swings the line across the thing
        J[:, nr:nr + 2] = -n
        if size:
            J[:, nr + 3] = -(n * L).sum(1)
        return (n * to).sum(1), J, (e * to).sum(1)

    for it in range(rounds):
        J, r, w = np.zeros((0, ncol)), np.zeros(0), np.zeros(0)
        if faces:
            lim = reach + (0.25 - reach) * min(1.0, it / (rounds * 0.6))    # close in from `reach` to 0.25 m
            Q = k * (P - c) @ R.T + c + t
            d, j = tree.query(Q, distance_upper_bound=lim)
            m = np.isfinite(d)
            if m.sum() < 30:
                raise SystemExit("too few points near the building to go on")
            q, n, s = Q[m], N[j[m]], S[j[m]]
            r = ((q - s) * n).sum(1)                                        # distance square on to the face
            w = 1.0 / (1.0 + (r / 0.08) ** 2)                               # points that stay off count for less
            lever = q - c
            cols = [np.cross(lever, n) if tilt else np.cross(lever, n)[:, 2:3], n]      # d r / d (turn, shift)
            if size:
                cols.append((lever * n).sum(1, keepdims=True))                            # ... and d r / d size
            J = np.concatenate(cols, axis=1)
            if it == 0:
                report["before"] = (float(np.sqrt((r ** 2).mean())), float(np.median(np.abs(r))), int(m.sum()))
        say = 0.5 * w.sum() if faces else None                              # what the faces have to say, halved
        if held is not None and len(held[0]):                               # cameras with known places: three pulls each
            now = k * (held[0] - c) @ R.T + c + t
            lv = now - c
            share = say / (3 * len(now)) if faces else 1.0
            for ax in range(3):
                e = np.zeros((len(now), 3))
                e[:, ax] = 1.0
                hc = [np.cross(lv, e) if tilt else np.cross(lv, e)[:, 2:3], e]
                if size:
                    hc.append(lv[:, ax:ax + 1])
                J = np.concatenate([J, np.concatenate(hc, axis=1)])
                r = np.concatenate([r, now[:, ax] - held[1][:, ax]])
                w = np.concatenate([w, np.full(len(now), share)])
        if sights is not None:
            rs, Js, _ = sight_rows(R, t, k)
            if it == 0:
                report["sights_before"] = rs
            J = np.concatenate([J, Js])
            r = np.concatenate([r, rs])
            w = np.concatenate([w, np.full(len(rs), say / len(rs) if faces else 1.0)])
        if len(r) < 3:
            raise SystemExit("nothing to go by: no faces, held frames or sights")
        free = np.ones(ncol, bool)
        if not turn and not tilt:
            free[0] = False                                                 # keep its direction
        if not lift and not tilt:
            free[nr + 2] = False                                            # keep its height
        A = (J * w[:, None]).T @ J + np.eye(ncol) * 1e-3 * w.sum()          # damping: what nothing pulls on, stays
        x = np.zeros(ncol)
        x[free] = np.linalg.solve(A[np.ix_(free, free)], -((J * w[:, None]).T @ r)[free])
        dw = x[:3] if tilt else np.array([0.0, 0.0, x[0]])
        dR = rot(dw)
        dk = 1.0 + (x[nr + 3] if size else 0.0)
        R, k, t = dR @ R, k * dk, dk * dR @ t + x[nr:nr + 3]
    if faces:
        Q = k * (P - c) @ R.T + c + t
        d, j = tree.query(Q, distance_upper_bound=0.25)
        m = np.isfinite(d)
        r = ((Q[m] - S[j[m]]) * N[j[m]]).sum(1)
        report["after"] = (float(np.sqrt((r ** 2).mean())), float(np.median(np.abs(r))), int(m.sum()))
        n = np.abs(N[j[m]])
        report["faces"] = (n / n.sum(1, keepdims=True)).mean(0)
        # how firmly each direction is held: the pull's stiffness along X, Y and Z, relative to the stiffest
        H = (N[j[m]][:, :, None] * N[j[m]][:, None, :]).sum(0)
        ev = np.linalg.eigvalsh(H)
        report["weakest"] = float(ev[0] / ev[-1])
    if held is not None and len(held[0]):
        report["held"] = np.linalg.norm(k * (held[0] - c) @ R.T + c + t - held[1], axis=1)
    if sights is not None:
        report["sights_after"], Js, report["sights_range"] = sight_rows(R, t, k)
        free = [nr - 1, nr, nr + 1] if turn else [nr, nr + 1]
        H = Js[:, free].T @ Js[:, free]
        dof = max(1, len(Js) - len(free))
        sd = np.sqrt(np.diag(np.linalg.inv(H + np.eye(len(free)) * 1e-9)) * float(report["sights_after"] @ report["sights_after"]) / dof)
        report["sights_sd"] = sd if turn else np.concatenate([[0.0], sd])   # how well the sights alone pin turn and shift
    return k * R, t, c, report


def main():
    a = sys.argv[1:]
    opt, planes = {}, []
    while "--plane" in a:
        i = a.index("--plane")
        planes.append([float(v) for v in a[i + 1].split(",")])
        del a[i:i + 2]
    for key in ("--to", "--clips", "--reach", "--hold", "--carry", "--sights"):
        if key in a:
            i = a.index(key)
            opt[key] = a[i + 1]
            del a[i:i + 2]
    tilt, size, lift, turn = "--tilt" in a, "--size" in a, "--lift" in a, "--no-turn" not in a
    a = [x for x in a if x not in ("--tilt", "--size", "--lift", "--no-turn")]
    sv = Survey(a[0], [])
    clips = set(opt["--clips"].split(","))
    wanted = [w for w in opt.get("--to", "").split(",") if w]
    target = [f for f in load_model(a[1]) if f["id"] in wanted] if wanted else []
    if len(target) != len(wanted):
        raise SystemExit(f"not in the model: {', '.join(sorted(set(wanted) - {f['id'] for f in target}))}")
    tris = [f["tri"][~f["soft"]] for f in target]
    for x0, y0, x1, y1, z0, z1 in planes:                 # an upright rectangle is two triangles
        q = np.array([[x0, y0, z0], [x1, y1, z0], [x1, y1, z1], [x0, y0, z1]])
        tris.append(np.array([q[[0, 1, 2]], q[[0, 2, 3]]]))
    if not tris and "--sights" not in opt and "--hold" not in opt:
        raise SystemExit("nothing to anchor to: give --to, --plane or --sights")
    S, N = surface(np.concatenate(tris), 0.04, np.random.default_rng(0)) if tris else (np.zeros((0, 3)), np.zeros((0, 3)))
    opt["--to"] = " and ".join(wanted + ([f"{len(planes)} given plane(s)"] if planes else []))
    mine = sv.points_of(clips)
    good = mine & (sv.track >= 3) & (sv.err < 1.5)
    reach = float(opt.get("--reach", 0.7))
    near = np.zeros(len(sv.P), bool)
    if len(S):
        d, _ = cKDTree(S).query(sv.P[good], distance_upper_bound=reach)
        near[np.flatnonzero(good)[np.isfinite(d)]] = True
        print(f"{int(mine.sum())} points live in {', '.join(sorted(clips))}; {int(near.sum())} of them lie within {reach} m of {opt['--to']}")
    held = None
    if "--hold" in opt:
        ref = np.load(opt["--hold"], allow_pickle=False)
        where = {str(n): k for k, n in enumerate(ref["names"])}
        both = [(i, where[n]) for i, n in enumerate(sv.names) if clip_of(n) in clips and n in where]
        held = (np.array([sv.C[i] for i, _ in both]).reshape(-1, 3), np.array([ref["C"][k] for _, k in both]).reshape(-1, 3))
        off = np.linalg.norm(held[0] - held[1], axis=1)
        print(f"{len(both)} frames of the group have known places in {os.path.basename(opt['--hold'])}; they stand {off.mean() * 100:.0f} cm from them now")
    sights, seen = None, []
    if "--sights" in opt:
        with open(opt["--sights"]) as fh:
            listed = json.load(fh)["sights"]
        T, C, D = [], [], []
        for thing in listed:
            for frame, u, v in thing["picks"]:
                i = sv.idx(frame)
                if clip_of(sv.names[i]) not in clips:
                    raise SystemExit(f"{frame} is not a frame of the group: sights are picked in the group's own frames")
                ray = sv.ray(i, u, v)
                T.append(thing["at"][:2])
                C.append(sv.centre(i))
                D.append(ray / np.linalg.norm(ray))
                seen.append((thing["what"], sv.names[i]))
        sights = (np.array(T, float), np.array(C, float), np.array(D, float))
        print(f"{len(seen)} picks of {len(listed)} known things in the group's frames")
    if len(S):
        mob = sv.P[near]
    else:                                                 # nothing to lay points on: turn about the middle of the cameras used
        mob = sights[1] if sights is not None else held[0]
    R, t, c, rep = fit(mob, S, N, reach, tilt, size, held=held, turn=turn, lift=lift, sights=sights)   # R carries the change of size, if any
    k = float(np.cbrt(np.linalg.det(R)))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    move = (mob - c) @ R.T + c + t - mob
    if "before" in rep:
        print(f"before: {rep['before'][2]} points, rms {rep['before'][0] * 100:.1f} cm off the faces (median {rep['before'][1] * 100:.1f})")
        print(f"after:  {rep['after'][2]} points within 0.25 m, rms {rep['after'][0] * 100:.1f} cm (median {rep['after'][1] * 100:.1f})")
    print(f"correction: turn {yaw:+.2f} deg about the vertical, " + (f"size x {k:.4f}, " if size else "")
          + f"shift ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}) m at the group's centre "
          f"({c[0]:.1f}, {c[1]:.1f}); the {'points on the building' if len(S) else 'cameras used'} move {np.linalg.norm(move, axis=1).mean() * 100:.0f} cm on average, "
          f"mean ({move[:, 0].mean():+.2f}, {move[:, 1].mean():+.2f}, {move[:, 2].mean():+.2f})")
    if "held" in rep:
        print(f"held frames: {rep['held'].mean() * 100:.1f} cm from their known places afterwards (worst {rep['held'].max() * 100:.1f})")
    if sights is not None:
        was, now, rng = rep["sights_before"], rep["sights_after"], rep["sights_range"]
        print("sights: how far each known thing lay to one side of the line of sight through its pick")
        for j, (what, frame) in enumerate(seen):
            print(f"    {frame[:-4]:12s} {what[:44]:44s} {rng[j]:5.1f} m off   {was[j] * 100:+7.1f} cm  ->  {now[j] * 100:+6.1f} cm")
        sd = rep["sights_sd"]
        print(f"    rms {np.sqrt((was ** 2).mean()) * 100:.1f} cm before, {np.sqrt((now ** 2).mean()) * 100:.1f} cm after; "
              f"by themselves the sights pin the turn to {np.degrees(sd[0]):.2f} deg and the place to {sd[1] * 100:.0f} and {sd[2] * 100:.0f} cm (one sigma)")
    if "faces" in rep:
        fx, fy, fz = rep["faces"]
        print(f"faces pulling: {100 * fx:.0f}% look along X, {100 * fy:.0f}% along Y, {100 * fz:.0f}% up or down; weakest direction held {rep['weakest']:.3f} as firmly as the strongest"
              + ("  (below about 0.02 that direction is not really pinned)" if rep["weakest"] < 0.02 else ""))

    carried = set(c for c in opt.get("--carry", "").split(",") if c)
    if carried:
        clips = clips | carried
        mine = sv.points_of(clips)
        print(f"carried along: {', '.join(sorted(carried))}")
    d = dict(np.load(a[0], allow_pickle=False))
    fr = np.array([clip_of(str(n)) in clips for n in d["names"]])
    d["C"][fr] = (d["C"][fr] - c) @ R.T + c + t
    d["R"][fr] = d["R"][fr] @ (R / k).T                                     # world -> camera, for the moved world
    d["P"][mine] = (d["P"][mine] - c) @ R.T + c + t
    Rw = d["R"][fr]
    d["yaw"][fr] = np.degrees(np.unwrap(np.radians(np.degrees(np.arctan2(Rw[:, 2, 1], Rw[:, 2, 0])))))
    d["pitch"][fr] = np.degrees(np.arcsin(np.clip(Rw[:, 2, 2], -1, 1)))
    d["roll"][fr] = np.degrees(np.arctan2(-Rw[:, 0, 2], -Rw[:, 1, 2]))
    np.savez(a[2], **d)
    print(f"moved {int(fr.sum())} frames and {int(mine.sum())} points; saved {a[2]}")


if __name__ == "__main__":
    main()
