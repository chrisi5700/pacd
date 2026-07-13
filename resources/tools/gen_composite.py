#!/usr/bin/env python3
"""Generate compound test subjects: unions of simple primitives.

Each scene is a union of boxes / cylinders / spheres placed in a common frame
(Z is up, ground plane at z = 0). For every scene we emit, under
``resources/composite/``:

    <name>.stl   -- binary STL of the merged mesh (one solid, welded triangles)
    <name>.json  -- ground-truth parts list (type + dims + rotation + position)

The JSON sidecar is the reason these make good fitting fixtures: it is the exact
answer a primitive-fitting pass should recover, so it can be scored directly.

Run:  python3 resources/tools/gen_composite.py
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

Vec = tuple[float, float, float]
Tri = tuple[Vec, Vec, Vec]

OUT_DIR = Path(__file__).resolve().parent.parent / "composite"

CYL_SEGMENTS = 48
SPHERE_STACKS = 24
SPHERE_SLICES = 48


# --- linear algebra ---------------------------------------------------------

def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def normalize(v: Vec) -> Vec:
    n = math.sqrt(sum(c * c for c in v))
    return (0.0, 0.0, 0.0) if n == 0.0 else (v[0] / n, v[1] / n, v[2] / n)


def facet_normal(t: Tri) -> Vec:
    return normalize(cross(sub(t[1], t[0]), sub(t[2], t[0])))


def _rx(v: Vec, c: float, s: float) -> Vec:
    x, y, z = v
    return (x, y * c - z * s, y * s + z * c)


def _ry(v: Vec, c: float, s: float) -> Vec:
    x, y, z = v
    return (x * c + z * s, y, -x * s + z * c)


def _rz(v: Vec, c: float, s: float) -> Vec:
    x, y, z = v
    return (x * c - y * s, x * s + y * c, z)


def place(v: Vec, rot: Vec, pos: Vec) -> Vec:
    """Apply intrinsic rotation (Rz.Ry.Rx, degrees) then translate by pos."""
    rx, ry, rz = (math.radians(a) for a in rot)
    v = _rx(v, math.cos(rx), math.sin(rx))
    v = _ry(v, math.cos(ry), math.sin(ry))
    v = _rz(v, math.cos(rz), math.sin(rz))
    return (v[0] + pos[0], v[1] + pos[1], v[2] + pos[2])


# --- centred primitive meshes -----------------------------------------------

def mesh_box(sx: float, sy: float, sz: float) -> list[Tri]:
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    tris: list[Tri] = []
    for a, b, c, d in quads:
        tris.append((v[a], v[b], v[c]))
        tris.append((v[a], v[c], v[d]))
    return tris


def mesh_cylinder(d: float, h: float, seg: int = CYL_SEGMENTS) -> list[Tri]:
    r, hz = d / 2, h / 2
    top_c: Vec = (0.0, 0.0, hz)
    bot_c: Vec = (0.0, 0.0, -hz)
    tris: list[Tri] = []
    for i in range(seg):
        a0 = 2 * math.pi * i / seg
        a1 = 2 * math.pi * (i + 1) / seg
        p0 = (r * math.cos(a0), r * math.sin(a0))
        p1 = (r * math.cos(a1), r * math.sin(a1))
        t0, t1 = (*p0, hz), (*p1, hz)
        b0, b1 = (*p0, -hz), (*p1, -hz)
        tris.append((b0, b1, t1))
        tris.append((b0, t1, t0))
        tris.append((top_c, t0, t1))
        tris.append((bot_c, b1, b0))
    return tris


def mesh_sphere(d: float, stacks: int = SPHERE_STACKS,
                slices: int = SPHERE_SLICES) -> list[Tri]:
    r = d / 2
    tris: list[Tri] = []

    def pt(i: int, j: int) -> Vec:
        phi = math.pi * i / stacks
        theta = 2 * math.pi * j / slices
        return (r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi))

    for i in range(stacks):
        for j in range(slices):
            a, b = pt(i, j), pt(i + 1, j)
            c, e = pt(i + 1, j + 1), pt(i, j + 1)
            if i != 0:
                tris.append((a, b, c))
            if i != stacks - 1:
                tris.append((a, c, e))
    return tris


# --- part description -------------------------------------------------------

def box(sx, sy, sz, pos=(0, 0, 0), rot=(0, 0, 0)):
    return {"kind": "box", "sx": sx, "sy": sy, "sz": sz, "rot": rot, "pos": pos}


def cyl(d, h, pos=(0, 0, 0), rot=(0, 0, 0)):
    return {"kind": "cylinder", "d": d, "h": h, "rot": rot, "pos": pos}


def sph(d, pos=(0, 0, 0)):
    return {"kind": "sphere", "d": d, "rot": (0, 0, 0), "pos": pos}


def part_mesh(part: dict) -> list[Tri]:
    k = part["kind"]
    if k == "box":
        base = mesh_box(part["sx"], part["sy"], part["sz"])
    elif k == "cylinder":
        base = mesh_cylinder(part["d"], part["h"])
    elif k == "sphere":
        base = mesh_sphere(part["d"])
    else:
        raise ValueError(k)
    rot, pos = part["rot"], part["pos"]
    return [tuple(place(v, rot, pos) for v in t) for t in base]  # type: ignore[misc]


# --- scenes (each returns a list of parts) ----------------------------------

def cylinder_on_cube():
    return [box(40, 40, 40, pos=(0, 0, 20)),
            cyl(24, 40, pos=(0, 0, 60))]


def snowman():
    return [sph(40, pos=(0, 0, 20)),
            sph(28, pos=(0, 0, 50)),
            sph(18, pos=(0, 0, 72))]


def dumbbell():
    return [cyl(12, 50, pos=(0, 0, 13), rot=(0, 90, 0)),   # bar along X
            sph(26, pos=(-25, 0, 13)),
            sph(26, pos=(25, 0, 13))]


def table():
    parts = [box(60, 60, 6, pos=(0, 0, 44))]               # top
    for sx in (-24, 24):
        for sy in (-24, 24):
            parts.append(cyl(8, 42, pos=(sx, sy, 21)))     # legs
    return parts


def sphere_chain():
    return [sph(20, pos=(x, 0, 10)) for x in (-36, -18, 0, 18, 36)]


def rocket():
    parts = [cyl(20, 50, pos=(0, 0, 25)),                  # body
             sph(22, pos=(0, 0, 52))]                      # nose dome
    parts += [box(8, 2, 18, pos=(12, 0, 9)),               # fins
              box(8, 2, 18, pos=(-12, 0, 9)),
              box(2, 8, 18, pos=(0, 12, 9)),
              box(2, 8, 18, pos=(0, -12, 9))]
    return parts


def tree():
    return [cyl(10, 40, pos=(0, 0, 20)),                   # trunk
            sph(44, pos=(0, 0, 52))]                       # canopy


def ring_of_spheres():
    parts = []
    for k in range(6):
        a = 2 * math.pi * k / 6
        parts.append(sph(16, pos=(30 * math.cos(a), 30 * math.sin(a), 8)))
    return parts


def l_bracket():
    return [box(16, 16, 50, pos=(0, 0, 25)),               # vertical arm
            box(40, 16, 16, pos=(20, 0, 8))]               # horizontal arm


def wheel_axle():
    return [cyl(6, 44, pos=(0, 0, 15), rot=(0, 90, 0)),    # axle along X
            cyl(30, 6, pos=(-20, 0, 15), rot=(0, 90, 0)),  # wheels
            cyl(30, 6, pos=(20, 0, 15), rot=(0, 90, 0))]


def jenga_tower():
    parts = []
    for i in range(6):
        rot = (0, 0, 90) if i % 2 else (0, 0, 0)
        parts.append(box(48, 16, 8, pos=(0, 0, 4 + 8 * i), rot=rot))
    return parts


SCENES = {
    "cylinder_on_cube": cylinder_on_cube,
    "snowman": snowman,
    "dumbbell": dumbbell,
    "table": table,
    "sphere_chain": sphere_chain,
    "rocket": rocket,
    "tree": tree,
    "ring_of_spheres": ring_of_spheres,
    "l_bracket": l_bracket,
    "wheel_axle": wheel_axle,
    "jenga_tower": jenga_tower,
}


# --- binary STL writer ------------------------------------------------------

def write_stl(path: Path, tris: list[Tri]) -> None:
    with path.open("wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", *facet_normal(t)))
            for v in t:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, int]] = []
    for name, scene in SCENES.items():
        parts = scene()
        tris: list[Tri] = []
        for p in parts:
            tris.extend(part_mesh(p))
        write_stl(OUT_DIR / f"{name}.stl", tris)
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps({"name": name, "parts": parts}, indent=2) + "\n")
        rows.append((name, len(parts), len(tris)))
    for name, nparts, ntri in rows:
        print(f"  {name:20s} {nparts:2d} parts {ntri:6d} tris")
    print(f"{len(rows)} scenes -> {OUT_DIR}")


if __name__ == "__main__":
    main()
