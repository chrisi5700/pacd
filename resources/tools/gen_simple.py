#!/usr/bin/env python3
"""Generate simple analytic primitives as binary STLs for the fitting test bed.

Produces, under ``resources/simple/``:

    {cube,cylinder} x {0deg,45deg} x {large,small}  (8 files)  +  sphere.stl

All meshes are centred on the origin. The 45-degree variants are rotated about
the X axis so the tilt is visible on the cylinder too (a cylinder spun about its
own Z symmetry axis would be indistinguishable from the unrotated one).

Sizes (edge length / diameter, in model units):
    large = 40, small = 20        sphere: diameter 40

Run:  python3 resources/tools/gen_simple.py
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

Vec = tuple[float, float, float]
Tri = tuple[Vec, Vec, Vec]

OUT_DIR = Path(__file__).resolve().parent.parent / "simple"

LARGE = 40.0
SMALL = 20.0
CYL_SEGMENTS = 64
SPHERE_STACKS = 32   # latitude bands
SPHERE_SLICES = 64   # longitude bands


# --- linear algebra helpers -------------------------------------------------

def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def normalize(v: Vec) -> Vec:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def facet_normal(t: Tri) -> Vec:
    return normalize(cross(sub(t[1], t[0]), sub(t[2], t[0])))


def rotate_x(v: Vec, deg: float) -> Vec:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y, z = v
    return (x, y * c - z * s, y * s + z * c)


# --- shape builders (centred on origin, +normals point outward) -------------

def make_cube(size: float) -> list[Tri]:
    h = size / 2.0
    # 8 corners
    v = [
        (-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),  # z = -h
        (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h),      # z = +h
    ]
    # each face as a CCW quad viewed from outside -> two triangles
    quads = [
        (0, 3, 2, 1),  # -Z
        (4, 5, 6, 7),  # +Z
        (0, 1, 5, 4),  # -Y
        (2, 3, 7, 6),  # +Y
        (1, 2, 6, 5),  # +X
        (0, 4, 7, 3),  # -X
    ]
    tris: list[Tri] = []
    for a, b, c, d in quads:
        tris.append((v[a], v[b], v[c]))
        tris.append((v[a], v[c], v[d]))
    return tris


def make_cylinder(diameter: float, height: float, segments: int) -> list[Tri]:
    r = diameter / 2.0
    hz = height / 2.0
    tris: list[Tri] = []
    top_c: Vec = (0.0, 0.0, hz)
    bot_c: Vec = (0.0, 0.0, -hz)
    for i in range(segments):
        a0 = 2.0 * math.pi * i / segments
        a1 = 2.0 * math.pi * (i + 1) / segments
        x0, y0 = r * math.cos(a0), r * math.sin(a0)
        x1, y1 = r * math.cos(a1), r * math.sin(a1)
        bt0: Vec = (x0, y0, hz)
        bt1: Vec = (x1, y1, hz)
        bb0: Vec = (x0, y0, -hz)
        bb1: Vec = (x1, y1, -hz)
        # side wall (outward normals)
        tris.append((bb0, bb1, bt1))
        tris.append((bb0, bt1, bt0))
        # top cap (+Z outward): CCW seen from above
        tris.append((top_c, bt0, bt1))
        # bottom cap (-Z outward): CCW seen from below
        tris.append((bot_c, bb1, bb0))
    return tris


def make_sphere(diameter: float, stacks: int, slices: int) -> list[Tri]:
    r = diameter / 2.0
    tris: list[Tri] = []

    def pt(i: int, j: int) -> Vec:
        phi = math.pi * i / stacks            # 0..pi  (latitude)
        theta = 2.0 * math.pi * j / slices    # 0..2pi (longitude)
        return (r * math.sin(phi) * math.cos(theta),
                r * math.sin(phi) * math.sin(theta),
                r * math.cos(phi))

    for i in range(stacks):
        for j in range(slices):
            a = pt(i, j)
            b = pt(i + 1, j)
            c = pt(i + 1, j + 1)
            d = pt(i, j + 1)
            if i != 0:                        # top cap degenerates to a point
                tris.append((a, b, c))
            if i != stacks - 1:               # bottom cap degenerates to a point
                tris.append((a, c, d))
    return tris


# --- binary STL writer ------------------------------------------------------

def write_stl(path: Path, tris: list[Tri]) -> None:
    with path.open("wb") as f:
        f.write(b"\0" * 80)                   # header
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            n = facet_normal(t)
            f.write(struct.pack("<3f", *n))
            for vtx in t:
                f.write(struct.pack("<3f", *vtx))
            f.write(struct.pack("<H", 0))     # attribute byte count


def rotated(tris: list[Tri], deg: float) -> list[Tri]:
    if deg == 0.0:
        return tris
    return [tuple(rotate_x(v, deg) for v in t) for t in tris]  # type: ignore[misc]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, int]] = []

    builders = {
        "cube": lambda s: make_cube(s),
        "cylinder": lambda s: make_cylinder(s, s, CYL_SEGMENTS),
    }
    sizes = {"large": LARGE, "small": SMALL}
    angles = {"0deg": 0.0, "45deg": 45.0}

    for shape, build in builders.items():
        for size_name, size in sizes.items():
            base = build(size)
            for ang_name, ang in angles.items():
                tris = rotated(base, ang)
                name = f"{shape}_{ang_name}_{size_name}.stl"
                write_stl(OUT_DIR / name, tris)
                written.append((name, len(tris)))

    sphere = make_sphere(LARGE, SPHERE_STACKS, SPHERE_SLICES)
    write_stl(OUT_DIR / "sphere.stl", sphere)
    written.append(("sphere.stl", len(sphere)))

    for name, ntri in sorted(written):
        print(f"  {name:28s} {ntri:5d} triangles")
    print(f"{len(written)} files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
