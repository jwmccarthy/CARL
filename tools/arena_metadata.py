"""Match CARL triangles to the native RocketSim mesh and edge metadata."""

import argparse
from pathlib import Path

import numpy as np


def load_obj(path):
    vertices = []
    triangles = []

    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue

        if fields[0] == "v":
            x, y, z = map(float, fields[1:4])
            vertices.append((y, x, z))
        elif fields[0] == "f":
            triangles.append([int(vertex.split("/")[0]) - 1 for vertex in fields[1:4]])

    return np.asarray(vertices), triangles


def triangle_key(vertices):
    return tuple(sorted(tuple(vertex) for vertex in np.round(vertices, 2)))


def index_reference(reference):
    lookup = {}

    for row in reference:
        key = triangle_key(row[2:11].reshape(3, 3))
        if key in lookup:
            raise ValueError("Duplicate reference triangle")

        lookup[key] = row

    return lookup


def find_reference(triangle, reference, lookup, centers):
    row = lookup.get(triangle_key(triangle))
    if row is not None:
        return row

    # Allow small coordinate differences after matching triangle centers
    distances = np.linalg.norm(centers - triangle.mean(axis=0), axis=1)
    for index in np.flatnonzero(distances < .005):
        candidate = reference[index, 2:11].reshape(3, 3)
        vertex_distances = np.linalg.norm(candidate[:, None] - triangle[None], axis=-1)
        if vertex_distances.min(axis=1).max() < .002:
            return reference[index]

    return None


def is_arena_plane(triangle):
    planes = ((2, 0), (2, 2048), (0, -4096), (0, 4096))
    return any(np.all(np.abs(triangle[:, axis] - value) < .001) for axis, value in planes)


def format_plane(triangle):
    vertices = " ".join(map(str, (triangle * .02).ravel()))
    return "-1 -1 6.2831853 6.2831853 6.2831853 0 " + vertices


def format_reference(triangle, row):
    native = row[2:11].reshape(3, 3)
    permutation = [int(np.linalg.norm(native - vertex, axis=1).argmin()) for vertex in triangle]

    if any(permutation[(edge + 1) % 3] != (permutation[edge] + 1) % 3 for edge in range(3)):
        raise ValueError("Reference and CARL winding disagree")

    angles = row[11:14][permutation]
    flags = int(row[14])
    mapped_flags = sum(
        ((flags >> edge) & 1) << index | ((flags >> (edge + 3)) & 1) << (index + 3)
        for index, edge in enumerate(permutation)
    )
    native_vertices = row[15:24].reshape(3, 3)[permutation].ravel()
    values = [int(row[0]), int(row[1]), *angles, mapped_flags, *native_vertices]

    return " ".join(map(str, values))


def export(obj_path, reference_path, output):
    vertices, triangles = load_obj(obj_path)
    reference = np.loadtxt(reference_path)
    lookup = index_reference(reference)
    centers = reference[:, 2:11].reshape(-1, 3, 3).mean(axis=1)
    result = [str(len(triangles))]
    matched = 0

    for indices in triangles:
        triangle = vertices[indices]
        row = find_reference(triangle, reference, lookup, centers)

        if row is None:
            if not is_arena_plane(triangle):
                raise ValueError(f"Unmatched non-plane triangle: {triangle}")

            result.append(format_plane(triangle))
            continue

        result.append(format_reference(triangle, row))
        matched += 1

    if matched != len(reference):
        raise ValueError(f"Matched {matched}/{len(reference)} reference triangles")

    Path(output).write_text("\n".join(result) + "\n")
    print(f"Exported metadata for {matched} native triangles and {len(triangles) - matched} plane triangles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obj")
    parser.add_argument("reference")
    parser.add_argument("output")
    args = parser.parse_args()
    export(args.obj, args.reference, args.output)
