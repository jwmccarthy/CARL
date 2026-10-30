"""Match CARL triangles to the native RocketSim mesh and edge metadata."""
import argparse
from pathlib import Path
import numpy as np


def export(obj_path, reference_path, output):
    vertices, triangles = [], []
    for line in Path(obj_path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'v':
            x, y, z = map(float, fields[1:4])
            vertices.append((y, x, z))
        elif fields and fields[0] == 'f':
            triangles.append([int(v.split('/')[0])-1 for v in fields[1:4]])
    vertices = np.asarray(vertices)
    def key(v):
        return tuple(sorted(tuple(x) for x in np.round(v, 2)))
    reference = np.loadtxt(reference_path)
    lookup = {}
    for row in reference:
        k = key(row[2:11].reshape(3, 3))
        if k in lookup:
            raise ValueError('Duplicate reference triangle')
        lookup[k] = row
    result = [str(len(triangles))]
    centers = reference[:, 2:11].reshape(-1, 3, 3).mean(axis=1)
    matched = 0
    for ids in triangles:
        triangle = vertices[ids]
        row = lookup.get(key(triangle))
        if row is None:
            nearby = np.flatnonzero(np.linalg.norm(centers-triangle.mean(axis=0), axis=1) < .005)
            for idx in nearby:
                candidate = reference[idx, 2:11].reshape(3, 3)
                if np.linalg.norm(candidate[:, None]-triangle[None], axis=-1).min(axis=1).max() < .002:
                    row = reference[idx]
                    break
        if row is None:
            if not any(np.all(np.abs(triangle[:, axis]-value) < .001)
                       for axis, value in ((2, 0), (2, 2048), (0, -4096), (0, 4096))):
                raise ValueError(f'Unmatched non-plane triangle: {triangle}')
            result.append('-1 -1 6.2831853 6.2831853 6.2831853 0 '+' '.join(map(str,(triangle*.02).ravel())))
            continue
        native = row[2:11].reshape(3, 3)
        permutation = [int(np.linalg.norm(native-v, axis=1).argmin()) for v in triangle]
        if any(permutation[(e+1) % 3] != (permutation[e]+1) % 3 for e in range(3)):
            raise ValueError('Reference and CARL winding disagree')
        angles = row[11:14][permutation]
        flags = int(row[14])
        mapped_flags = sum(((flags >> e) & 1) << j | ((flags >> (e+3)) & 1) << (j+3)
                           for j, e in enumerate(permutation))
        native_bt = row[15:24].reshape(3,3)[permutation].ravel()
        result.append(' '.join(map(str, [int(row[0]), int(row[1]), *angles, mapped_flags, *native_bt])))
        matched += 1
    if matched != len(reference):
        raise ValueError(f'Matched {matched}/{len(reference)} reference triangles')
    Path(output).write_text('\n'.join(result)+'\n')
    print(f'Exported metadata for {matched} native triangles and {len(triangles)-matched} plane triangles')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('obj'); p.add_argument('reference'); p.add_argument('output')
    args = p.parse_args()
    export(args.obj, args.reference, args.output)
