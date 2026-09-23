"""Read the triangle meshes and Lambert colors in the pinned AgileX DAEs.

This deliberately supports the upstream asset format, not arbitrary COLLADA.
Node transforms are baked into vertices; separate normal indices are expanded
for MJCF's per-vertex normals. Equal colors share one mesh per link. No generated
files or third-party importer are needed.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def _floats(values) -> str:
    return " ".join(format(float(value), ".9g") for value in np.asarray(values).flat)


def visual_meshes(path: Path) -> tuple[dict[str, str], ...]:
    """Return inline MJCF mesh arrays plus RGBA, invalidating on source edits."""
    path = path.resolve()
    stat = path.stat()
    return _read_meshes(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=32)
def _read_meshes(path: Path, mtime_ns: int, size: int) -> tuple[dict[str, str], ...]:
    root = ET.parse(path).getroot()
    if root.findtext("c:asset/c:up_axis", namespaces=NS) != "Z_UP":
        raise ValueError(f"Expected a Z_UP AgileX visual mesh: {path}")
    if root.findall(".//c:texture", NS):
        raise ValueError(f"Image textures are not supported by the Piper DAE adapter: {path}")
    unit = root.find("c:asset/c:unit", NS)
    meter = float(unit.get("meter", "1")) if unit is not None else 1.0
    elements = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}

    def reference(url: str) -> ET.Element:
        if not url.startswith("#"):
            raise ValueError(f"Expected a local COLLADA reference in {path}: {url}")
        return elements[url[1:]]

    colors = {}
    for material in root.findall("c:library_materials/c:material", NS):
        effect = reference(material.find("c:instance_effect", NS).attrib["url"])
        lambert = effect.find("c:profile_COMMON/c:technique/c:lambert", NS)
        if lambert is None:
            raise ValueError(f"Expected a Lambert material in {path}")
        diffuse = lambert.findtext("c:diffuse/c:color", namespaces=NS)
        if diffuse is None:
            raise ValueError(f"Missing diffuse color in {path}")
        if lambert.find("c:transparent", NS) is not None:
            raise ValueError(f"Transparent COLLADA materials are not supported: {path}")
        colors[material.attrib["id"]] = tuple(float(v) for v in diffuse.split())

    def source(url: str) -> np.ndarray:
        element = reference(url)
        accessor = element.find("c:technique_common/c:accessor", NS)
        values = np.fromstring(reference(accessor.attrib["source"]).text, sep=" ")
        offset = int(accessor.get("offset", "0"))
        stride = int(accessor.get("stride", "1"))
        count = int(accessor.attrib["count"])
        params = [p.get("name") for p in accessor.findall("c:param", NS)]
        return values[offset:offset + count * stride].reshape(count, stride)[:,
                       [params.index(axis) for axis in ("X", "Y", "Z")]]

    groups: dict[tuple[float, ...], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    def visit(node: ET.Element, parent: np.ndarray) -> None:
        transform = parent.copy()
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "matrix":
                # COLLADA serializes the mathematical matrix in row order.
                transform = transform @ np.fromstring(child.text, sep=" ").reshape(4, 4)
            elif tag in ("translate", "rotate", "scale", "lookat", "skew", "instance_node"):
                raise ValueError(f"Unsupported upstream COLLADA node element {tag}: {path}")
        for instance in node.findall("c:instance_geometry", NS):
            bindings = {item.attrib["symbol"]: item.attrib["target"][1:]
                        for item in instance.findall("c:bind_material/c:technique_common/c:instance_material", NS)}
            mesh = reference(instance.attrib["url"]).find("c:mesh", NS)
            for primitive in mesh:
                tag = primitive.tag.split("}")[-1]
                if tag in ("source", "vertices"):
                    continue
                if tag != "triangles":
                    raise ValueError(f"Expected triangle primitives, got {tag}: {path}")
                inputs = {item.attrib["semantic"]: item for item in primitive.findall("c:input", NS)}
                width = max(int(item.get("offset", "0")) for item in inputs.values()) + 1
                indices = np.fromstring(primitive.findtext("c:p", namespaces=NS), sep=" ", dtype=int)
                indices = indices.reshape(int(primitive.attrib["count"]), 3, width).reshape(-1, width)
                vertex_input, normal_input = inputs["VERTEX"], inputs["NORMAL"]
                positions = reference(vertex_input.attrib["source"]).find("c:input[@semantic='POSITION']", NS)
                # Preserve hard edges: a position may have multiple normals.
                pairs, faces = np.unique(indices[:, [int(vertex_input.get("offset", "0")),
                                                     int(normal_input.get("offset", "0"))]],
                                         axis=0, return_inverse=True)
                vertices = source(positions.attrib["source"])[pairs[:, 0]]
                normals = source(normal_input.attrib["source"])[pairs[:, 1]]
                vertices = (vertices @ transform[:3, :3].T + transform[:3, 3]) * meter
                normals = normals @ np.linalg.inv(transform[:3, :3])
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                faces = faces.reshape(-1, 3)
                if np.linalg.det(transform[:3, :3]) < 0:
                    faces = faces[:, ::-1]
                color = colors[bindings[primitive.attrib["material"]]]
                groups.setdefault(color, []).append((vertices, normals, faces))
        for child in node.findall("c:node", NS):
            visit(child, transform)

    scene = reference(root.find("c:scene/c:instance_visual_scene", NS).attrib["url"])
    for node in scene.findall("c:node", NS):
        visit(node, np.eye(4))
    result = []
    for color, parts in groups.items():
        faces, offset = [], 0
        for vertices, _, part_faces in parts:
            faces.append(part_faces + offset)
            offset += len(vertices)
        result.append({
            "rgba": _floats(color),
            "vertex": _floats(np.concatenate([part[0] for part in parts])),
            "normal": _floats(np.concatenate([part[1] for part in parts])),
            "face": " ".join(str(int(v)) for v in np.concatenate(faces).flat),
        })
    if not result:
        raise ValueError(f"No visual triangles found in {path}")
    return tuple(result)
