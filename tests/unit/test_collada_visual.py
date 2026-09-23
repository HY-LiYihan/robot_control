"""Exercise scene transforms, units, normals and bindings through a real DAE."""
import xml.etree.ElementTree as ET

import numpy as np

from robot_control.backends.collada_visual import NS, visual_meshes
from robot_control.backends.piper_model import ASSET_ROOT


def test_nested_transform_units_and_mirrored_normals(tmp_path):
    source = ASSET_ROOT / "piper/meshes/dae/link6.dae"
    original, = visual_meshes(source)
    tree = ET.parse(source)
    root = tree.getroot()
    root.find("c:asset/c:unit", NS).set("meter", "0.01")
    scene = root.find("c:library_visual_scenes/c:visual_scene", NS)
    node = scene.find("c:node", NS)
    # Reflection + nonuniform scale, with a parent translation. This catches
    # matrix transposition, ignored scene instances and incorrect normal math.
    node.find("c:matrix", NS).text = "0 2 0 .1 3 0 0 .2 0 0 4 .3 0 0 0 1"
    scene.remove(node)
    ns = "{" + NS["c"] + "}"
    parent = ET.SubElement(scene, ns + "node")
    ET.SubElement(parent, ns + "matrix").text = "1 0 0 .5 0 1 0 0 0 0 1 0 0 0 0 1"
    parent.append(node)
    path = tmp_path / "transformed.dae"
    tree.write(path)
    changed, = visual_meshes(path)
    vertices = np.fromstring(original["vertex"], sep=" ").reshape(-1, 3)
    expected = np.column_stack((2 * vertices[:, 1] + .6,
                                3 * vertices[:, 0] + .2, 4 * vertices[:, 2] + .3)) * .01
    np.testing.assert_allclose(np.fromstring(changed["vertex"], sep=" ").reshape(-1, 3),
                               expected, atol=1e-10)
    normals = np.fromstring(original["normal"], sep=" ").reshape(-1, 3)
    expected_normals = np.column_stack((normals[:, 1] / 2, normals[:, 0] / 3, normals[:, 2] / 4))
    expected_normals /= np.linalg.norm(expected_normals, axis=1, keepdims=True)
    np.testing.assert_allclose(np.fromstring(changed["normal"], sep=" ").reshape(-1, 3),
                               expected_normals, atol=1e-8)
    faces = np.fromstring(original["face"], sep=" ", dtype=int).reshape(-1, 3)
    np.testing.assert_array_equal(np.fromstring(changed["face"], sep=" ", dtype=int).reshape(-1, 3),
                                  faces[:, ::-1])
    assert changed["rgba"] == original["rgba"]
