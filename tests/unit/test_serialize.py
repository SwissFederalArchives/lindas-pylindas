"""Unit tests for ``Cube.serialize`` (Requirements 2.1, 2.2, 2.6).

Note on 2.2: the file variant returns the cube (``self``), a deliberate
deviation from the requirement text (which says ``None``); it is documented
in the ``serialize`` docstring.
"""
import copy
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml
from rdflib import Graph
from rdflib.compare import isomorphic

from pylindas.pycube.cube import Cube

_DESCRIPTION_PATH = (
    Path(__file__).resolve().parents[2] / "example" / "Cubes" / "mock" / "description.yml"
)
with open(_DESCRIPTION_PATH, encoding="utf-8") as _f:
    _DESCRIPTION = yaml.safe_load(_f)

_NON_ASCII = "Zürich äöü é"


@pytest.fixture
def cube() -> Cube:
    df = pd.DataFrame(
        {
            "Jahr": [2020, 2021],
            "Station": ["Bern", "Zürich"],
            "Wert": [1.5, 2.5],
            "Standardfehler": [0.1, 0.2],
            "Status": [_NON_ASCII, "plain"],
        }
    )
    c = Cube(
        dataframe=df,
        cube_yaml=copy.deepcopy(_DESCRIPTION),
        environment="TEST",
        local=True,
    )
    return c.prepare_data().write_cube().write_observations().write_shape()


def test_no_file_created_without_filename(cube, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = cube.serialize()
    assert isinstance(result, str) and result
    assert list(tmp_path.iterdir()) == []


def test_file_variant_returns_cube(cube, tmp_path):
    path = tmp_path / "cube.ttl"
    assert cube.serialize(str(path)) is cube
    assert path.exists()
    assert isomorphic(Graph().parse(str(path), format="turtle"), cube._graph)


def test_file_variant_accepts_pathlike(cube, tmp_path):
    path = tmp_path / "cube.ttl"
    assert cube.serialize(path) is cube
    assert path.exists()


def test_write_replaces_existing_content(cube, tmp_path):
    path = tmp_path / "cube.ttl"
    path.write_text("OLD CONTENT " * 1000, encoding="utf-8")
    cube.serialize(str(path))
    content = path.read_text(encoding="utf-8")
    assert "OLD CONTENT" not in content
    assert content == cube.serialize()


def test_file_is_utf8_and_non_ascii_round_trips(cube, tmp_path):
    path = tmp_path / "cube.ttl"
    cube.serialize(str(path))
    raw = path.read_bytes()
    assert _NON_ASCII.encode("utf-8") in raw
    parsed = Graph().parse(data=raw.decode("utf-8"), format="turtle")
    assert any(str(o) == _NON_ASCII for o in parsed.objects())


def test_unwritable_path_names_path_and_leaves_graph_unchanged(cube, tmp_path):
    bad = tmp_path / "missing_dir" / "cube.ttl"
    before_count = len(cube._graph)
    before_copy = Graph()
    for t in cube._graph:
        before_copy.add(t)

    with pytest.raises(OSError) as excinfo:
        cube.serialize(str(bad))

    assert str(bad) in str(excinfo.value)
    assert not bad.exists()
    assert len(cube._graph) == before_count
    assert isomorphic(cube._graph, before_copy)
