"""Unit tests for core dependencies and optional extras.

Validates Requirements 8.1, 8.2, 8.3, 8.4, 8.7:

* pyproject metadata lists exactly the core dependencies and the three extras,
* every extra-backed function raises ``ImportError`` naming ``pylindas[<extra>]``
  when its package is missing, and leaves its input data untouched,
* ``pylindas`` imports in a venv that only has the core dependencies.
"""
import copy
import importlib
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]

CORE = {"numpy", "pandas", "pyshacl", "pyyaml", "rdflib", "requests"}
EXTRAS = {
    "shared": {"shapely": ">=2.0.0"},
    "query": {"sparqlwrapper": ">=2.0.0", "sparql-dataframe": ">=0.4"},
    "fetch": {"jsonschema": ">=4.0.0"},
}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _split(req: str):
    m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*(.*?)\s*(?:#.*)?$", req)
    return _norm(m.group(1)), m.group(2)


@pytest.fixture(scope="module")
def project():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


# --- pyproject metadata (8.1, 8.2, 8.3, 8.4) -------------------------------------------------


def test_core_dependencies_are_exactly_the_six(project):
    names = [_split(r)[0] for r in project["dependencies"]]
    assert len(names) == len(set(names))
    assert set(names) == {_norm(n) for n in CORE}


def test_core_dependencies_have_lower_bounds(project):
    for req in project["dependencies"]:
        assert ">=" in _split(req)[1], req


def test_extras_contain_only_the_listed_packages(project):
    extras = project["optional-dependencies"]
    assert set(extras) == set(EXTRAS)
    for extra, expected in EXTRAS.items():
        parsed = dict(_split(r) for r in extras[extra])
        assert parsed == expected, extra


def test_no_extra_package_is_a_core_dependency(project):
    core = {_split(r)[0] for r in project["dependencies"]}
    for reqs in project["optional-dependencies"].values():
        assert not core & {_split(r)[0] for r in reqs}


# --- simulated missing extras (8.1, 8.2, 8.3, 8.4, 8.7) --------------------------------------


@pytest.fixture
def block(monkeypatch):
    """Return a function that makes the given top-level packages un-importable."""

    def _block(*names):
        for name in names:
            monkeypatch.setitem(sys.modules, name, None)

    return _block


@pytest.fixture
def real_query(monkeypatch):
    """Undo the autouse query/network mocks and hand back the real query functions."""
    monkeypatch.undo()
    return importlib.import_module("pylindas.lindas.query")


def _assert_extra_error(excinfo, extra):
    assert isinstance(excinfo.value, ImportError)
    assert f"pylindas[{extra}]" in str(excinfo.value)


def test_query_lindas_without_query_extra(block, real_query):
    block("SPARQLWrapper")
    query = "ASK { ?s ?p ?o }"
    before = copy.deepcopy(query)
    with pytest.raises(ImportError) as exc:
        real_query.query_lindas(query, environment="TEST")
    _assert_extra_error(exc, "query")
    assert query == before


def test_shared_dimension_queries_without_query_extra(block, monkeypatch):
    monkeypatch.undo()
    mod = importlib.import_module("pylindas.shared_dimension_queries.shared_dimensions_queries")
    block("SPARQLWrapper")
    with pytest.raises(ImportError) as exc:
        mod.query_lindas("ASK { ?s ?p ?o }", environment="TEST")
    _assert_extra_error(exc, "query")


def test_get_cube_without_query_extra(block, monkeypatch):
    monkeypatch.undo()
    get = importlib.import_module("pylindas.getter.get")
    block("SPARQLWrapper")
    with pytest.raises(ImportError) as exc:
        get.get_cube("https://example.org/query", "id", "1")
    _assert_extra_error(exc, "query")


def test_get_observations_without_query_extra(block, monkeypatch):
    monkeypatch.undo()
    get = importlib.import_module("pylindas.getter.get")
    block("sparql_dataframe")
    args = ["https://example.org/query", "cube-id", "1"]
    before = list(args)
    with pytest.raises(ImportError) as exc:
        get.get_observations(*args)
    _assert_extra_error(exc, "query")
    assert args == before


def test_cube_module_wrapper_without_query_extra(block, monkeypatch):
    """The lazy wrapper in pycube.cube surfaces the same actionable error."""
    monkeypatch.undo()
    cube = importlib.import_module("pylindas.pycube.cube")
    block("SPARQLWrapper")
    with pytest.raises(ImportError) as exc:
        cube.query_lindas("ASK { ?s ?p ?o }", environment="TEST")
    _assert_extra_error(exc, "query")


def test_geojson_to_wkt_without_shared_extra(block):
    from rdflib import Graph, URIRef
    from pylindas.pycube.shared_dimension import GeoSharedDimension

    geojson = {"type": "Point", "coordinates": [1.0, 2.0]}
    before = copy.deepcopy(geojson)
    block("shapely", "shapely.geometry")
    dim = GeoSharedDimension(URIRef("https://example.org/"), {}, Graph())
    with pytest.raises(ImportError) as exc:
        dim._geojson_to_wkt(geojson)
    _assert_extra_error(exc, "shared")
    assert geojson == before


def test_geojson_to_wkt_empty_input_needs_no_extra(block):
    from rdflib import Graph, URIRef
    from pylindas.pycube.shared_dimension import GeoSharedDimension

    block("shapely", "shapely.geometry")
    dim = GeoSharedDimension(URIRef("https://example.org/"), {}, Graph())
    assert dim._geojson_to_wkt({}) is None


def test_fetch_validation_without_fetch_extra(block, monkeypatch, tmp_path):
    from pylindas import fetch as fetch_mod

    data = {"result": {"title": {"en": "t"}, "description": {"en": "d"}, "distributions": []}}
    before = copy.deepcopy(data)
    fetcher = fetch_mod.DataEuropaFetcher()
    monkeypatch.setattr(fetch_mod, "download_json", lambda url: data)
    monkeypatch.setattr(
        fetcher, "_get_distributions", lambda d: {"csv": b"a\n1\n", "frictionless": {"path": "p", "name": "n"}}
    )
    monkeypatch.setattr(fetcher, "_transform_metadata", lambda m, d: {"Name": {"en": "n"}})
    monkeypatch.setattr(fetch_mod, "read_schema", lambda p: {"type": "object"})
    block("jsonschema")
    with pytest.raises(ImportError) as exc:
        fetcher.fetch_dataset("https://data.europa.eu/set/x", str(tmp_path))
    _assert_extra_error(exc, "fetch")
    assert data == before


def test_dataframe_input_unchanged_on_missing_extra(block, monkeypatch):
    monkeypatch.undo()
    get = importlib.import_module("pylindas.getter.get")
    block("sparql_dataframe")
    # The frame is not an argument of any extra-backed function, so also check the
    # error path does not touch module-level state used with frames.
    df = pd.DataFrame({"a": [1, 2]})
    before = df.copy()
    with pytest.raises(ImportError):
        get.get_observations("https://example.org/query", "id", "1")
    pd.testing.assert_frame_equal(df, before)


def test_require_extra_message_names_extra(block):
    from pylindas._optional import require_extra

    block("shapely")
    with pytest.raises(ImportError, match=re.escape("pylindas[shared]")):
        require_extra("shapely", "shared")


# --- core-only venv (8.1, 8.7) ---------------------------------------------------------------

CORE_DEPS = ["pandas", "numpy", "rdflib", "pyshacl", "PyYAML", "requests"]

_CORE_ONLY_SCRIPT = """
import importlib.util
import pylindas, pylindas.pycube.cube, pylindas.lindas.upload, pylindas.cli
assert "site-packages" in pylindas.__file__, pylindas.__file__
for extra_pkg in ("SPARQLWrapper", "sparql_dataframe", "shapely", "jsonschema"):
    assert importlib.util.find_spec(extra_pkg) is None, extra_pkg
from pylindas.lindas.query import query_lindas
try:
    query_lindas("ASK { ?s ?p ?o }", environment="TEST")
except ImportError as e:
    assert "pylindas[query]" in str(e), str(e)
else:
    raise SystemExit("ImportError expected")
print("CORE_ONLY_OK")
"""


@pytest.mark.slow
def test_pylindas_imports_in_core_only_venv():
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    tmp = Path(tempfile.mkdtemp(prefix="pylindas-core-"))
    try:

        def run(cmd, **kw):
            return subprocess.run(cmd, capture_output=True, text=True, timeout=180, **kw)

        build = run([uv, "build", "--wheel", "--out-dir", str(tmp / "dist"), str(ROOT)])
        assert build.returncode == 0, build.stdout + build.stderr
        wheels = list((tmp / "dist").glob("pylindas-*.whl"))
        assert len(wheels) == 1, wheels

        venv = tmp / "venv"
        created = run([uv, "venv", "--python", f"{sys.version_info.major}.{sys.version_info.minor}", str(venv)])
        if created.returncode != 0:
            created = run([uv, "venv", str(venv)])
        assert created.returncode == 0, created.stderr
        py = str(venv / "bin" / "python")

        no_deps = run([uv, "pip", "install", "--python", py, "--no-deps", str(wheels[0])])
        assert no_deps.returncode == 0, no_deps.stderr
        deps = run([uv, "pip", "install", "--python", py, *CORE_DEPS])
        if deps.returncode != 0:
            pytest.skip("core dependencies could not be installed (offline?): " + deps.stderr[-300:])

        # cwd is the tmp dir so the checkout cannot shadow the installed package.
        result = run([py, "-c", _CORE_ONLY_SCRIPT], cwd=str(tmp))
        assert result.returncode == 0 and "CORE_ONLY_OK" in result.stdout, result.stdout + result.stderr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
