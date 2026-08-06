from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test_impact_tool", REPO_ROOT / "scripts" / "test_impact.py"
)
assert SPEC is not None and SPEC.loader is not None
test_impact = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = test_impact
SPEC.loader.exec_module(test_impact)


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repo(tmp_path: Path, *, rules=None):
    _write(tmp_path / "pkg" / "__init__.py")
    _write(tmp_path / "pkg" / "core.py", "VALUE = 1\n")
    _write(tmp_path / "pkg" / "feature.py", "from .core import VALUE\n")
    _write(tmp_path / "pkg" / "other.py", "VALUE = 2\n")
    _write(tmp_path / "tests" / "test_feature.py", "import pkg.feature\n")
    _write(tmp_path / "tests" / "test_other.py", "import pkg.other\n")
    return test_impact.build_graph(
        tmp_path,
        revision="base-sha",
        source_roots=("pkg",),
        test_roots=("tests",),
        rules=rules,
    )


def _browser_selection(graph, changed_files: list[str], *, repo_root: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for browser-selector parity tests")
    repository_files = sorted(
        path.relative_to(repo_root).as_posix()
        for path in repo_root.rglob("*")
        if path.is_file()
    )
    payload = {
        "graph": graph.to_dict(),
        "changed_files": changed_files,
        "repository_files": repository_files,
    }
    javascript = """
const fs = require("fs");
const selector = require(process.argv[1]);
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const result = selector.selectTests(
  input.graph,
  input.changed_files,
  input.repository_files,
);
process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        [
            node,
            "-e",
            javascript,
            str(REPO_ROOT / "docs" / "test-impact" / "selector.js"),
        ],
        input=json.dumps(payload),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _assert_selection_parity(python_selection: dict, browser_selection: dict) -> None:
    for field in (
        "mode",
        "fallback",
        "reason_code",
        "graph_revision",
        "changed_files",
        "test_files",
    ):
        assert browser_selection[field] == python_selection[field]


def test_build_graph_selects_transitive_python_dependents(tmp_path: Path) -> None:
    graph = _repo(tmp_path)

    selection = test_impact.select_tests(graph, ["pkg/core.py"], repo_root=tmp_path)

    assert selection["mode"] == "selected"
    assert selection["fallback"] is False
    assert selection["test_files"] == ["tests/test_feature.py"]


def test_browser_selector_matches_python_and_explains_each_changed_file(
    tmp_path: Path,
) -> None:
    graph = _repo(tmp_path)
    changed_files = ["pkg/core.py", "pkg/feature.py"]

    python_selection = test_impact.select_tests(
        graph, changed_files, repo_root=tmp_path
    )
    browser_selection = _browser_selection(graph, changed_files, repo_root=tmp_path)

    _assert_selection_parity(python_selection, browser_selection)
    assert browser_selection["test_reasons"] == {
        "tests/test_feature.py": [
            {
                "changed_file": "pkg/core.py",
                "kind": "dependency_path",
                "path": [
                    "pkg/core.py",
                    "pkg/feature.py",
                    "tests/test_feature.py",
                ],
            },
            {
                "changed_file": "pkg/feature.py",
                "kind": "dependency_path",
                "path": ["pkg/feature.py", "tests/test_feature.py"],
            },
        ]
    }


def test_browser_selector_explains_direct_tests_and_dependency_overrides(
    tmp_path: Path,
) -> None:
    rules = test_impact.ImpactRules(
        dependency_overrides=(
            test_impact.DependencyOverride(
                source_patterns=("generated/**",),
                test_patterns=("tests/test_other.py",),
            ),
        )
    )
    graph = _repo(tmp_path, rules=rules)

    direct = _browser_selection(graph, ["tests/test_feature.py"], repo_root=tmp_path)
    overridden = _browser_selection(graph, ["generated/kernel.inc"], repo_root=tmp_path)

    assert direct["test_reasons"]["tests/test_feature.py"] == [
        {
            "changed_file": "tests/test_feature.py",
            "kind": "direct_test",
            "path": ["tests/test_feature.py"],
        }
    ]
    assert overridden["test_reasons"]["tests/test_other.py"] == [
        {
            "changed_file": "generated/kernel.inc",
            "kind": "dependency_override",
            "path": ["generated/kernel.inc", "tests/test_other.py"],
        }
    ]


@pytest.mark.parametrize(
    ("changed_file", "expected_mode"),
    [
        ("docs/guide.rst", "none"),
        ("ci/config.yml", "all"),
        ("unknown/file.txt", "all"),
    ],
)
def test_browser_selector_matches_python_rule_and_fallback_modes(
    tmp_path: Path, changed_file: str, expected_mode: str
) -> None:
    rules = test_impact.ImpactRules(
        global_patterns=("ci/**",),
        ignored_patterns=("docs/**",),
    )
    graph = _repo(tmp_path, rules=rules)

    python_selection = test_impact.select_tests(
        graph, [changed_file], repo_root=tmp_path
    )
    browser_selection = _browser_selection(graph, [changed_file], repo_root=tmp_path)

    _assert_selection_parity(python_selection, browser_selection)
    assert browser_selection["mode"] == expected_mode


def test_native_cuda_source_reaches_tests_through_python_jit_module(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pkg" / "__init__.py")
    _write(tmp_path / "pkg" / "feature.py", 'SOURCE = "kernel.cu"\n')
    _write(tmp_path / "tests" / "test_feature.py", "import pkg.feature\n")
    _write(tmp_path / "csrc" / "kernel.cu", "__global__ void kernel() {}\n")
    graph = test_impact.build_graph(
        tmp_path,
        revision="base-sha",
        source_roots=("pkg",),
        test_roots=("tests",),
        native_roots=("csrc",),
    )

    selection = test_impact.select_tests(graph, ["csrc/kernel.cu"], repo_root=tmp_path)

    assert selection["mode"] == "selected"
    assert selection["test_files"] == ["tests/test_feature.py"]


def test_native_header_include_is_transitive(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py")
    _write(tmp_path / "pkg" / "feature.py", 'SOURCE = "kernels/kernel.cu"\n')
    _write(tmp_path / "tests" / "test_feature.py", "import pkg.feature\n")
    _write(
        tmp_path / "csrc" / "kernels" / "kernel.cu",
        '#include "detail/common.cuh"\n',
    )
    _write(tmp_path / "csrc" / "kernels" / "detail" / "common.cuh")
    graph = test_impact.build_graph(
        tmp_path,
        revision="base-sha",
        source_roots=("pkg",),
        test_roots=("tests",),
        native_roots=("csrc",),
    )

    selection = test_impact.select_tests(
        graph,
        ["csrc/kernels/detail/common.cuh"],
        repo_root=tmp_path,
    )

    assert selection["mode"] == "selected"
    assert selection["test_files"] == ["tests/test_feature.py"]


def test_jinja_include_is_transitive(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py")
    _write(tmp_path / "pkg" / "feature.py", 'TEMPLATE = "kernel.jinja"\n')
    _write(tmp_path / "tests" / "test_feature.py", "import pkg.feature\n")
    _write(
        tmp_path / "csrc" / "kernel.jinja",
        '{% include "detail/common.jinja" %}\n',
    )
    _write(tmp_path / "csrc" / "detail" / "common.jinja", "// shared\n")
    graph = test_impact.build_graph(
        tmp_path,
        revision="base-sha",
        source_roots=("pkg",),
        test_roots=("tests",),
        native_roots=("csrc",),
    )

    selection = test_impact.select_tests(
        graph, ["csrc/detail/common.jinja"], repo_root=tmp_path
    )

    assert selection["mode"] == "selected"
    assert selection["test_files"] == ["tests/test_feature.py"]


def test_dynamic_native_filename_matches_bounded_files(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py")
    _write(
        tmp_path / "pkg" / "feature.py",
        'SOURCE = f"kernel_{variant}.cu"\n',
    )
    _write(tmp_path / "tests" / "test_feature.py", "import pkg.feature\n")
    _write(tmp_path / "csrc" / "kernel_a.cu")
    _write(tmp_path / "csrc" / "kernel_b.cu")
    graph = test_impact.build_graph(
        tmp_path,
        revision="base-sha",
        source_roots=("pkg",),
        test_roots=("tests",),
        native_roots=("csrc",),
    )

    selection = test_impact.select_tests(
        graph, ["csrc/kernel_b.cu"], repo_root=tmp_path
    )

    assert selection["mode"] == "selected"
    assert selection["test_files"] == ["tests/test_feature.py"]


def test_unreferenced_native_file_falls_back_to_all(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py")
    _write(tmp_path / "tests" / "test_feature.py", "def test_feature(): pass\n")
    _write(tmp_path / "csrc" / "unreferenced.cu")
    graph = test_impact.build_graph(
        tmp_path,
        revision="base-sha",
        source_roots=("pkg",),
        test_roots=("tests",),
        native_roots=("csrc",),
    )

    selection = test_impact.select_tests(
        graph, ["csrc/unreferenced.cu"], repo_root=tmp_path
    )

    assert selection["mode"] == "all"
    assert selection["reason_code"] == "uncovered_dependency"


def test_native_graph_serialization_round_trip(tmp_path: Path) -> None:
    _write(tmp_path / "csrc" / "kernel.cu")
    graph = _repo(tmp_path)

    restored = test_impact.ImpactGraph.from_dict(graph.to_dict())

    assert restored == graph
    assert restored.native_roots == ("csrc",)
    assert "csrc/kernel.cu" in restored.nodes


def test_build_graph_models_pytest_conftest_as_implicit_dependency(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pkg" / "__init__.py")
    _write(tmp_path / "tests" / "conftest.py", "HELPER = 1\n")
    _write(tmp_path / "tests" / "a" / "test_a.py", "def test_a(): pass\n")
    _write(tmp_path / "tests" / "b" / "test_b.py", "def test_b(): pass\n")
    graph = test_impact.build_graph(
        tmp_path,
        revision="base-sha",
        source_roots=("pkg",),
        test_roots=("tests",),
    )

    selection = test_impact.select_tests(
        graph, ["tests/conftest.py"], repo_root=tmp_path
    )

    assert selection["mode"] == "selected"
    assert selection["test_files"] == [
        "tests/a/test_a.py",
        "tests/b/test_b.py",
    ]


def test_package_reexports_resolve_at_the_consumer_without_selecting_every_test(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "from .feature import FEATURE\n")
    _write(tmp_path / "pkg" / "feature.py", "FEATURE = 1\n")
    _write(tmp_path / "pkg" / "other.py", "OTHER = 2\n")
    _write(tmp_path / "tests" / "test_feature.py", "import pkg\nassert pkg.FEATURE\n")
    _write(tmp_path / "tests" / "test_other.py", "import pkg.other\n")
    graph = test_impact.build_graph(
        tmp_path,
        revision="base-sha",
        source_roots=("pkg",),
        test_roots=("tests",),
    )

    selection = test_impact.select_tests(graph, ["pkg/feature.py"], repo_root=tmp_path)

    assert selection["mode"] == "selected"
    assert selection["test_files"] == ["tests/test_feature.py"]


@pytest.mark.parametrize(
    ("changed_file", "reason_code"),
    [
        ("csrc/kernel.cu", "unknown_change"),
        ("pkg/not_imported.py", "uncovered_dependency"),
    ],
)
def test_unknown_or_uncovered_changes_fall_back_to_all(
    tmp_path: Path, changed_file: str, reason_code: str
) -> None:
    graph = _repo(tmp_path)
    if changed_file == "pkg/not_imported.py":
        _write(tmp_path / changed_file, "VALUE = 3\n")
        graph = test_impact.build_graph(
            tmp_path,
            revision="base-sha",
            source_roots=("pkg",),
            test_roots=("tests",),
        )

    selection = test_impact.select_tests(graph, [changed_file], repo_root=tmp_path)

    assert selection["mode"] == "all"
    assert selection["fallback"] is True
    assert selection["reason_code"] == reason_code


def test_rules_support_ignored_files_and_explicit_native_overrides(
    tmp_path: Path,
) -> None:
    rules = test_impact.ImpactRules(
        ignored_patterns=("docs/**",),
        dependency_overrides=(
            test_impact.DependencyOverride(
                source_patterns=("csrc/feature/**",),
                test_patterns=("tests/test_feature.py",),
            ),
        ),
    )
    graph = _repo(tmp_path, rules=rules)

    ignored = test_impact.select_tests(graph, ["docs/guide.rst"], repo_root=tmp_path)
    overridden = test_impact.select_tests(
        graph, ["csrc/feature/kernel.cu"], repo_root=tmp_path
    )

    assert ignored["mode"] == "none"
    assert overridden["mode"] == "selected"
    assert overridden["test_files"] == ["tests/test_feature.py"]


def test_dependency_override_takes_precedence_over_ignored_pattern(
    tmp_path: Path,
) -> None:
    rules = test_impact.ImpactRules(
        ignored_patterns=("docs/**",),
        dependency_overrides=(
            test_impact.DependencyOverride(
                source_patterns=("docs/test-impact/selector.js",),
                test_patterns=("tests/test_feature.py",),
            ),
        ),
    )
    graph = _repo(tmp_path, rules=rules)

    python_selection = test_impact.select_tests(
        graph, ["docs/test-impact/selector.js"], repo_root=tmp_path
    )
    browser_selection = _browser_selection(
        graph, ["docs/test-impact/selector.js"], repo_root=tmp_path
    )

    _assert_selection_parity(python_selection, browser_selection)
    assert browser_selection["mode"] == "selected"
    assert (
        browser_selection["test_reasons"]["tests/test_feature.py"][0]["kind"]
        == "dependency_override"
    )


def test_revision_mismatch_falls_back_to_all(tmp_path: Path) -> None:
    graph = _repo(tmp_path)

    selection = test_impact.select_tests(
        graph,
        ["pkg/core.py"],
        repo_root=tmp_path,
        expected_revision="different-sha",
    )

    assert selection["mode"] == "all"
    assert selection["reason_code"] == "graph_revision_mismatch"


def test_deleted_test_does_not_force_a_full_suite(tmp_path: Path) -> None:
    graph = _repo(tmp_path)
    (tmp_path / "tests" / "test_feature.py").unlink()

    selection = test_impact.select_tests(
        graph, ["tests/test_feature.py"], repo_root=tmp_path
    )

    assert selection["mode"] == "none"


def test_selection_manifest_resolves_multiple_files(tmp_path: Path) -> None:
    graph = _repo(tmp_path)
    selection = test_impact.select_tests(
        graph,
        ["tests/test_feature.py", "tests/test_other.py"],
        repo_root=tmp_path,
    )
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(selection), encoding="utf-8")

    mode, files = test_impact.resolve_selection_manifest(manifest, repo_root=tmp_path)

    assert mode == "selected"
    assert files == ("tests/test_feature.py", "tests/test_other.py")


def test_stale_selection_manifest_is_rejected(tmp_path: Path) -> None:
    graph = _repo(tmp_path)
    selection = test_impact.select_tests(
        graph, ["tests/test_feature.py"], repo_root=tmp_path
    )
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(test_impact.ImpactError, match="does not match"):
        test_impact.resolve_selection_manifest(
            manifest,
            repo_root=tmp_path,
            expected_revision="different-sha",
        )


def test_invalid_selection_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": test_impact.SELECTION_SCHEMA_VERSION,
                "mode": "selected",
                "test_files": ["../outside.py"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(test_impact.ImpactError, match="unsafe path"):
        test_impact.resolve_selection_manifest(manifest, repo_root=tmp_path)


def test_select_command_writes_all_manifest_when_graph_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "selection.json"

    exit_code = test_impact.main(
        [
            "select",
            "--repository-root",
            str(tmp_path),
            "--graph",
            str(tmp_path / "missing-graph.json"),
            "--changed-file",
            "pkg/core.py",
            "--output",
            str(output),
        ]
    )

    selection = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert selection["mode"] == "all"
    assert selection["reason_code"] == "selection_error"
    assert "fell back to all tests" in capsys.readouterr().err


def test_resolve_command_falls_back_to_all_for_invalid_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "selection.json"
    manifest.write_text("not json", encoding="utf-8")

    exit_code = test_impact.main(
        [
            "resolve-selection",
            "--repository-root",
            str(tmp_path),
            "--manifest",
            str(manifest),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "all\n\n"
    assert "running all tests" in captured.err


def test_unit_test_runner_skips_installation_for_none_manifest(tmp_path: Path) -> None:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": test_impact.SELECTION_SCHEMA_VERSION,
                "mode": "none",
                "graph_revision": revision,
                "test_files": [],
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("TEST_PATH", None)
    environment.pop("TEST_SELECTION_MANIFEST", None)
    environment.pop("SOURCE_GIT_SHA", None)
    environment.pop("GITHUB_SHA", None)
    environment.pop("CI_COMMIT_SHA", None)
    environment.update({"MAX_JOBS": "1", "PIP_CONSTRAINT": os.devnull})

    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts" / "task_run_unit_tests.sh"),
            "--test-selection-manifest",
            str(manifest),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "no unit tests are affected" in result.stdout
    assert "pip install" not in result.stdout
