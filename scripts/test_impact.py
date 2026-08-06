#!/usr/bin/env python3
"""Build and query a conservative Python/native test-impact graph.

The selector is deliberately fail-open: if it cannot prove that a change is
covered by the graph or an explicit rule, it emits a manifest requesting the
full test suite.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import sys
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


GRAPH_SCHEMA_VERSION = 2
RULES_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1

NATIVE_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".inl",
        ".j2",
        ".jinja",
        ".jinja2",
    }
)
_NATIVE_REFERENCE_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./*{}%+-]+\.(?:jinja2|jinja|cuh|cpp|hpp|inl|cc|cu|j2|c|h))"
)
_NATIVE_INCLUDE_RE = re.compile(
    r"^\s*#\s*include\s*[<\"](?P<path>[^>\"]+)[>\"]", re.MULTILINE
)
_JINJA_REFERENCE_RE = re.compile(
    r"\{%\s*(?:include|import|extends)\s+[\"'](?P<path>[^\"']+)[\"']"
    r"|\{%\s*from\s+[\"'](?P<from_path>[^\"']+)[\"']\s+import\b"
)


class ImpactError(RuntimeError):
    """Raised when an impact artifact cannot be built or validated."""


def _normalized_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImpactError(f"{field} must contain non-empty path strings")
    raw = value.strip().replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ImpactError(f"{field} contains an unsafe path: {value!r}")
    normalized = path.as_posix()
    if normalized in ("", ".") or ":" in path.parts[0]:
        raise ImpactError(f"{field} contains an unsafe path: {value!r}")
    return normalized


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ImpactError(f"{field} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class DependencyOverride:
    source_patterns: tuple[str, ...]
    test_patterns: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Any, *, field: str) -> DependencyOverride:
        if not isinstance(value, dict):
            raise ImpactError(f"{field} must be an object")
        sources = _string_list(
            value.get("source_patterns"), field=f"{field}.source_patterns"
        )
        tests = _string_list(value.get("test_patterns"), field=f"{field}.test_patterns")
        if not sources or not tests:
            raise ImpactError(f"{field} patterns must not be empty")
        return cls(sources, tests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_patterns": list(self.source_patterns),
            "test_patterns": list(self.test_patterns),
        }


@dataclass(frozen=True)
class ImpactRules:
    global_patterns: tuple[str, ...] = ()
    ignored_patterns: tuple[str, ...] = ()
    dependency_overrides: tuple[DependencyOverride, ...] = ()

    @classmethod
    def from_dict(cls, value: Any) -> ImpactRules:
        if not isinstance(value, dict):
            raise ImpactError("impact rules must be a JSON object")
        if value.get("schema_version") != RULES_SCHEMA_VERSION:
            raise ImpactError(
                "unsupported impact-rules schema_version: "
                f"{value.get('schema_version')!r}"
            )
        overrides_value = value.get("dependency_overrides", [])
        if not isinstance(overrides_value, list):
            raise ImpactError("dependency_overrides must be a list")
        return cls(
            global_patterns=_string_list(
                value.get("global_patterns", []), field="global_patterns"
            ),
            ignored_patterns=_string_list(
                value.get("ignored_patterns", []), field="ignored_patterns"
            ),
            dependency_overrides=tuple(
                DependencyOverride.from_dict(
                    item, field=f"dependency_overrides[{index}]"
                )
                for index, item in enumerate(overrides_value)
            ),
        )

    @classmethod
    def load(cls, path: Path | None) -> ImpactRules:
        if path is None:
            return cls()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ImpactError(f"cannot read impact rules {path}: {error}") from error
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RULES_SCHEMA_VERSION,
            "global_patterns": list(self.global_patterns),
            "ignored_patterns": list(self.ignored_patterns),
            "dependency_overrides": [
                override.to_dict() for override in self.dependency_overrides
            ],
        }


@dataclass(frozen=True)
class ImpactGraph:
    revision: str
    source_roots: tuple[str, ...]
    test_roots: tuple[str, ...]
    native_roots: tuple[str, ...]
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    tests: tuple[str, ...]
    rules: ImpactRules
    generated_at: str

    @classmethod
    def from_dict(cls, value: Any) -> ImpactGraph:
        if not isinstance(value, dict):
            raise ImpactError("impact graph must be a JSON object")
        if value.get("schema_version") != GRAPH_SCHEMA_VERSION:
            raise ImpactError(
                "unsupported impact-graph schema_version: "
                f"{value.get('schema_version')!r}"
            )
        revision = value.get("revision")
        generated_at = value.get("generated_at")
        if not isinstance(revision, str) or not revision.strip():
            raise ImpactError("impact graph revision must be a non-empty string")
        if not isinstance(generated_at, str) or not generated_at:
            raise ImpactError("impact graph generated_at must be a string")

        source_roots = tuple(
            _normalized_relative_path(item, field="source_roots")
            for item in _string_list(value.get("source_roots"), field="source_roots")
        )
        test_roots = tuple(
            _normalized_relative_path(item, field="test_roots")
            for item in _string_list(value.get("test_roots"), field="test_roots")
        )
        native_roots = tuple(
            _normalized_relative_path(item, field="native_roots")
            for item in _string_list(value.get("native_roots"), field="native_roots")
        )
        nodes = tuple(
            _normalized_relative_path(item, field="nodes")
            for item in _string_list(value.get("nodes"), field="nodes")
        )
        tests = tuple(
            _normalized_relative_path(item, field="tests")
            for item in _string_list(value.get("tests"), field="tests")
        )
        if len(set(nodes)) != len(nodes) or len(set(tests)) != len(tests):
            raise ImpactError(
                "impact graph nodes and tests must not contain duplicates"
            )
        node_set = set(nodes)
        if not set(tests).issubset(node_set):
            raise ImpactError("impact graph tests must also be graph nodes")

        raw_edges = value.get("edges")
        if not isinstance(raw_edges, list):
            raise ImpactError("impact graph edges must be a list")
        edges: list[tuple[str, str]] = []
        for index, edge in enumerate(raw_edges):
            if not isinstance(edge, list) or len(edge) != 2:
                raise ImpactError(f"edges[{index}] must be a two-element list")
            importer = _normalized_relative_path(edge[0], field=f"edges[{index}]")
            dependency = _normalized_relative_path(edge[1], field=f"edges[{index}]")
            if importer not in node_set or dependency not in node_set:
                raise ImpactError(f"edges[{index}] references a node not in the graph")
            edges.append((importer, dependency))

        return cls(
            revision=revision.strip(),
            source_roots=source_roots,
            test_roots=test_roots,
            native_roots=native_roots,
            nodes=nodes,
            edges=tuple(edges),
            tests=tests,
            rules=ImpactRules.from_dict(value.get("rules")),
            generated_at=generated_at,
        )

    @classmethod
    def load(cls, path: Path) -> ImpactGraph:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ImpactError(f"cannot read impact graph {path}: {error}") from error
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "revision": self.revision,
            "generated_at": self.generated_at,
            "source_roots": list(self.source_roots),
            "test_roots": list(self.test_roots),
            "native_roots": list(self.native_roots),
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "tests": list(self.tests),
            "rules": self.rules.to_dict(),
        }


def _pytest_norecurse_names(repo_root: Path) -> set[str]:
    pytest_ini = repo_root / "pytest.ini"
    if not pytest_ini.exists():
        return set()
    for line in pytest_ini.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("norecursedirs") and "=" in line:
            return set(line.split("=", 1)[1].split())
    return set()


def _python_files(
    repo_root: Path, roots: Sequence[str], *, excluded_dir_names: set[str]
) -> set[str]:
    files: set[str] = set()
    for root_name in roots:
        root = repo_root / root_name
        if not root.exists():
            raise ImpactError(f"graph root does not exist: {root_name}")
        for path in root.rglob("*.py"):
            relative = path.relative_to(repo_root)
            if any(part in excluded_dir_names for part in relative.parts[:-1]):
                continue
            files.add(relative.as_posix())
    return files


def _native_files(repo_root: Path, roots: Sequence[str]) -> set[str]:
    files: set[str] = set()
    for root_name in roots:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in NATIVE_SUFFIXES:
                files.add(path.relative_to(repo_root).as_posix())
    return files


def _native_reference_patterns(value: str) -> set[str]:
    patterns: set[str] = set()
    for match in _NATIVE_REFERENCE_RE.finditer(value.replace("\\", "/")):
        pattern = match.group("path")
        pattern = re.sub(r"\{[^}]*\}|%[a-zA-Z]", "*", pattern)
        patterns.add(pattern)
    return patterns


def _joined_string_pattern(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            parts.append("*")
    return "".join(parts)


def _python_native_reference_patterns(tree: ast.AST) -> set[str]:
    patterns: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            patterns.update(_native_reference_patterns(node.value))
        elif isinstance(node, ast.JoinedStr):
            patterns.update(_native_reference_patterns(_joined_string_pattern(node)))
    return patterns


def _matching_native_paths(pattern: str, native_paths: set[str]) -> set[str]:
    normalized = pattern.replace("\\", "/").lstrip("./")
    csrc_index = normalized.find("csrc/")
    if csrc_index >= 0:
        normalized = normalized[csrc_index:]
    return {
        path
        for path in native_paths
        if fnmatch.fnmatchcase(path, normalized)
        or fnmatch.fnmatchcase(path, f"*/{normalized}")
        or (
            "/" not in normalized
            and fnmatch.fnmatchcase(PurePosixPath(path).name, normalized)
        )
    }


def _resolve_native_reference(
    reference: str,
    *,
    importer: str,
    repo_root: Path,
    native_roots: Sequence[str],
    native_paths: set[str],
) -> set[str]:
    normalized = reference.strip().replace("\\", "/")
    candidates: set[str] = set()
    importer_parent = (repo_root / importer).parent
    for base in (importer_parent, *(repo_root / root for root in native_roots)):
        candidate = (base / normalized).resolve()
        try:
            relative = candidate.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if relative in native_paths:
            candidates.add(relative)
    candidates.update(_matching_native_paths(normalized, native_paths))
    return candidates


def _native_dependency_edges(
    repo_root: Path,
    *,
    trees: dict[str, ast.AST],
    native_roots: Sequence[str],
    native_paths: set[str],
) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for importer, tree in trees.items():
        if PurePosixPath(importer).name == "__init__.py":
            continue
        for pattern in _python_native_reference_patterns(tree):
            for dependency in _matching_native_paths(pattern, native_paths):
                edges.add((importer, dependency))

    for importer in sorted(native_paths):
        try:
            content = (repo_root / importer).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ImpactError(
                f"cannot parse native dependency source {importer}: {error}"
            ) from error
        references = [
            match.group("path") for match in _NATIVE_INCLUDE_RE.finditer(content)
        ]
        references.extend(
            match.group("path") or match.group("from_path")
            for match in _JINJA_REFERENCE_RE.finditer(content)
        )
        for reference in references:
            for dependency in _resolve_native_reference(
                reference,
                importer=importer,
                repo_root=repo_root,
                native_roots=native_roots,
                native_paths=native_paths,
            ):
                if dependency != importer:
                    edges.add((importer, dependency))
    return edges


def _module_name(path: str) -> str:
    file_path = PurePosixPath(path)
    parts = list(file_path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _from_import_base(node: ast.ImportFrom, *, module: str, is_package: bool) -> str:
    package = module if is_package else module.rpartition(".")[0]
    if not node.level:
        return node.module or ""
    package_parts = package.split(".") if package else []
    keep = len(package_parts) - (node.level - 1)
    if keep < 0:
        return ""
    base_parts = package_parts[:keep]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _dotted_attribute(node: ast.Attribute) -> tuple[str, ...] | None:
    parts: deque[str] = deque()
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.appendleft(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.appendleft(current.id)
    return tuple(parts)


def _import_candidates(tree: ast.AST, *, module: str, is_package: bool) -> set[str]:
    candidates: set[str] = set()
    local_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidates.add(alias.name)
                local_name = alias.asname or alias.name.split(".")[0]
                local_aliases[local_name] = alias.name if alias.asname else local_name
        elif isinstance(node, ast.ImportFrom):
            base = _from_import_base(node, module=module, is_package=is_package)
            if base:
                candidates.add(base)
            for alias in node.names:
                if alias.name != "*":
                    imported = f"{base}.{alias.name}" if base else alias.name
                    candidates.add(imported)
                    local_aliases[alias.asname or alias.name] = imported
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted_attribute(node)
        if dotted is None or dotted[0] not in local_aliases:
            continue
        candidates.add(".".join((local_aliases[dotted[0]], *dotted[1:])))
    return candidates


def _most_specific_target(candidate: str, targets: dict[str, str]) -> str | None:
    parts = candidate.split(".")
    while parts:
        name = ".".join(parts)
        target = targets.get(name)
        if target is not None:
            return target
        parts.pop()
    return None


def _resolved_module_files(
    candidate: str, targets: dict[str, str], modules: dict[str, str]
) -> set[str]:
    resolved: set[str] = set()
    target = _most_specific_target(candidate, targets)
    if target is not None:
        resolved.add(target)
    # Importing a module executes each package initializer on its path.
    candidate_parts = candidate.split(".")
    for index in range(1, len(candidate_parts)):
        initializer = modules.get(".".join(candidate_parts[:index]))
        if initializer is not None:
            resolved.add(initializer)
    return resolved


def _package_reexports(
    trees: dict[str, ast.AST], modules: dict[str, str]
) -> dict[str, str]:
    reexports: dict[str, str] = {}
    for path, tree in trees.items():
        if PurePosixPath(path).name != "__init__.py":
            continue
        module = _module_name(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = _from_import_base(node, module=module, is_package=True)
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{base}.{alias.name}" if base else alias.name
                target = _most_specific_target(candidate, modules)
                if target is not None:
                    public_name = alias.asname or alias.name
                    reexports[f"{module}.{public_name}"] = target
    return reexports


def build_graph(
    repo_root: Path,
    *,
    revision: str,
    source_roots: Sequence[str] = ("flashinfer",),
    test_roots: Sequence[str] = ("tests",),
    native_roots: Sequence[str] = ("csrc",),
    rules: ImpactRules | None = None,
) -> ImpactGraph:
    repo_root = repo_root.resolve()
    if not revision.strip():
        raise ImpactError("revision must not be empty")
    normalized_sources = tuple(
        _normalized_relative_path(item, field="source_roots") for item in source_roots
    )
    normalized_tests = tuple(
        _normalized_relative_path(item, field="test_roots") for item in test_roots
    )
    normalized_native = tuple(
        _normalized_relative_path(item, field="native_roots") for item in native_roots
    )
    source_files = _python_files(
        repo_root, normalized_sources, excluded_dir_names=set()
    )
    excluded_tests = _pytest_norecurse_names(repo_root)
    test_files = _python_files(
        repo_root, normalized_tests, excluded_dir_names=excluded_tests
    )
    native_files = _native_files(repo_root, normalized_native)
    python_files = source_files | test_files
    nodes = python_files | native_files
    modules = {_module_name(path): path for path in sorted(python_files)}
    trees: dict[str, ast.AST] = {}
    edges: set[tuple[str, str]] = set()

    for path in sorted(python_files):
        absolute = repo_root / path
        try:
            tree = ast.parse(absolute.read_text(encoding="utf-8"), filename=path)
        except (OSError, SyntaxError, UnicodeError) as error:
            raise ImpactError(
                f"cannot parse Python dependency source {path}: {error}"
            ) from error
        trees[path] = tree

    targets = dict(modules)
    targets.update(_package_reexports(trees, modules))
    for path, tree in trees.items():
        module = _module_name(path)
        is_package = PurePosixPath(path).name == "__init__.py"
        # Package initializers frequently re-export most of FlashInfer's public
        # API. Treating those re-exports as ordinary imports turns every module
        # into a dependency of every test. Re-exports are resolved at their use
        # sites above; changes to __init__.py itself still reach its importers.
        if is_package:
            continue
        for candidate in _import_candidates(tree, module=module, is_package=is_package):
            for dependency in _resolved_module_files(candidate, targets, modules):
                # The shared root conftest imports FlashInfer to configure test
                # hooks. Do not turn that setup import into an all-suite edge;
                # conftest.py itself remains an implicit dependency of tests.
                if (
                    PurePosixPath(path).name == "conftest.py"
                    and dependency in source_files
                ):
                    continue
                if dependency != path:
                    edges.add((path, dependency))

    edges.update(
        _native_dependency_edges(
            repo_root,
            trees=trees,
            native_roots=normalized_native,
            native_paths=native_files,
        )
    )

    tests = {
        path for path in test_files if PurePosixPath(path).name.startswith("test_")
    }
    # Pytest loads conftest.py files implicitly, so model those edges even when
    # a test module does not import them.
    for test in tests:
        parent = PurePosixPath(test).parent
        while parent.parts:
            conftest = (parent / "conftest.py").as_posix()
            if conftest in nodes and conftest != test:
                edges.add((test, conftest))
            parent = parent.parent

    return ImpactGraph(
        revision=revision.strip(),
        source_roots=normalized_sources,
        test_roots=normalized_tests,
        native_roots=normalized_native,
        nodes=tuple(sorted(nodes)),
        edges=tuple(sorted(edges)),
        tests=tuple(sorted(tests)),
        rules=rules or ImpactRules(),
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _existing_test(repo_root: Path, path: str) -> bool:
    absolute = (repo_root / path).resolve()
    try:
        absolute.relative_to(repo_root.resolve())
    except ValueError:
        return False
    return (
        absolute.is_file()
        and absolute.name.startswith("test_")
        and absolute.suffix == ".py"
    )


def _looks_like_test(graph: ImpactGraph, path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        candidate.name.startswith("test_")
        and candidate.suffix == ".py"
        and any(
            candidate.is_relative_to(PurePosixPath(root)) for root in graph.test_roots
        )
    )


def _selection(
    mode: str,
    reason_code: str,
    reason: str,
    *,
    changed_files: Sequence[str],
    graph_revision: str | None,
    tests: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "mode": mode,
        "fallback": mode == "all",
        "reason_code": reason_code,
        "reason": reason,
        "graph_revision": graph_revision,
        "changed_files": list(changed_files),
        "test_files": sorted(set(tests)),
    }


def fallback_selection(
    reason_code: str,
    reason: str,
    *,
    changed_files: Sequence[str] = (),
    graph_revision: str | None = None,
) -> dict[str, Any]:
    return _selection(
        "all",
        reason_code,
        reason,
        changed_files=changed_files,
        graph_revision=graph_revision,
    )


def select_tests(
    graph: ImpactGraph,
    changed_files: Sequence[str],
    *,
    repo_root: Path,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    try:
        changed = tuple(
            sorted(
                {
                    _normalized_relative_path(item, field="changed_files")
                    for item in changed_files
                    if item.strip()
                }
            )
        )
    except ImpactError as error:
        return fallback_selection("invalid_changed_path", str(error))
    if not changed:
        return fallback_selection(
            "no_changed_files",
            "no changed files were provided",
            graph_revision=graph.revision,
        )
    if expected_revision is not None and graph.revision != expected_revision:
        return fallback_selection(
            "graph_revision_mismatch",
            f"graph revision {graph.revision!r} does not match {expected_revision!r}",
            changed_files=changed,
            graph_revision=graph.revision,
        )

    reverse_edges: dict[str, set[str]] = defaultdict(set)
    for importer, dependency in graph.edges:
        reverse_edges[dependency].add(importer)
    graph_nodes = set(graph.nodes)
    graph_tests = set(graph.tests)
    selected: set[str] = set()

    for path in changed:
        if _matches(path, graph.rules.global_patterns):
            return fallback_selection(
                "global_change",
                f"{path} matches a global test-impact rule",
                changed_files=changed,
                graph_revision=graph.revision,
            )

        override_matches = [
            override
            for override in graph.rules.dependency_overrides
            if _matches(path, override.source_patterns)
        ]
        if override_matches:
            override_tests = {
                test
                for override in override_matches
                for test in graph.tests
                if _matches(test, override.test_patterns)
                and _existing_test(repo_root, test)
            }
            if not override_tests:
                return fallback_selection(
                    "empty_dependency_override",
                    f"dependency override for {path} matched no existing tests",
                    changed_files=changed,
                    graph_revision=graph.revision,
                )
            selected.update(override_tests)
            continue

        if _matches(path, graph.rules.ignored_patterns):
            continue

        if _looks_like_test(graph, path) and not _existing_test(repo_root, path):
            # A deleted test cannot be executed and has no downstream consumers.
            continue
        if _existing_test(repo_root, path):
            selected.add(path)
            continue
        if path not in graph_nodes:
            return fallback_selection(
                "unknown_change",
                f"{path} is not represented by the impact graph or rules",
                changed_files=changed,
                graph_revision=graph.revision,
            )

        queue: deque[str] = deque([path])
        visited = {path}
        impacted: set[str] = set()
        while queue:
            dependency = queue.popleft()
            for importer in reverse_edges.get(dependency, ()):
                if importer in visited:
                    continue
                visited.add(importer)
                if importer in graph_tests:
                    impacted.add(importer)
                queue.append(importer)
        existing_impacted = {
            test for test in impacted if _existing_test(repo_root, test)
        }
        if not existing_impacted:
            return fallback_selection(
                "uncovered_dependency",
                f"{path} has no existing dependent tests in the impact graph",
                changed_files=changed,
                graph_revision=graph.revision,
            )
        selected.update(existing_impacted)

    if selected:
        return _selection(
            "selected",
            "impacted_tests",
            f"selected {len(selected)} impacted test file(s)",
            changed_files=changed,
            graph_revision=graph.revision,
            tests=selected,
        )
    return _selection(
        "none",
        "no_impacted_tests",
        "all changes were ignored or removed tests",
        changed_files=changed,
        graph_revision=graph.revision,
    )


def _write_json(path: Path | None, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        sys.stdout.write(serialized)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_changed_files(paths: Sequence[Path], inline: Sequence[str]) -> list[str]:
    changed = list(inline)
    for path in paths:
        try:
            changed.extend(path.read_text(encoding="utf-8").splitlines())
        except OSError as error:
            raise ImpactError(
                f"cannot read changed-files input {path}: {error}"
            ) from error
    return [item.strip() for item in changed if item.strip()]


def resolve_selection_manifest(
    path: Path, *, repo_root: Path, expected_revision: str | None = None
) -> tuple[str, tuple[str, ...]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImpactError(
            f"cannot read test-selection manifest {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ImpactError("test-selection manifest must be a JSON object")
    if value.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ImpactError(
            "unsupported test-selection schema_version: "
            f"{value.get('schema_version')!r}"
        )
    mode = value.get("mode")
    if mode not in ("all", "selected", "none"):
        raise ImpactError(f"invalid test-selection mode: {mode!r}")
    files = tuple(
        _normalized_relative_path(item, field="test_files")
        for item in _string_list(value.get("test_files"), field="test_files")
    )
    if mode == "selected" and not files:
        raise ImpactError("selected test-selection manifest has no test files")
    if mode != "selected" and files:
        raise ImpactError(f"{mode} test-selection manifest must not list test files")
    if len(set(files)) != len(files):
        raise ImpactError("test-selection manifest contains duplicate test files")
    graph_revision = value.get("graph_revision")
    if mode in ("selected", "none"):
        if not isinstance(graph_revision, str) or not graph_revision:
            raise ImpactError(f"{mode} test-selection manifest has no graph revision")
        if expected_revision is not None and graph_revision != expected_revision:
            raise ImpactError(
                f"test-selection graph revision {graph_revision!r} does not match "
                f"checkout revision {expected_revision!r}"
            )
    invalid = [path for path in files if not _existing_test(repo_root, path)]
    if invalid:
        raise ImpactError(
            "test-selection manifest contains missing or invalid test files: "
            + ", ".join(invalid[:5])
        )
    return mode, tuple(sorted(files))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="build a Python and native-source dependency graph"
    )
    build.add_argument("--repository-root", type=Path, default=Path.cwd())
    build.add_argument("--revision", required=True)
    build.add_argument("--source-root", action="append", default=[])
    build.add_argument("--test-root", action="append", default=[])
    build.add_argument("--native-root", action="append", default=[])
    build.add_argument("--rules", type=Path)
    build.add_argument("--output", type=Path)

    select = subparsers.add_parser("select", help="select tests for changed files")
    select.add_argument("--repository-root", type=Path, default=Path.cwd())
    select.add_argument("--graph", type=Path, required=True)
    select.add_argument("--changed-files", type=Path, action="append", default=[])
    select.add_argument("--changed-file", action="append", default=[])
    select.add_argument("--expected-revision")
    select.add_argument("--output", type=Path)

    resolve = subparsers.add_parser(
        "resolve-selection", help="print runner-safe mode and selected files"
    )
    resolve.add_argument("--repository-root", type=Path, default=Path.cwd())
    resolve.add_argument("--manifest", type=Path, required=True)
    resolve.add_argument("--expected-revision")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        try:
            graph = build_graph(
                args.repository_root,
                revision=args.revision,
                source_roots=args.source_root or ("flashinfer",),
                test_roots=args.test_root or ("tests",),
                native_roots=args.native_root or ("csrc",),
                rules=ImpactRules.load(args.rules),
            )
            _write_json(args.output, graph.to_dict())
        except ImpactError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        return 0

    if args.command == "select":
        changed: list[str] = []
        graph_revision: str | None = None
        try:
            changed = _load_changed_files(args.changed_files, args.changed_file)
            graph = ImpactGraph.load(args.graph)
            graph_revision = graph.revision
            selection = select_tests(
                graph,
                changed,
                repo_root=args.repository_root,
                expected_revision=args.expected_revision,
            )
        except ImpactError as error:
            selection = fallback_selection(
                "selection_error",
                str(error),
                changed_files=changed,
                graph_revision=graph_revision,
            )
        if selection["fallback"]:
            print(
                f"WARNING: test-impact selection fell back to all tests: {selection['reason']}",
                file=sys.stderr,
            )
        _write_json(args.output, selection)
        return 0

    try:
        mode, files = resolve_selection_manifest(
            args.manifest,
            repo_root=args.repository_root,
            expected_revision=args.expected_revision,
        )
    except ImpactError as error:
        print(
            f"WARNING: invalid test-selection manifest; running all tests: {error}",
            file=sys.stderr,
        )
        mode, files = "all", ()
    print(mode)
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
