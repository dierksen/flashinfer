(function (root, factory) {
  "use strict";

  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.TestImpactSelector = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const GRAPH_SCHEMA_VERSION = 2;
  const SELECTION_SCHEMA_VERSION = 1;

  function normalizePath(value) {
    if (typeof value !== "string" || value.trim() === "") {
      throw new Error("changed files must contain non-empty path strings");
    }
    const raw = value.trim().replaceAll("\\", "/");
    if (raw.startsWith("/")) {
      throw new Error(`changed files contains an unsafe path: ${value}`);
    }
    const rawParts = raw.split("/");
    if (rawParts.includes("..") || (rawParts[0] || "").includes(":")) {
      throw new Error(`changed files contains an unsafe path: ${value}`);
    }
    const parts = rawParts.filter((part) => part !== "" && part !== ".");
    if (parts.length === 0) {
      throw new Error(`changed files contains an unsafe path: ${value}`);
    }
    return parts.join("/");
  }

  function escapeRegularExpression(value) {
    return value.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
  }

  // Match Python fnmatch.fnmatchcase for the path globs used by impact rules.
  // Like fnmatch, '*' may cross '/' because repository paths are plain strings.
  function globToRegularExpression(pattern) {
    let expression = "";
    let index = 0;
    while (index < pattern.length) {
      const character = pattern[index];
      if (character === "*") {
        while (pattern[index + 1] === "*") {
          index += 1;
        }
        expression += ".*";
      } else if (character === "?") {
        expression += ".";
      } else if (character === "[") {
        let end = index + 1;
        if (pattern[end] === "!" || pattern[end] === "^") {
          end += 1;
        }
        if (pattern[end] === "]") {
          end += 1;
        }
        while (end < pattern.length && pattern[end] !== "]") {
          end += 1;
        }
        if (end === pattern.length) {
          expression += "\\[";
        } else {
          let content = pattern.slice(index + 1, end).replaceAll("\\", "\\\\");
          if (content.startsWith("!")) {
            content = `^${content.slice(1)}`;
          } else if (content.startsWith("^")) {
            content = `\\${content}`;
          }
          expression += `[${content}]`;
          index = end;
        }
      } else {
        expression += escapeRegularExpression(character);
      }
      index += 1;
    }
    return new RegExp(`^${expression}$`);
  }

  const globCache = new Map();

  function matches(path, patterns) {
    return (patterns || []).some((pattern) => {
      if (!globCache.has(pattern)) {
        globCache.set(pattern, globToRegularExpression(pattern));
      }
      return globCache.get(pattern).test(path);
    });
  }

  function validateGraph(graph) {
    if (!graph || typeof graph !== "object") {
      throw new Error("impact graph must be an object");
    }
    if (graph.schema_version !== GRAPH_SCHEMA_VERSION) {
      throw new Error(`unsupported impact graph schema: ${graph.schema_version}`);
    }
    for (const field of ["nodes", "edges", "tests", "test_roots"]) {
      if (!Array.isArray(graph[field])) {
        throw new Error(`impact graph ${field} must be an array`);
      }
    }
    if (!graph.rules || typeof graph.rules !== "object") {
      throw new Error("impact graph rules must be an object");
    }
    return graph;
  }

  function looksLikeTest(graph, path) {
    const name = path.split("/").at(-1);
    return (
      name.startsWith("test_") &&
      name.endsWith(".py") &&
      graph.test_roots.some((root) => path === root || path.startsWith(`${root}/`))
    );
  }

  function isExistingTest(path, repositoryFiles) {
    const name = path.split("/").at(-1);
    return (
      repositoryFiles.has(path) && name.startsWith("test_") && name.endsWith(".py")
    );
  }

  function makeSelection(mode, reasonCode, reason, graph, changedFiles, testFiles, testReasons) {
    return {
      schema_version: SELECTION_SCHEMA_VERSION,
      mode,
      fallback: mode === "all",
      reason_code: reasonCode,
      reason,
      graph_revision: graph ? graph.revision : null,
      changed_files: [...changedFiles],
      test_files: [...testFiles].sort(),
      test_reasons: testReasons || {},
    };
  }

  function fallback(reasonCode, reason, graph, changedFiles) {
    return makeSelection("all", reasonCode, reason, graph, changedFiles, [], {});
  }

  function buildReverseEdges(graph) {
    const reverseEdges = new Map();
    for (const edge of graph.edges) {
      if (!Array.isArray(edge) || edge.length !== 2) {
        throw new Error("impact graph edges must contain importer/dependency pairs");
      }
      const [importer, dependency] = edge;
      if (!reverseEdges.has(dependency)) {
        reverseEdges.set(dependency, []);
      }
      reverseEdges.get(dependency).push(importer);
    }
    for (const importers of reverseEdges.values()) {
      importers.sort();
    }
    return reverseEdges;
  }

  function reconstructPath(test, parents) {
    const path = [];
    let current = test;
    while (current !== null && current !== undefined) {
      path.push(current);
      current = parents.get(current);
    }
    return path.reverse();
  }

  function impactedTestPaths(changedFile, reverseEdges, graphTests) {
    const parents = new Map([[changedFile, null]]);
    const queue = [changedFile];
    const paths = new Map();
    for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
      const dependency = queue[queueIndex];
      for (const importer of reverseEdges.get(dependency) || []) {
        if (parents.has(importer)) {
          continue;
        }
        parents.set(importer, dependency);
        queue.push(importer);
        if (graphTests.has(importer)) {
          paths.set(importer, reconstructPath(importer, parents));
        }
      }
    }
    return paths;
  }

  function addReason(testReasons, test, reason) {
    if (!testReasons[test]) {
      testReasons[test] = [];
    }
    const identity = `${reason.changed_file}\0${reason.kind}\0${reason.path.join("\0")}`;
    const duplicate = testReasons[test].some(
      (candidate) =>
        `${candidate.changed_file}\0${candidate.kind}\0${candidate.path.join("\0")}` ===
        identity,
    );
    if (!duplicate) {
      testReasons[test].push(reason);
      testReasons[test].sort((left, right) =>
        `${left.changed_file}\0${left.kind}`.localeCompare(
          `${right.changed_file}\0${right.kind}`,
        ),
      );
    }
  }

  function selectTests(graphValue, changedFileValues, repositoryFileValues) {
    let graph;
    try {
      graph = validateGraph(graphValue);
    } catch (error) {
      return fallback("selection_error", error.message, null, []);
    }

    let changedFiles;
    try {
      changedFiles = [...new Set(
        (changedFileValues || [])
          .filter((path) => typeof path === "string" && path.trim() !== "")
          .map(normalizePath),
      )].sort();
    } catch (error) {
      return fallback("invalid_changed_path", error.message, graph, []);
    }
    if (changedFiles.length === 0) {
      return fallback(
        "no_changed_files",
        "no changed files were provided",
        graph,
        changedFiles,
      );
    }

    const graphNodes = new Set(graph.nodes);
    const graphTests = new Set(graph.tests);
    const repositoryFiles = new Set(repositoryFileValues || graph.nodes);
    const reverseEdges = buildReverseEdges(graph);
    const selected = new Set();
    const testReasons = {};
    const rules = graph.rules;

    for (const path of changedFiles) {
      if (matches(path, rules.global_patterns)) {
        return fallback(
          "global_change",
          `${path} matches a global test-impact rule`,
          graph,
          changedFiles,
        );
      }

      const overrides = (rules.dependency_overrides || []).filter((override) =>
        matches(path, override.source_patterns),
      );
      if (overrides.length > 0) {
        const overrideTests = graph.tests.filter(
          (test) =>
            overrides.some((override) => matches(test, override.test_patterns)) &&
            isExistingTest(test, repositoryFiles),
        );
        if (overrideTests.length === 0) {
          return fallback(
            "empty_dependency_override",
            `dependency override for ${path} matched no existing tests`,
            graph,
            changedFiles,
          );
        }
        for (const test of overrideTests) {
          selected.add(test);
          addReason(testReasons, test, {
            changed_file: path,
            kind: "dependency_override",
            path: [path, test],
          });
        }
        continue;
      }

      if (matches(path, rules.ignored_patterns)) {
        continue;
      }

      if (looksLikeTest(graph, path) && !isExistingTest(path, repositoryFiles)) {
        continue;
      }
      if (isExistingTest(path, repositoryFiles)) {
        selected.add(path);
        addReason(testReasons, path, {
          changed_file: path,
          kind: "direct_test",
          path: [path],
        });
        continue;
      }
      if (!graphNodes.has(path)) {
        return fallback(
          "unknown_change",
          `${path} is not represented by the impact graph or rules`,
          graph,
          changedFiles,
        );
      }

      const impacted = impactedTestPaths(path, reverseEdges, graphTests);
      const existingImpacted = [...impacted.keys()].filter((test) =>
        isExistingTest(test, repositoryFiles),
      );
      if (existingImpacted.length === 0) {
        return fallback(
          "uncovered_dependency",
          `${path} has no existing dependent tests in the impact graph`,
          graph,
          changedFiles,
        );
      }
      for (const test of existingImpacted) {
        selected.add(test);
        addReason(testReasons, test, {
          changed_file: path,
          kind: "dependency_path",
          path: impacted.get(test),
        });
      }
    }

    if (selected.size > 0) {
      return makeSelection(
        "selected",
        "impacted_tests",
        `selected ${selected.size} impacted test file(s)`,
        graph,
        changedFiles,
        selected,
        testReasons,
      );
    }
    return makeSelection(
      "none",
      "no_impacted_tests",
      "all changes were ignored or removed tests",
      graph,
      changedFiles,
      [],
      {},
    );
  }

  return {
    GRAPH_SCHEMA_VERSION,
    globToRegularExpression,
    matches,
    normalizePath,
    selectTests,
  };
});
