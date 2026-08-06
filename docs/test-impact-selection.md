# Test impact selection

FlashInfer's unit-test runner can consume a versioned selection manifest and
run only the affected test files. The selection tooling is shared by GitHub
Actions and GitLab API-triggered jobs; CI-specific wiring is intentionally kept
outside the graph and manifest formats.

## Safety model

`scripts/test_impact.py` is conservative and fail-open. It emits `mode: all`
when the graph is missing, malformed, stale, or does not cover a changed file.
The unit-test runner also falls back to the full suite if its selection manifest
cannot be validated. An empty or broken selection therefore cannot silently
skip tests.

The graph covers Python imports, including transitive imports, package
re-exports used by consumers, and pytest's implicit `conftest.py` relationship.
It also discovers repository-local native dependencies without importing
FlashInfer:

- `.cu`, `.cuh`, C/C++, header, inline, and Jinja files under `csrc/` become
  graph nodes.
- Native/template filenames found in Python literals and bounded f-strings
  connect JIT generator modules to their inputs.
- Local C/CUDA `#include` directives are followed transitively.
- Jinja `include`, `import`, `from`, and `extends` relationships are followed
  transitively.

Dynamic filenames that cannot be matched to a bounded set of existing files,
compiler-only dependencies outside the configured native roots, and native
files with no path to an existing test remain uncovered and trigger the full
suite. Compiler depfiles can be added as a later observed-dependency layer.

`ci/test-impact-rules.json` contains the small amount of maintained metadata:

- `global_patterns` always run the full suite.
- `ignored_patterns` are changes proven not to affect tests.
- `dependency_overrides` map source globs to one or more test globs.

Global rules take precedence over overrides, and overrides take precedence over
ignored rules. This lets a focused executable or test under an otherwise
ignored documentation tree retain its own coverage. An override that matches no
existing test fails open to the full suite.

## Build and query a graph

Build the graph from the exact checkout that will be tested and record that
checkout's revision:

```bash
revision=$(git rev-parse HEAD)
python3 scripts/test_impact.py build \
  --repository-root . \
  --revision "${revision}" \
  --rules ci/test-impact-rules.json \
  --output test-impact-graph.json
```

The default roots are `flashinfer/`, `tests/`, and `csrc/`. Supplying any
`--source-root`, `--test-root`, or `--native-root` values replaces that kind's
default; repeat an option to configure multiple roots.

Provide changed paths one per line and create a runner manifest:

```bash
git diff --name-only "${base_revision}...HEAD" > changed-files.txt
python3 scripts/test_impact.py select \
  --repository-root . \
  --graph test-impact-graph.json \
  --changed-files changed-files.txt \
  --expected-revision "${revision}" \
  --output test-selection.json
```

The result has one of three modes:

- `all`: run normal full-suite discovery; `fallback` is `true`.
- `selected`: run the sorted, validated `test_files` list.
- `none`: skip installation and test execution.

The existing runner accepts the manifest through either interface:

```bash
bash scripts/task_run_unit_tests.sh \
  --test-selection-manifest test-selection.json

TEST_SELECTION_MANIFEST=test-selection.json \
  bash scripts/task_run_unit_tests.sh
```

`--test-path` and `--test-selection-manifest` are mutually exclusive.
For `selected` and `none` modes, the runner verifies `graph_revision` against
`SOURCE_GIT_SHA`, `GITHUB_SHA`, `CI_COMMIT_SHA`, or the local Git `HEAD` (in
that order). If no checkout identity is available, it runs the full suite.

## Artifact schemas

All three JSON artifacts have independent `schema_version` fields. Graphs also
record their source `revision`; selectors should pass `--expected-revision`
whenever a graph is restored from an artifact or cache. The selection manifest
records its reason code, changed paths, graph revision, and final test-file list
so a shadow-mode CI job can publish and audit decisions without enforcing them.

The graph stores repository-relative paths only. Its directed edges have the
form `[importer, dependency]` for Python imports, Python-to-native references,
native includes, and template composition. This makes the artifact portable
across runner workspaces and inspectable without importing FlashInfer.

## Browser explorer

The documentation Pages site includes a static test-impact explorer at
`test-impact/`. It accepts repository paths selected from the current checkout
or pasted one per line, then performs the same conservative selection in the
browser. No Python service is involved at query time.

`docs/build_docs.sh` builds the graph from the exact revision being published
and writes both `graph.json` and the repository-file list alongside the
page. Each selected test includes every changed file that caused its selection.
Graph-based reasons show one deterministic shortest path from the changed file
through its import/include/template dependents to the test; direct test edits
and dependency overrides are labeled separately.
