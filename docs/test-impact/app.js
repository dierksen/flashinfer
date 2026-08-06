(function () {
  "use strict";

  const selector = globalThis.TestImpactSelector;
  const elements = {
    addFile: document.querySelector("#add-file"),
    analyze: document.querySelector("#analyze"),
    changedFiles: document.querySelector("#changed-files"),
    clear: document.querySelector("#clear"),
    copy: document.querySelector("#copy-tests"),
    filePicker: document.querySelector("#file-picker"),
    fileOptions: document.querySelector("#repository-file-options"),
    form: document.querySelector("#impact-form"),
    graphMetadata: document.querySelector("#graph-metadata"),
    loadStatus: document.querySelector("#load-status"),
    results: document.querySelector("#results"),
  };

  let graph = null;
  let repositoryFiles = [];
  let lastResult = null;

  function element(name, attributes, ...children) {
    const node = document.createElement(name);
    for (const [key, value] of Object.entries(attributes || {})) {
      if (key === "className") {
        node.className = value;
      } else {
        node.setAttribute(key, value);
      }
    }
    for (const child of children) {
      node.append(child instanceof Node ? child : document.createTextNode(child));
    }
    return node;
  }

  function changedFiles() {
    return elements.changedFiles.value.split(/\r?\n/).filter((line) => line.trim());
  }

  function setChangedFiles(paths) {
    elements.changedFiles.value = paths.join("\n");
  }

  function addSelectedFile() {
    const candidate = elements.filePicker.value.trim();
    if (!candidate) {
      return;
    }
    try {
      const normalized = selector.normalizePath(candidate);
      setChangedFiles([...new Set([...changedFiles(), normalized])].sort());
      elements.filePicker.value = "";
      elements.filePicker.focus();
    } catch (error) {
      elements.filePicker.setCustomValidity(error.message);
      elements.filePicker.reportValidity();
    }
  }

  function renderPath(path) {
    const container = element("div", { className: "impact-path" });
    path.forEach((part, index) => {
      if (index > 0) {
        container.append(element("span", { className: "path-arrow", "aria-hidden": "true" }, "→"));
      }
      container.append(element("code", {}, part));
    });
    return container;
  }

  function reasonLabel(kind) {
    if (kind === "direct_test") {
      return "Test file edited directly";
    }
    if (kind === "dependency_override") {
      return "Explicit dependency rule";
    }
    return "Shortest dependency path";
  }

  function renderSelected(result) {
    const fragment = document.createDocumentFragment();
    fragment.append(
      element(
        "div",
        { className: "result-heading" },
        element("div", { className: "mode mode-selected" }, "Selected"),
        element(
          "div",
          {},
          element("h2", {}, `${result.test_files.length} of ${graph.tests.length} test files`),
          element("p", {}, "Expand a test to see every edited file that reaches it."),
        ),
      ),
    );

    const list = element("div", { className: "test-results" });
    for (const test of result.test_files) {
      const reasons = result.test_reasons[test] || [];
      const details = element("details", { className: "test-result" });
      details.append(
        element(
          "summary",
          {},
          element("code", {}, test),
          element(
            "span",
            { className: "reason-count" },
            `${reasons.length} changed ${reasons.length === 1 ? "file" : "files"}`,
          ),
        ),
      );
      const body = element("div", { className: "test-result-body" });
      for (const reason of reasons) {
        body.append(
          element(
            "section",
            { className: "impact-reason" },
            element(
              "div",
              { className: "reason-title" },
              element("strong", {}, reason.changed_file),
              element("span", { className: "reason-kind" }, reasonLabel(reason.kind)),
            ),
            renderPath(reason.path),
          ),
        );
      }
      details.append(body);
      list.append(details);
    }
    fragment.append(list);
    return fragment;
  }

  function renderResult(result) {
    lastResult = result;
    elements.results.replaceChildren();
    elements.results.classList.remove("empty");
    elements.copy.hidden = result.mode !== "selected";

    if (result.mode === "selected") {
      elements.results.append(renderSelected(result));
      return;
    }

    const all = result.mode === "all";
    elements.results.append(
      element(
        "div",
        { className: "result-heading" },
        element(
          "div",
          { className: `mode ${all ? "mode-all" : "mode-none"}` },
          all ? "All" : "None",
        ),
        element(
          "div",
          {},
          element(
            "h2",
            {},
            all ? `Run the full suite (${graph.tests.length} known test files)` : "No tests selected",
          ),
          element("p", {}, result.reason),
          element("code", { className: "reason-code" }, result.reason_code),
        ),
      ),
    );
  }

  function analyze(event) {
    event.preventDefault();
    if (!graph) {
      return;
    }
    renderResult(selector.selectTests(graph, changedFiles(), repositoryFiles));
  }

  function populateRepositoryFiles() {
    const fragment = document.createDocumentFragment();
    for (const path of repositoryFiles) {
      fragment.append(element("option", { value: path }));
    }
    elements.fileOptions.append(fragment);
  }

  async function copyTests() {
    if (!lastResult || lastResult.mode !== "selected") {
      return;
    }
    await navigator.clipboard.writeText(lastResult.test_files.join("\n"));
    const original = elements.copy.textContent;
    elements.copy.textContent = "Copied";
    window.setTimeout(() => {
      elements.copy.textContent = original;
    }, 1200);
  }

  async function loadData() {
    try {
      const [graphResponse, filesResponse] = await Promise.all([
        fetch("./graph.json", { cache: "no-cache" }),
        fetch("./repository-files.txt", { cache: "no-cache" }),
      ]);
      if (!graphResponse.ok || !filesResponse.ok) {
        throw new Error("The generated graph data is unavailable.");
      }
      graph = await graphResponse.json();
      repositoryFiles = (await filesResponse.text())
        .split(/\r?\n/)
        .filter(Boolean)
        .sort();
      // Validate the graph before enabling queries.
      const validation = selector.selectTests(graph, ["docs/"], repositoryFiles);
      if (validation.reason_code === "selection_error") {
        throw new Error(validation.reason);
      }
      populateRepositoryFiles();
      elements.analyze.disabled = false;
      elements.addFile.disabled = false;
      elements.loadStatus.textContent = "Ready";
      elements.loadStatus.className = "status status-ready";

      const revision = graph.revision.slice(0, 12);
      const revisionLink = element(
        "a",
        {
          href: `https://github.com/flashinfer-ai/flashinfer/tree/${graph.revision}`,
          rel: "noreferrer",
        },
        revision,
      );
      elements.graphMetadata.replaceChildren(
        document.createTextNode(`${graph.tests.length} tests · ${graph.nodes.length} nodes · `),
        revisionLink,
        document.createTextNode(` · generated ${new Date(graph.generated_at).toLocaleString()}`),
      );
    } catch (error) {
      elements.loadStatus.textContent = "Graph unavailable";
      elements.loadStatus.className = "status status-error";
      elements.graphMetadata.textContent = error.message;
    }
  }

  elements.form.addEventListener("submit", analyze);
  elements.addFile.addEventListener("click", addSelectedFile);
  elements.filePicker.addEventListener("input", () => elements.filePicker.setCustomValidity(""));
  elements.filePicker.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addSelectedFile();
    }
  });
  elements.clear.addEventListener("click", () => {
    setChangedFiles([]);
    lastResult = null;
    elements.copy.hidden = true;
    elements.results.classList.add("empty");
    elements.results.replaceChildren(
      element("p", {}, "Choose or paste changed files, then analyze their impact."),
    );
  });
  elements.copy.addEventListener("click", copyTests);

  loadData();
})();
