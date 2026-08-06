#!/bin/bash
set -eo pipefail
set -x
echo "Building FlashInfer documentation..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Install flashinfer package first
echo "Installing FlashInfer package..."
pip install -e ..

make clean
make SPHINXOPTS='-T -v' html

# The explorer itself is a static Sphinx extra. Generate its data from the
# exact checkout being published so browser queries use the same graph as CI.
TEST_IMPACT_OUTPUT="${SCRIPT_DIR}/_build/html/test-impact"
mkdir -p "${TEST_IMPACT_OUTPUT}"
REVISION="$(git -C "${REPOSITORY_ROOT}" rev-parse HEAD)"
python3 "${REPOSITORY_ROOT}/scripts/test_impact.py" build \
  --repository-root "${REPOSITORY_ROOT}" \
  --revision "${REVISION}" \
  --rules "${REPOSITORY_ROOT}/ci/test-impact-rules.json" \
  --output "${TEST_IMPACT_OUTPUT}/graph.json"
git -C "${REPOSITORY_ROOT}" ls-files --cached --others --exclude-standard \
  > "${TEST_IMPACT_OUTPUT}/repository-files.txt"

# Add RunLLM widget to generated HTML files
echo "Adding RunLLM widget to documentation..."
python3 wrap_run_llm.py

echo "Documentation build complete!"
