#!/bin/bash

set -eo pipefail

export PARALLEL_TESTS=true  # Enable parallel test execution for unit tests (auto-discovery mode)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common test functions
# shellcheck disable=SC1091  # File exists, checked separately
source "${SCRIPT_DIR}/test_utils.sh"

# Find and filter test files based on pytest.ini exclusions
find_test_files() {
    TEST_SELECTION_MODE="all"
    TEST_SELECTION_COUNT=0

    if [ -n "${TEST_SELECTION_MANIFEST:-}" ]; then
        local selection_output
        local -a selection_lines
        local expected_revision
        expected_revision="${SOURCE_GIT_SHA:-${GITHUB_SHA:-${CI_COMMIT_SHA:-}}}"
        if [ -z "${expected_revision}" ]; then
            expected_revision=$(git rev-parse HEAD 2>/dev/null || true)
        fi
        if [ -z "${expected_revision}" ]; then
            echo "⚠️  TEST IMPACT: cannot determine checkout revision; running the full unit-test suite"
            echo ""
            TEST_SELECTION_MODE="all"
            TEST_SELECTION_MANIFEST=""
        else
            selection_output=$(python3 "${SCRIPT_DIR}/test_impact.py" resolve-selection \
                --repository-root "$(pwd)" \
                --manifest "${TEST_SELECTION_MANIFEST}" \
                --expected-revision "${expected_revision}")
            mapfile -t selection_lines <<< "${selection_output}"
            TEST_SELECTION_MODE="${selection_lines[0]}"

            if [ "${TEST_SELECTION_MODE}" = "none" ]; then
                TEST_FILES=""
                echo "✅ TEST IMPACT: no unit tests are affected"
                return
            fi
            if [ "${TEST_SELECTION_MODE}" = "selected" ]; then
                local -a selected_files=("${selection_lines[@]:1}")
                TEST_SELECTION_COUNT=${#selected_files[@]}
                TEST_FILES="${selected_files[*]}"
                echo "🎯 TEST IMPACT: selected ${TEST_SELECTION_COUNT} test file(s)"
                for test_file in "${selected_files[@]}"; do
                    echo "  ${test_file}"
                done
                echo ""
                return
            fi
            echo "⚠️  TEST IMPACT: manifest requests the full unit-test suite"
            echo ""
        fi
    fi

    SEARCH_DIR="${TEST_PATH:-tests/}"

    if [ -n "$TEST_PATH" ]; then
        if [ ! -d "${SEARCH_DIR}" ]; then
            echo "ERROR: TEST_PATH '${SEARCH_DIR}' does not exist or is not a directory."
            echo "Available test directories:"
            find tests/ -maxdepth 1 -type d | sort | tail -n +2 | sed 's/^/  /'
            exit 1
        fi
        echo "🎯 TEST_PATH set: scoping test discovery to ${SEARCH_DIR}"
        echo ""
    fi

    echo "Reading pytest.ini for excluded directories..."
    EXCLUDED_DIRS=""
    if [ -f "./pytest.ini" ]; then
        # Extract norecursedirs from pytest.ini and convert to array
        NORECURSEDIRS=$(grep "^norecursedirs" ./pytest.ini | sed 's/norecursedirs\s*=\s*//' | sed 's/#.*//')
        if [ -n "$NORECURSEDIRS" ]; then
            EXCLUDED_DIRS=$(echo "$NORECURSEDIRS" | tr ',' ' ' | tr -s ' ')
            echo "⚠️  WARNING: Excluding directories from pytest.ini: $EXCLUDED_DIRS"
            echo ""
        fi
    fi

    echo "Finding all test_*.py files in ${SEARCH_DIR} directory..."

    # Find all test_*.py files
    ALL_TEST_FILES=$(find "${SEARCH_DIR}" -name "test_*.py" -type f | sort)

    # Filter out excluded files based on directory exclusions
    TEST_FILES=""
    for test_file in $ALL_TEST_FILES; do
        exclude_file=false
        test_dir=$(dirname "$test_file")

        for excluded_dir in $EXCLUDED_DIRS; do
            excluded_dir=$(echo "$excluded_dir" | xargs)  # trim whitespace
            if [ -n "$excluded_dir" ]; then
                # Check if this file's directory should be excluded
                if [[ "$test_dir" == *"/$excluded_dir" ]] || [[ "$test_dir" == "tests/$excluded_dir" ]] || [[ "$test_dir" == *"/$excluded_dir/"* ]]; then
                    exclude_file=true
                    break
                fi
            fi
        done

        if [ "$exclude_file" = false ]; then
            TEST_FILES="$TEST_FILES $test_file"
        fi
    done

    # Clean up whitespace
    TEST_FILES=$(echo "$TEST_FILES" | xargs)

    if [ -z "$TEST_FILES" ]; then
        echo "No test files found in ${SEARCH_DIR} directory (after exclusions)"
        exit 1
    fi

    echo "Found test files:"
    for test_file in $TEST_FILES; do
        echo "  $test_file"
    done
    echo ""
}

# Main execution
main() {
    # Parse command line arguments
    parse_args "$@"

    # Resolve the impacted test set before performing any package installation.
    # A "none" manifest exits without consuming a GPU runner.
    find_test_files
    if [ "${TEST_SELECTION_MODE}" = "none" ]; then
        exit 0
    fi

    # Print test mode banner
    print_test_mode_banner

    # Keep this after impact selection so a validated "none" result can exit
    # without installing test-only dependencies. Full and selected runs still
    # perform the same compatibility check and conditional install as before.
    # nvshmem4py-cu12 pins cuda-python<=12.9; letting pip resolve its deps on a
    # cu13 container downgrades cuda-python/cuda-bindings and makes the next
    # requirements resolution evict CUDA torch. Install only if missing, and
    # --no-deps: the image already ships compatible CUDA and NVSHMEM libraries.
    # TODO: Remove once CI container ships with nvshmem4py pre-installed.
    python -c "import nvshmem.core" 2>/dev/null || pip install --no-deps nvshmem4py-cu12

    # Install and verify (includes precompiled kernels)
    install_and_verify

    # apply dependency overrides after installation since pip may overwrite
    source "${SCRIPT_DIR}/setup_test_env.sh"

    # tests/moe_ep needs the EP runtime stack (nvidia-nccl >= 2.30.7 + nccl4py,
    # nvshmem4py, DeepGEMM) and a --no-build-isolation FlashInfer install so the
    # build hook's NCCL floor upgrade isn't lost in a throwaway PEP 517 env.
    # Without it the split-path tests fail validate_arch_for_backend on
    # Blackwell (NCCL 2.28.9 from the torch pin) and the deep_gemm multirank
    # file exits 5 (module-level importorskip collects nothing).
    # The ~25-min trtllm fused-MoE JIT prewarm stays off (FI_EP_PREWARM
    # defaults to 0); the torchrun-only tests that would need it auto-skip
    # here (no WORLD_SIZE).
    #
    # build_flashinfer_ep_pytorch.sh pins cuda-bindings==13.2.0 (a cu13 package)
    # and is designed for the nvcr.io/nvidia/pytorch:26.05 base image. On cu12
    # CI images that ship CUDA 12.x torch, that pin conflicts with the image's
    # cuda-python~=12.x and breaks nccl.ep's CUDA-major consistency check.
    # cu12 CI images already have nccl4py pre-installed as a base dependency of
    # flashinfer-python, so EP is available (or unavailable due to the cu12/cu13
    # libnccl_ep mismatch — in either case the tests handle it via auto-skip).
    _cuda_major=$(python -c \
        'import torch; v=torch.version.cuda; print(v.split(".")[0] if v else "0")' \
        2>/dev/null || echo 0)
    if [ "$DRY_RUN" != "true" ] && [[ " ${TEST_FILES} " == *tests/moe_ep/* ]] && [ "${_cuda_major}" -ge 13 ]; then
        FI_SRC="$(pwd)" bash docker/install/build_flashinfer_ep_pytorch.sh
    fi

    # Execute tests or dry run
    if [ "$DRY_RUN" == "true" ]; then
        execute_dry_run "$TEST_FILES"
    else
        execute_tests "$TEST_FILES"
    fi

    exit "$EXIT_CODE"
}

main "$@"
