<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.io to discuss further. -->

## Motivation

Add **DOCA_MEMOS** support for NIXL storage backend. DOCA_MEMOS implements low‑latency, high-throughput, infinite capacity, scalable sharing data path for large‑scale inference workloads.

## Modifications

- **NIXL / HiCacheNixl**
  - Add **DOCA_MEMOS** to object plugin selection; require `use_host_hugepages=true` and readable `/proc/meminfo` hugepage stats at backend creation.
  - Backend-specific cache key formatting (SHA-256 hex prefix) and OBJ registration tuples for DOCA_MEMOS vs default OBJ.
  - Enforce hugetlb host pools and `page_first` / `page_first_direct` layout when DOCA_MEMOS is active.
  - Parse optional `nixl_enable_prog_thread`, `nixl_sync_mode`, and pass them into `nixl_agent_config`.
- **Host memory**
  - Add `NixlHostTensorAllocator`, `HugepageUtil` (mmap `MAP_HUGETLB`, meminfo checks, CUDA host register/unregister), and `CudaHostRegisterUtil`.
  - Thread `use_host_hugepages` from extra config through `HiRadixCache`, hybrid pool assemblers, and all host pool types (MHA/MLA/Mamba/NSA indexer).
  - Use hugepage free-page accounting instead of `psutil` when `use_host_hugepages` is enabled.
- **HiCache controller**
  - Make `storage_batch_size` configurable via extra config (default 128); document tuning for DOCA task pool size.
- **Config & tests**
  - Extend `nixl.config.toml.sample` with DOCA_MEMOS, hugepages, sync mode, and batch-size notes.
  - Expand `test_hicache_nixl_storage.py`.

## Accuracy Tests

N/A — storage/allocator plumbing only; no model forward or kernel math changes.

## Speed Tests and Profiling

Not run in this PR. Intended follow-up: HiCache L3 backup/prefetch benchmarks with DOCA_MEMOS vs POSIX/OBJ on hardware with `vm.nr_hugepages` reserved.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
