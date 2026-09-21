# Multi-host Evidence Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an honest NewsLens multi-host receipt from stages executed on their owning EC2 hosts.

**Architecture:** The app host runs load against explicit state-host endpoints. The state host runs recovery and DLQ because it owns Kafka and its Docker Compose project. A receipt builder validates the three raw reports, a redacted topology manifest, and artifact SHA-256 checksums.

**Tech Stack:** Python standard library, JSON, SHA-256, Docker Compose, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-multihost-evidence-design.md`

## Global Constraints

- Do not print, store, or commit passwords, private IP addresses, SSH credentials, or `.env` data.
- Do not use SSH or remote Docker control in the evidence code.
- Reject any topology that claims Kafka and PostgreSQL are highly available.
- A failed stage, malformed artifact, missing field, or checksum mismatch prevents a passing receipt.

## Review Focus

- Explicit endpoint arguments must replace local defaults in multi-host mode.
- HA claims must be rejected even when stage reports pass.
- Valid JSON with a missing field must fail validation clearly.
- A modified artifact after checksum generation must be detected.
- Recovery must restore both consumers in its `finally` block after a failure.

---

### Task 1: Host-aware raw reports

**Files:**
- Modify: `scripts/realtime/benchmark_ingestion.py`
- Modify: `scripts/realtime/failure_recovery.py`
- Modify: `scripts/realtime/verify_dlq.py`
- Test: `tests/test_realtime_soak.py`

**Interfaces:** Each report gains `execution_host_role: str`; load gains `endpoints: dict[str, object]` with the ingestion, search, and consumer URLs.

- [ ] Write a failing test that executes a load stage with `--execution-host-role application`, then asserts `report["execution_host_role"] == "application"` and `report["endpoints"]["ingestion_url"] == "http://state:8080"`.
- [ ] Run `python -m pytest tests/test_realtime_soak.py -q`; verify it fails because the fields are absent.
- [ ] Add `--execution-host-role` to all three scripts. Add `endpoints` to load, and record state role plus the selected Compose file and consumer URLs in recovery and DLQ reports.
- [ ] Run `python -m pytest tests/test_realtime_soak.py -q && python -m pytest`.
- [ ] Commit with `git add scripts/realtime tests/test_realtime_soak.py && git commit -m "Add host-aware realtime evidence reports"`.

### Task 2: Receipt and checksum builder

**Files:**
- Create: `scripts/realtime/build_multihost_receipt.py`
- Create: `tests/test_realtime_receipt.py`

**Interfaces:** `build_receipt(load: dict, recovery: dict, dlq: dict, topology: dict) -> dict`; CLI requires `--load`, `--recovery`, `--dlq`, `--topology`, and `--output-dir`.

- [ ] Write failing tests: `build_receipt` raises `ValueError("stateful services are not highly available in this topology")` for a true HA flag; `verify_checksums()` returns false after an artifact changes; a missing required report field raises `ValueError`.
- [ ] Run `python -m pytest tests/test_realtime_receipt.py -q`; verify module-import failure.
- [ ] Implement `build_receipt`, required-field validation, `write_checksums(paths, output_dir)`, `verify_checksums(manifest)`, and JSON CLI output. Store only artifact file names and SHA-256 digests. Embed `stateful_services_highly_available: false` and the single-instance limitation.
- [ ] Run `python -m pytest tests/test_realtime_receipt.py -q && python -m pytest`.
- [ ] Commit with `git add scripts/realtime/build_multihost_receipt.py tests/test_realtime_receipt.py && git commit -m "Add multi-host evidence receipt builder"`.

### Task 3: Runbook and deployment execution

**Files:**
- Modify: `docs/MULTIHOST_EVIDENCE_RUNBOOK.md`
- Test: `tests/test_realtime_receipt.py`

**Interfaces:** The runbook supplies separate app-host load and state-host recovery/DLQ commands, then builds a receipt from their JSON files.

- [ ] Write a failing test that calls the receipt CLI with only `--output-dir` and asserts a non-zero exit code caused by missing required input paths.
- [ ] Run `python -m pytest tests/test_realtime_receipt.py -q`; verify required-argument failure.
- [ ] Document exact staged commands using `<state-host-private-dns>` placeholders, not host IPs. Document the redacted topology manifest, receipt command, checksum verification, and the explicit non-HA limitation.
- [ ] Run `python -m ruff check . && python -m pytest`.
- [ ] Commit with `git add docs/MULTIHOST_EVIDENCE_RUNBOOK.md tests/test_realtime_receipt.py && git commit -m "Document host-specific evidence run"`.

### Task 4: EC2 evidence run

**Files:** No source changes.

- [ ] Confirm both hosts have the same commit and healthy required services with `git rev-parse HEAD` and the matching Compose `ps` command.
- [ ] Run the app-host load stage with explicit remote ingestion/search endpoints and both consumer metric endpoints; preserve the raw JSON regardless of result.
- [ ] Run recovery and DLQ on the state host against its local Compose project, using its local consumer endpoint and the app consumer endpoint.
- [ ] Build the receipt from the three raw reports and redacted topology manifest; verify checksums.
- [ ] Commit only the redacted receipt and checksum manifest after checking that no secret or host address is present.

## Self-review

- Tasks 1-3 cover every spec requirement: host-local execution, no remote control, report validation, checksums, redaction, and non-HA wording.
- Every review-focus failure mode has a Task 1 or Task 2 test, while the existing recovery test covers consumer restoration.
- The interfaces used by Task 3 are defined in Task 2.
