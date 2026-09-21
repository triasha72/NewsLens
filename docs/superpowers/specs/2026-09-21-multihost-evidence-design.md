# Multi-host evidence workflow

**Status:** Proposed  
**Date:** 2026-09-21  
**Decider:** Triasha Sarkar

## Context

NewsLens runs the application tier and the state tier on separate EC2 hosts.
The current `run_soak_evidence.py` wrapper can label a result as two-host, but
it invokes every child command from the machine where it is started. That is
not valid for recovery or DLQ checks: those checks must operate Kafka and the
state-host Docker Compose project locally.

The workflow must produce evidence that can be inspected later without
including credentials, private IP addresses, or a claim of high availability.

## Decision

Run the evidence in three host-local stages and then combine the immutable JSON
artifacts into a receipt:

1. The app host runs load generation against the state-host ingestion and
   search APIs. It queries consumer metrics from both hosts.
2. The state host runs recovery and DLQ checks locally against its Kafka and
   Compose project. It queries the state consumer locally and the app consumer
   through its private application endpoint.
3. A receipt builder validates the input schemas, records only host roles,
   commit SHA, image digests, instance type, and service health, then writes a
   checksum manifest and readiness result. It rejects a receipt that calls the
   topology highly available.

The receipt builder is a local file operation. It does not use SSH, does not
copy secrets, and does not attempt to start or stop remote containers.

## Options considered

### One app-host orchestration script with remote Docker control

This would require SSH credentials, remote command execution, host-key
management, and failure handling. It is unnecessary for the current evidence
goal and makes the workflow less portable.

### Three host-local stages with a receipt builder

This keeps operational actions on the machine that owns the affected
containers. JSON artifacts form the boundary between stages. This is the
chosen option.

### Keep the current wrapper and change labels only

This would produce a misleading result and is rejected.

## Interfaces

### App-host load stage

`benchmark_ingestion.py` receives explicit ingestion, search, and both
consumer metric URLs. Its report includes an `execution_host_role` field of
`application` and a topology reference with `application_hosts: 2`.

### State-host recovery and DLQ stages

`failure_recovery.py` and `verify_dlq.py` receive the state-host Compose file
and explicit consumer metric URLs. Their reports include an
`execution_host_role` field of `state`.

### Receipt builder

The new command receives paths to the three raw reports and a redacted topology
manifest. It validates required fields, writes `multihost_receipt_v0_1.json`,
and writes SHA-256 checksums for every artifact. The output states that Kafka
and PostgreSQL are single-instance stateful services.

## Failure handling

- A failed stage does not create a passing receipt.
- A malformed report or checksum mismatch stops receipt creation.
- Recovery tests restart the consumers in a `finally` block.
- Operators preserve raw failed reports alongside successful attempts; they do
  not merge planned and unplanned failures into one passing receipt.

## Verification

- Unit tests prove explicit URLs and Compose paths are present in each stage
  command.
- Unit tests prove the receipt builder rejects a topology claiming stateful HA.
- On EC2, each stage is run from its designated host and its service health is
  recorded before and after the run.
- The final receipt is checked against its checksum manifest.

## Consequences

This closes the multi-host application-evidence gap once a successful receipt
is published. It does not close the separate high-availability gap: Kafka and
PostgreSQL remain single-instance services.
