# Local model lifecycle

This POSIX runner connects immutable MIND snapshots to chronological recipe validation,
content-addressed receipts, artifact integrity checks, manual promotion and rollback.
Use a training/development snapshot that excludes the official final holdout.
The cutoff filters behavior events; the caller must supply the catalog available at
that cutoff. Article publication times cannot be reconstructed from these TSV files.

```sh
python scripts/manage_lifecycle.py --root runs/lifecycle train --snapshot data/snapshot --cutoff 2019-11-14 --minimum-ndcg 0.3 --minimum-validation 100
python scripts/manage_lifecycle.py --root runs/lifecycle promote RUN_ID
NEWSLENS_RELEASE_ROOT=runs/lifecycle uvicorn newslens.api.app:app
python scripts/manage_lifecycle.py --root runs/lifecycle rollback
```

Replace RUN_ID with the receipt ID. Repeating the same specification reuses its
verified result. Use successive immutable snapshots/cutoffs for backfills; each
produces a separate receipt. Rejected candidates cannot be promoted. Release
pointers are written atomically under a process lock. Restart API workers after
promotion or rollback; each worker resolves and verifies the pointer at startup.
Set only one of NEWSLENS_RELEASE_ROOT and NEWSLENS_ARTIFACT_PATH.

The gate assesses the training recipe on a chronological validation split and
then refits on available snapshot events. It does not establish final-test quality
of that refit. This local implementation does not establish cloud orchestration,
a long-running production deployment, user lift, or a sustained operational SLO.
Keep the root writable only by trusted operators; artifact loading uses joblib.
