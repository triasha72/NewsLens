import json

import pytest
from test_artifact_export import make_behaviors, make_news

from newslens.artifacts import export_fallback_artifact
from newslens.operations.lifecycle import atomic_json, digest, release, serving_path


def candidate(root, run_id):
    path = root / "runs" / run_id
    export_fallback_artifact(make_news(), make_behaviors(), path / "artifact")
    atomic_json(
        path / "receipt.json",
        {
            "run_id": run_id,
            "status": "candidate",
            "manifest_sha256": digest(path / "artifact/manifest.json"),
        },
    )
    return path


def test_promote_rollback_and_corruption(tmp_path):
    a, b = "a" * 24, "b" * 24
    candidate(tmp_path, a)
    path = candidate(tmp_path, b)
    release(tmp_path, a)
    release(tmp_path, b)
    assert serving_path(tmp_path) == (path / "artifact").resolve()
    assert release(tmp_path, rollback=True)["current"] == a
    (path / "artifact/model.joblib").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        release(tmp_path, b)
    assert json.loads((tmp_path / "release.json").read_text())["current"] == a


def test_rejected_candidate_cannot_promote(tmp_path):
    atomic_json(tmp_path / "runs" / ("a" * 24) / "receipt.json", {"status": "rejected"})
    with pytest.raises(ValueError, match="passing candidate"):
        release(tmp_path, "a" * 24)


def test_training_retry_cutoff_and_api_release(tmp_path, monkeypatch):
    from newslens.api.settings import ApiSettings, ApiSettingsError
    from newslens.operations.lifecycle import train

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "news.tsv").write_text(
        "N1\tscience\tspace\tMars water\tPlanet exploration\turl\t[]\t[]\nN2\tsport\tball\tMars team\tChampionship wins\turl\t[]\t[]\n"
    )
    (snapshot / "behaviors.tsv").write_text(
        "".join(f"{i}\tU1\t01/{i:02d}/2020 12:00:00 AM\tN1\tN1-0 N2-1\n" for i in range(1, 11))
    )
    root = tmp_path / "registry"
    args = {"cutoff": "2020-01-09", "minimum_ndcg": 0, "minimum_validation": 1}
    result = train(snapshot, root, **args)
    assert result["status"] == "candidate"
    assert train(snapshot, root, **args) == result
    release(root, result["run_id"])
    monkeypatch.delenv("NEWSLENS_ARTIFACT_PATH", raising=False)
    monkeypatch.setenv("NEWSLENS_RELEASE_ROOT", str(root))
    assert ApiSettings.from_environment().artifact_path == serving_path(root)
    monkeypatch.setenv("NEWSLENS_ARTIFACT_PATH", "other")
    with pytest.raises(ApiSettingsError):
        ApiSettings.from_environment()
    with pytest.raises(ValueError, match="no behaviors"):
        train(snapshot, root, **{**args, "cutoff": "1999-01-01"})


def test_rehearsal_refuses_live_registry(tmp_path):
    from scripts.rehearse_lifecycle import rehearse

    with pytest.raises(ValueError, match="isolated registry"):
        rehearse(tmp_path, tmp_path, ["2020-01-01", "2020-01-02"], 0, 1)


def test_complete_release_rehearsal(tmp_path):
    from scripts.rehearse_lifecycle import rehearse

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "news.tsv").write_text(
        "N1\tscience\tspace\tMars water\tPlanet\turl\t[]\t[]\n"
        "N2\tscience\tspace\tMars team\tPlanet\turl\t[]\t[]\n"
    )
    (snapshot / "behaviors.tsv").write_text(
        "".join(f"{i}\tU1\t01/{i:02d}/2020 12:00:00 AM\tN1\tN1-0 N2-1\n" for i in range(1, 11))
    )
    result = rehearse(snapshot, tmp_path / "registry", ["2020-01-08", "2020-01-09"], 0, 1)
    assert result["status"] == "completed"
    assert [e["step"] for e in result["events"]][-1] == "rollback_and_resolve"
    assert result["events"][-1]["run_id"] == result["events"][0]["run_id"]
