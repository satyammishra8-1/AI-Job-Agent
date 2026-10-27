from fastapi.testclient import TestClient

from job_agent.dashboard.app import create_app
from job_agent.profile_intelligence import analyze_profile
from job_agent.tailor.career_facts import CareerFacts, Project


def test_profile_analysis_counts_actionable_gaps_and_evidence():
    result = analyze_profile(CareerFacts(
        skills_inventory={"Frontend": ("React", "Docker")},
        skill_evidence={"React": ("QuickChat repository",)},
        projects=(Project(name="QuickChat", technologies=("React",),
                          evidence=("GitHub repository",)),),
        target_roles=("Software Engineering Intern",),
    ))
    assert result["completeness"] == 38
    assert result["evidence_backed_skills"] == ["React"]
    assert result["unknown_or_unsupported_skills"] == ["Docker"]
    assert any("GitHub" in item for item in result["missing"])


def test_profile_create_update_version_and_stable_identity(tmp_path):
    client = TestClient(create_app(data_dir=tmp_path))
    first = client.put("/api/profile", json={"name": "", "target_roles": ["Data Engineer"]})
    assert first.status_code == 200
    profile1 = first.json()["profile"]
    assert profile1["profile_version"] == 1
    second = client.put("/api/profile", json=profile1)
    assert second.json()["changed"] is False
    assert second.json()["profile"]["profile_version"] == 1
    changed = {**profile1, "target_roles": ["Research Engineer"]}
    third = client.put("/api/profile", json=changed)
    assert third.json()["profile"]["profile_version"] == 2
    assert third.json()["profile"]["profile_id"] == profile1["profile_id"]


def test_profile_api_analyzes_unconfigured_profile_without_inventing_data(tmp_path):
    result = TestClient(create_app(data_dir=tmp_path)).get("/api/profile").json()
    assert result["analysis"]["completeness"] == 0
    assert result["analysis"]["target_roles"] == []
    assert result["profile"]["name"] == ""


def test_profile_change_invalidates_preparation_and_pending_approval(tmp_path):
    from job_agent.apply.tracker import update_application, upsert_job_state
    import json

    client = TestClient(create_app(data_dir=tmp_path))
    created = client.put("/api/profile", json={"target_roles": ["Developer"]}).json()["profile"]
    prep_path = tmp_path / "application_preparations.json"
    prep_path.write_text(json.dumps({"job-1": {"status": "READY_TO_APPLY", "reviewed": True,
                                                     "approved_fingerprint": "approval"}}))
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    (out_dir / "tailoring_status.json").write_text(json.dumps({"job-1": {
        "job_id": "job-1", "status": "READY", "filename": "resume",
    }}))
    record = upsert_job_state(tmp_path / "applications.json", job_id="job-1",
                              company="Example", title="Developer", status="ready_to_apply")
    update_application(tmp_path / "applications.json", record.attempt_id,
                       approval_state="approved", approval_identity="approval")

    response = client.put("/api/profile", json={**created, "target_roles": ["Research Engineer"]})
    assert response.status_code == 200
    assert response.json()["profile"]["profile_version"] == 2
    preparation = json.loads(prep_path.read_text())["job-1"]
    assert preparation["status"] == "NEEDS_REVIEW"
    assert preparation["approved_fingerprint"] is None
    tailoring = json.loads((out_dir / "tailoring_status.json").read_text())["job-1"]
    assert tailoring["status"] == "NEEDS_REVIEW"


def test_dashboard_search_adds_profile_roles_skills_and_locations(tmp_path, monkeypatch):
    import yaml
    from job_agent.config import load_profile
    from job_agent.dashboard import service

    search_profile = tmp_path / "search.yaml"
    search_profile.write_text(yaml.safe_dump({
        "keywords": ["software"], "sources": [{"ats": "ashby", "board": "example"}],
    }))
    facts_path = tmp_path / "career_facts.yaml"
    facts_path.write_text(yaml.safe_dump({
        "target_roles": ["AI Engineer"], "preferred_locations": ["Bengaluru"],
        "remote_preference": "remote", "skills_inventory": {"AI/ML": ["PyTorch"]},
        "skill_evidence": {"PyTorch": ["Vision project"]},
    }))
    captured = {}

    def fake_run(_handler, args):
        captured["profile"] = load_profile(args.profile)
        return {"ok": True}

    monkeypatch.setattr(service, "_run_cli", fake_run)
    assert service.run_search_cli(search_profile, facts_path=facts_path)["ok"]
    profile = captured["profile"]
    assert "AI Engineer" in profile.keywords
    assert "PyTorch" in profile.keywords
    assert "Bengaluru" in profile.location.allowed_countries
    assert profile.location.remote_ok
