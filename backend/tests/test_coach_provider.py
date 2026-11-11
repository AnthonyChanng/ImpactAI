from types import SimpleNamespace

from backend.app.services import coach


def test_optional_provider_can_only_reorder_known_evidence(monkeypatch):
    report = {
        "evidence": [{"id": "combat", "value": 0.7}, {"id": "openings", "value": -0.05}],
        "mode": "deterministic_evidence",
    }
    monkeypatch.setattr(
        coach,
        "settings",
        SimpleNamespace(
            coach_llm_url="https://provider.invalid/chat",
            coach_llm_api_key="test-key",
            coach_llm_model="test-model",
        ),
    )

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"order":["openings","combat"]}'}}]}

    monkeypatch.setattr(coach.httpx, "post", lambda *args, **kwargs: Response())
    result = coach.optional_narration(report)
    assert result["mode"] == "llm_ordered_verified_evidence"
    assert result["evidence"] == report["evidence"][::-1]
    monkeypatch.setattr(
        Response,
        "json",
        lambda self: {
            "choices": [{"message": {"content": '{"order":["invented statistic","combat"]}'}}]
        },
    )
    result = coach.optional_narration(report)
    assert result["evidence"] == report["evidence"]
    assert result["mode"] == "deterministic_evidence"
    assert "invalid" in result["llm_status"]
