from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from backend.app import main

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize(
    "url",
    [
        "/health",
        "/api/overview",
        "/api/matches",
        "/api/matches/60894",
        "/api/matches/60894/players",
        "/api/matches/60894/rounds",
        "/api/players",
        "/api/players/8419",
        "/api/players/8419/matches",
        "/api/players/8419/impact",
        "/api/players/8419/trends",
        "/api/models",
        "/api/models/metrics",
        "/api/analysis/global-features",
        "/api/agents",
    ],
)
def test_major_endpoints_serve_real_records(client, url):
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.json()["data"]
    assert response.json()["meta"]["model_version"].startswith("v1-")


def test_search_filters_pagination_and_missing(client):
    found = client.get("/api/players?q=Reduxx&agent=jett&map=Breeze").json()
    assert found["meta"]["total"] == 1
    assert found["data"][0]["id"] == 8419
    assert client.get("/api/players?q=unavailable-player").json()["data"] == []
    assert len(client.get("/api/players?page_size=3").json()["data"]) == 3
    for url in ["/api/players/-1", "/api/matches/99999999", "/api/players/8419/impact?game_id=99"]:
        assert client.get(url).status_code == 404
    assert client.get("/api/matches?page_size=999").status_code == 422
    assert client.get("/api/matches?date_from=2022-01-09&date_to=2021-01-01").status_code == 422


def test_scoreboard_and_round_consistency(client):
    players = client.get("/api/matches/60894/players").json()["data"]
    assert len(players) == 10
    assert sum(p["team_side"] == 1 for p in players) == 5
    rounds = client.get("/api/matches/60894/rounds").json()["data"]
    assert len(rounds) == 20
    assert all(0 <= r["probability"] <= 1 for r in rounds)
    assert rounds[-1]["team1_pre_score"] + int(rounds[-1]["team1_won"]) == 13
    assert all(p["impact"]["peer_count"] >= 50 for p in players)


def test_inference_swaps_and_rejects_leakage(client):
    body = {
        "map": "Ascent",
        "patch": "Patch 3.0",
        "round_number": 5,
        "team1_pre_score": 3,
        "team2_pre_score": 1,
        "team1_side": "attack",
        "previous_win": 1,
    }
    a = client.post("/api/analysis/round", json=body)
    assert a.status_code == 200
    b = client.post(
        "/api/analysis/round",
        json={
            **body,
            "team1_pre_score": 1,
            "team2_pre_score": 3,
            "team1_side": "defend",
            "previous_win": 0,
        },
    )
    assert a.json()["data"]["team1_probability"] + b.json()["data"][
        "team1_probability"
    ] == pytest.approx(1, abs=1e-6)
    assert (
        client.post("/api/analysis/round", json={**body, "WinType_POST_ROUND": "elim"}).status_code
        == 422
    )
    assert client.post("/api/analysis/round", json={**body, "round_number": 25}).status_code == 422


def test_coach_evidence_is_grounded_and_cache_is_idempotent(client):
    body = {"player_id": 8419, "game_id": 60894}
    response = client.post("/api/analysis/coach", json=body)
    assert response.status_code == 200
    report = response.json()["data"]
    performance = client.get("/api/players/8419/impact?game_id=60894").json()["data"]
    for evidence in report["evidence"]:
        component = performance["impact"]["components"][evidence["component"]]
        assert evidence["value"] == component["value"]
        assert evidence["peer_mean"] == component["peer_mean"]
        assert evidence["source"]["game_id"] == 60894
    assert report == client.post("/api/analysis/coach", json=body).json()["data"]
    fallback = client.post("/api/analysis/coach", json={**body, "use_llm": True}).json()["data"]
    assert fallback["evidence"] == report["evidence"]
    assert (
        client.post("/api/analysis/coach", json={"player_id": 1, "game_id": 99}).status_code == 404
    )


def test_cached_coach_provider_does_not_block_other_database_requests(client, monkeypatch):
    body = {"player_id": 8419, "game_id": 60894}
    assert client.post("/api/analysis/coach", json=body).status_code == 200
    entered, release = Event(), Event()

    def slow_provider(report):
        entered.set()
        assert release.wait(timeout=5), "Test did not release the provider"
        return report

    monkeypatch.setattr(main, "optional_narration", slow_provider)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(
            client.post, "/api/analysis/coach", json={**body, "use_llm": True}
        )
        try:
            assert entered.wait(timeout=2)
            health = executor.submit(client.get, "/health")
            assert health.result(timeout=2).status_code == 200
        finally:
            release.set()
        assert pending.result(timeout=2).status_code == 200
