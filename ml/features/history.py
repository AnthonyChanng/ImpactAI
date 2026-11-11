"""Observable past information for research; no current-game final statistics or raw ID inputs."""

from collections import defaultdict

import numpy as np
import pandas as pd

from ml.features.build import round_features

RECENT_FEATURES = ["RecentWinDiff3", "RecentWinDiff5", "PreviousStreak", "HalfScoreDiff"]
TEAM_FEATURES = [
    "HistoricalEloDiff",
    "HistoricalMapEloDiff",
    "HistoricalExperienceDiff",
    "HistoricalMinGames",
]


def recent_round_features(raw):
    """All added outcomes are shifted within game before use; preserve caller row order."""
    ordered = raw.sort_values(["GameID", "RoundNumber"])
    x = round_features(ordered)
    previous = ordered.groupby("GameID").Team1_RoundWon.shift(1)
    for window in (3, 5):
        mean = (
            previous.groupby(ordered.GameID)
            .transform(lambda group: group.rolling(window, min_periods=1).mean())
            .fillna(0.5)
        )
        x[f"RecentWinDiff{window}"] = 2 * mean - 1
    phase_keys = [ordered.GameID, x.Phase]
    prior_in_phase = ordered.groupby(phase_keys).Team1_RoundWon.shift(1).fillna(0)
    x["HalfScoreDiff"] = (
        2 * prior_in_phase.groupby(phase_keys).cumsum() - ordered.groupby(phase_keys).cumcount()
    )
    streaks = []
    last_game, streak = None, 0
    for game_id, outcome in zip(ordered.GameID, ordered.Team1_RoundWon):
        if game_id != last_game:
            streak = 0
        streaks.append(streak)
        sign = 1 if outcome else -1
        streak = streak + sign if streak * sign > 0 else sign
        last_game = game_id
    x["PreviousStreak"] = streaks
    return x.reindex(raw.index)


class TeamHistory:
    """Elo updates after whole timestamp batches; average maps per series before updating.

    Training features use strictly earlier timestamps. Transform freezes the final training
    state, even when validation outcomes are supplied. IDs are lookup keys, never model inputs.
    """

    def __init__(self, k=24.0):
        self.k = k
        self.global_ratings = {}
        self.map_ratings = {}
        self.counts = {}

    def _row(self, row):
        a, b = row.Team1ID, row.Team2ID
        ga, gb = self.global_ratings.get(a, 0.0), self.global_ratings.get(b, 0.0)
        ma = self.map_ratings.get((a, row.Map), ga)
        mb = self.map_ratings.get((b, row.Map), gb)
        na, nb = self.counts.get(a, 0), self.counts.get(b, 0)
        return {
            "GameID": row.GameID,
            "HistoricalEloDiff": (ga - gb) / 400,
            "HistoricalMapEloDiff": (ma - mb) / 400,
            "HistoricalExperienceDiff": np.log1p(na) - np.log1p(nb),
            "HistoricalMinGames": min(na, nb),
        }

    def fit_transform(self, games):
        if games.empty or games.GameID.duplicated().any():
            raise ValueError("Training history needs unique nonempty game rows")
        if games.groupby("MatchID").Date.nunique().max() != 1:
            raise ValueError("All maps of a series must have the same timestamp")
        self.global_ratings, self.map_ratings, self.counts = {}, {}, {}
        rows = []
        for _, batch in games.sort_values(["Date", "GameID"]).groupby("Date", sort=True):
            global_deltas, map_deltas, counts = (
                defaultdict(list),
                defaultdict(list),
                defaultdict(int),
            )
            for game in batch.itertuples():
                features = self._row(game)
                rows.append(features)
                reliable = (
                    game.Team1_Rounds != game.Team2_Rounds
                    and max(game.Team1_Rounds, game.Team2_Rounds) >= 13
                )
                if not reliable:
                    continue
                outcome = float(game.Team1_Rounds > game.Team2_Rounds)
                for rating_key, deltas in [
                    ("HistoricalEloDiff", global_deltas),
                    ("HistoricalMapEloDiff", map_deltas),
                ]:
                    expected = 1 / (1 + 10 ** (-features[rating_key]))
                    delta = self.k * (outcome - expected)
                    for team, sign in [(game.Team1ID, 1), (game.Team2ID, -1)]:
                        key = (
                            (game.MatchID, team)
                            if rating_key == "HistoricalEloDiff"
                            else (game.MatchID, team, game.Map)
                        )
                        deltas[key].append(sign * delta)
                counts[game.Team1ID] += 1
                counts[game.Team2ID] += 1
            # New map ratings inherit pre-batch global state, not outcomes in this batch.
            for (_, team, map_name), values in map_deltas.items():
                key = (team, map_name)
                self.map_ratings[key] = self.map_ratings.get(
                    key, self.global_ratings.get(team, 0.0)
                ) + float(np.mean(values))
            for (_, team), values in global_deltas.items():
                self.global_ratings[team] = self.global_ratings.get(team, 0.0) + float(
                    np.mean(values)
                )
            for team, n in counts.items():
                self.counts[team] = self.counts.get(team, 0) + n
        self.last_training_timestamp = str(games.Date.max())
        return pd.DataFrame(rows).set_index("GameID").reindex(games.GameID)

    def transform(self, games):
        if games.empty:
            return pd.DataFrame(columns=TEAM_FEATURES, index=pd.Index([], name="GameID"))
        if games.Date.le(pd.Timestamp(self.last_training_timestamp)).any():
            raise ValueError("Frozen team history may only be used after the training period")
        return pd.DataFrame([self._row(row) for row in games.itertuples()]).set_index("GameID")
