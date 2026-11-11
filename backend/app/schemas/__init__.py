from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    data: T
    meta: dict[str, Any] = Field(default_factory=dict)


class GameOut(BaseModel):
    id: int
    match_id: int
    map: str
    date: str
    event: str
    patch: str | None
    split: str
    team1_name: str
    team2_name: str
    team1_score: int
    team2_score: int
    num_rounds: int
    outcome_reliable: bool


class PlayerOut(BaseModel):
    id: int
    name: str
    games: int
    acs: float
    adr: float
    kills: float
    deaths: float
    score: float


class RoundOut(BaseModel):
    number: int
    team1_pre_score: int
    team2_pre_score: int
    team1_side: str | None
    team1_won: bool
    win_type: str
    probability: float | None


class NamedCount(BaseModel):
    name: str
    count: int


class ComponentOut(BaseModel):
    z: float
    value: float
    peer_mean: float


class ContributionOut(BaseModel):
    feature: str
    contribution: float


class ExplanationOut(BaseModel):
    method: str
    base_log_odds: float
    raw_log_odds: float
    temperature: float
    top: list[ContributionOut]
    other_contribution: float


class ImpactOut(BaseModel):
    game_id: int
    player_id: int
    model_version: str
    score: float = Field(ge=0, le=100)
    percentile: float = Field(ge=0, le=100)
    peer_count: int
    peer_level: str
    exact_context_count: int
    components: dict[str, ComponentOut]
    association_probability: float = Field(ge=0, le=1)
    explanation: ExplanationOut


class PerformanceOut(BaseModel):
    player_id: int
    player_name: str
    agent: str
    team_side: int
    team: str
    team_won: bool | None
    kills: float
    deaths: float
    assists: float
    acs: float
    adr: float
    first_kills: float
    first_deaths: float
    hs_percent: float | None
    game: GameOut
    impact: ImpactOut


class PlayerDetailOut(PlayerOut):
    agents: list[NamedCount]
    historical_team_ids: list[int]


class TrendOut(BaseModel):
    game_id: int
    date: str
    map: str
    agent: str
    score: float
    acs: float
    adr: float


class MonthCount(BaseModel):
    month: str
    games: int


class OverviewOut(BaseModel):
    games: int
    series: int
    players: int
    rounds: int
    player_records: int
    agents: list[NamedCount]
    trends: list[MonthCount]
    distribution: list[NamedCount]
    maps: list[str]
    coverage_start: str
    coverage_end: str
    round_model: str
    player_model: str
    warnings: list[str]


class AgentOut(BaseModel):
    name: str
    selections: int
    adr: float
    score: float
    fkpr: float
    fdpr: float
    min_score: float
    max_score: float
    distribution: list[NamedCount]


class EvidenceSource(BaseModel):
    game_id: int
    player_id: int
    metric: str
    model_version: str


class EvidenceOut(BaseModel):
    id: str
    component: str
    category: str
    value: float
    peer_mean: float
    z: float
    statement: str
    suggestion: str
    source: EvidenceSource
    pattern: str
    prior_count: int
    prior_below: int
    prior_median_z: float | None


class CoachOut(BaseModel):
    player_id: int
    game_id: int
    player_name: str
    agent: str
    mode: str
    model_version: str
    score: float
    percentile: float
    peer_count: int
    peer_level: str
    evidence: list[EvidenceOut]
    limitations: list[str]
    llm_status: str | None = None
    coach_version: str
    history_count: int
    history_game_ids: list[int]
    assessment: str


class CoachRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    player_id: int = Field(gt=0)
    game_id: int = Field(gt=0)
    use_llm: bool = False


class RoundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    map: str = Field(min_length=1, max_length=40)
    patch: str | None = Field(default=None, max_length=300)
    round_number: int = Field(ge=1, le=100)
    team1_pre_score: int = Field(ge=0, le=99)
    team2_pre_score: int = Field(ge=0, le=99)
    team1_side: str = Field(pattern="^(attack|defend|unknown)$")
    previous_win: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def coherent_history(self):
        if self.team1_pre_score + self.team2_pre_score != self.round_number - 1:
            raise ValueError("Pre-scores must sum to round_number - 1")
        if self.round_number == 1 and self.previous_win != 0.5:
            raise ValueError("Round one previous_win must be 0.5 (unavailable)")
        if self.round_number > 1 and self.previous_win not in [0.0, 1.0]:
            raise ValueError("Previous round outcome must be 0 or 1")
        scores = [self.team1_pre_score, self.team2_pre_score]
        if max(scores) >= 13 and abs(scores[0] - scores[1]) >= 2:
            raise ValueError("Pre-scores describe a completed game")
        if self.round_number > 1:
            previous_winner = 0 if self.previous_win == 1 else 1
            if scores[previous_winner] == 0:
                raise ValueError("The previous round winner must have at least one round win")
            scores[previous_winner] -= 1
            if max(scores) >= 13 and abs(scores[0] - scores[1]) >= 2:
                raise ValueError("The game would have ended before the previous round")
        if self.round_number > 24 and self.team1_side != "unknown":
            raise ValueError("Overtime sides must remain unknown")
        return self


class PeerRangeComponent(BaseModel):
    low: float
    median: float
    high: float
    measured_coverage: float = Field(ge=0, le=1)


class PeerRangesOut(BaseModel):
    available: bool
    reason: str | None
    research_run: str
    nominal_coverage: float = Field(gt=0, lt=1)
    components: dict[str, PeerRangeComponent]
    limitation: str
