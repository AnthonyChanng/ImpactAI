"""Cover summary queries without fetching large per-record JSON payloads."""

from alembic import op

revision = "b91537ac901e"
down_revision = "a83b91e274fc"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_index("ix_impact_version_player", table_name="impact_predictions")
    op.create_index(
        "ix_impact_version_player",
        "impact_predictions",
        ["version", "player_id", "game_id", "score"],
    )
    op.create_index(
        "ix_game_players_summary",
        "game_players",
        [
            "player_id",
            "game_id",
            "agent",
            "acs",
            "adr",
            "kills",
            "deaths",
            "first_kills",
            "first_deaths",
        ],
    )
    if op.get_bind().dialect.name == "sqlite":
        op.execute("ANALYZE")


def downgrade():
    op.drop_index("ix_game_players_summary", table_name="game_players")
    op.drop_index("ix_impact_version_player", table_name="impact_predictions")
    op.create_index("ix_impact_version_player", "impact_predictions", ["version", "player_id"])
