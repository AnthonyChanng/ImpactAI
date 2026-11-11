"""Require derived outputs to reference observed performances and rounds."""

from alembic import op

revision = "a83b91e274fc"
down_revision = "6d8c25a5c354"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("impact_predictions") as batch:
        batch.create_foreign_key(
            "fk_impact_performance",
            "game_players",
            ["game_id", "player_id"],
            ["game_id", "player_id"],
        )
        batch.create_check_constraint("ck_impact_score", "score BETWEEN 0 AND 100")
        batch.create_check_constraint("ck_impact_percentile", "percentile BETWEEN 0 AND 100")
        batch.create_check_constraint(
            "ck_impact_probability", "association_probability BETWEEN 0 AND 1"
        )
    with op.batch_alter_table("round_predictions") as batch:
        batch.create_foreign_key(
            "fk_prediction_round",
            "rounds",
            ["game_id", "round_number"],
            ["game_id", "number"],
        )
        batch.create_check_constraint("ck_round_probability", "probability BETWEEN 0 AND 1")
    with op.batch_alter_table("coaching_reports") as batch:
        batch.create_foreign_key(
            "fk_coach_performance",
            "game_players",
            ["game_id", "player_id"],
            ["game_id", "player_id"],
        )


def downgrade():
    with op.batch_alter_table("coaching_reports") as batch:
        batch.drop_constraint("fk_coach_performance", type_="foreignkey")
    with op.batch_alter_table("round_predictions") as batch:
        batch.drop_constraint("ck_round_probability", type_="check")
        batch.drop_constraint("fk_prediction_round", type_="foreignkey")
    with op.batch_alter_table("impact_predictions") as batch:
        for name in ["ck_impact_probability", "ck_impact_percentile", "ck_impact_score"]:
            batch.drop_constraint(name, type_="check")
        batch.drop_constraint("fk_impact_performance", type_="foreignkey")
