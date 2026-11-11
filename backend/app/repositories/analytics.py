from sqlalchemy import and_, func, select

from backend.app.models import Game, GamePlayer, ImpactPrediction, Match, Player


def game_out(game, match):
    return {
        "id": game.id,
        "match_id": game.match_id,
        "map": game.map,
        "team1_name": game.team1_name,
        "team2_name": game.team2_name,
        "team1_score": game.team1_score,
        "team2_score": game.team2_score,
        "num_rounds": game.num_rounds,
        "outcome_reliable": game.outcome_reliable,
        "date": match.date.isoformat(),
        "event": match.event,
        "patch": match.patch,
        "split": match.split,
    }


def player_query(version):
    return (
        select(
            Player.id,
            Player.name,
            func.count(GamePlayer.game_id).label("games"),
            func.avg(GamePlayer.acs).label("acs"),
            func.avg(GamePlayer.adr).label("adr"),
            func.sum(GamePlayer.kills).label("kills"),
            func.sum(GamePlayer.deaths).label("deaths"),
            func.avg(ImpactPrediction.score).label("score"),
        )
        .join(GamePlayer, Player.id == GamePlayer.player_id)
        .join(
            ImpactPrediction,
            and_(
                ImpactPrediction.game_id == GamePlayer.game_id,
                ImpactPrediction.player_id == GamePlayer.player_id,
                ImpactPrediction.version == version,
            ),
        )
        .group_by(Player.id, Player.name)
    )


def player_filters(query, q=None, agent=None, map_name=None):
    if q:
        query = query.where(Player.name.icontains(q, autoescape=True))
    if agent:
        query = query.where(GamePlayer.agent == agent)
    if map_name:
        query = query.join(Game, Game.id == GamePlayer.game_id).where(Game.map == map_name)
    return query


def impact_rows(db, version, player_id=None, game_id=None):
    query = (
        select(GamePlayer, Player, Game, Match, ImpactPrediction)
        .join(Player, Player.id == GamePlayer.player_id)
        .join(Game, Game.id == GamePlayer.game_id)
        .join(Match, Match.id == Game.match_id)
        .join(
            ImpactPrediction,
            and_(
                ImpactPrediction.game_id == GamePlayer.game_id,
                ImpactPrediction.player_id == GamePlayer.player_id,
                ImpactPrediction.version == version,
            ),
        )
    )
    if player_id is not None:
        query = query.where(
            GamePlayer.player_id == player_id, ImpactPrediction.player_id == player_id
        )
    if game_id is not None:
        # Make the prediction-key restriction explicit. SQLite otherwise may scan
        # every prediction in a version before applying the transitive game filter.
        query = query.where(GamePlayer.game_id == game_id, ImpactPrediction.game_id == game_id)
    return query.order_by(
        Match.date.desc(), Game.id.desc(), GamePlayer.team_side, ImpactPrediction.score.desc()
    )


def performance_out(row):
    gp, player, game, match, impact = row
    return {
        "player_id": player.id,
        "player_name": gp.stats["PlayerName"],
        "agent": gp.agent,
        "team_side": gp.team_side,
        "team": game.team1_name if gp.team_side == 1 else game.team2_name,
        "team_won": gp.team_won,
        "kills": gp.kills,
        "deaths": gp.deaths,
        "assists": gp.assists,
        "acs": gp.acs,
        "adr": gp.adr,
        "first_kills": gp.first_kills,
        "first_deaths": gp.first_deaths,
        "hs_percent": gp.stats["HS_Percent"],
        "game": game_out(game, match),
        "impact": impact.detail,
    }


def earlier_performances(db, version, row):
    """Strictly earlier series only; same-series maps cannot leak into the comparison."""
    gp, _, _, match, _ = row
    query = (
        impact_rows(db, version, player_id=gp.player_id)
        .where(Match.date < match.date, Match.id != match.id, GamePlayer.agent == gp.agent)
        .limit(9)
    )
    return [performance_out(previous) for previous in db.execute(query)]
