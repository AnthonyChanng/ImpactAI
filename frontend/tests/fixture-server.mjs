// Artificial API fixtures for browser tests only. Never imported by the application.
import { createServer } from 'node:http';
const version = 'fixture-only';
const meta = {model_version: version};
const components = Object.fromEntries(['Combat', 'Openings', 'Damage', 'Assists', 'Objectives', 'Clutches'].map((name, i) => [name, {z: i === 0 ? 1 : -0.7, value: i === 2 ? 130 : 0.1, peer_mean: i === 2 ? 150 : 0.2}]));
const games = [60894, 60895].map((id, i) => ({id, match_id: i + 1, date: `2021-10-0${2-i}T12:00:00`, map: i ? 'Ascent' : 'Breeze', event: 'Browser test fixture', patch: 'Fixture patch', split: 'validation', team1_name: 'Example One', team2_name: 'Example Two', team1_score: 13, team2_score: 7, num_rounds: 20, outcome_reliable: true}));
const players = Array.from({length: 10}, (_, i) => ({id: 8419 + i, name: i ? `Example ${i}` : 'Example Player', games: 2, kills: 30, deaths: 20, acs: 210, adr: 130, score: 55, agents: [{name: i % 2 ? 'sage' : 'jett', count: 2}]}));
function performance(player, game) {
  return {player_id: player.id, player_name: player.name, agent: player.agents[0].name, team: 'Example One', team_side: player.id < 8424 ? 1 : 2, team_won: true, kills: 15, deaths: 10, assists: 5, acs: 210, adr: 130, first_kills: 2, first_deaths: 3, hs_percent: 20, game,
    impact: {game_id: game.id, player_id: player.id, model_version: version, score: 55, percentile: 60, peer_count: 120, peer_level: 'agent_map', exact_context_count: 20, association_probability: 0.6, components, explanation: {method: 'Test fixture', base_log_odds: 0, raw_log_odds: 0.2, temperature: 1, other_contribution: 0, top: [{feature: 'ADR', contribution: 0.2}]}}};
}
const distribution = [{name: '0–25', count: 2}, {name: '25–50', count: 5}, {name: '50–75', count: 10}, {name: '75–100', count: 3}];
const metric = {n: 20, log_loss: 0.6, brier: 0.2, roc_auc: 0.7, accuracy: 0.6, calibration: [{predicted: 0.3, observed: 0.3}, {predicted: 0.7, observed: 0.7}]};
const model = {selected: 'fixture', temperature: 1, validation_selection: {fixture: metric}, test_predeclared_models_uncalibrated: {fixture: metric}, test_selected: metric, test_log_loss_95pct_series_bootstrap: [0.5, 0.7], slices: {Map: {Breeze: metric}}};
const overview = {games: 2, series: 2, players: 10, rounds: 40, player_records: 20, agents: [{name: 'jett', count: 10}, {name: 'sage', count: 10}], maps: ['Ascent', 'Breeze'], trends: [{month: '2021-10', games: 2}], distribution, coverage_start: '2021-10-01', coverage_end: '2021-10-02', round_model: 'fixture', player_model: 'fixture', warnings: ['Artificial browser test data']};
const envelope = (data, extra = {}) => ({data, meta: {...meta, ...extra}});
function paginate(rows, url) {
  const page = Number(url.searchParams.get('page') || 1), size = Number(url.searchParams.get('page_size') || 20);
  return envelope(rows.slice((page-1)*size, page*size), {total: rows.length, page, page_size: size});
}
function ranges(gameId) {
  return {available: gameId === 60894, reason: gameId === 60894 ? null : 'This performance is outside the saved research sample; experimental ranges are unavailable.', research_run: 'browser-fixture', nominal_coverage: 0.8, limitation: 'Experimental fixture: reused development validation; coverage is not guaranteed.', components: gameId === 60894 ? Object.fromEntries(Object.keys(components).map(c => [c, {low: 0, median: c === 'Damage' ? 130 : 0.2, high: c === 'Damage' ? 180 : 0.4, measured_coverage: 0.78}])) : {}};
}
async function respond(req, url) {
  const path = url.pathname;
  const q = url.searchParams;
  if (path === '/health') return envelope({status: 'fixture'});
  if (path === '/api/overview') return envelope(overview);
  if (path === '/api/models/metrics') return envelope({version, round: model, player: model, global_features: [], xgboost_shap: [{feature: 'ADR', importance: 0.2}], global_explanation_method: 'fixture', protocol: 'Artificial browser fixture.', limitations: ['diagnostic controls only']});
  if (path === '/api/agents') return envelope(overview.agents.map(a => ({...a, selections: 10, adr: 130, score: 55, fkpr: 0.1, fdpr: 0.1, min_score: 20, max_score: 80, distribution})));
  if (path === '/api/players') return paginate(players.filter(p => p.name.toLowerCase().includes((q.get('q') || '').toLowerCase()) && (!q.get('agent') || p.agents[0].name === q.get('agent'))), url);
  if (path === '/api/matches') return paginate(games.filter(g => (!q.get('map') || g.map === q.get('map')) && g.event.toLowerCase().includes((q.get('event') || '').toLowerCase())), url);
  const match = path.match(/^\/api\/matches\/(\d+)(?:\/(players|rounds))?$/);
  if (match) {
    const game = games.find(g => g.id === Number(match[1]));
    if (!game) return null;
    if (match[2] === 'players') return envelope(players.map(p => performance(p, game)));
    if (match[2] === 'rounds') return envelope(Array.from({length: 20}, (_, i) => ({number: i+1, team1_pre_score: Math.min(i, 12), team2_pre_score: Math.max(0, i-12), team1_side: i < 12 ? 'attack' : 'defend', team1_won: i < 12 || i === 19, win_type: 'elim', probability: i % 2 ? 0.6 : 0.5})));
    return envelope(game);
  }
  const playerRoute = path.match(/^\/api\/players\/(\d+)(?:\/(matches|impact|trends|peer-ranges))?$/);
  if (playerRoute) {
    const player = players.find(p => p.id === Number(playerRoute[1]));
    if (!player) return null;
    if (!playerRoute[2]) return envelope(player);
    if (playerRoute[2] === 'matches') return paginate(games.map(g => performance(player, g)), url);
    if (playerRoute[2] === 'trends') return envelope(games.map(g => ({game_id: g.id, date: g.date, score: 55})));
    const game = games.find(g => g.id === Number(q.get('game_id') || 60894));
    if (!game) return null;
    if (playerRoute[2] === 'peer-ranges') return envelope(ranges(game.id));
    return envelope(performance(player, game));
  }
  if (path === '/api/analysis/coach' && req.method === 'POST') {
    let body = '';
    for await (const chunk of req) body += chunk;
    const {player_id, game_id} = JSON.parse(body);
    const player = players.find(p => p.id === player_id), game = games.find(g => g.id === game_id);
    if (!player || !game) return null;
    return envelope({player_id, game_id, player_name: player.name, agent: player.agents[0].name, mode: 'deterministic_evidence', model_version: version, coach_version: 'patterns-v2', history_count: 0, history_game_ids: [], assessment: 'Review priorities are descriptive heuristics.', score: 55, percentile: 60, peer_count: 120, peer_level: 'agent_map', limitations: ['Artificial browser test data.'], evidence: ['Combat', 'Openings', 'Damage'].map((component, i) => ({id: component.toLowerCase(), component, category: i ? 'review_priority' : 'strength', ...components[component], pattern: 'limited_history', prior_count: 0, prior_below: 0, prior_median_z: null, statement: `${component}: fixture observation.`, suggestion: 'Review the source record.', source: {game_id, player_id, metric: component, model_version: version}}))});
  }
  return null;
}
const server = createServer(async (req, res) => {
  try {
    const result = await respond(req, new URL(req.url, 'http://127.0.0.1'));
    res.writeHead(result ? 200 : 404, {'Content-Type': 'application/json'});
    res.end(JSON.stringify(result || {error: {message: 'Fixture route not found'}}));
  } catch {
    res.writeHead(500, {'Content-Type': 'application/json'});
    res.end(JSON.stringify({error: {message: 'Invalid fixture request'}}));
  }
});
server.listen(18001, '127.0.0.1');
for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => server.close());
