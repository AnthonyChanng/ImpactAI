"use client";
import { Suspense, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Bot, Crosshair } from "lucide-react";
import { useQueryState } from "@/hooks/use-query-state";
import { useApi } from "@/hooks/use-api";
import { Card, Heading, Note, Scoreboard, State } from "@/components/ui";
import { ImpactBreakdown } from "@/components/impact";
import { TrendChart } from "@/components/charts";
import { date, percent } from "@/lib/format";
import type { Game, Performance, Round } from "@/types/api";

function Analysis() {
  const {id} = useParams<{id: string}>();
  const {params, update} = useQueryState();
  const player = params.get("player") || "";
  const [roundNumber, setRoundNumber] = useState(1);
  const game = useApi<Game>(`/api/matches/${id}`);
  const players = useApi<Performance[]>(`/api/matches/${id}/players`);
  const rounds = useApi<Round[]>(`/api/matches/${id}/rounds`);
  const g = game.data;
  const performance = players.data?.find(p => String(p.player_id) === player) || players.data?.[0];
  const round = rounds.data?.find(r => r.number === roundNumber);
  return <><Heading eyebrow="INDIVIDUAL MATCH ANALYSIS" title={g ? `${g.map} / Match analysis` : "Match analysis"} description={g ? `${g.event} · ${date(g.date)}` : "Loading historical game"}/><State loading={game.loading} error={game.error} retry={game.retry}/>{g && <>
    <div className="match-banner"><div><span className="eyebrow">SOURCE TEAM 1</span><h2>{g.team1_name}</h2></div><div className="final-score"><strong>{g.team1_score}</strong><span>:</span><strong>{g.team2_score}</strong><small>{g.outcome_reliable ? "FINAL SCORE" : "UNRELIABLE FINAL OUTCOME"}</small></div><div><span className="eyebrow">SOURCE TEAM 2</span><h2>{g.team2_name}</h2></div></div>
    <div className="match-context"><span><Crosshair size={14}/>{g.map}</span><span>{g.patch || "Patch unavailable"}</span><span>{g.num_rounds} rounds</span><span className="partition">{g.split}</span></div>
    <Note>{g.split === "test" ? "This series belongs to the held-out test partition." : `This series belongs to the ${g.split} partition; displayed model estimates are not held-out test predictions.`} Team numbers identify the source team ordering, not attacker/defender.</Note>
    <Card title="Round-by-round" subtitle="Line: pre-round probability for team 1 · gaps: unavailable estimates · tiles: observed round winner"><State loading={rounds.loading} error={rounds.error} retry={rounds.retry}/>{rounds.data && <><TrendChart data={rounds.data.map(r => ({label: `R${r.number}`, value: r.probability === null ? null : r.probability * 100}))} label="Team 1 pre-round probability (%)" percent/><div className="round-timeline">{rounds.data.map(r => <button key={r.number} aria-label={`Round ${r.number}: team ${r.team1_won ? 1 : 2} won`} aria-pressed={r.number === roundNumber} onClick={() => setRoundNumber(r.number)} className={`${r.team1_won ? "round-team1" : "round-team2"} ${r.number === roundNumber ? "round-selected" : ""}`}>{r.number}</button>)}</div><div className="round-legend"><span><i className="team1-dot"/>{g.team1_name}</span><span><i className="team2-dot"/>{g.team2_name}</span></div>{round && <div className="round-detail"><strong>ROUND {round.number}</strong><span>Pre-score <b>{round.team1_pre_score} : {round.team2_pre_score}</b></span><span>Team 1 side <b>{round.team1_side || "Unknown (overtime)"}</b></span><span>Estimated win probability <b>{round.probability === null ? "Unavailable" : percent(round.probability)}</b></span><span>Recorded outcome <b>Team {round.team1_won ? 1 : 2} · {round.win_type}</b></span></div>}</>}</Card>
    <State loading={players.loading} error={players.error} retry={players.retry}/>{players.data && [1, 2].map(side => <Card key={side} title={side === 1 ? g.team1_name : g.team2_name} subtitle={players.data?.filter(p => p.team_side === side).map(p => p.agent).join(" · ")} action={<span className="pill">TEAM {side}</span>}><Scoreboard players={players.data!.filter(p => p.team_side === side)} selected={performance?.player_id}/></Card>)}
    <div className="section-title"><div><h2>Player breakdown</h2><p>Choose a player to inspect their contextual estimate.</p></div><div className="inline-controls"><select aria-label="Select player for breakdown" value={performance?.player_id || ""} onChange={e => update({player: e.target.value})}>{players.data?.map(p => <option key={p.player_id} value={p.player_id}>{p.player_name} · {p.agent}</option>)}</select>{performance && <Link className="button primary" href={`/coach?player=${performance.player_id}&game=${id}`}><Bot size={16}/> Coach</Link>}</div></div>{performance && <ImpactBreakdown impact={performance.impact}/>}
  </>}</>;
}
export default function MatchPage() {return <Suspense fallback={<State loading/>}><Analysis/></Suspense>;}
