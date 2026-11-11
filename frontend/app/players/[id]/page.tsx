"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Suspense } from "react";
import { useQueryState } from "@/hooks/use-query-state";
import { ArrowUpRight, Bot } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { Card, Heading, Pagination, Stat, State } from "@/components/ui";
import { ImpactBreakdown } from "@/components/impact";
import { TrendChart } from "@/components/charts";
import { date, number } from "@/lib/format";
import type { Performance, Player } from "@/types/api";

function PlayerProfile() {
  const {id} = useParams<{id: string}>();
  const {params, update, page} = useQueryState();
  const selected = params.get("game") || "";
  const player = useApi<Player>(`/api/players/${id}`);
  const matches = useApi<Performance[]>(`/api/players/${id}/matches?page=${page}`);
  const impact = useApi<Performance>(`/api/players/${id}/impact${selected ? `?game_id=${selected}` : ""}`);
  const trends = useApi<{game_id: number; date: string; score: number}[]>(`/api/players/${id}/trends`);
  const p = player.data;
  return <><Heading eyebrow="HISTORICAL PLAYER PROFILE" title={p?.name || "Player profile"} description={p ? `${p.agents?.map(a => a.name).join(" · ")} · Professional match archive` : "Loading player context"} action={impact.data && <Link className="button primary" href={`/coach?player=${id}&game=${impact.data.game.id}`}><Bot size={16}/> Open AI Coach</Link>}/><State loading={player.loading} error={player.error} retry={player.retry}/>{p && <>
    <div className="stats-grid"><Stat label="Recorded games" value={number(p.games)} sub="Across historical teams"/><Stat label="Overall K/D" value={number(p.kills/Math.max(p.deaths, 1), 2)} sub="Total kills / total deaths"/><Stat label="Average ACS" value={number(p.acs, 1)} sub="End-of-game statistic"/><Stat label="Average ADR" value={number(p.adr, 1)} sub="Damage per round" accent/></div>
    <Card title="Performance over time" subtitle="Latest 200 recorded games · contextual index"><State loading={trends.loading} error={trends.error} retry={trends.retry}/>{trends.data && <TrendChart data={trends.data.map(t => ({label: date(t.date), value: t.score}))} label="Context index"/>}</Card>
    <div className="section-title"><div><h2>Inspect a performance</h2><p>Select a match below to update the explanation.</p></div>{impact.data && <Link className="button secondary" href={`/matches/${impact.data.game.id}?player=${id}`}>Full match <ArrowUpRight size={15}/></Link>}</div>
    {impact.data && <div className="selected-performance"><span className="map-chip">{impact.data.game.map}</span><strong>{impact.data.team}</strong><span className="muted">{date(impact.data.game.date)} · {impact.data.agent}</span><span className="partition">{impact.data.game.split}</span><span className="mono">{impact.data.kills} / {impact.data.deaths} / {impact.data.assists} KDA</span><span className="muted">FK {impact.data.first_kills} · FD {impact.data.first_deaths}</span></div>}
    <State loading={impact.loading} error={impact.error} retry={impact.retry}/>{impact.data && <ImpactBreakdown impact={impact.data.impact}/>}
    <Card title="Match history" subtitle="Recorded observations, with team and agent context"><State loading={matches.loading} error={matches.error} retry={matches.retry}/>{matches.data && <><div className="table-scroll"><table><thead><tr><th>Date / Team</th><th>Map / Agent</th><th>Result</th><th>ACS</th><th>Index</th><th/></tr></thead><tbody>{matches.data.map(m => <tr key={m.game.id} className={impact.data?.game.id === m.game.id ? "selected-row" : ""}><td><strong>{date(m.game.date)}</strong><small className="table-sub">{m.team}</small></td><td>{m.game.map}<small className="table-sub capitalize">{m.agent}</small></td><td><span className={m.team_won ? "teal" : "red"}>{m.team_won === null ? "Unreliable outcome" : m.team_won ? "WIN" : "LOSS"}</span></td><td>{number(m.acs)}</td><td><span className="score-chip">{number(m.impact.score, 1)}</span></td><td><button className="button secondary" aria-label={`Analyze game ${m.game.id}`} onClick={() => update({game: m.game.id})}>Analyze <ArrowUpRight size={14}/></button></td></tr>)}</tbody></table></div><Pagination page={page} total={matches.meta?.total || 0} setPage={page => update({page})}/></>}</Card>
  </>}</>;
}

export default function PlayerPage() {return <Suspense fallback={<State loading/>}><PlayerProfile/></Suspense>;}
