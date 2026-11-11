"use client";
import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useQueryState } from "@/hooks/use-query-state";
import { useDebounced } from "@/hooks/use-debounced";
import { ArrowUpRight, Bot, Check, Search, Sparkles } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { Card, Heading, Note, Stat, State } from "@/components/ui";
import { date, number } from "@/lib/format";
import type { Coach, Performance, Player } from "@/types/api";

function CoachView() {
  const {params, update} = useQueryState();
  const player = params.get("player") || "";
  const game = params.get("game") || "";
  const query = params.get("q") || "";
  const settledQuery = useDebounced(query);
  const [report, setReport] = useState<Coach>();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const activeRequest = useRef<AbortController | null>(null);
  useEffect(() => () => activeRequest.current?.abort(), []);
  const search = useApi<Player[]>(`/api/players?q=${encodeURIComponent(settledQuery)}&page_size=8`, {keepPreviousData: true});
  const history = useApi<Performance[]>(player ? `/api/players/${player}/matches?page_size=100` : null);
  const gameId = game || String(history.data?.[0]?.game.id || "");
  function resetReport() {
    activeRequest.current?.abort();
    activeRequest.current = null;
    setPending(false); setError(""); setReport(undefined);
  }
  async function generate() {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    setPending(true); setError(""); setReport(undefined);
    try {
      const response = await fetch("/api/analysis/coach", {method: "POST", signal: controller.signal, headers: {"Content-Type": "application/json"}, body: JSON.stringify({player_id: Number(player), game_id: Number(gameId)})});
      const body = await response.json();
      if (activeRequest.current !== controller || controller.signal.aborted) return;
      if (!response.ok) throw new Error(body.error?.message || "Could not create report");
      setReport(body.data);
    } catch (e) {
      if (activeRequest.current === controller && !controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "Unexpected error");
      }
    } finally {
      if (activeRequest.current === controller && !controller.signal.aborted) setPending(false);
    }
  }
  return <><Heading eyebrow="TURN EVIDENCE INTO A REVIEW PLAN" title="Your performance, explained." description="A focused coaching report grounded in measured historical performances." action={<span className="pill"><Bot size={15}/> Evidence-based coach</span>}/>
    <div className="coach-layout"><Card title="Choose a performance" subtitle="Historical players and matches"><label className="field-label" htmlFor="coach-search">Find a player</label><label className="search-input"><Search size={17}/><input id="coach-search" value={query} placeholder="Search player name…" onChange={e => update({q: e.target.value}, true)}/></label><div className="coach-player-list"><State loading={search.loading && !search.data} error={search.error} retry={search.retry}/>{search.data?.map(p => <button key={p.id} className={player === String(p.id) ? "chosen" : ""} onClick={() => {resetReport(); update({player: p.id, game: ""});}}><strong>{p.name}</strong><small>{p.games} games</small>{player === String(p.id) && <Check size={15}/>}</button>)}</div><label className="field-label" htmlFor="coach-match">Select a match</label><State error={history.error} retry={history.retry}/><select id="coach-match" value={gameId} onChange={e => {resetReport(); update({game: e.target.value});}} disabled={!player || history.loading}>{!gameId && <option value="">Choose a player first</option>}{game && !history.data?.some(p => String(p.game.id) === game) && <option value={game}>Game {game}</option>}{history.data?.map(p => <option key={p.game.id} value={p.game.id}>{p.game.map} · {date(p.game.date)} · {p.agent}</option>)}</select><button className="button primary full-width" disabled={!player || !gameId || pending} onClick={generate}><Sparkles size={16}/>{pending ? "Preparing evidence…" : "Analyze performance"}</button><p className="small-caption">Works without an LLM subscription. All statistics and advice come from verified evidence templates.</p></Card>
    <div><State loading={pending} error={error} retry={generate}/>{!report && !pending && !error && <div className="coach-empty"><span className="coach-symbol"><Bot size={38}/></span><h2>A clearer next step.</h2><p>Select a player and a match to see strengths, review priorities, and the numbers supporting each observation.</p><div className="coach-guarantees"><span><Check size={14}/> Real source statistics</span><span><Check size={14}/> Historical peer comparisons</span><span><Check size={14}/> No invented gameplay events</span></div></div>}{report && <div className="coach-report"><div className="section-title"><div><div className="eyebrow">PERFORMANCE REVIEW</div><h2>{report.player_name} <span className="muted">/ {report.agent}</span></h2></div><Link className="button secondary" href={`/matches/${report.game_id}?player=${report.player_id}`}>Source match <ArrowUpRight size={14}/></Link></div><div className="stats-grid compact"><Stat label="Contextual index" value={number(report.score, 1)} sub="Heuristic composite"/><Stat label="Peer percentile" value={`${number(report.percentile)}th`} sub={`${number(report.peer_count)} historical references`}/></div><Note><strong>{report.assessment}</strong><br/>Compared with {report.history_count} earlier same-agent games. {report.history_game_ids.map(id => <Link key={id} href={`/matches/${id}?player=${report.player_id}`}>Game #{id}</Link>)}</Note>{report.evidence.map((e, i) => <Card className="evidence-card" key={e.id}><div className="evidence-heading"><span className={`evidence-number ${e.category === "strength" ? "teal" : e.category === "context" ? "muted" : "red"}`}>0{i+1}</span><div><div className="eyebrow">{e.category === "strength" ? "RELATIVE STRENGTH" : e.category === "context" ? "CONTEXT" : e.pattern === "repeated" ? "REPEATED REVIEW PRIORITY" : e.pattern === "isolated" ? "ISOLATED DIP" : "REVIEW PRIORITY · LIMITED HISTORY"}</div><h3>{e.component}</h3></div></div><p>{e.statement}</p><div className="suggestion">{e.suggestion}</div><Link className="evidence-source" href={`/matches/${e.source.game_id}?player=${e.source.player_id}`}>Evidence: game #{e.source.game_id} · {e.source.metric} <ArrowUpRight size={12}/></Link></Card>)}<Note>{report.limitations.join(" ")}</Note></div>}</div></div>
  </>;
}
function SelectedCoach() {
  const {params} = useQueryState();
  // Remount the report/request lifecycle when a link or browser history changes performance.
  return <CoachView key={`${params.get("player")}:${params.get("game")}`}/>;
}
export default function CoachPage() {return <Suspense fallback={<State loading/>}><SelectedCoach/></Suspense>;}
