"use client";
import Link from "next/link";
import { ArrowRight, ArrowUpRight, Crosshair, Database, FlaskConical, Layers3, Swords, Users } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { BarPlot, TrendChart } from "@/components/charts";
import { Card, Heading, MatchTable, Note, Stat, State } from "@/components/ui";
import { number } from "@/lib/format";
import type { Game, Overview } from "@/types/api";

export default function OverviewPage() {
  const {data, meta, loading, error, retry} = useApi<Overview>("/api/overview");
  const recent = useApi<Game[]>("/api/matches?page_size=5");
  return <><Heading eyebrow="THE BIGGER PICTURE" title="Performance, in context." description="Look beyond the scoreboard. Understand what the numbers can tell you." action={<Link href="/players" className="button primary">Explore players <ArrowUpRight size={16}/></Link>}/>
    <div className="coverage-strip"><div><span className="status-dot"/><strong>Historical professional esports</strong><span className="muted">Jul 2020 — Jan 2022</span></div><span className="coverage-tag">TRAIN / VALIDATION / TEST</span></div>
    <State loading={loading} error={error} retry={retry}/>{data && <>
      <div className="stats-grid"><Stat label="Games analyzed" value={number(data.games)} sub={`${number(data.series)} professional series`} icon={<Swords size={18}/>}/><Stat label="Players in the dataset" value={number(data.players)} sub={`${number(data.player_records)} player performances`} icon={<Users size={18}/>}/><Stat label="Rounds observed" value={number(data.rounds)} sub="Verified score progression" icon={<Layers3 size={18}/>}/><Stat label="Model pipeline" value="Trained" sub={meta?.model_version || "Versioned artifacts"} icon={<FlaskConical size={18}/>} accent/></div>
      <div className="overview-main"><Card title="Contextual performance distribution" subtitle="Player-game observations · heuristic index" action={<span className="pill">All games</span>}><div className="distribution-intro"><span className="chart-number">Beyond K/D<span className="red-dot">.</span></span><p>Six standardized components.<br/>Training-only historical peer references.</p></div><BarPlot data={data.distribution.map(d => ({label: d.name, value: d.count}))} label="Player performances"/><div className="chart-foot"><span>Below peer reference</span><span>Contextual index →</span><span>Above peer reference</span></div></Card>
      <Card className="agent-card" title="Most selected agents" subtitle="Historical professional picks" action={<Crosshair size={17}/>}><div className="agent-list">{data.agents.slice(0, 6).map((agent, i) => <Link key={agent.name} href={`/players?agent=${agent.name}`} className="agent-item"><span className="rank-number">0{i+1}</span><span className={`agent-avatar agent-${agent.name}`}>{agent.name.slice(0, 2).toUpperCase()}</span><div className="agent-info"><div><strong>{agent.name}</strong><span>{number(agent.count)}</span></div><div className="agent-track"><span style={{width: `${agent.count / data.agents[0].count * 100}%`}}/></div></div></Link>)}</div><Link href="/agents" className="text-link">View agent analytics <ArrowRight size={15}/></Link></Card></div>
      <div className="grid-two"><Card title="A historical view" subtitle="Games recorded per month" action={<span className="chart-legend"><i/> Games</span>}><TrendChart data={data.trends.map(d => ({label: d.month, value: d.games}))}/></Card><Card className="model-story"><div className="eyebrow"><FlaskConical size={15}/> EXPLAINABLE BY DESIGN</div><h2>Every estimate<br/>has a paper trail.</h2><p>Compare trained models with simple baselines. Inspect calibration, feature contributions, and held-out results.</p><div className="model-mini"><div><span>Round outcome</span><strong>{data.round_model.replaceAll("_", " ")}</strong></div><div><span>Player association</span><strong>{data.player_model}</strong></div></div><Link className="text-link" href="/models">Inside the models <ArrowUpRight size={16}/></Link></Card></div>
      <Card title="Latest in the archive" subtitle="Explore the games behind the data" action={<Link className="text-link" href="/matches">All matches <ArrowRight size={15}/></Link>}><State loading={recent.loading} error={recent.error} retry={recent.retry}/>{recent.data && <MatchTable games={recent.data}/>}</Card>
      <Note><strong>Context, with clear limits.</strong> This is historical esports data, not modern ranked gameplay. The index is a heuristic peer comparison. It does not measure causal contribution, rank, or kill-by-kill win probability added. <Link href="/models">Read the methodology <Database size={12}/></Link></Note>
    </>}
  </>;
}
