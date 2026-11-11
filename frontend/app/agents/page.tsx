"use client";
import Link from "next/link";
import { Suspense } from "react";
import { useQueryState } from "@/hooks/use-query-state";
import { ArrowUpRight } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { Card, Heading, Note, State } from "@/components/ui";
import { BarPlot } from "@/components/charts";
import { initials, number } from "@/lib/format";
import type { Agent, Overview } from "@/types/api";

function AgentView() {
  const {params, update} = useQueryState();
  const map = params.get("map") || "";
  const requestedAgent = params.get("agent") || "jett";
  const api = useApi<Agent[]>(`/api/agents?map=${encodeURIComponent(map)}`);
  const selected = api.data?.some(a => a.name === requestedAgent) ? requestedAgent : api.data?.[0]?.name || "";
  const overview = useApi<Overview>("/api/overview");
  return <><Heading eyebrow="ROLE & CONTEXT" title="Agent analytics" description="Compare selections, opening engagements, and observed damage across historical agents." action={<select aria-label="Filter agents by map" value={map} onChange={e => update({map: e.target.value})}><option value="">All maps</option>{overview.data?.maps.map(m => <option key={m}>{m}</option>)}</select>}/><State loading={api.loading} error={api.error} retry={api.retry}/>{api.data && <><div className="grid-two"><Card title="Agent selections" subtitle="Player-game appearances"><BarPlot data={api.data.slice(0, 10).map(a => ({label: a.name, value: a.selections}))}/></Card><Card title="Damage per round" subtitle="Observed means · affected by role and selection"><BarPlot data={api.data.slice(0, 10).map(a => ({label: a.name, value: a.adr}))} label="ADR"/></Card></div><Card title="Historical agent comparison" subtitle="Contextual scores use agent-aware training references"><div className="table-scroll"><table><thead><tr><th>Agent</th><th>Selections</th><th>ADR</th><th>FK / round</th><th>FD / round</th><th>Avg. index</th><th>Observed range</th><th/></tr></thead><tbody>{api.data.map(a => <tr key={a.name}><td><span className="identity"><span className={`agent-avatar agent-${a.name}`}>{initials(a.name)}</span><strong className="capitalize">{a.name}</strong></span></td><td>{number(a.selections)}</td><td>{number(a.adr, 1)}</td><td>{number(a.fkpr, 3)}</td><td>{number(a.fdpr, 3)}</td><td>{number(a.score, 1)}</td><td className="muted">{number(a.min_score, 1)} – {number(a.max_score, 1)}</td><td><Link className="icon-button" aria-label={`Compare ${a.name} players`} href={`/players?${new URLSearchParams({agent: a.name, ...(map ? {map} : {})})}`}><ArrowUpRight size={16}/></Link></td></tr>)}</tbody></table></div></Card></>}
    {api.data && <Card title="Contextual performance distribution" subtitle="Player-game heuristic scores for the selected agent and map" action={<select aria-label="Distribution agent" value={selected} onChange={e => update({agent: e.target.value})}>{api.data.map(a => <option key={a.name}>{a.name}</option>)}</select>}><BarPlot label="Player performances" data={(api.data.find(a => a.name === selected)?.distribution || []).map(b => ({label: b.name, value: b.count}))}/></Card>}
    <Note>Agent selection and performance are associated with team strategy, players, maps, and historical patches. These observations do not establish causal agent strength or describe the current meta.</Note>
  </>;
}

export default function AgentsPage() {return <Suspense fallback={<State loading/>}><AgentView/></Suspense>;}
