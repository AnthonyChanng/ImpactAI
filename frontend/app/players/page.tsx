"use client";
import { Suspense } from "react";
import Link from "next/link";
import { useQueryState } from "@/hooks/use-query-state";
import { useDebounced } from "@/hooks/use-debounced";
import { ArrowUpRight, Search, Users } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { Card, Heading, Pagination, State } from "@/components/ui";
import { initials, number } from "@/lib/format";
import type { Overview, Player } from "@/types/api";

function PlayerSearch() {
  const {params, update, page} = useQueryState();
  const query = params.get("q") || "";
  const agent = params.get("agent") || "";
  const map = params.get("map") || "";
  const settledQuery = useDebounced(query);
  const overview = useApi<Overview>("/api/overview");
  const api = useApi<Player[]>(`/api/players?${new URLSearchParams({q: settledQuery, agent, map, page: String(page)})}`, {keepPreviousData: true});
  return <><Heading eyebrow="THE PEOPLE BEHIND THE PLAYS" title="Player explorer" description="Find a historical player and explore their performances in context."/>
    <Card><div className="filters"><label className="search-input"><Search size={18}/><input aria-label="Search players" placeholder="Search a player name…" value={query} onChange={e => {update({q: e.target.value, page: 1}, true);}}/></label><select aria-label="Filter by agent" value={agent} onChange={e => {update({agent: e.target.value, page: 1});}}><option value="">All agents</option>{overview.data?.agents.map(a => <option key={a.name}>{a.name}</option>)}</select><select aria-label="Filter by map" value={map} onChange={e => {update({map: e.target.value, page: 1});}}><option value="">All maps</option>{overview.data?.maps.map(m => <option key={m}>{m}</option>)}</select></div>
    <div role="status" className="refresh-status">{api.refreshing || query !== settledQuery ? "Updating results…" : ""}</div><State loading={api.loading && !api.data} error={api.error} retry={api.retry} empty={api.data?.length === 0}/>{api.data && api.data.length > 0 && <><div className="table-scroll"><table><thead><tr><th>Player</th><th>Games</th><th>K/D</th><th>Avg. ACS</th><th>Avg. ADR</th><th>Avg. index</th><th/></tr></thead><tbody>{api.data.map(p => <tr key={p.id}><td><Link className="identity" href={`/players/${p.id}`}><span className="player-avatar">{initials(p.name)}</span><span><strong>{p.name}</strong><small>Historical professional player</small></span></Link></td><td>{p.games}</td><td>{number(p.kills / Math.max(p.deaths, 1), 2)}</td><td>{number(p.acs, 1)}</td><td>{number(p.adr, 1)}</td><td><span className="score-chip positive">{number(p.score, 1)}</span></td><td><Link className="icon-button" aria-label={`View ${p.name}`} href={`/players/${p.id}`}><ArrowUpRight size={17}/></Link></td></tr>)}</tbody></table></div><Pagination page={page} total={api.meta?.total || 0} setPage={page => update({page})} disabled={api.loading}/></>}</Card><div className="small-caption"><Users size={14}/> Sorted by available historical game count. Averages reflect the selected filters.</div>
  </>;
}
export default function PlayersPage() {return <Suspense fallback={<State loading/>}><PlayerSearch/></Suspense>;}
