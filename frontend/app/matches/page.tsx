"use client";
import { Suspense } from "react";
import { useQueryState } from "@/hooks/use-query-state";
import { useDebounced } from "@/hooks/use-debounced";
import { Search } from "lucide-react";
import { useApi } from "@/hooks/use-api";
import { Card, Heading, MatchTable, Pagination, State } from "@/components/ui";
import type { Game, Overview } from "@/types/api";

function MatchSearch() {
  const {params: query, update, page} = useQueryState();
  const map = query.get("map") || "";
  const event = query.get("event") || "";
  const from = query.get("date_from") || "";
  const to = query.get("date_to") || "";
  const settledEvent = useDebounced(event);
  const params = new URLSearchParams({page: String(page), map, event: settledEvent});
  if (from) params.set("date_from", from);
  if (to) params.set("date_to", to);
  const api = useApi<Game[]>(`/api/matches?${params}`, {keepPreviousData: true});
  const overview = useApi<Overview>("/api/overview");
  return <><Heading eyebrow="THE MATCH ARCHIVE" title="Every game has a story." description="Explore scoreboards, contextual performances, and pre-round estimates."/>
    <Card><div className="filters"><label className="search-input"><Search size={17}/><input aria-label="Search events" placeholder="Search an event…" value={event} onChange={e => {update({event: e.target.value, page: 1}, true);}}/></label><select aria-label="Filter by map" value={map} onChange={e => {update({map: e.target.value, page: 1});}}><option value="">All maps</option>{overview.data?.maps.map(m => <option key={m}>{m}</option>)}</select><label className="date-filter">From<input type="date" aria-label="From date" value={from} onChange={e => {update({date_from: e.target.value, page: 1});}}/></label><label className="date-filter">To<input type="date" aria-label="To date" value={to} onChange={e => {update({date_to: e.target.value, page: 1});}}/></label></div><div role="status" className="refresh-status">{api.refreshing || event !== settledEvent ? "Updating results…" : ""}</div><State loading={api.loading && !api.data} error={api.error} retry={api.retry} empty={api.data?.length === 0}/>{api.data && <><MatchTable games={api.data}/><Pagination page={page} total={api.meta?.total || 0} setPage={page => update({page})} disabled={api.loading}/></>}</Card>
  </>;
}

export default function MatchesPage() {return <Suspense fallback={<State loading/>}><MatchSearch/></Suspense>;}
