"use client";
import Link from "next/link";
import { ArrowRight, ArrowUpRight, Info, LoaderCircle, RefreshCw } from "lucide-react";
import type { Game, Performance } from "@/types/api";
import { date, initials, number } from "@/lib/format";

export function Heading({eyebrow, title, description, action}: {eyebrow: string; title: string; description: string; action?: React.ReactNode}) {
  return <div className="page-heading"><div><div className="eyebrow"><span/>{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}
export function Card({title, subtitle, action, children, className = ""}: {title?: string; subtitle?: string; action?: React.ReactNode; children: React.ReactNode; className?: string}) {
  return <section className={`card ${className}`}>{title && <div className="card-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>}{children}</section>;
}
export function Stat({label, value, sub, icon, accent}: {label: string; value: string; sub: string; icon?: React.ReactNode; accent?: boolean}) {
  return <div className={`stat-card ${accent ? "stat-accent" : ""}`}><div className="stat-top"><span>{label}</span>{icon}</div><strong>{value}</strong><small>{sub}</small></div>;
}
export function State({loading, error, retry, empty}: {loading?: boolean; error?: string; retry?: () => void; empty?: boolean}) {
  if (error) return <div className="state-panel" role="alert"><Info size={24}/><h2>Data is unavailable</h2><p>{error}</p>{retry && <button className="button secondary" onClick={retry}><RefreshCw size={14}/> Retry</button>}</div>;
  if (loading) return <div className="state-panel" role="status"><LoaderCircle className="spin" size={24}/><p>Loading historical analysis…</p></div>;
  if (empty) return <div className="state-panel"><h2>No results found</h2><p>Try a different player, map, or agent.</p></div>;
  return null;
}
export function Note({children}: {children: React.ReactNode}) { return <div className="note"><Info size={16}/><div>{children}</div></div>; }
export function Pagination({page, total, size = 20, setPage, disabled = false}: {page: number; total: number; size?: number; disabled?: boolean; setPage: (page: number) => void}) {
  return <div className="pagination"><span>{number(total)} results · Page {page} of {Math.max(1, Math.ceil(total / size))}</span><div><button className="button secondary" disabled={disabled || page === 1} onClick={() => setPage(page - 1)}>Previous</button><button className="button secondary" disabled={disabled || page * size >= total} onClick={() => setPage(page + 1)}>Next <ArrowRight size={14}/></button></div></div>;
}
export function MatchTable({games}: {games: Game[]}) {
  return <div className="table-scroll"><table><thead><tr><th>Match</th><th>Map</th><th>Score</th><th>Date</th><th>Partition</th><th aria-label="View match"/></tr></thead><tbody>{games.map(g => <tr key={g.id}><td><Link className="match-name" href={`/matches/${g.id}`}><span>{g.team1_name}</span><span className="versus">vs</span><span>{g.team2_name}</span></Link><small className="table-sub">{g.event}</small></td><td><span className={`map-chip map-${g.map.toLowerCase()}`}>{g.map}</span></td><td className="mono"><strong>{g.team1_score}</strong><span className="muted"> : </span>{g.team2_score}</td><td className="muted nowrap">{date(g.date)}</td><td><span className="partition">{g.split}</span></td><td><Link className="icon-button" aria-label={`Analyze ${g.team1_name} versus ${g.team2_name}`} href={`/matches/${g.id}`}><ArrowUpRight size={17}/></Link></td></tr>)}</tbody></table></div>;
}
export function Scoreboard({players, selected}: {players: Performance[]; selected?: number}) {
  return <div className="table-scroll"><table><thead><tr><th>Player / Agent</th><th>K / D / A</th><th>ACS</th><th>ADR</th><th>FK / FD</th><th>Index <span title="Equal-weight heuristic contextual score">ⓘ</span></th><th>Peer percentile</th></tr></thead><tbody>{players.map(p => <tr className={p.player_id === selected ? "selected-row" : ""} key={p.player_id}><td><Link className="identity" href={`/players/${p.player_id}`}><span className={`agent-avatar agent-${p.agent}`}>{initials(p.agent)}</span><span><strong>{p.player_name}</strong><small>{p.agent}</small></span></Link></td><td className="mono nowrap">{p.kills} <span className="muted">/</span> {p.deaths} <span className="muted">/</span> {p.assists}</td><td>{number(p.acs)}</td><td>{number(p.adr)}</td><td className="mono">{p.first_kills} <span className="muted">/</span> {p.first_deaths}</td><td><span className={`score-chip ${p.impact.score >= 50 ? "positive" : "negative"}`}>{number(p.impact.score, 1)}</span></td><td><div className="percentile-cell"><div><span style={{width: `${p.impact.percentile}%`}}/></div><span>{number(p.impact.percentile)}th</span></div></td></tr>)}</tbody></table></div>;
}
