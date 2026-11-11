"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, ArrowUpRight, Bot, ChartNoAxesCombined, ChevronRight, CircleHelp, Crosshair, Database, FlaskConical, LayoutDashboard, Menu, Swords, Users, X } from "lucide-react";
import { useState } from "react";

const links = [
  {href: "/", label: "Overview", icon: LayoutDashboard},
  {href: "/players", label: "Players", icon: Users},
  {href: "/matches", label: "Match analysis", icon: Swords},
  {href: "/agents", label: "Agent analytics", icon: Crosshair},
  {href: "/models", label: "Model insights", icon: FlaskConical},
  {href: "/coach", label: "AI Coach", icon: Bot},
];

export function Shell({children}: {children: React.ReactNode}) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const current = links.find(l => l.href === "/" ? path === "/" : path.startsWith(l.href));
  return <div className="app-shell">
    <a className="skip-link" href="#main">Skip to content</a>
    {open && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setOpen(false)} />}
    <aside className={`sidebar ${open ? "is-open" : ""}`}>
      <Link className="brand" href="/" onClick={() => setOpen(false)}><span className="brand-mark"><Activity size={25}/></span><span>impact<span className="brand-ai">ai</span><small>PERFORMANCE INTELLIGENCE</small></span></Link>
      <button className="mobile-close icon-button" onClick={() => setOpen(false)} aria-label="Close navigation"><X size={20}/></button>
      <div className="workspace"><span className="workspace-icon"><Crosshair size={18}/></span><div>VALORANT<small>Historical esports</small></div><span className="tiny-badge">PRO</span></div>
      <div className="nav-label">WORKSPACE</div>
      <nav aria-label="Main navigation">{links.map(({href, label, icon: Icon}) => <Link href={href} key={href} onClick={() => setOpen(false)} className={(href === "/" ? path === "/" : path.startsWith(href)) ? "nav-link active" : "nav-link"}><Icon size={18}/><span>{label}</span>{label === "AI Coach" && <span className="new-badge">BETA</span>}</Link>)}</nav>
      <div className="sidebar-bottom"><div className="research-card"><FlaskConical size={19}/><strong>Built on evidence.</strong><p>Real match data. Transparent models. Measurable performance.</p><Link href="/models">Explore the methodology <ArrowUpRight size={14}/></Link></div>
        <div className="dataset-status"><span className="status-dot"/> Historical dataset<small>JUL 2020 — JAN 2022</small></div>
        <Link className="method-link" href="/models"><CircleHelp size={15}/> Methodology & limitations</Link>
      </div>
    </aside>
    <div className="main-wrap"><header className="topbar"><div className="breadcrumb"><button className="mobile-toggle icon-button" onClick={() => setOpen(true)} aria-label="Open navigation"><Menu size={20}/></button><ChartNoAxesCombined size={16}/><span>Workspace</span><ChevronRight size={14}/><strong>{current?.label || "Analysis"}</strong></div><div className="topbar-right"><span className="historical-badge"><Database size={12}/> HISTORICAL ESPORTS</span><span className="avatar">IA</span></div></header>
      <main id="main">{children}</main><footer><span>ImpactAI <span className="muted">/</span> Historical performance intelligence</span><span>Independent research project · Not affiliated with Riot Games</span></footer>
    </div>
  </div>;
}
