"use client";
import type { Impact, PeerRanges } from "@/types/api";
import { useApi } from "@/hooks/use-api";
import { Card, Note, State } from "@/components/ui";
import { BarPlot } from "@/components/charts";
import { featureName, number, percent } from "@/lib/format";

export function ImpactBreakdown({impact}: {impact: Impact}) {
  const ranges = useApi<PeerRanges>(`/api/players/${impact.player_id}/peer-ranges?game_id=${impact.game_id}`);
  return <><div className="grid-two"><Card title="Contextual performance" subtitle="Equal-weight heuristic · training-only reference"><div className="impact-hero"><div className="score-orbit" style={{"--score": `${impact.percentile * 3.6}deg`} as React.CSSProperties}><div><strong>{number(impact.score, 1)}</strong><span>CONTEXT INDEX</span></div></div><div><span className="percentile-large">{number(impact.percentile)}<small>th percentile</small></span><p>{number(impact.peer_count)} historical peer performances</p><span className="pill">{impact.peer_level.replaceAll("_", " + ")}</span></div></div><div className="components-list">{Object.entries(impact.components).map(([name, c]) => <div key={name}><span>{name}</span><div className="component-meter"><span style={{left: `${Math.min(50, 50+c.z/3*50)}%`, width: `${Math.abs(c.z)/3*50}%`, background: c.z >= 0 ? "#6ebdb4" : "#e66679"}}/><i/></div><strong className={c.z >= 0 ? "teal" : "red"}>{c.z >= 0 ? "+" : ""}{number(c.z, 2)}<small> σ</small></strong></div>)}</div><p className="small-caption">Zero represents a smoothed agent/map/patch/length reference. Sparse groups fall back to broader training peers.</p></Card>
    <Card title="What shaped the model estimate" subtitle="Local feature contributions · uncalibrated log-odds"><BarPlot horizontal data={impact.explanation.top.slice(0, 7).map(f => ({label: featureName(f.feature), value: f.contribution}))} label="Log-odds contribution"/><div className="association-footer"><span>Retrospective team-win association</span><strong>{percent(impact.association_probability)}</strong></div><p className="small-caption">{impact.explanation.method}. Positive values increase the model output relative to its baseline. This probability is separate from the index.</p></Card></div>
    <Card title="Observed performance and peer references" subtitle={`${number(impact.peer_count)} index peers · ${impact.peer_level.replaceAll("_", " + ")} · ${number(impact.exact_context_count)} exact-context records`}>
      <State loading={ranges.loading} error={ranges.error} retry={ranges.retry}/>
      <div className="table-scroll"><table><thead><tr><th>Component</th><th>Observed</th><th>Peer mean</th>{ranges.data?.available && <><th>Research median</th><th>Expected {percent(ranges.data.nominal_coverage)} range</th><th>Measured coverage</th></>}</tr></thead>
        <tbody>{Object.entries(impact.components).map(([name, c]) => {
          const interval = ranges.data?.available ? ranges.data.components[name] : undefined;
          return <tr key={name}><td>{name}</td><td>{number(c.value, 3)}</td><td>{number(c.peer_mean, 3)}</td>{interval && <><td>{number(interval.median, 3)}</td><td>{number(interval.low, 3)} – {number(interval.high, 3)}</td><td>{percent(interval.measured_coverage)}</td></>}</tr>;
        })}</tbody></table></div>
      <p className="small-caption">Component units: combat and openings are net events per round; damage is ADR; assists and objectives are per round; clutches are wins per round. Peer means are smoothed training references.</p>
      {ranges.data && <Note>{ranges.data.available ? <><strong>Experimental · {ranges.data.research_run}.</strong> Measured coverage describes the saved development sample. Index peer counts above do not describe the learned range model. </> : ranges.data.reason} {ranges.data.available && ranges.data.limitation}</Note>}
    </Card>
    <Note>The index combines combat, openings, damage, assists, objectives, and clutch counts equally. Components overlap and opportunities are unobserved. The score is a descriptive heuristic; SHAP explains model behavior, not causal contributions.</Note></>;
}
