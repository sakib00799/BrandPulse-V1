"use client";

import { useEffect, useState } from "react";
import { API_URL, apiFetch } from "@/lib/api";
import { Badge, PageHeader, Panel } from "@/components/shell";

type TargetMetrics = { test_metrics: { macro_f1: number; weighted_f1: number; accuracy: number }; review_threshold: { threshold: number }; confusion_matrix_url: string };
type Performance = { selected_model: string; targets: Record<string, TargetMetrics>; latency: { median_ms: number; p95_ms: number; model_size_bytes: number }; subgroups: Record<string, Record<string, Array<{ group: string; support: number; accuracy: number; macro_f1_present_classes: number }>>> };

export default function PerformancePage() {
  const [data, setData] = useState<Performance | null>(null);
  useEffect(() => { apiFetch<Performance>("/analytics/performance").then(setData); }, []);
  return (
    <>
      <PageHeader eyebrow="Evidence, not hype" title="Model performance" description="Fixed grouped-test metrics, confusion matrices, subgroup checks and measured runtime for the selected serving model." />
      {!data ? <Panel><p className="animate-pulse text-sm text-ink/45">Loading evaluation artifacts…</p></Panel> : <div className="space-y-5">
        <Panel className="flex flex-wrap items-center justify-between gap-4 bg-ink text-white"><div><div className="text-[10px] uppercase tracking-[.25em] text-lime">Selected model</div><div className="mt-2 text-2xl font-semibold">{data.selected_model}</div></div><div className="flex gap-6 text-sm"><div><span className="text-white/45">P95</span><strong className="ml-2">{data.latency.p95_ms.toFixed(3)} ms</strong></div><div><span className="text-white/45">Size</span><strong className="ml-2">{(data.latency.model_size_bytes / 1e6).toFixed(2)} MB</strong></div></div></Panel>
        <div className="grid gap-4 md:grid-cols-3">{Object.entries(data.targets).map(([target, value]) => <Panel key={target}><div className="flex justify-between"><h2 className="capitalize font-semibold">{target}</h2><Badge tone={target === "category" ? "danger" : "neutral"}>threshold {value.review_threshold.threshold.toFixed(2)}</Badge></div><div className="mt-6 text-4xl font-semibold">{value.test_metrics.macro_f1.toFixed(3)}</div><div className="text-xs text-ink/40">macro-F1</div><div className="mt-5 grid grid-cols-2 gap-2 text-xs"><div className="rounded-lg bg-canvas p-3">Weighted-F1<br/><strong className="text-base">{value.test_metrics.weighted_f1.toFixed(3)}</strong></div><div className="rounded-lg bg-canvas p-3">Accuracy<br/><strong className="text-base">{value.test_metrics.accuracy.toFixed(3)}</strong></div></div></Panel>)}</div>
        <div className="grid gap-5 xl:grid-cols-3">{Object.entries(data.targets).map(([target, value]) => <Panel key={target}><h3 className="mb-3 capitalize font-semibold">{target} confusion matrix</h3><img src={`${API_URL}${value.confusion_matrix_url}`} alt={`${target} confusion matrix`} className="w-full rounded-xl border border-ink/5" /></Panel>)}</div>
        <Panel><h2 className="text-lg font-semibold">Company subgroup snapshot</h2><p className="mt-1 text-xs text-ink/45">Macro-F1 uses classes present in each subgroup; small samples are uncertain.</p><div className="mt-5 overflow-x-auto"><table className="w-full min-w-[700px] text-sm"><thead className="text-left text-[10px] uppercase tracking-widest text-ink/40"><tr><th className="py-2">Target</th><th>Company</th><th>Support</th><th>Accuracy</th><th>Macro-F1</th></tr></thead><tbody className="divide-y divide-ink/5">{Object.entries(data.subgroups).flatMap(([target, dimensions]) => dimensions.company.map((row) => <tr key={`${target}-${row.group}`}><td className="py-2 capitalize">{target}</td><td>{row.group}</td><td>{row.support}</td><td>{row.accuracy.toFixed(3)}</td><td>{row.macro_f1_present_classes.toFixed(3)}</td></tr>))}</tbody></table></div></Panel>
      </div>}
    </>
  );
}
