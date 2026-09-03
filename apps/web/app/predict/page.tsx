"use client";

import { FormEvent, useState } from "react";
import { apiFetch, Prediction } from "@/lib/api";
import { Badge, PageHeader, Panel } from "@/components/shell";

export default function PredictPage() {
  const [text, setText] = useState("Payment korechi kintu internet active hoy nai");
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); setLoading(true);
    try { setResult(await apiFetch<Prediction>("/predict", { method: "POST", body: JSON.stringify({ text }) })); }
    finally { setLoading(false); }
  };
  return (
    <>
      <PageHeader eyebrow="Try the model" title="Live prediction" description="Paste a Bangla, Banglish, English or code-mixed comment. Confidence is shown as measured model probability, with explicit review reasons." />
      <div className="grid gap-5 xl:grid-cols-[1fr_.85fr]">
        <Panel>
          <form onSubmit={submit}>
            <label className="text-xs font-bold uppercase tracking-widest text-ink/50">Customer comment</label>
            <textarea className="field mt-3 min-h-52 resize-y leading-6" value={text} onChange={(event) => setText(event.target.value)} maxLength={5000} />
            <div className="mt-4 flex items-center justify-between"><span className="text-xs text-ink/35">{text.length} / 5,000 characters</span><button className="button-primary" disabled={loading}>{loading ? "Analyzing…" : "Analyze feedback"}</button></div>
          </form>
        </Panel>
        <Panel className={!result ? "flex min-h-80 items-center justify-center" : ""}>
          {!result ? <p className="max-w-xs text-center text-sm leading-6 text-ink/40">Your three-head prediction and review status will appear here.</p> : (
            <div>
              <div className="flex items-center justify-between"><h2 className="text-lg font-semibold">Prediction</h2><Badge tone={result.needs_human_review ? "warn" : "good"}>{result.needs_human_review ? "Human review" : "Auto-accept"}</Badge></div>
              <div className="mt-6 space-y-5">{(["category", "sentiment", "priority"] as const).map((target) => (
                <div key={target}><div className="flex justify-between text-sm"><span className="capitalize text-ink/50">{target}</span><strong>{result[target].label} · {(result[target].confidence * 100).toFixed(1)}%</strong></div><div className="mt-2 h-2 overflow-hidden rounded-full bg-canvas"><div className="h-full rounded-full bg-moss" style={{ width: `${result[target].confidence * 100}%` }} /></div></div>
              ))}</div>
              {result.review_reasons.length > 0 && <div className="mt-7 rounded-xl bg-amber-50 p-4"><div className="text-xs font-bold uppercase tracking-wider text-amber-900">Why review?</div><ul className="mt-2 space-y-1 text-xs text-amber-800">{result.review_reasons.map((reason) => <li key={reason}>• {reason.replaceAll("_", " ")}</li>)}</ul></div>}
              <div className="mt-5 text-[11px] text-ink/35">Model {result.model_version}</div>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
