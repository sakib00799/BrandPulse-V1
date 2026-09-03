"use client";

import { useEffect, useState } from "react";
import { apiFetch, CommentPage, CommentRecord } from "@/lib/api";
import { Badge, PageHeader, Panel } from "@/components/shell";

export default function ReviewPage() {
  const [data, setData] = useState<CommentPage | null>(null);
  const [selected, setSelected] = useState<CommentRecord | null>(null);
  const [category, setCategory] = useState(""); const [sentiment, setSentiment] = useState(""); const [priority, setPriority] = useState("");
  const [saved, setSaved] = useState(false);
  useEffect(() => { apiFetch<CommentPage>("/review-queue?page_size=20").then(setData); }, []);
  const choose = (item: CommentRecord) => { setSelected(item); setCategory(item.predicted_category ?? ""); setSentiment(item.predicted_sentiment ?? ""); setPriority(item.predicted_priority ?? ""); setSaved(false); };
  const save = async () => {
    if (!selected) return;
    await apiFetch("/feedback", { method: "POST", body: JSON.stringify({ comment_id: selected.id, text: selected.text, original_category: selected.predicted_category, corrected_category: category, original_sentiment: selected.predicted_sentiment, corrected_sentiment: sentiment, original_priority: selected.predicted_priority, corrected_priority: priority, reviewer_note: "Dashboard human review" }) });
    setSaved(true);
  };
  return (
    <>
      <PageHeader eyebrow="Human in the loop" title="Review queue" description="Low-confidence and predicted High-priority comments stay visible. Corrections are stored as feedback and never trigger automatic retraining." />
      <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
        <Panel className="p-0"><div className="border-b border-ink/10 px-5 py-4 text-sm font-semibold">{data?.total ?? "—"} comments need review</div><div className="divide-y divide-ink/5">{data?.items.map((item) => <button key={item.id} onClick={() => choose(item)} className={`block w-full p-5 text-left transition hover:bg-canvas/60 ${selected?.id === item.id ? "bg-canvas" : ""}`}><div className="flex items-center justify-between gap-3"><div className="text-xs font-semibold text-moss">{item.company} · {item.source_platform}</div><Badge tone={item.predicted_priority === "High" ? "danger" : "warn"}>{item.predicted_priority}</Badge></div><p className="mt-2 line-clamp-2 text-sm leading-5">{item.text}</p><div className="mt-2 text-[10px] text-ink/40">{item.review_reasons.join(" · ")}</div></button>)}</div></Panel>
        <Panel className="h-fit xl:sticky xl:top-8">{!selected ? <p className="py-20 text-center text-sm text-ink/40">Select a comment to review.</p> : <div><div className="text-[10px] font-bold uppercase tracking-widest text-moss">Record {selected.source_record_id}</div><p className="mt-3 text-sm leading-6">{selected.text}</p><div className="mt-6 space-y-3"><label className="block text-xs font-semibold">Category<input className="field mt-1" value={category} onChange={(e) => setCategory(e.target.value)} /></label><label className="block text-xs font-semibold">Sentiment<select className="field mt-1" value={sentiment} onChange={(e) => setSentiment(e.target.value)}><option>Negative</option><option>Neutral</option><option>Positive</option></select></label><label className="block text-xs font-semibold">Priority<select className="field mt-1" value={priority} onChange={(e) => setPriority(e.target.value)}><option>Low</option><option>Medium</option><option>High</option></select></label></div><button onClick={save} className="button-primary mt-5 w-full">Save reviewer feedback</button>{saved && <p className="mt-3 text-center text-xs font-semibold text-moss">Saved. No automatic retraining triggered.</p>}</div>}</Panel>
      </div>
    </>
  );
}
