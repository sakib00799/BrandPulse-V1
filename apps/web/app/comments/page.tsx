"use client";

import { FormEvent, useEffect, useState } from "react";
import { apiFetch, CommentPage } from "@/lib/api";
import { Badge, PageHeader, Panel } from "@/components/shell";

export default function CommentsPage() {
  const [data, setData] = useState<CommentPage | null>(null);
  const [search, setSearch] = useState("");
  const [company, setCompany] = useState("");
  const [priority, setPriority] = useState("");
  const [loading, setLoading] = useState(true);
  const load = (query = "") => {
    setLoading(true);
    apiFetch<CommentPage>(`/comments?page_size=30${query}`).then(setData).finally(() => setLoading(false));
  };
  useEffect(() => load(), []);
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const params = new URLSearchParams();
    if (search) params.set("search", search);
    if (company) params.set("company", company);
    if (priority) params.set("priority", priority);
    load(`&${params.toString()}`);
  };
  return (
    <>
      <PageHeader eyebrow="Evidence browser" title="Comment explorer" description="Search source comments and compare supplied ground truth with model predictions. Source links remain attached for traceability." />
      <Panel className="mb-5">
        <form onSubmit={submit} className="grid gap-3 md:grid-cols-[1fr_180px_160px_auto]">
          <input className="field" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search comment text…" />
          <select className="field" value={company} onChange={(e) => setCompany(e.target.value)}>
            <option value="">All companies</option>
            {["bKash", "Nagad", "Grameenphone", "Robi", "Banglalink", "Daraz", "foodpanda", "Pathao"].map((value) => <option key={value}>{value}</option>)}
          </select>
          <select className="field" value={priority} onChange={(e) => setPriority(e.target.value)}>
            <option value="">All priorities</option><option>High</option><option>Medium</option><option>Low</option>
          </select>
          <button className="button-primary">Apply filters</button>
        </form>
      </Panel>
      <Panel className="overflow-hidden p-0">
        <div className="flex items-center justify-between border-b border-ink/10 px-5 py-4">
          <span className="text-sm font-semibold">{loading ? "Loading…" : `${data?.total ?? 0} comments`}</span>
          <span className="text-xs text-ink/40">Showing up to 30</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] text-left text-sm">
            <thead className="bg-canvas/70 text-[10px] uppercase tracking-widest text-ink/45">
              <tr><th className="px-5 py-3">Comment</th><th className="px-4 py-3">Company</th><th className="px-4 py-3">Category: actual / predicted</th><th className="px-4 py-3">Sentiment</th><th className="px-4 py-3">Priority</th></tr>
            </thead>
            <tbody className="divide-y divide-ink/5">
              {data?.items.map((item) => (
                <tr key={item.id} className="align-top hover:bg-canvas/40">
                  <td className="max-w-md px-5 py-4"><p className="line-clamp-3 leading-5">{item.text}</p>{item.source_url && <a href={item.source_url} target="_blank" rel="noreferrer" className="mt-2 inline-block text-xs font-semibold text-moss underline">Open source ↗</a>}</td>
                  <td className="px-4 py-4"><div className="font-medium">{item.company}</div><div className="mt-1 text-xs text-ink/40">{item.source_platform}</div></td>
                  <td className="px-4 py-4"><div>{item.actual_category}</div><div className="mt-2 text-xs text-moss">→ {item.predicted_category} ({Math.round((item.category_confidence ?? 0) * 100)}%)</div></td>
                  <td className="px-4 py-4"><div>{item.actual_sentiment}</div><div className="mt-2 text-xs text-moss">→ {item.predicted_sentiment}</div></td>
                  <td className="px-4 py-4"><Badge tone={item.predicted_priority === "High" ? "danger" : item.predicted_priority === "Medium" ? "warn" : "neutral"}>{item.predicted_priority}</Badge>{item.needs_human_review && <div className="mt-2 text-[10px] font-bold uppercase tracking-wide text-coral">Review</div>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}
