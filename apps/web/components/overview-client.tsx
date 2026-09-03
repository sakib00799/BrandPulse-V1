"use client";

import { useEffect, useState } from "react";
import { Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { apiFetch, DistributionItem } from "@/lib/api";
import { Panel } from "@/components/shell";

type Overview = {
  total_comments: number;
  high_priority_count: number;
  sentiment_distribution: DistributionItem[];
  category_distribution: DistributionItem[];
  company_distribution: DistributionItem[];
  platform_distribution: DistributionItem[];
};

const colors = ["#315c4c", "#cde85b", "#f0785f", "#7f9c8d", "#d7b66f", "#7889a5", "#b77d8c", "#92a944"];

export function OverviewClient() {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    apiFetch<Overview>("/analytics/overview").then(setData).catch((value: Error) => setError(value.message));
  }, []);
  if (error) return <Panel><p className="text-sm text-red-700">API unavailable: {error}</p></Panel>;
  if (!data) return <Panel><p className="animate-pulse text-sm text-ink/50">Loading verified analytics…</p></Panel>;
  const negative = data.sentiment_distribution.find((item) => item.label === "Negative")?.count ?? 0;
  const stats = [
    ["Total comments", data.total_comments.toLocaleString(), "Audited records"],
    ["High priority", data.high_priority_count.toLocaleString(), "Predicted queue"],
    ["Negative", negative.toLocaleString(), `${((negative / Math.max(1, data.total_comments)) * 100).toFixed(1)}% of comments`],
    ["Companies", data.company_distribution.length.toString(), "Observed brands"],
  ];
  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map(([label, value, note], index) => (
          <Panel key={label} className={index === 1 ? "!bg-ink !text-white" : ""}>
            <div className={`text-[11px] font-bold uppercase tracking-widest ${index === 1 ? "text-lime" : "text-ink/45"}`}>{label}</div>
            <div className="mt-4 text-4xl font-semibold tracking-tight">{value}</div>
            <div className={`mt-2 text-xs ${index === 1 ? "text-white/55" : "text-ink/45"}`}>{note}</div>
          </Panel>
        ))}
      </div>
      <div className="grid gap-5 xl:grid-cols-[1.25fr_.75fr]">
        <Panel>
          <div className="mb-5">
            <h2 className="text-lg font-semibold">Top complaint categories</h2>
            <p className="text-xs text-ink/45">Model-predicted distribution</p>
          </div>
          <div className="h-[340px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.category_distribution.slice(0, 8)} layout="vertical" margin={{ left: 15, right: 20 }}>
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="label" width={125} tick={{ fontSize: 11, fill: "#42524d" }} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: "#f3f1e9" }} />
                <Bar dataKey="count" fill="#315c4c" radius={[0, 7, 7, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel>
          <h2 className="text-lg font-semibold">Sentiment mix</h2>
          <p className="text-xs text-ink/45">Current predicted labels</p>
          <div className="h-[270px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={data.sentiment_distribution} dataKey="count" nameKey="label" innerRadius={60} outerRadius={94} paddingAngle={3} isAnimationActive={false}>
                  {data.sentiment_distribution.map((item, index) => <Cell key={item.label} fill={colors[index]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-2">
            {data.sentiment_distribution.map((item, index) => (
              <div key={item.label} className="rounded-xl bg-canvas p-3 text-center">
                <div className="mx-auto mb-2 h-2 w-2 rounded-full" style={{ backgroundColor: colors[index] }} />
                <div className="text-xs text-ink/50">{item.label}</div>
                <div className="mt-1 font-semibold">{item.count}</div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
      <Panel className="border-amber-300/60 bg-amber-50">
        <div className="text-sm font-semibold text-amber-900">Time trends intentionally unavailable</div>
        <p className="mt-1 text-xs leading-5 text-amber-800/75">All supplied dates are relative and lack a trustworthy collection timestamp. The dashboard does not invent a time series.</p>
      </Panel>
    </div>
  );
}
