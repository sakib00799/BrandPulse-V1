"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const links = [
  ["/", "Overview", "01"],
  ["/comments", "Comment explorer", "02"],
  ["/predict", "Live prediction", "03"],
  ["/performance", "Model performance", "04"],
  ["/review", "Review queue", "05"],
];

export function Shell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="min-h-screen bg-canvas text-ink lg:grid lg:grid-cols-[260px_1fr]">
      <aside className="border-b border-ink/10 bg-ink px-6 py-6 text-white lg:min-h-screen lg:border-b-0 lg:border-r lg:border-white/10 lg:px-7 lg:py-8">
        <div className="flex items-center justify-between lg:block">
          <Link href="/" className="block">
            <span className="text-[11px] font-bold uppercase tracking-[0.34em] text-lime">BrandPulse</span>
            <div className="mt-1 text-2xl font-semibold tracking-tight">BD / বাংলা</div>
          </Link>
          <span className="rounded-full border border-white/20 px-3 py-1 text-[10px] uppercase tracking-widest text-white/60">v1 prototype</span>
        </div>
        <nav className="mt-7 grid grid-cols-2 gap-2 sm:grid-cols-5 lg:mt-14 lg:grid-cols-1">
          {links.map(([href, label, index]) => {
            const active = href === "/" ? pathname === href : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={`group flex items-center gap-3 rounded-xl px-3 py-3 text-sm transition ${
                  active ? "bg-white text-ink" : "text-white/65 hover:bg-white/10 hover:text-white"
                }`}
              >
                <span className={`font-mono text-[10px] ${active ? "text-moss" : "text-lime/70"}`}>{index}</span>
                <span>{label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="mt-10 hidden border-t border-white/10 pt-6 text-xs leading-5 text-white/45 lg:block">
          Human-assisted intelligence for Bangla, Banglish and code-mixed feedback.
        </div>
      </aside>
      <main className="min-w-0 px-5 py-7 sm:px-8 lg:px-12 lg:py-10">{children}</main>
    </div>
  );
}

export function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <header className="mb-8 max-w-4xl">
      <div className="mb-3 text-[11px] font-bold uppercase tracking-[0.3em] text-moss">{eyebrow}</div>
      <h1 className="text-4xl font-semibold tracking-[-0.045em] sm:text-5xl">{title}</h1>
      <p className="mt-4 max-w-2xl text-sm leading-6 text-ink/60 sm:text-base">{description}</p>
    </header>
  );
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-2xl border border-ink/10 bg-white p-5 shadow-panel ${className}`}>{children}</section>;
}

export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "good" | "warn" | "danger" }) {
  const tones = {
    neutral: "bg-ink/5 text-ink/65",
    good: "bg-moss/10 text-moss",
    warn: "bg-amber-100 text-amber-800",
    danger: "bg-coral/15 text-red-800",
  };
  return <span className={`inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold ${tones[tone]}`}>{children}</span>;
}
