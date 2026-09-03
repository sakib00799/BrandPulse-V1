import { OverviewClient } from "@/components/overview-client";
import { PageHeader } from "@/components/shell";

export default function OverviewPage() {
  return (
    <>
      <PageHeader eyebrow="Intelligence overview" title="Feedback, made legible." description="A transparent view of category, sentiment and operational priority across Bangla, Banglish and English customer comments." />
      <OverviewClient />
    </>
  );
}
