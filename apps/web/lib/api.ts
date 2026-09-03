export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export type DistributionItem = { label: string; count: number };

export type Prediction = {
  category: { label: string; confidence: number };
  sentiment: { label: string; confidence: number };
  priority: { label: string; confidence: number };
  needs_human_review: boolean;
  review_reasons: string[];
  model_version: string;
};

export type CommentRecord = {
  id: number;
  source_record_id: string | null;
  text: string;
  company: string | null;
  source_platform: string | null;
  source_url: string | null;
  created_at_raw: string | null;
  actual_category: string | null;
  actual_sentiment: string | null;
  actual_priority: string | null;
  predicted_category: string | null;
  category_confidence: number | null;
  predicted_sentiment: string | null;
  sentiment_confidence: number | null;
  predicted_priority: string | null;
  priority_confidence: number | null;
  needs_human_review: boolean;
  review_reasons: string[];
  model_version: string | null;
};

export type CommentPage = {
  total: number;
  page: number;
  page_size: number;
  items: CommentRecord[];
};
