export type Document = {
  id: string;
  name: string;
  status: "queued" | "processing" | "ready" | "failed";
  size_bytes: number;
  page_count: number | null;
  chunk_count: number;
  warnings: string[];
  error_code: string | null;
  error_message: string | null;
  source_url?: string | null;
  kind?: "pdf" | "text" | "url";
  vector_status?: string | null;
  vector_error?: string | null;
  summary_status?: string | null;
  detailed_summary_status?: string | null;
};
export type Citation = {
  chunk_id: string;
  document_id: string;
  document_name: string;
  page: number;
  quote: string;
  extraction_method?: "native" | "ocr";
};
export type Evidence = Omit<Citation, "quote"> & {
  text: string;
  score: number;
};
export type Claim = {
  text: string;
  kind?: "quote" | "explanation";
  citations: Citation[];
};
export type SummaryContent = {
  document_type?: string;
  summary?: string;
  topics?: string[];
  key_points?: { text: string; pages: number[] }[];
  suggested_questions?: string[];
  sections?: { pages: number[]; bullets: { text: string }[] }[];
  key_points_dropped?: number;
  kind?: string;
};
export type QueryResult = {
  id: string;
  question: string;
  status:
    | "answered"
    | "no_relevant_passage"
    | "unsupported_question"
    | "clarification_needed"
    | "insufficient_evidence";
  reason: string;
  missing_information?: string;
  suggestion?: string;
  claims: Claim[];
  evidence: Evidence[];
  candidates?: { id: string; name: string }[];
  summary?: SummaryContent;
  coverage?: { complete?: boolean; pages_used?: number; pages_total?: number };
  timings: { retrieval_ms: number; generation_ms: number; total_ms: number };
  model: string | null;
  retrieval_mode: string;
  verification: string;
  intent?: string;
  scope?: { document_ids: string[]; source: string };
};
export type SummaryResponse = {
  document_id: string;
  document_name: string | null;
  kind: "overview" | "detailed";
  status: "ready" | "failed";
  content: SummaryContent | null;
  coverage: Record<string, number | boolean> | null;
  model: string | null;
  error: string | null;
  created_at?: number;
};
export type AssistantRun = {
  id: string;
  kind: "coding" | "chat";
  task: string;
  question: string;
  status: "answered";
  answer: string;
  language: string;
  grounded: boolean;
  sources: Evidence[];
  model: string | null;
  verification: string;
  retrieval_mode: string | null;
  timings: { retrieval_ms: number; generation_ms: number; total_ms: number };
};
export type Readiness = {
  worker_active: boolean;
  model: { reachable: boolean; model_available: boolean; model: string };
  default_retrieval_mode: string;
  answer_mode?: string;
  coding?: {
    provider: string;
    reachable: boolean;
    model_available: boolean;
    model: string;
    error?: string;
  };
  retrieval: {
    enabled: boolean;
    loaded: boolean;
    error?: string | null;
    documents?: Record<string, number>;
  };
};

let csrfToken = "";
export const setCsrfToken = (token: string) => {
  csrfToken = token;
};

export async function request<T>(
  path: string,
  options: RequestInit = {},
  timeoutMs = 180_000,
): Promise<T> {
  let response: Response;
  try {
    const headers = new Headers(options.headers);
    if (csrfToken && options.method && options.method !== "GET") {
      headers.set("X-CSRF-Token", csrfToken);
    }
    response = await fetch(`/api/v1${path}`, {
      ...options,
      headers,
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch {
    throw new Error(
      "Cannot reach the local API, or the request timed out. Check the API terminal.",
    );
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 401 && !path.startsWith("/auth/")) {
      window.dispatchEvent(new Event("omnirag-session-expired"));
    }
    const detail = data?.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : (detail?.message ??
            (Array.isArray(detail)
              ? detail.map((x) => x.msg).join("; ")
              : `Request failed (${response.status}).`)),
    );
  }
  return data as T;
}

export const fileUrl = (id: string, page = 1) =>
  `/api/v1/documents/${encodeURIComponent(id)}/file#page=${page}`;

export const pagePreviewUrl = (id: string, page: number) =>
  `/api/v1/documents/${encodeURIComponent(id)}/pages/${page}/preview`;
