// Talks to the FastAPI backend through Next's /api rewrite (see next.config.js).

export type Metric = { mae: number; rmse: number; r2: number };

export type Meta = {
  options: Record<string, string[]>;
  year_range: [number, number];
  metrics: { baseline: Metric; no_mmr: Metric; with_mmr: Metric };
  importance: [string, number][];
};

export type Prediction = {
  predicted_price: number;
  range_low: number;
  range_high: number;
  confidence_note: string;
};

export type Sale = {
  year: number; make: string; model: string; trim: string; body: string;
  odometer: number; condition: number; color: string;
  sellingprice: number; mmr: number; state: string;
};

export type CarInput = {
  year: number; make: string; model: string; trim: string; body: string;
  transmission: string; state: string; condition: number; odometer: number;
  color: string; interior: string; sale_year: number; sale_month: number;
};

const BASE = "/api";

async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const getMeta = () => j<Meta>("/meta");
export const getRecent = (limit = 60) => j<Sale[]>(`/recent?limit=${limit}`);
export const predict = (car: CarInput) =>
  j<Prediction>("/predict", { method: "POST", body: JSON.stringify(car) });

export const fmt$ = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
export const fmtN = (n: number) => n.toLocaleString("en-US");
export const titleCase = (s: string) =>
  (s || "").replace(/\b\w/g, (c) => c.toUpperCase());
