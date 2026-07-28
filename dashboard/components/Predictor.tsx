"use client";
import { useState } from "react";
import { predict, fmt$, titleCase, type Meta, type Sale, type CarInput, type Prediction } from "@/lib/api";

const SANE_MONTH = 6;

function Select({ label, value, opts, onChange, extra }:
  { label: string; value: string; opts: string[]; onChange: (v: string) => void; extra?: string }) {
  // ensure the current value is always selectable even if outside the top-N list
  const list = extra && !opts.includes(extra) ? [extra, ...opts] : opts;
  return (
    <div className="field">
      <label>{label}</label>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {list.map((o) => <option key={o} value={o}>{titleCase(o)}</option>)}
      </select>
    </div>
  );
}

export default function Predictor({ meta, pool }: { meta: Meta; pool: Sale[] }) {
  const o = meta.options;
  const [y0, y1] = meta.year_range;
  const [car, setCar] = useState<CarInput>({
    year: 2015, make: "kia", model: "sorento", trim: "lx", body: "suv",
    transmission: "automatic", state: o.state?.[0] || "ca",
    condition: 35, odometer: 30000, color: "white", interior: "black",
    sale_year: 2015, sale_month: SANE_MONTH,
  });
  const [res, setRes] = useState<Prediction | null>(null);
  const [actual, setActual] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const set = (k: keyof CarInput, v: string | number) => setCar((c) => ({ ...c, [k]: v }));

  async function run(next?: CarInput, actualPrice?: number | null) {
    setBusy(true); setErr(null);
    try {
      const p = await predict(next || car);
      setRes(p);
      setActual(actualPrice ?? null);
    } catch (e: any) {
      setErr("Backend not reachable — start the FastAPI server (see banner above).");
    } finally { setBusy(false); }
  }

  function loadRealCar() {
    if (!pool.length) return;
    const s = pool[Math.floor(Math.random() * pool.length)];
    const next: CarInput = {
      year: s.year, make: s.make?.toLowerCase(), model: s.model?.toLowerCase(),
      trim: (s.trim || "base").toLowerCase(), body: (s.body || "sedan").toLowerCase(),
      transmission: "automatic", state: (s.state || "ca").toLowerCase(),
      condition: s.condition || 30, odometer: Math.round(s.odometer || 30000),
      color: (s.color || "white").toLowerCase(), interior: "black",
      sale_year: s.year + 1, sale_month: SANE_MONTH,
    };
    setCar(next);
    run(next, s.sellingprice);
  }

  const diff = res && actual != null ? res.predicted_price - actual : null;
  const pct = diff != null && actual ? Math.abs(diff) / actual : null;

  return (
    <div className="card">
      <h2>Real-time price predictor</h2>
      <p className="sub">Enter a car — the trained XGBoost model prices it live.</p>

      <div className="form-grid">
        <Select label="Make" value={car.make} opts={o.make || []} extra={car.make} onChange={(v) => set("make", v)} />
        <Select label="Model" value={car.model} opts={o.model || []} extra={car.model} onChange={(v) => set("model", v)} />
        <Select label="Body" value={car.body} opts={o.body || []} extra={car.body} onChange={(v) => set("body", v)} />
        <Select label="Color" value={car.color} opts={o.color || []} extra={car.color} onChange={(v) => set("color", v)} />
        <div className="field">
          <label>Year ({y0}–{y1})</label>
          <input type="number" min={y0} max={y1} value={car.year}
            onChange={(e) => { const yr = +e.target.value; setCar((c) => ({ ...c, year: yr, sale_year: Math.max(yr, c.sale_year) })); }} />
        </div>
        <div className="field">
          <label>Odometer (miles)</label>
          <input type="number" min={0} step={1000} value={car.odometer}
            onChange={(e) => set("odometer", +e.target.value)} />
        </div>
        <div className="field full">
          <label>Condition: {car.condition} / 49</label>
          <input type="range" min={1} max={49} value={car.condition}
            onChange={(e) => set("condition", +e.target.value)} />
        </div>
      </div>

      <div className="btn-row">
        <button className="btn primary" onClick={() => run()} disabled={busy}>
          {busy ? "Pricing…" : "Predict price"}
        </button>
        <button className="btn ghost" onClick={loadRealCar} disabled={busy}>
          🎲 Load a real car
        </button>
      </div>

      {err && <div className="banner" style={{ marginTop: 14 }}>{err}</div>}

      {res && !err && (
        <div className="result">
          <div className="plabel">Predicted selling price</div>
          <div className="price">{fmt$(res.predicted_price)}</div>
          <div className="range">
            likely range <b>{fmt$(res.range_low)}</b> – <b>{fmt$(res.range_high)}</b> · {res.confidence_note}
          </div>
          {actual != null && diff != null && (
            <div className="vs">
              <span>Actual sale: <b>{fmt$(actual)}</b></span>
              <span className={`pill ${pct != null && pct <= 0.1 ? "good" : "warn"}`}>
                {diff >= 0 ? "+" : "−"}{fmt$(Math.abs(diff))} ({((pct || 0) * 100).toFixed(1)}% off)
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
