"use client";
import { fmt$, type Meta } from "@/lib/api";

const PRETTY: Record<string, string> = {
  make: "Make", model: "Model", trim: "Trim", body: "Body type",
  transmission: "Transmission", state: "State", color: "Exterior color",
  interior: "Interior", car_age: "Car age", odometer: "Mileage",
  condition: "Condition", sale_month: "Sale month", year: "Model year",
};

export function FeatureImportance({ meta }: { meta: Meta }) {
  const rows = meta.importance.slice(0, 8);
  const max = Math.max(...rows.map(([, v]) => v)) || 1;
  return (
    <div className="card">
      <h2>What drives the price</h2>
      <p className="sub">XGBoost feature importance (gain) — bigger = more influence on the prediction.</p>
      <div className="bars">
        {rows.map(([name, val]) => (
          <div className="bar-row" key={name}>
            <span className="name" title={PRETTY[name] || name}>{PRETTY[name] || name}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(val / max) * 100}%`, background: "var(--series-1)" }} />
            </div>
            <span className="val">{((val / max) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ModelComparison({ meta }: { meta: Meta }) {
  const m = meta.metrics;
  const rows = [
    { name: "Baseline (guess avg)", mae: m.baseline.mae, r2: m.baseline.r2, color: "var(--muted)" },
    { name: "XGBoost · specs only", mae: m.no_mmr.mae, r2: m.no_mmr.r2, color: "var(--series-1)" },
    { name: "XGBoost · specs + MMR", mae: m.with_mmr.mae, r2: m.with_mmr.r2, color: "var(--series-3)" },
  ];
  const max = Math.max(...rows.map((r) => r.mae));
  return (
    <div className="card">
      <h2>Model accuracy</h2>
      <p className="sub">Average dollar error per car (lower = better). The specs-only model is the deployable one.</p>
      <div className="bars">
        {rows.map((r) => (
          <div className="bar-row" key={r.name}>
            <span className="name" title={r.name}>{r.name}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(r.mae / max) * 100}%`, background: r.color }} />
            </div>
            <span className="val">{fmt$(r.mae)}</span>
          </div>
        ))}
      </div>
      <p className="sub" style={{ marginTop: 14, marginBottom: 0 }}>
        R² · specs-only <b style={{ color: "var(--text-primary)" }}>{m.no_mmr.r2.toFixed(3)}</b> — explains{" "}
        {(m.no_mmr.r2 * 100).toFixed(1)}% of price variation.
      </p>
    </div>
  );
}
