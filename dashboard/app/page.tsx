"use client";
import { useEffect, useState } from "react";
import { getMeta, getRecent, fmt$, fmtN, type Meta, type Sale } from "@/lib/api";
import Predictor from "@/components/Predictor";
import { FeatureImportance, ModelComparison } from "@/components/Charts";
import LiveSales from "@/components/LiveSales";

function Tile({ label, value, foot, up }: { label: string; value: string; foot?: string; up?: boolean }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {foot && <div className={`foot ${up ? "up" : ""}`}>{foot}</div>}
    </div>
  );
}

function ThemeToggle() {
  const [dark, setDark] = useState<boolean | null>(null);
  useEffect(() => {
    const t = document.documentElement.getAttribute("data-theme");
    setDark(t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches);
  }, []);
  const toggle = () => {
    const next = !dark; setDark(next);
    document.documentElement.setAttribute("data-theme", next ? "dark" : "light");
    try { localStorage.setItem("theme", next ? "dark" : "light"); } catch {}
  };
  return <button className="theme-btn" onClick={toggle} title="Toggle theme">{dark ? "☀️" : "🌙"}</button>;
}

export default function Page() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [recent, setRecent] = useState<Sale[]>([]);
  const [down, setDown] = useState(false);

  useEffect(() => {
    Promise.all([getMeta(), getRecent(60)])
      .then(([m, r]) => { setMeta(m); setRecent(r); })
      .catch(() => setDown(true));
  }, []);

  return (
    <div className="wrap">
      <div className="top">
        <div className="logo">🚗</div>
        <div className="title">
          <h1>Car Price Intelligence</h1>
          <p>Real-time used-car pricing · XGBoost model on 553K auction sales</p>
        </div>
        <div className="spacer" />
        <span className="live"><span className="dot" />LIVE MODEL</span>
        <ThemeToggle />
      </div>

      {down && (
        <div className="banner">
          ⚠️ <b>Backend not running.</b> Start it so the dashboard can price cars:{" "}
          <code>cd api &amp;&amp; uvicorn main:app --port 8000</code> — then refresh this page.
        </div>
      )}

      {meta && (
        <>
          <div className="kpis">
            <Tile label="Avg prediction error" value={fmt$(meta.metrics.no_mmr.mae)} foot="per car (MAE)" />
            <Tile label="Model accuracy (R²)" value={meta.metrics.no_mmr.r2.toFixed(3)}
              foot={`explains ${(meta.metrics.no_mmr.r2 * 100).toFixed(0)}% of price`} />
            <Tile label="Beats naive guess by"
              value={`${(meta.metrics.baseline.mae / meta.metrics.no_mmr.mae).toFixed(1)}×`}
              foot={`baseline was ${fmt$(meta.metrics.baseline.mae)} off`} up />
            <Tile label="Sales analyzed" value="553K" foot="auction records" />
          </div>

          <div className="grid-main">
            <Predictor meta={meta} pool={recent} />
            <FeatureImportance meta={meta} />
          </div>

          <div className="grid-2">
            <LiveSales pool={recent} />
            <ModelComparison meta={meta} />
          </div>
        </>
      )}

      {!meta && !down && <div className="skeleton">Loading model & market data…</div>}

      <p className="foot-note">
        Predictions from an XGBoost regressor trained on car specs only (no MMR leakage). For decision support, not a guaranteed valuation.
      </p>
    </div>
  );
}
