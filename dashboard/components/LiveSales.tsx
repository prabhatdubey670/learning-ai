"use client";
import { useEffect, useRef, useState } from "react";
import { fmt$, fmtN, titleCase, type Sale } from "@/lib/api";

type Row = Sale & { _id: number; _new?: boolean };

// Replays the most recent auction sales as a live-updating feed.
export default function LiveSales({ pool }: { pool: Sale[] }) {
  const [rows, setRows] = useState<Row[]>([]);
  const idx = useRef(0);
  const uid = useRef(0);

  useEffect(() => {
    if (!pool.length) return;
    // seed with the first 8
    const seed = pool.slice(0, 8).map((s) => ({ ...s, _id: uid.current++ }));
    setRows(seed);
    idx.current = 8 % pool.length;

    const t = setInterval(() => {
      const s = pool[idx.current % pool.length];
      idx.current++;
      setRows((prev) => [{ ...s, _id: uid.current++, _new: true }, ...prev].slice(0, 8));
    }, 2600);
    return () => clearInterval(t);
  }, [pool]);

  if (!pool.length) return <div className="skeleton">No recent sales available.</div>;

  return (
    <div className="card">
      <h2>Latest market sales <span className="live" style={{ marginLeft: 6, verticalAlign: "middle" }}><span className="dot" />LIVE</span></h2>
      <p className="sub">Most recent auction sales, replayed — a new car drops in every couple of seconds.</p>
      <div style={{ overflowX: "auto" }}>
        <table className="tbl">
          <thead>
            <tr>
              <th>Vehicle</th><th>Miles</th><th>Cond.</th>
              <th style={{ textAlign: "right" }}>Sold</th>
              <th style={{ textAlign: "right" }}>vs MMR</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const delta = r.sellingprice - r.mmr;
              const good = delta >= 0;
              return (
                <tr key={r._id} className={r._new ? "flash" : undefined}>
                  <td>
                    <div className="car">{r.year} {titleCase(r.make)} {titleCase(r.model)}</div>
                    <div className="meta">{titleCase(r.trim)} · {titleCase(r.body)} · {titleCase(r.color)}</div>
                  </td>
                  <td className="num">{fmtN(Math.round(r.odometer))}</td>
                  <td className="num">{r.condition ?? "—"}</td>
                  <td className="num" style={{ fontWeight: 600 }}>{fmt$(r.sellingprice)}</td>
                  <td className="num">
                    <span className={good ? "delta-up" : "delta-dn"}>
                      {good ? "▲" : "▼"} {fmt$(Math.abs(delta))}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
