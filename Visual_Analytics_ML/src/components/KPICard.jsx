import React from 'react';

/**
 * Blue KPI card (matches wireframe blue pill style).
 *
 * Props:
 *   label   -  string
 *   value   -  string (pre-formatted)
 *   delta   -  number (%)
 *   color   -  accent / top-border color
 */
export default function KPICard({ label, value, delta, color }) {
  const positive = delta >= 0;

  return (
    <div
      className="kpi"
      style={{ '--kc': color }}
    >
      <div className="kpi-lbl">{label}</div>
      <div className="kpi-val mono">{value}</div>
      <div
        className="kpi-delta"
        style={{ color: positive ? '#34d399' : '#f87171' }}
      >
        <span>{positive ? '^' : 'v'}</span>
        <span>{Math.abs(delta).toFixed(1)}%</span>
        <span style={{ color: 'var(--muted)', fontWeight: 400, fontSize: 9, marginLeft: 3 }}>
          vs prior
        </span>
      </div>
    </div>
  );
}
