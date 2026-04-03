import React from 'react';

export default function KPICard({ label, value, color }) {
  return (
    <div className="kpi" style={{ '--kc': color }}>
      <div className="kpi-lbl">{label}</div>
      <div className="kpi-val mono">{value}</div>
    </div>
  );
}
