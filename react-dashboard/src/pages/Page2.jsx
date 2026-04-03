import React, { useMemo } from 'react';
import {
  ResponsiveContainer, ComposedChart, BarChart, Bar, Line, Area, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts';
import { FEATURE_META, CLASS_NAMES, CLASS_COLORS } from '../ml/models';
import { computeCorrelations } from '../ml/engine';

const TT = {
  contentStyle: { background: '#0c1525', border: '1px solid #1e3050', borderRadius: 8, fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, padding: '8px 12px' },
  labelStyle: { color: '#e4eeff', fontWeight: 700, marginBottom: 4 },
};
const tick = { fontSize: 8, fill: 'var(--muted)' };
const monoTick = { fontSize: 8, fill: 'var(--muted)', fontFamily: "'IBM Plex Mono'" };
const fmt$ = v => '$' + Number(v).toFixed(0);

const BASE_CORRELATIONS = computeCorrelations();

export default function Page2({ params, result }) {
  const { monthlySeries, fi, metrics, q1, q2 } = result;

  /* Correlation data  -  re-ranks active features to show effect of feature toggles */
  const corrData = useMemo(() => {
    return BASE_CORRELATIONS.map(c => {
      // Find if this feature is active
      const fIdx = FEATURE_META.findIndex(fm => fm.col === c.col);
      const active = fIdx === -1 || params.activeFeatures.includes(fIdx);
      return { ...c, r: +Math.abs(c.r).toFixed(3), active };
    });
  }, [params.activeFeatures]);

  /* Feature importance sorted  -  responds to activeFeatures + classWeights + depth */
  const fiData = useMemo(() =>
    FEATURE_META.map((fm, i) => ({
      short: fm.short, name: fm.name, col: fm.col,
      importance: fi[i],
      active: params.activeFeatures.includes(i),
    })).sort((a, b) => b.importance - a.importance),
    [fi, params.activeFeatures]
  );

  /* Per-class accuracy */
  const classAccData = CLASS_NAMES.map((n, i) => ({
    name: n,
    shortName: ['Low', 'Med', 'High'][i],
    accuracy: result.perClassAcc[i],
    range: i === 0 ? `<$${Math.round(q1)}` : i === 1 ? `$${Math.round(q1)}-$${Math.round(q2)}` : `>$${Math.round(q2)}`,
    color: CLASS_COLORS[i],
  }));

  /* Top misclassifications from confusion matrix */
  const errorData = useMemo(() => {
    const rows = [];
    const confMatrix = result.confMatrix;
    CLASS_NAMES.forEach((actual, ai) => {
      CLASS_NAMES.forEach((pred, pi) => {
        if (ai !== pi && confMatrix[ai][pi] > 0) {
          rows.push({
            label: CLASS_NAMES[ai] + ' => ' + CLASS_NAMES[pi],
            shortLabel: ['Low', 'Med', 'High'][ai] + '->' + ['Low', 'Med', 'High'][pi],
            count: confMatrix[ai][pi],
            actualIdx: ai, predIdx: pi,
            explanation: `${confMatrix[ai][pi]} real "${CLASS_NAMES[ai]}" transactions were wrongly predicted as "${CLASS_NAMES[pi]}"`,
          });
        }
      });
    });
    return rows.sort((a, b) => b.count - a.count);
  }, [result.confMatrix]);

  return (
    <div className="page-anim">

      {/* Row 1: Correlation - full width, interactive with right panel */}
      <div className="card" style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 4 }}>
          <div>
            <div className="card-title">Feature Correlation with Sales Amount</div>
            <div className="card-sub" style={{ marginBottom: 0 }}>
              How strongly each dataset column is associated with the total sale amount.
              <span style={{ color: '#fbbf24', marginLeft: 5, fontWeight: 600 }}>
                Unit Price (r=0.63) and Quantity (r=0.69) dominate - these drive the tree's first splits.
              </span>
              <span style={{ color: '#4d6a8a', marginLeft: 8 }}>
                Toggle features OFF in the right panel to see dimmed (inactive) bars - showing what the model ignores.
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: 14 }}>
          {/* Horizontal bar chart */}
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={corrData} layout="vertical" margin={{ top: 2, right: 24, left: 4, bottom: 18 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
              <XAxis type="number" domain={[0, 0.75]} tick={tick} axisLine={false} tickLine={false}
                tickFormatter={v => v.toFixed(1)}
                label={{ value: 'Correlation strength (0 = no link, 1 = perfect link)', position: 'insideBottom', offset: -12, fill: 'var(--muted)', fontSize: 8 }} />
              <YAxis type="category" dataKey="name" width={104}
                tick={{ fontSize: 9, fill: 'var(--muted)', fontFamily: "'Plus Jakarta Sans'" }} axisLine={false} tickLine={false} />
              <Tooltip {...TT}
                formatter={(v, _, entry) => [v.toFixed(3) + ' (' + entry.payload.type + ')', entry.payload.col + ' column']}
                labelFormatter={l => l} />
              <Bar dataKey="r" name="Correlation" radius={[0, 4, 4, 0]} maxBarSize={18}>
                {corrData.map((c, i) => {
                  const baseColor = c.r > 0.3 ? '#f59e0b' : c.r > 0.1 ? '#38bdf8' : '#4d6a8a';
                  return <Cell key={i} fill={baseColor} fillOpacity={c.active ? 1 : 0.2} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Interpretation list */}
          <div style={{ overflowY: 'auto', maxHeight: 230 }}>
            <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 7 }}>
              What each column means for sale size
            </div>
            {corrData.map((c, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 7, paddingBottom: 7, borderBottom: '1px solid var(--border)', opacity: c.active ? 1 : 0.35, transition: 'opacity .2s' }}>
                <div style={{ width: 34, height: 34, borderRadius: 7, flexShrink: 0, background: (c.r > 0.3 ? '#f59e0b' : c.r > 0.1 ? '#38bdf8' : '#4d6a8a') + '20', border: '1px solid ' + (c.r > 0.3 ? '#f59e0b' : c.r > 0.1 ? '#38bdf8' : '#4d6a8a') + '44', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <span className="mono" style={{ fontSize: 9, fontWeight: 700, color: c.r > 0.3 ? '#f59e0b' : c.r > 0.1 ? '#38bdf8' : '#4d6a8a' }}>{c.r.toFixed(2)}</span>
                </div>
                <div>
                  <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--text2)', marginBottom: 1 }}>
                    {c.name} {!c.active && <span style={{ color: '#f87171', fontSize: 8 }}>(disabled)</span>}
                  </div>
                  <div style={{ fontSize: 8, color: 'var(--muted)', lineHeight: 1.4 }}>{c.interp}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row 2: Actual vs Predicted time series */}
      <div className="card" style={{ marginBottom: 10 }}>
        <div className="card-title">Actual vs Predicted Monthly Sales - 2019 to 2025</div>
        <div className="card-sub">
          Blue filled area = actual total monthly sales. Amber dashed line = Decision Tree prediction (each transaction
          mapped to its class mean: Low=${Math.round(result.classMeans[0])}, Med=${Math.round(result.classMeans[1])}, High=${Math.round(result.classMeans[2])} per transaction).
          Training period ends at the vertical reference line (Jan 2021).
        </div>
        <ResponsiveContainer width="100%" height={195}>
          <ComposedChart data={monthlySeries} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="actGrad2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6ea8fe" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#6ea8fe" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="label" tick={monoTick} interval={5} axisLine={false} tickLine={false} />
            <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={fmt$} width={56} />
            <Tooltip {...TT} formatter={(v, n) => [fmt$(v), n]} />
            <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }} />
            <ReferenceLine x={"Jan'21"} stroke="var(--border2)" strokeDasharray="5 3"
              label={{ value: 'Prediction Zone starts', position: 'insideTopLeft', fill: 'var(--muted)', fontSize: 8 }} />
            <Area type="monotone" dataKey="actual" name="Actual Monthly Sales"
              stroke="#6ea8fe" strokeWidth={2} fill="url(#actGrad2)" dot={false} />
            <Line type="monotone" dataKey="pred" name="DT Prediction (class mean)"
              stroke="#f59e0b" strokeWidth={2.5} strokeDasharray="8 4" dot={false} connectNulls />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Row 3: Feature Importance + Per-Class Accuracy + Top Misclassifications */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: 10 }}>

        {/* Feature Importance */}
        <div className="card">
          <div className="card-title">Feature Importance</div>
          <div className="card-sub">
            How much each feature contributes to the tree's splits (information gain).
            Toggle features OFF in the right panel  -  their bar drops to zero and the tree redistributes importance.
            <span style={{ color: '#fbbf24', marginLeft: 4 }}>Orange = active. Grey = disabled.</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={fiData} layout="vertical" margin={{ top: 2, right: 10, left: 4, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
              <XAxis type="number" tick={tick} axisLine={false} tickLine={false} tickFormatter={v => v + '%'} />
              <YAxis type="category" dataKey="short" width={68}
                tick={{ fontSize: 9, fill: 'var(--muted)', fontFamily: "'Plus Jakarta Sans'" }} axisLine={false} tickLine={false} />
              <Tooltip {...TT} formatter={(v, _, entry) => [v.toFixed(2) + '%', entry.payload.name + ' (' + entry.payload.col + ')']} />
              <Bar dataKey="importance" name="Importance %" radius={[0, 4, 4, 0]} maxBarSize={16}>
                {fiData.map((d, i) => (
                  <Cell key={i} fill={d.active ? '#f59e0b' : '#4d6a8a'} fillOpacity={d.active ? 0.9 : 0.4} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Per-Class Accuracy */}
        <div className="card">
          <div className="card-title">Per-Class Accuracy (Test Set)</div>
          <div className="card-sub">
            How accurately the model predicted each sale category on 2021-2025 test data.
            Low accuracy on a class = adjust that class's weight in the right panel to improve it.
          </div>
          <ResponsiveContainer width="100%" height={130}>
            <BarChart data={classAccData} margin={{ top: 4, right: 4, left: -14, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="shortName" tick={{ fontSize: 10, fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tick={tick} axisLine={false} tickLine={false} tickFormatter={v => v + '%'} />
              <Tooltip {...TT} formatter={(v, _, entry) => [v + '%', CLASS_NAMES[entry.payload.color === CLASS_COLORS[0] ? 0 : entry.payload.color === CLASS_COLORS[1] ? 1 : 2] + ' (' + entry.payload.range + ')']} />
              <Bar dataKey="accuracy" name="Accuracy %" radius={[4, 4, 0, 0]} maxBarSize={46}>
                {classAccData.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {/* Summary row */}
          <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {classAccData.map((d, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: d.color, flexShrink: 0 }} />
                <span style={{ fontSize: 9, color: 'var(--muted)', flex: 1 }}>{d.name} ({d.range})</span>
                <span className="mono" style={{ fontSize: 10, fontWeight: 700, color: d.color }}>{d.accuracy}%</span>
                <div style={{ width: 48, height: 3, background: 'var(--border)', borderRadius: 2 }}>
                  <div style={{ height: '100%', width: d.accuracy + '%', background: d.color, borderRadius: 2 }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Top Misclassifications */}
        <div className="card">
          <div className="card-title">Top Misclassifications</div>
          <div className="card-sub">
            Which pairs of categories does the model confuse most? Adjust class weights or class boundaries in the right panel to reduce these errors.
          </div>
          {errorData.length === 0 ? (
            <div style={{ color: '#34d399', fontSize: 12, padding: '16px 0', textAlign: 'center', fontWeight: 700 }}>
              Perfect classification - no errors!
            </div>
          ) : (
            <>
              <ResponsiveContainer width="100%" height={130}>
                <BarChart data={errorData} margin={{ top: 4, right: 4, left: -14, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="shortLabel" tick={{ fontSize: 8.5, fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
                  <YAxis tick={tick} axisLine={false} tickLine={false} label={{ value: 'Errors', angle: -90, position: 'insideLeft', fill: 'var(--muted)', fontSize: 8 }} />
                  <Tooltip {...TT} formatter={(v, _, entry) => [v + ' transactions', entry.payload.label]} />
                  <Bar dataKey="count" name="Misclassified" radius={[4, 4, 0, 0]} maxBarSize={36}>
                    {errorData.map((d, i) => <Cell key={i} fill={CLASS_COLORS[d.actualIdx]} fillOpacity={0.75} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 5 }}>
                {errorData.slice(0, 4).map((d, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid var(--border)', fontSize: 9 }}>
                    <span style={{ color: 'var(--muted)', flex: 1, lineHeight: 1.4 }}>
                      <span style={{ color: CLASS_COLORS[d.actualIdx], fontWeight: 600 }}>{CLASS_NAMES[d.actualIdx]}</span>
                      {' predicted as '}
                      <span style={{ color: CLASS_COLORS[d.predIdx], fontWeight: 600 }}>{CLASS_NAMES[d.predIdx]}</span>
                    </span>
                    <span className="mono" style={{ color: '#f87171', fontWeight: 700, marginLeft: 8 }}>{d.count}x</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
