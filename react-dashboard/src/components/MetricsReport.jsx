import React from 'react';
import { CLASS_NAMES, CLASS_COLORS } from '../ml/models';

function pct(v) { return (v * 100).toFixed(1) + '%'; }

function Bar({ val, color = 'var(--c0)', h = 5 }) {
  return (
    <div style={{ width: '100%', height: h, background: 'var(--border)', borderRadius: h / 2 }}>
      <div style={{ height: '100%', width: (val * 100) + '%', background: color, borderRadius: h / 2, transition: 'width .4s' }} />
    </div>
  );
}

export default function MetricsReport({ metrics, trainAcc }) {
  if (!metrics) return null;
  const { accuracy, ciLo, ciHi, perClass } = metrics;

  return (
    <div>
      {/* Top 2 summary cards: Accuracy + CI, Train vs Test */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>

        {/* Accuracy + CI */}
        <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
          <div style={{ fontSize: 9, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.1em', marginBottom: 4 }}>
            Test Accuracy
          </div>
          <div className="mono" style={{ fontSize: 26, fontWeight: 800, color: '#34d399', lineHeight: 1 }}>
            {pct(accuracy)}
          </div>
          <div style={{ fontSize: 9, color: 'var(--muted)', marginTop: 4 }}>95% Confidence Interval</div>
          <div className="mono" style={{ fontSize: 11, color: 'var(--text2)', marginTop: 1, fontWeight: 700 }}>
            {ciLo.toFixed(1)}% to {ciHi.toFixed(1)}%
          </div>
          <div style={{ marginTop: 5, fontSize: 8, color: 'var(--muted)', lineHeight: 1.5 }}>
            We are 95% sure the model's real-world accuracy on any new transactions will fall inside this range.
          </div>
        </div>

        {/* Train vs Test gap */}
        <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px' }}>
          <div style={{ fontSize: 9, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.1em', marginBottom: 6 }}>
            Train vs Test Gap
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
            <div className="mono" style={{ fontSize: 18, fontWeight: 800, color: '#38bdf8' }}>{trainAcc}%</div>
            <div style={{ fontSize: 9, color: 'var(--muted)' }}>on training data (2019-20)</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 6 }}>
            <div className="mono" style={{ fontSize: 18, fontWeight: 800, color: '#34d399' }}>{pct(accuracy)}</div>
            <div style={{ fontSize: 9, color: 'var(--muted)' }}>on new test data (2021-25)</div>
          </div>
          {(() => {
            const gap = trainAcc - accuracy * 100;
            const col = gap > 15 ? '#f87171' : gap > 8 ? '#fbbf24' : '#34d399';
            const msg = gap > 15 ? 'High gap - model may have memorised training data. Reduce Max Depth.'
              : gap > 8 ? 'Moderate gap - watch for overfitting.'
                : 'Small gap - model generalises well.';
            return (
              <div style={{ padding: '4px 8px', background: col + '15', border: '1px solid ' + col + '44', borderRadius: 5, fontSize: 8, color: col, lineHeight: 1.5 }}>
                Gap: {gap.toFixed(1)} pts. {msg}
              </div>
            );
          })()}
        </div>
      </div>

      {/* Per-class table  -  no Support, no Macro/Weighted */}
      <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>

        {/* Header */}
        <div style={{ display: 'grid', gridTemplateColumns: '120px repeat(4, 1fr)', background: 'var(--surface)', borderBottom: '1px solid var(--border)', padding: '7px 10px', gap: 4 }}>
          {['Sale Category', 'Precision', 'Recall', 'F1 Score', 'Specificity'].map((h, i) => (
            <div key={i} style={{ fontSize: 8, fontWeight: 800, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.07em', textAlign: i > 0 ? 'center' : 'left' }}>
              {h}
            </div>
          ))}
        </div>

        {/* Rows */}
        {perClass.map((m, ci) => (
          <div key={ci} style={{ display: 'grid', gridTemplateColumns: '120px repeat(4, 1fr)', padding: '9px 10px', gap: 4, borderBottom: '1px solid var(--border)' }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: CLASS_COLORS[ci] }}>{CLASS_NAMES[ci]}</div>
              <div style={{ fontSize: 8, color: 'var(--muted)', marginTop: 1 }}>
                {ci === 0 ? 'Low-value transactions' : ci === 1 ? 'Medium-value transactions' : 'High-value transactions'}
              </div>
            </div>
            {[
              { key: 'precision',   tip: 'When it predicts this class, how often is it right?' },
              { key: 'recall',      tip: 'Of all real instances, how many did it catch?' },
              { key: 'f1',          tip: 'Overall class score - balance of Precision and Recall' },
              { key: 'specificity', tip: 'Of non-instances, how many did it correctly exclude?' },
            ].map(({ key, tip }) => (
              <div key={key} style={{ textAlign: 'center' }}>
                <div className="mono" style={{ fontSize: 14, fontWeight: 700, color: CLASS_COLORS[ci] }}>
                  {pct(m[key])}
                </div>
                <Bar val={m[key]} color={CLASS_COLORS[ci]} h={3} />
                <div style={{ fontSize: 7, color: 'var(--muted)', marginTop: 2, lineHeight: 1.3 }}>{tip}</div>
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Metric glossary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 6, marginTop: 8 }}>
        {[
          ['Precision', '#38bdf8', 'When the model says "High Sale", how often is it correct? High precision = fewer false alarms raised to the sales team.'],
          ['Recall',    '#34d399', 'Of all real High Sale transactions, how many did the model catch? High recall = fewer missed opportunities.'],
          ['F1 Score',  '#f59e0b', 'Single score that balances Precision and Recall. Use this when you need one number to compare settings.'],
          ['Specificity', '#a78bfa', 'Of all transactions that are NOT High Sale, how many did the model correctly skip? Complements Recall.'],
        ].map(([title, color, text]) => (
          <div key={title} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderLeft: '3px solid ' + color, borderRadius: '0 7px 7px 0', padding: '7px 10px' }}>
            <div style={{ fontSize: 9, fontWeight: 700, color, marginBottom: 3 }}>{title}</div>
            <div style={{ fontSize: 8, color: 'var(--muted)', lineHeight: 1.6 }}>{text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
