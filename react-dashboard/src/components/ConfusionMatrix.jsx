import React from 'react';
import { CLASS_COLORS } from '../ml/models';

// Market-analyst-friendly class labels with dollar context
const classLabel = (idx, q1, q2) => {
  const labels = [
    { name:'Low-value',    range:'below $'  + q1.toFixed(0),        short:'Low'    },
    { name:'Medium-value', range:'$'+q1.toFixed(0)+' - $'+q2.toFixed(0), short:'Medium' },
    { name:'High-value',   range:'above $'  + q2.toFixed(0),        short:'High'   },
  ];
  return labels[idx];
};

/**
 * Redesigned confusion matrix for market analysts.
 * Rows = what actually happened, Cols = what the model predicted.
 */
export default function ConfusionMatrix({ matrix, q1 = 168, q2 = 384 }) {
  if (!matrix) return null;

  const total  = matrix.flat().reduce((s, v) => s + v, 0) || 1;
  const maxVal = Math.max(...matrix.flat());

  // Summary sentences per class
  const classSummaries = matrix.map((row, ai) => {
    const rowTotal  = row.reduce((s, v) => s + v, 0) || 1;
    const correct   = row[ai];
    const pct       = (correct / rowTotal * 100).toFixed(0);
    const cl        = classLabel(ai, q1, q2);
    return { cl, correct, total:rowTotal, pct };
  });

  return (
    <div>
      {/* Explanation header */}
      <div style={{ background:'var(--bg)', border:'1px solid var(--border)', borderRadius:8,
        padding:'9px 11px', marginBottom:12, fontSize:9, color:'var(--muted)', lineHeight:1.7 }}>
        <strong style={{color:'var(--text2)'}}>How to read this table:</strong> Each row is a real sales category.
        Each column is what the model predicted. Numbers on the{' '}
        <span style={{color:'#34d399',fontWeight:700}}>green diagonal</span> = the model was right.
        Numbers <span style={{color:'#f87171',fontWeight:700}}>off-diagonal</span> = the model confused two categories.
        Bigger numbers on the diagonal = better model.
      </div>

      {/* Column header label */}
      <div style={{ display:'flex', alignItems:'center', marginBottom:4 }}>
        <div style={{ width:130 }}/>
        <div style={{ flex:1, textAlign:'center', fontSize:9, color:'var(--muted)',
          fontWeight:700, letterSpacing:'.08em', textTransform:'uppercase' }}>
          What the Model Predicted
        </div>
      </div>

      {/* Column sub-headers */}
      <div style={{ display:'grid', gridTemplateColumns:'130px repeat(3,1fr)', gap:3, marginBottom:4 }}>
        <div style={{ display:'flex', alignItems:'flex-end', justifyContent:'flex-end',
          paddingRight:8, fontSize:9, color:'var(--muted)', fontWeight:700,
          textTransform:'uppercase', letterSpacing:'.06em', paddingBottom:2 }}>
          Actual
        </div>
        {[0,1,2].map((c) => {
          const cl = classLabel(c, q1, q2);
          return (
            <div key={c} style={{ textAlign:'center', padding:'5px 3px',
              background:CLASS_COLORS[c]+'14', borderRadius:'6px 6px 0 0',
              border:'1px solid '+CLASS_COLORS[c]+'33', borderBottom:'none' }}>
              <div style={{ fontSize:10, fontWeight:800, color:CLASS_COLORS[c] }}>{cl.short}</div>
              <div style={{ fontSize:8, color:'var(--muted)', marginTop:1 }}>{cl.range}</div>
            </div>
          );
        })}
      </div>

      {/* Matrix rows */}
      {matrix.map((row, ai) => {
        const cl = classLabel(ai, q1, q2);
        return (
          <div key={ai} style={{ display:'grid', gridTemplateColumns:'130px repeat(3,1fr)', gap:3, marginBottom:3 }}>
            {/* Row label */}
            <div style={{ display:'flex', flexDirection:'column', justifyContent:'center',
              alignItems:'flex-end', paddingRight:8, paddingLeft:4 }}>
              <div style={{ fontSize:10, fontWeight:800, color:CLASS_COLORS[ai] }}>{cl.short} sale</div>
              <div style={{ fontSize:8, color:'var(--muted)', marginTop:1 }}>{cl.range}</div>
            </div>

            {/* Cells */}
            {row.map((count, pi) => {
              const isDiag    = ai === pi;
              const intensity = count / (maxVal || 1);
              const bg = isDiag
                ? 'rgba(52,211,153,' + (0.1 + intensity * 0.5) + ')'
                : 'rgba(248,113,113,' + (intensity * 0.45) + ')';
              const pct = (count / total * 100).toFixed(1);

              return (
                <div key={pi} style={{
                  background: bg,
                  border: '1px solid ' + (isDiag ? CLASS_COLORS[ai]+'44' : 'var(--border)'),
                  borderRadius: 7, padding:'10px 6px', textAlign:'center',
                  transition:'background 0.3s',
                }}>
                  <div className="mono" style={{
                    fontSize:20, fontWeight:800,
                    color: isDiag ? CLASS_COLORS[ai] : (count > 0 ? '#f87171' : 'var(--muted)'),
                    lineHeight:1,
                  }}>
                    {count}
                  </div>
                  <div style={{ fontSize:8, color:'var(--muted)', marginTop:3 }}>
                    {pct}% of all
                  </div>
                  {isDiag && count > 0 && (
                    <div style={{ fontSize:7, color:'#34d399', marginTop:2, fontWeight:700 }}>
                      CORRECT
                    </div>
                  )}
                  {!isDiag && count > 0 && (
                    <div style={{ fontSize:7, color:'#f87171aa', marginTop:2 }}>
                      predicted as<br/>{classLabel(pi,q1,q2).short}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        );
      })}

      {/* Plain-English summary */}
      <div style={{ marginTop:12, display:'flex', flexDirection:'column', gap:5 }}>
        <div style={{ fontSize:9, fontWeight:700, color:'var(--text2)',
          textTransform:'uppercase', letterSpacing:'.08em', marginBottom:2 }}>
          Plain-English Summary
        </div>
        {classSummaries.map(({ cl, correct, total: rowTot, pct }, i) => (
          <div key={i} style={{ background:'var(--bg)', border:'1px solid var(--border)',
            borderLeft:'3px solid '+CLASS_COLORS[i], borderRadius:'0 6px 6px 0',
            padding:'7px 10px', fontSize:9, color:'var(--muted)', lineHeight:1.6 }}>
            For every <strong style={{color:CLASS_COLORS[i]}}>{cl.name} transaction</strong> (
            {cl.range}), the model correctly identified{' '}
            <strong style={{color:'#34d399'}}>{pct}%</strong> of them
            ({correct} out of {rowTot} transactions).
          </div>
        ))}
      </div>
    </div>
  );
}
