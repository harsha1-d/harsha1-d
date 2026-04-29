import React, { useState } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import { CLASS_NAMES, CLASS_COLORS } from '../ml/models';

const CLASS_LABELS = [
  'Low-value sale',
  'Medium-value sale',
  'High-value sale',
];

// Custom dot: filled = correct prediction, hollow ring = wrong prediction
const CustomDot = (props) => {
  const { cx, cy, fill, payload } = props;
  if (payload.correct) {
    return <circle cx={cx} cy={cy} r={3.5} fill={fill} fillOpacity={0.75} stroke="none"/>;
  }
  return (
    <g>
      <circle cx={cx} cy={cy} r={4.5} fill="none" stroke="#f87171" strokeWidth={1.5}/>
      <line x1={cx-3} y1={cy-3} x2={cx+3} y2={cy+3} stroke="#f87171" strokeWidth={1.2}/>
      <line x1={cx+3} y1={cy-3} x2={cx-3} y2={cy+3} stroke="#f87171" strokeWidth={1.2}/>
    </g>
  );
};

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div style={{ background:'#0c1525', border:'1px solid #1e3050', borderRadius:8,
      padding:'10px 13px', fontFamily:"'IBM Plex Mono',monospace", fontSize:10 }}>
      <div style={{ fontWeight:700, color:'white', marginBottom:6 }}>Transaction Detail</div>
      <div style={{ color:'#8ba3c7' }}>Unit Price: <span style={{color:'white'}}>${d.unitPrice}</span></div>
      <div style={{ color:'#8ba3c7' }}>Total Sale: <span style={{color:'white'}}>${d.sales}</span></div>
      <div style={{ color:'#8ba3c7' }}>Quantity: <span style={{color:'white'}}>{d.qty} items</span></div>
      <div style={{ color:'#8ba3c7', marginTop:4 }}>
        Actual class:
        <span style={{ color:CLASS_COLORS[d.actualClass], fontWeight:700, marginLeft:4 }}>
          {CLASS_NAMES[d.actualClass]} ({CLASS_LABELS[d.actualClass]})
        </span>
      </div>
      <div style={{ color:'#8ba3c7' }}>
        Model said:
        <span style={{ color: d.correct ? CLASS_COLORS[d.predictedClass] : '#f87171', fontWeight:700, marginLeft:4 }}>
          {CLASS_NAMES[d.predictedClass]} ({d.correct ? 'CORRECT' : 'WRONG'})
        </span>
      </div>
      <div style={{ color:'#8ba3c7' }}>Year: <span style={{color:'white'}}>{d.year}</span></div>
    </div>
  );
};

export default function ScatterPlot({ scatterData, q1, q2 }) {
  const [highlight, setHighlight] = useState('all'); // 'all' | 'correct' | 'wrong'

  if (!scatterData?.length) return (
    <div style={{ color:'var(--muted)', padding:20 }}>No scatter data available.</div>
  );

  // Split data by class AND correct/wrong
  const filtered = highlight === 'correct' ? scatterData.filter(d => d.correct)
    : highlight === 'wrong' ? scatterData.filter(d => !d.correct)
    : scatterData;

  const byClass = [0, 1, 2].map((c) => filtered.filter((d) => d.actualClass === c));
  const wrongCount   = scatterData.filter(d => !d.correct).length;
  const correctCount = scatterData.filter(d =>  d.correct).length;
  const accuracy     = (correctCount / scatterData.length * 100).toFixed(1);

  return (
    <div>
      {/* Context bar */}
      <div style={{ display:'flex', alignItems:'center', gap:16, marginBottom:10, flexWrap:'wrap' }}>
        <div style={{ fontSize:10, color:'var(--muted)' }}>
          Showing <span style={{color:'var(--text)',fontWeight:700}}>{scatterData.length}</span> sampled transactions.
          Each dot = one sale. <span style={{color:'#34d399'}}>Filled = model got it right</span>,{' '}
          <span style={{color:'#f87171'}}>X = model got it wrong</span>.
        </div>
        <div style={{ marginLeft:'auto', display:'flex', gap:6 }}>
          {[
            ['all',     'Show All',     'var(--border2)',  'var(--text2)'],
            ['correct', 'Correct Only', '#1a4030',         '#34d399'],
            ['wrong',   'Wrong Only',   '#2e1a1a',         '#f87171'],
          ].map(([val, lbl, bg, col]) => (
            <button key={val} onClick={() => setHighlight(val)} style={{
              padding:'4px 10px', borderRadius:5, fontSize:10, fontWeight:600,
              border:'1px solid ' + (highlight===val ? col : 'var(--border)'),
              background: highlight===val ? bg : 'transparent',
              color: highlight===val ? col : 'var(--muted)',
              cursor:'pointer',
            }}>{lbl}</button>
          ))}
        </div>
      </div>

      {/* Summary chips */}
      <div style={{ display:'flex', gap:10, marginBottom:8 }}>
        <div style={{ fontSize:11, padding:'4px 12px', borderRadius:6,
          background:'#0a2018', border:'1px solid #1a4030', color:'#34d399' }}>
          {correctCount} correct ({accuracy}%)
        </div>
        <div style={{ fontSize:11, padding:'4px 12px', borderRadius:6,
          background:'#2a1010', border:'1px solid #4a2020', color:'#f87171' }}>
          {wrongCount} wrong ({(100-accuracy).toFixed(1)}%)
        </div>
        {CLASS_NAMES.map((n, i) => (
          <div key={n} style={{ fontSize:11, padding:'4px 12px', borderRadius:6,
            background:CLASS_COLORS[i]+'18', border:'1px solid '+CLASS_COLORS[i]+'44',
            color:CLASS_COLORS[i] }}>
            {n}: {scatterData.filter(d=>d.actualClass===i).length}
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top:6, right:16, left:0, bottom:16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis type="number" dataKey="unitPrice" name="Unit Price" domain={[0,110]}
            tick={{ fontSize:9, fill:'var(--muted)' }} axisLine={false} tickLine={false}
            tickFormatter={(v) => '$'+v}
            label={{ value:'Unit Price (per item)', position:'insideBottom', offset:-10, fill:'var(--muted)', fontSize:9 }}/>
          <YAxis type="number" dataKey="sales" name="Total Sale" domain={[0,1100]}
            tick={{ fontSize:9, fill:'var(--muted)', fontFamily:"'IBM Plex Mono'" }}
            axisLine={false} tickLine={false} tickFormatter={(v) => '$'+v} width={52}
            label={{ value:'Total Sale Value', angle:-90, position:'insideLeft', offset:14, fill:'var(--muted)', fontSize:9 }}/>
          <ZAxis range={[22,22]}/>
          <Tooltip content={<CustomTooltip />}/>
          <Legend
            formatter={(value) => <span style={{fontSize:10,color:'var(--text2)'}}>{value}</span>}
            wrapperStyle={{ paddingTop:8 }}/>

          {/* Horizontal reference lines at class boundaries */}
          <ReferenceLine y={q1} stroke="#38bdf850" strokeDasharray="6 3"
            label={{ value:'Low / Med: $'+q1.toFixed(0), position:'right', fill:'#38bdf8', fontSize:8 }}/>
          <ReferenceLine y={q2} stroke="#f59e0b50" strokeDasharray="6 3"
            label={{ value:'Med / High: $'+q2.toFixed(0), position:'right', fill:'#f59e0b', fontSize:8 }}/>

          {byClass.map((data, c) => (
            <Scatter
              key={c}
              name={CLASS_NAMES[c] + ' (' + CLASS_LABELS[c] + ')'}
              data={data}
              fill={CLASS_COLORS[c]}
              shape={<CustomDot />}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>

      <div style={{ fontSize:9, color:'var(--muted)', marginTop:4, lineHeight:1.6 }}>
        <strong style={{color:'var(--text2)'}}>How to read this:</strong>{' '}
        Each dot is a real transaction from 2021-2025. The colour shows what sales category it actually belonged to.
        A filled dot means the Decision Tree correctly predicted that category.
        An X dot means it predicted the wrong category.
        The dashed lines show where the model draws the boundary between Low, Medium, and High sales.
        If you see many X dots clustered near a boundary line, that is where the model struggles most.
      </div>
    </div>
  );
}
