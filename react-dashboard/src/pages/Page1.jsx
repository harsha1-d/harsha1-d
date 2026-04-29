import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, BarChart, Bar, Line, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceDot,
} from 'recharts';
import TreeViz        from '../components/TreeViz';
import MetricsReport  from '../components/MetricsReport';
import ConfusionMatrix from '../components/ConfusionMatrix';
import { computeDepthCurve } from '../ml/engine';
import { CLASS_NAMES, CLASS_COLORS } from '../ml/models';
import { TEST } from '../data/dataset';

const TT = {
  contentStyle: { background: '#0c1525', border: '1px solid #1e3050', borderRadius: 8, fontFamily: "'IBM Plex Mono',monospace", fontSize: 10, padding: '8px 12px' },
  labelStyle: { color: '#e4eeff', fontWeight: 700, marginBottom: 4 },
};

function buildClassHistogram(testPreds, q1, q2, classIdx) {
  const salesForClass = TEST.map((r, i) => ({ sales: r[9], pred: testPreds[i] }))
    .filter(d => d.pred === classIdx)
    .map(d => d.sales);
  if (!salesForClass.length) return [];
  const lo = Math.min(...salesForClass), hi = Math.max(...salesForClass);
  const binCount = 12;
  const step = (hi - lo) / binCount || 1;
  const bins = Array.from({ length: binCount }, (_, i) => ({
    range: '$' + Math.round(lo + i * step) + '-$' + Math.round(lo + (i + 1) * step),
    lo: lo + i * step,
    hi: lo + (i + 1) * step,
    count: 0, correct: 0,
  }));
  salesForClass.forEach((s, idx) => {
    const bi = Math.min(Math.floor((s - lo) / step), binCount - 1);
    bins[bi].count++;
    const actualClass = s <= q1 ? 0 : s <= q2 ? 1 : 2;
    if (actualClass === classIdx) bins[bi].correct++;
  });
  return bins.filter(b => b.count > 0).map(b => ({
    ...b, accuracy: b.count ? Math.round(b.correct / b.count * 100) : 0,
  }));
}

export default function Page1({ params, result }) {
  const [displayDepth, setDisplayDepth] = useState(3);

  const depthCurve = useMemo(
    () => computeDepthCurve(params),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [params.minSamplesSplit, params.classWeights, params.lowBoundaryPct,
     params.highBoundaryPct, params.activeFeatures]
  );

  const classHistograms = useMemo(() =>
    [0, 1, 2].map(ci => buildClassHistogram(result.testPreds, result.q1, result.q2, ci)),
    [result.testPreds, result.q1, result.q2]
  );

  const observations = useMemo(() => {
    const { metrics, trainAcc, confMatrix } = result;
    const obs = [];
    const gap = trainAcc - metrics.accuracy * 100;
    const cNames = ['Low-value', 'Medium-value', 'High-value'];
    if (gap > 15) obs.push({ type: 'warn', icon: '!', title: 'Overfitting detected',
      text: 'Train accuracy is ' + gap.toFixed(0) + 'pts above test. Reduce Max Depth or increase Min Samples to Split in the right panel.' });
    else obs.push({ type: 'good', icon: 'OK', title: 'Model is healthy',
      text: 'Train-test gap is only ' + gap.toFixed(1) + 'pts. The model generalises well to unseen transactions.' });
    const minC = result.perClassAcc.indexOf(Math.min(...result.perClassAcc));
    const maxC = result.perClassAcc.indexOf(Math.max(...result.perClassAcc));
    if (result.perClassAcc[minC] < result.perClassAcc[maxC] - 18)
      obs.push({ type: 'warn', icon: '!', title: 'Class imbalance',
        text: cNames[minC] + ' accuracy is weakest at ' + result.perClassAcc[minC] + '%. Increase the ' + cNames[minC].split('-')[0] + ' class weight in the right panel.' });
    let maxErr = 0, errPair = [0, 1];
    confMatrix.forEach((row, ai) => row.forEach((v, pi) => { if (ai !== pi && v > maxErr) { maxErr = v; errPair = [ai, pi]; } }));
    if (maxErr > 0) obs.push({ type: 'info', icon: 'i', title: 'Top confusion',
      text: cNames[errPair[0]] + ' vs ' + cNames[errPair[1]] + ': mixed up ' + maxErr + ' times. Adjust class boundaries to separate them.' });
    return obs;
  }, [result]);

  const obsColor = { good: '#34d399', warn: '#fbbf24', info: '#38bdf8' };
  const classDesc = [
    'Transactions the model labelled LOW SALE (below $' + Math.round(result.q1) + ')',
    'Transactions labelled MEDIUM SALE ($' + Math.round(result.q1) + ' - $' + Math.round(result.q2) + ')',
    'Transactions labelled HIGH SALE (above $' + Math.round(result.q2) + ')',
  ];

  return (
    <div className="page-anim">

      {/* Health banners */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(' + observations.length + ', 1fr)', gap: 8, marginBottom: 10 }}>
        {observations.map((obs, i) => (
          <div key={i} style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderLeft: '4px solid ' + obsColor[obs.type], borderRadius: '0 8px 8px 0', padding: '9px 12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
              <div style={{ width: 18, height: 18, borderRadius: '50%', background: obsColor[obs.type] + '22', border: '1px solid ' + obsColor[obs.type], display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 8, fontWeight: 800, color: obsColor[obs.type] }}>
                {obs.icon}
              </div>
              <span style={{ fontSize: 10, fontWeight: 700, color: obsColor[obs.type] }}>{obs.title}</span>
            </div>
            <div style={{ fontSize: 9, color: 'var(--muted)', lineHeight: 1.65 }}>{obs.text}</div>
          </div>
        ))}
      </div>

      {/* Tree card  -  histograms sit INSIDE this card, above the SVG */}
      <div className="card" style={{ marginBottom: 10 }}>

        {/* Card header */}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 10 }}>
          <div>
            <div className="card-title">Decision Tree - How the Model Decides Each Sale Category</div>
            <div className="card-sub" style={{ marginBottom: 0 }}>
              Trained on <strong style={{ color: '#38bdf8' }}>865 transactions (2019-2020)</strong>.
              Tested on <strong style={{ color: '#34d399' }}>2,135 transactions (2021-2025)</strong>.
              &nbsp;| Depth: {result.maxActualDepth} | Nodes: {result.nNodes} | Leaves: {result.nLeaves}
              &nbsp;| <span style={{ color: 'var(--c0)', fontWeight: 600 }}>Blue-outlined = KEY DRIVER nodes</span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0, marginLeft: 14 }}>
            <span style={{ fontSize: 9, color: 'var(--muted)' }}>Levels:</span>
            {[1, 2, 3, 4, 5].map(d => (
              <button key={d} onClick={() => setDisplayDepth(d)} style={{
                width: 28, height: 28, borderRadius: 6, border: '1px solid',
                borderColor: displayDepth === d ? 'var(--c0)' : 'var(--border)',
                background: displayDepth === d ? 'color-mix(in srgb,var(--c0) 14%,var(--bg))' : 'var(--bg)',
                color: displayDepth === d ? 'var(--c0)' : 'var(--muted)',
                fontSize: 11, fontWeight: 700, cursor: 'pointer',
              }}>{d}</button>
            ))}
          </div>
        </div>

        {/* 3 CLASS HISTOGRAMS  -  replace the 4 path cards */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text2)', marginBottom: 3 }}>
            How the Model Distributes Predictions Across Sale Values
          </div>
          <div style={{ fontSize: 9, color: 'var(--muted)', marginBottom: 8, lineHeight: 1.5 }}>
            Each histogram shows the actual sale amounts of transactions the model assigned to that class.
            <strong style={{ color: 'var(--text2)', marginLeft: 4 }}>Green = correctly classified. Red = mislabelled transactions in that bin.</strong>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
            {[0, 1, 2].map(ci => {
              const data = classHistograms[ci];
              const color = CLASS_COLORS[ci];
              const total = data.reduce((s, b) => s + b.count, 0);
              const correct = data.reduce((s, b) => s + b.correct, 0);
              const precision = total ? Math.round(correct / total * 100) : 0;
              return (
                <div key={ci} style={{ background: 'var(--bg)', border: '1px solid ' + color + '44', borderRadius: 8, padding: '10px 12px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
                    <div>
                      <div style={{ fontSize: 11, fontWeight: 800, color }}>{CLASS_NAMES[ci].toUpperCase()}</div>
                      <div style={{ fontSize: 8, color: 'var(--muted)', marginTop: 1 }}>{classDesc[ci]}</div>
                    </div>
                    <div style={{ textAlign: 'right', flexShrink: 0 }}>
                      <div className="mono" style={{ fontSize: 16, fontWeight: 700, color, lineHeight: 1 }}>{precision}%</div>
                      <div style={{ fontSize: 7, color: 'var(--muted)' }}>precision</div>
                    </div>
                  </div>
                  <div style={{ fontSize: 7, color: 'var(--muted)', marginBottom: 4, lineHeight: 1.4 }}>
                    {ci === 0 && 'Correct zone: sales below $' + Math.round(result.q1)}
                    {ci === 1 && 'Correct zone: $' + Math.round(result.q1) + ' to $' + Math.round(result.q2)}
                    {ci === 2 && 'Correct zone: sales above $' + Math.round(result.q2)}
                    {' | Bars outside zone = misclassified'}
                  </div>
                  <ResponsiveContainer width="100%" height={130}>
                    <BarChart data={data} margin={{ top: 2, right: 4, left: -18, bottom: 0 }} barCategoryGap="8%">
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis dataKey="range" tick={false} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 7, fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
                      <Tooltip
                        contentStyle={TT.contentStyle}
                        formatter={(v, _, entry) => [v + ' txns (' + entry.payload.accuracy + '% correct)', entry.payload.range]}
                      />
                      <Bar dataKey="correct" name="Correctly classified" stackId="a" fill={color} fillOpacity={0.9} />
                      <Bar dataKey={(d) => d.count - d.correct} name="Misclassified" stackId="a" fill="#f87171" fillOpacity={0.5} radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 8 }}>
                      <div style={{ width: 10, height: 5, background: color, borderRadius: 1 }} />
                      <span style={{ color: 'var(--muted)' }}>Correct</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 8 }}>
                      <div style={{ width: 10, height: 5, background: '#f87171', opacity: 0.6, borderRadius: 1 }} />
                      <span style={{ color: 'var(--muted)' }}>Misclassified</span>
                    </div>
                    <span className="mono" style={{ fontSize: 8, color: 'var(--muted)', marginLeft: 'auto' }}>{total} predictions</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* SVG Decision Tree */}
        <TreeViz root={result.root} displayDepth={displayDepth} q1={result.q1} q2={result.q2} />
      </div>

      {/* Confusion matrix + Metrics report */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: 10, marginBottom: 10 }}>
        <div className="card">
          <div className="card-title">Did the Model Predict the Right Sale Category?</div>
          <div className="card-sub">
            Every test transaction (2021-2025) falls into one cell.
            Cells on the <strong style={{ color: '#34d399' }}>green diagonal</strong> = model was correct.
            <strong style={{ color: '#f87171' }}> Red off-diagonal</strong> = model confused two categories.
            Rows = what actually happened; Columns = what the model guessed.
          </div>
          <ConfusionMatrix matrix={result.confMatrix} q1={result.q1} q2={result.q2} />
        </div>
        <div className="card">
          <div className="card-title">Model Performance Report</div>
          <div className="card-sub">
            Precision, Recall, F1-Score and Specificity per sale category.
            Tested on 2,135 transactions from 2021-2025.
            <span style={{ color: 'var(--c0)', marginLeft: 4 }}>
              95% CI: [{result.metrics?.ciLo.toFixed(1)}% - {result.metrics?.ciHi.toFixed(1)}%]
            </span>
          </div>
          <MetricsReport metrics={result.metrics} trainAcc={result.trainAcc} />
        </div>
      </div>

      {/* Depth vs accuracy */}
      <div className="card">
        <div className="card-title">Does a Deeper Tree Mean Better Accuracy?</div>
        <div className="card-sub">
          <span style={{ color: '#38bdf8', fontWeight: 600 }}>Training accuracy</span> always rises with depth.
          But <span style={{ color: '#34d399', fontWeight: 600 }}>test accuracy</span> peaks then flattens or drops.
          That drop = overfitting. The <span style={{ color: 'var(--c0)' }}>blue dot</span> marks your current setting.
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 14, alignItems: 'center' }}>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={depthCurve} margin={{ top: 8, right: 24, left: 0, bottom: 28 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="depth" tick={{ fontSize: 10, fill: 'var(--muted)' }} axisLine={false} tickLine={false}
                label={{ value: 'Tree Depth (Max Levels Allowed)', position: 'insideBottom', offset: -14, fill: 'var(--muted)', fontSize: 9 }} />
              <YAxis domain={[40, 100]} tick={{ fontSize: 9, fill: 'var(--muted)', fontFamily: "'IBM Plex Mono'" }}
                axisLine={false} tickLine={false} tickFormatter={v => v + '%'} width={38} />
              <Tooltip {...TT} formatter={(v, n) => [v + '%', n]} />
              <Legend wrapperStyle={{ fontSize: 10, paddingTop: 8 }} />
              {depthCurve.length > 0 && (
                <ReferenceDot x={params.maxDepth} y={depthCurve.find(d => d.depth === params.maxDepth)?.testAcc || 0}
                  r={7} fill="var(--c0)" stroke="var(--bg)" strokeWidth={2}
                  label={{ value: 'Current', position: 'top', fill: 'var(--c0)', fontSize: 8 }} />
              )}
              <Line type="monotone" dataKey="trainAcc" name="Training Accuracy" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3, fill: '#38bdf8' }} />
              <Line type="monotone" dataKey="testAcc" name="Test Accuracy (new data)" stroke="#34d399" strokeWidth={2.5} dot={{ r: 3, fill: '#34d399' }} />
            </LineChart>
          </ResponsiveContainer>
          {depthCurve.length > 0 && (() => {
            const best = depthCurve.reduce((b, d) => d.testAcc > b.testAcc ? d : b, depthCurve[0]);
            const isCur = params.maxDepth === best.depth;
            return (
              <div style={{ width: 180, padding: '10px 12px', background: 'var(--bg)', border: '1px solid var(--border)', borderLeft: '3px solid #34d399', borderRadius: '0 8px 8px 0', fontSize: 9, color: 'var(--muted)', lineHeight: 1.7 }}>
                <strong style={{ color: '#34d399', fontSize: 10 }}>Best depth: {best.depth}</strong><br />
                Test accuracy: {best.testAcc}%<br />
                {isCur
                  ? <span style={{ color: '#34d399' }}>Your setting is optimal.</span>
                  : <span>Set Max Depth to <strong style={{ color: 'var(--c0)' }}>{best.depth}</strong> for best results.</span>}
              </div>
            );
          })()}
        </div>
      </div>

    </div>
  );
}
