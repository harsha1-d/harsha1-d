import React, { useMemo } from 'react';
import { FEATURE_META, CLASS_NAMES, CLASS_COLORS, businessQuestion, getFeatures } from '../ml/models';

// -- Walk tree for a single row, collecting the decision path --
function traceTree(root, features) {
  const path = [];
  let node = root;
  while (node && !node.isLeaf) {
    const bq = businessQuestion(node.feat, node.thresh);

    // Physical navigation: left branch = feature <= threshold
    const physicalGoLeft = features[node.feat] <= node.thresh;

    // For "above / more-than" questions (Unit Price feat 7, Quantity feat 8):
    //   left  (<=)  = NO   -  value does NOT exceed the threshold
    //   right (>)   = YES  -  value DOES exceed the threshold
    // For all other features the left branch IS the YES answer.
    const isAboveQuestion = node.feat === 7 || node.feat === 8;
    const answeredYes = isAboveQuestion ? !physicalGoLeft : physicalGoLeft;

    path.push({ bq, answeredYes, nSamples: node.nSamples, infoGain: node.infoGain });

    // Navigate physically (always left = <=, right = >)
    node = physicalGoLeft ? node.left : node.right;
  }
  return { path, leaf: node };
}

export default function TweakPanel({ params, setParams, result, wiQty, setWiQty, wiPrice, setWiPrice }) {
  const upd       = (key, val) => setParams((p) => ({ ...p, [key]: val }));
  const updWeight = (idx, val) =>
    setParams((p) => { const w = [...p.classWeights]; w[idx] = +val; return { ...p, classWeights: w }; });
  const togFeat   = (idx) =>
    setParams((p) => {
      const next = p.activeFeatures.includes(idx)
        ? p.activeFeatures.filter((x) => x !== idx)
        : [...p.activeFeatures, idx];
      return next.length >= 1 ? { ...p, activeFeatures: next } : p;
    });
  const selectTopN = (n) => {
    if (!result) return;
    const sorted = result.fi.map((imp, i) => ({ imp, i }))
      .sort((a, b) => b.imp - a.imp).slice(0, n).map((x) => x.i);
    setParams((p) => ({ ...p, activeFeatures: sorted }));
  };

  // wiQty and wiPrice come from App.jsx props (shared with TreeViz for path highlighting)

  // -- Build a synthetic row from what-if inputs and predict --
  const wiPrediction = useMemo(() => {
    if (!result || !result.root) return null;
    // Build a synthetic row matching the dataset format:
    // [year, month, branch, custType, gender, productLine, payment, unitPrice, qty, sales, grossIncome, rating]
    // We use 2023 (mid-prediction period) as year, June as month, rating 7 as neutral
    const syntheticRow = [2023, 6, 1, 0, 0, 2, 2, wiPrice, wiQty, 0, 0, 7.0]; // branch=Cairo, Member, Female, Food&Bev, Ewallet (neutral defaults)
    const features = getFeatures(syntheticRow);
    const { path, leaf } = traceTree(result.root, features);
    const predictedClass  = leaf ? leaf.majority : 1;
    const predictedSale   = result.classMeans ? result.classMeans[predictedClass] : 0;
    const actualSale      = wiPrice * wiQty * 1.05;
    const purity          = leaf ? Math.round(leaf.classDist[leaf.majority] / leaf.classDist.reduce((s,v)=>s+v,0) * 100) : 0;
    return { path, leaf, predictedClass, predictedSale, actualSale, purity, features };
  }, [result, wiQty, wiPrice]);

  // -- Shared sub-components --
  const Slider = ({ label, helpText, min, max, step = 1, value, onChange, color = 'var(--c0)', unit = '' }) => (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text2)' }}>{label}</span>
        <span className="mono" style={{ fontSize: 11, fontWeight: 700, color }}>{value}{unit}</span>
      </div>
      {helpText && <div style={{ fontSize: 8, color: 'var(--muted)', marginBottom: 5, lineHeight: 1.5 }}>{helpText}</div>}
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(+e.target.value)} style={{ width: '100%', accentColor: color }}/>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 7, color: 'var(--muted)', marginTop: 1 }}>
        <span>{min}{unit}</span><span>{max}{unit}</span>
      </div>
    </div>
  );

  const Section = ({ title, hint, children }) => (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 8, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '.12em',
        color: 'var(--muted)', borderBottom: '1px solid var(--border)', paddingBottom: 5, marginBottom: 8, marginTop: 14 }}>
        {title}
      </div>
      {hint && <div style={{ fontSize: 8, color: 'var(--muted)', marginBottom: 9, lineHeight: 1.5,
        padding: '5px 7px', background: 'var(--bg)', borderRadius: 5, borderLeft: '2px solid var(--border2)' }}>
        {hint}
      </div>}
      {children}
    </div>
  );

  const mc = wiPrediction ? CLASS_COLORS[wiPrediction.predictedClass] : 'var(--muted)';

  return (
    <aside className="fp" style={{ width: 232 }}>

      {/* Title */}
      <div style={{ fontSize: 12, fontWeight: 800, color: 'var(--text)', marginBottom: 3 }}>Model Controls</div>
      <div style={{ fontSize: 9, color: 'var(--muted)', marginBottom: 10, lineHeight: 1.5 }}>
        Adjust any control below to re-train the model instantly. Watch how the scatter plot and accuracy change.
      </div>

      {/* Live metrics */}
      {result && (
        <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 11px', marginBottom: 4 }}>
          <div style={{ fontSize: 8, fontWeight: 700, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.1em', marginBottom: 8 }}>
            Current Model Performance
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5, marginBottom: 8 }}>
            {[
              ['Training Score', result.trainAcc + '%', '#38bdf8', 'How well it learned from 2019-2020 data'],
              ['Test Score',     result.testAcc  + '%', '#34d399', 'How well it predicts 2021-2025 data'],
              ['Decision Points', result.nNodes,         'var(--text2)', 'Total decision branches in the tree'],
              ['Final Answers',   result.nLeaves,        'var(--text2)', 'Number of outcome leaf nodes'],
            ].map(([lbl, val, col, tip]) => (
              <div key={lbl} title={tip} style={{ background: 'var(--surface)', borderRadius: 6, padding: '6px 8px', cursor: 'help' }}>
                <div style={{ fontSize: 8, color: 'var(--muted)', lineHeight: 1.3 }}>{lbl}</div>
                <div className="mono" style={{ fontSize: 14, fontWeight: 700, color: col, marginTop: 2 }}>{val}</div>
              </div>
            ))}
          </div>
          {CLASS_NAMES.map((n, i) => (
            <div key={n} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: CLASS_COLORS[i], flexShrink: 0 }}/>
              <span style={{ fontSize: 9, color: 'var(--muted)', flex: 1 }}>{n} sales</span>
              <span className="mono" style={{ fontSize: 9, color: CLASS_COLORS[i], width: 32, textAlign: 'right' }}>{result.perClassAcc[i]}%</span>
              <div style={{ width: 52, height: 4, background: 'var(--border)', borderRadius: 2 }}>
                <div style={{ height: '100%', width: result.perClassAcc[i] + '%', background: CLASS_COLORS[i], borderRadius: 2, transition: 'width .4s' }}/>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ======================================
          WHAT-IF PREDICTOR
      ====================================== */}
      <Section
        title="What-If Transaction Predictor"
        hint="Type in items and price to see what the current model would predict for that transaction  -  and exactly why.">

        {/* Qty + Price number inputs (prominent) */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10 }}>
          <div>
            <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--text2)', marginBottom: 4 }}>
              No. of Items
            </div>
            <input
              type="number" min={1} max={10} step={0.5} value={wiQty}
              onChange={(e) => setWiQty(Math.max(1, Math.min(10, +e.target.value)))}
              style={{
                width: '100%', background: 'var(--bg)', border: '2px solid var(--c0)',
                borderRadius: 7, color: 'var(--text)', fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 18, fontWeight: 700, padding: '7px 10px', outline: 'none',
                textAlign: 'center',
              }}
            />
            <div style={{ fontSize: 8, color: 'var(--muted)', marginTop: 3, textAlign: 'center' }}>items (1-10)</div>
          </div>
          <div>
            <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--text2)', marginBottom: 4 }}>
              Unit Price ($)
            </div>
            <input
              type="number" min={10} max={100} step={1} value={wiPrice}
              onChange={(e) => setWiPrice(Math.max(10, Math.min(100, +e.target.value)))}
              style={{
                width: '100%', background: 'var(--bg)', border: '2px solid var(--c0)',
                borderRadius: 7, color: 'var(--text)', fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 18, fontWeight: 700, padding: '7px 10px', outline: 'none',
                textAlign: 'center',
              }}
            />
            <div style={{ fontSize: 8, color: 'var(--muted)', marginTop: 3, textAlign: 'center' }}>dollars (10-100)</div>
          </div>
        </div>


        {/* Computed sale preview */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 7, padding: '6px 10px', marginBottom: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9 }}>
            <span style={{ color: 'var(--muted)' }}>Transaction total</span>
            <span className="mono" style={{ color: 'var(--text)', fontWeight: 700 }}>
              ${wiPrediction ? wiPrediction.actualSale.toFixed(2) : (wiQty * wiPrice * 1.05).toFixed(2)}
            </span>
          </div>
          <div style={{ fontSize: 8, color: 'var(--muted)', marginTop: 2 }}>
            ({wiQty} items × ${wiPrice} × 1.05 tax)
          </div>
        </div>

        {/* PREDICTION RESULT */}
        {wiPrediction && (
          <div style={{
            background: mc + '18',
            border: '2px solid ' + mc,
            borderRadius: 10, padding: '12px 12px 10px',
            marginBottom: 10,
          }}>
            {/* Predicted class */}
            <div style={{ fontSize: 9, fontWeight: 700, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.1em', marginBottom: 6 }}>
              Model Prediction
            </div>
            <div style={{ fontSize: 20, fontWeight: 800, color: mc, lineHeight: 1, marginBottom: 4 }}>
              {CLASS_NAMES[wiPrediction.predictedClass].toUpperCase()}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text2)', marginBottom: 6 }}>
              {wiPrediction.predictedClass === 0 && `Below $${result?.q1.toFixed(0)}`}
              {wiPrediction.predictedClass === 1 && `$${result?.q1.toFixed(0)} - $${result?.q2.toFixed(0)}`}
              {wiPrediction.predictedClass === 2 && `Above $${result?.q2.toFixed(0)}`}
              {' '}per transaction
            </div>

            {/* Confidence bar */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
              <div style={{ flex: 1, height: 5, background: 'var(--border)', borderRadius: 3 }}>
                <div style={{ height: '100%', width: wiPrediction.purity + '%', background: mc, borderRadius: 3, transition: 'width .4s' }}/>
              </div>
              <span className="mono" style={{ fontSize: 10, fontWeight: 700, color: mc }}>
                {wiPrediction.purity}%
              </span>
            </div>
            <div style={{ fontSize: 8, color: mc }}>
              {wiPrediction.purity}% of transactions on this exact path are truly {CLASS_NAMES[wiPrediction.predictedClass]}
            </div>

            {/* Decision path trace */}
            <div style={{ marginTop: 10 }}>
              <div style={{ fontSize: 8, fontWeight: 700, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 6 }}>
                How the tree decided  -  step by step
              </div>
              {wiPrediction.path.length === 0 && (
                <div style={{ fontSize: 9, color: 'var(--muted)', fontStyle: 'italic' }}>
                  Prediction made at root node (depth 0 tree).
                </div>
              )}
              {wiPrediction.path.map((step, i) => (
                <div key={i} style={{
                  display: 'flex', gap: 6, alignItems: 'flex-start',
                  marginBottom: 6, padding: '5px 7px',
                  background: step.answeredYes ? '#34d39914' : '#f8717114',
                  border: '1px solid ' + (step.answeredYes ? '#34d39944' : '#f8717144'),
                  borderRadius: 6,
                }}>
                  {/* Step number */}
                  <div style={{
                    width: 16, height: 16, borderRadius: '50%', flexShrink: 0,
                    background: step.answeredYes ? '#34d399' : '#f87171',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 8, fontWeight: 800, color: 'white', marginTop: 1,
                  }}>
                    {i + 1}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    {/* Column tag */}
                    <div style={{ fontSize: 7, color: 'var(--muted)', marginBottom: 1, textTransform: 'uppercase', letterSpacing: '.07em' }}>
                      {step.bq.col}
                    </div>
                    {/* Question */}
                    <div style={{ fontSize: 9, fontWeight: 600, color: 'var(--text)', lineHeight: 1.3, marginBottom: 2 }}>
                      {step.bq.question}
                    </div>
                    {/* Answer */}
                    <div style={{
                      fontSize: 9, fontWeight: 700,
                      color: step.answeredYes ? '#34d399' : '#f87171',
                    }}>
                      {step.answeredYes
                        ? '-> YES  -  ' + step.bq.yesLabel.replace(/^YES\s*[-]\s*/,'')
                        : '-> NO  -  '  + step.bq.noLabel.replace(/^NO\s*[-]\s*/,'')}
                    </div>
                  </div>
                </div>
              ))}

              {/* Final leaf summary */}
              {wiPrediction.leaf && (
                <div style={{
                  marginTop: 4, padding: '7px 10px',
                  background: mc + '22', border: '1.5px solid ' + mc,
                  borderRadius: 7, textAlign: 'center',
                }}>
                  <div style={{ fontSize: 8, color: mc, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 2 }}>
                    Reaches leaf
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 800, color: mc }}>
                    {CLASS_NAMES[wiPrediction.predictedClass]}
                  </div>
                  <div style={{ fontSize: 9, color: 'var(--text2)', marginTop: 2 }}>
                    {wiPrediction.leaf.nSamples} training transactions in this group
                  </div>
                  {/* Class mix bar */}
                  <div style={{ display: 'flex', height: 5, borderRadius: 3, overflow: 'hidden', marginTop: 5 }}>
                    {[0,1,2].map(c => {
                      const tot = wiPrediction.leaf.classDist.reduce((s,v)=>s+v,0)||1;
                      const pct = wiPrediction.leaf.classDist[c]/tot*100;
                      return pct > 0 ? (
                        <div key={c} style={{ width: pct+'%', background: CLASS_COLORS[c], transition: 'width .4s' }}/>
                      ) : null;
                    })}
                  </div>
                  <div style={{ fontSize: 7, color: 'var(--muted)', marginTop: 3 }}>
                    Class mix: {wiPrediction.leaf.classDist.map((n,i)=>CLASS_NAMES[i].split(' ')[0]+' '+n).join(' | ')}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </Section>

      {/* ======================================
          TREE COMPLEXITY
      ====================================== */}
      <Section title="Tree Complexity" hint="These two controls determine how deep and detailed the model's decision process gets.">
        <Slider
          label="How many decisions deep"
          helpText="Each level is one question the model asks (e.g. 'Is unit price over $55?'). More levels = smarter but risks memorising the data."
          min={1} max={8} value={params.maxDepth}
          onChange={(v) => upd('maxDepth', v)} color="var(--c0)"/>
        <Slider
          label="Minimum transactions per branch"
          helpText="A group must have at least this many transactions before the model splits it further. Higher = simpler, more general model."
          min={2} max={60} value={params.minSamplesSplit}
          onChange={(v) => upd('minSamplesSplit', v)} color="var(--c0)"/>

      </Section>



      {/* ======================================
          BIAS REDUCTION
      ====================================== */}
      <Section title="Bias Reduction (Class Weights)" hint="If the model ignores a sales category, increase its weight. This forces the model to try harder on that category.">
        {CLASS_NAMES.map((n, i) => (
          <Slider key={n}
            label={n + ' sales sensitivity'}
            helpText={i === 0 ? 'Increase if the model keeps under-predicting low-value transactions.'
              : i === 1 ? 'Increase if medium-value predictions are often wrong.'
              : 'Increase if the model misses high-value transactions (most costly errors).'}
            min={0.5} max={4} step={0.1}
            value={params.classWeights[i]}
            onChange={(v) => updWeight(i, v)}
            color={CLASS_COLORS[i]}/>
        ))}
      </Section>

      {/* ======================================
          FEATURE SELECTION
      ====================================== */}
      <Section title="Transaction Attributes to Use" hint="Choose which transaction details the model is allowed to consider when making decisions.">
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 8, color: 'var(--muted)', marginBottom: 4 }}>Quick-select top attributes by importance:</div>
          <div style={{ display: 'flex', gap: 4 }}>
            {[3, 5, 7, 10].map((n) => (
              <button key={n} onClick={() => selectTopN(n)} style={{
                flex: 1, padding: '3px 0', borderRadius: 4, fontSize: 9, fontWeight: 700,
                border: '1px solid var(--border)', background: 'var(--bg)',
                color: 'var(--muted)', cursor: 'pointer',
              }}>
                Top {n}
              </button>
            ))}
          </div>
        </div>

        {FEATURE_META.map((fm, idx) => {
          const active = params.activeFeatures.includes(idx);
          const fi = result ? result.fi[idx] : 0;
          const isTop = result && fi >= (result.fi.slice().sort((a,b)=>b-a)[2] || 0);
          return (
            <div key={idx} onClick={() => togFeat(idx)} style={{
              display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5,
              cursor: 'pointer', padding: '5px 7px', borderRadius: 6,
              background: active ? (isTop ? CLASS_COLORS[2]+'08' : 'var(--bg)') : 'transparent',
              border: '1px solid ' + (active ? (isTop ? CLASS_COLORS[2]+'44' : 'var(--border2)') : 'var(--border)'),
              opacity: active ? 1 : 0.5, transition: 'all .15s',
            }}>
              <div style={{ width: 10, height: 10, borderRadius: 3, flexShrink: 0,
                background: active ? 'var(--green)' : 'var(--muted)',
                border: '1px solid ' + (active ? 'var(--green)' : 'var(--border)') }}/>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 9, color: active ? 'var(--text2)' : 'var(--muted)',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {fm.name}
                </div>
                {result && (
                  <div style={{ height: 2, background: 'var(--border)', borderRadius: 1, marginTop: 2 }}>
                    <div style={{ height: '100%', width: fi + '%', background: 'var(--c0)', borderRadius: 1, transition: 'width .4s' }}/>
                  </div>
                )}
              </div>
              {result && (
                <span className="mono" style={{ fontSize: 8, color: fi > 5 ? 'var(--c0)' : 'var(--muted)', width: 28, textAlign: 'right', flexShrink: 0 }}>
                  {fi.toFixed(1)}%
                </span>
              )}
            </div>
          );
        })}
      </Section>

      <div style={{ marginTop: 8, padding: '7px 9px', background: 'var(--bg)', borderRadius: 6,
        border: '1px solid var(--border)', fontSize: 8, color: 'var(--muted)', lineHeight: 1.6 }}>
        The % next to each attribute = how much it contributes to the model's decisions.
        A high % means that attribute drives most of the model's choices.
      </div>
    </aside>
  );
}
