import React, { useState, useMemo } from 'react';
import Header    from './components/Header';
import TweakPanel from './components/TweakPanel';
import Page1     from './pages/Page1';
import Page2     from './pages/Page2';
import { buildAndEval } from './ml/engine';

const DEFAULT_PARAMS = {
  maxDepth:          4,
  minSamplesSplit:   10,
  classWeights:      [1, 1, 1],
  lowBoundaryPct:    33,
  highBoundaryPct:   66,
  activeFeatures:      [0,1,2,3,4,5,6,7,8,9],
  maxFeaturesPerSplit: null,
};

export default function App() {
  const [page,    setPage]    = useState(0);
  const [params,  setParams]  = useState(DEFAULT_PARAMS);
  // What-If predictor state  -  lives here so it persists across page switches
  const [wiQty,   setWiQty]   = useState(5);
  const [wiPrice, setWiPrice] = useState(55);

  const result = useMemo(() => buildAndEval(params), [params]);

  return (
    <div className="app-shell">
      <Header page={page} setPage={setPage} result={result} />
      <div className="body-wrap">
        <main className="content">
          {page === 0
            ? <Page1 params={params} result={result} />
            : <Page2 params={params} result={result} />}
        </main>
        <TweakPanel
          params={params} setParams={setParams} result={result}
          wiQty={wiQty}   setWiQty={setWiQty}
          wiPrice={wiPrice} setWiPrice={setWiPrice}
        />
      </div>
    </div>
  );
}
