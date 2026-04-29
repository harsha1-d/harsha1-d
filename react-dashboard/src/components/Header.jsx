import React from 'react';

export default function Header({ page, setPage, result }) {
  return (
    <header className="hdr">
      <div className="hdr-logo">
        Decision Tree <span>Model Visualizer</span>
      </div>

      {/* Live stats strip */}
      {result && (
        <div style={{ display:'flex', gap:10, marginLeft:14 }}>
          {[
            ['Nodes',  result.nNodes],
            ['Leaves', result.nLeaves],
            ['Depth',  result.maxActualDepth],
          ].map(([lbl,val]) => (
            <div key={lbl} style={{ fontSize:9, color:'var(--muted)' }}>
              {lbl}: <span className="mono" style={{color:'var(--text2)',fontWeight:700}}>{val}</span>
            </div>
          ))}
          <div style={{ fontSize:9, color:'var(--muted)', marginLeft:4, paddingLeft:10, borderLeft:'1px solid var(--border)' }}>
            Train: <span className="mono" style={{color:'#38bdf8',fontWeight:700}}>{result.trainAcc}%</span>
          </div>
          <div style={{ fontSize:9, color:'var(--muted)' }}>
            Test: <span className="mono" style={{color:'#34d399',fontWeight:700}}>{result.testAcc}%</span>
          </div>
        </div>
      )}

      <div style={{ flex:1 }}/>

      {/* Page navigation */}
      <nav className="nav-wrap">
        <button className={'nav-btn' + (page===0?' active':'')} onClick={()=>setPage(0)}>
          Tree Visualizer
        </button>
        <button className={'nav-btn' + (page===1?' active':'')} onClick={()=>setPage(1)}>
          Prediction Analysis
        </button>
      </nav>

      {/* Train/Pred badges */}
      <div style={{ display:'flex', gap:8, marginLeft:12, fontSize:9, color:'var(--muted)' }}>
        <span style={{ padding:'2px 8px', background:'var(--bg)', border:'1px solid var(--border)', borderRadius:5 }}>
          Train: 2019-20
        </span>
        <span style={{ padding:'2px 8px', background:'color-mix(in srgb,#34d399 10%,var(--bg))',
          border:'1px solid color-mix(in srgb,#34d399 35%,transparent)', borderRadius:5, color:'#34d399' }}>
          Predict: 2021-25
        </span>
      </div>
    </header>
  );
}
