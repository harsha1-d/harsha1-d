import React, { useMemo } from 'react';
import {
  ResponsiveContainer, ComposedChart, LineChart, AreaChart,
  Area, Line, Bar, BarChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts';

import KPICard   from '../components/KPICard';
import DropFilter from '../components/DropFilter';
import { MODEL_NAMES, MODEL_COLORS } from '../components/ModelBar';
import { runModel, ALL_SALES } from '../ml/engine';
import { mean } from '../ml/models';
import { RAW_DATA, TEST, BRANCH_NAMES, PRODUCT_NAMES, PAYMENT_NAMES, MONTHS } from '../data/dataset';

const TT = {
  contentStyle: { background:'#131c30', border:'1px solid #253657', borderRadius:8, fontFamily:"'IBM Plex Mono',monospace", fontSize:10, padding:'8px 12px' },
  labelStyle:   { color:'#e4eeff', fontWeight:700, marginBottom:4 },
};
const fmt$ = (v) => '$' + Number(v).toFixed(0);
const tick  = { fontSize: 8, fill: 'var(--muted)' };
const monoTick = { fontSize: 8, fill: 'var(--muted)', fontFamily:"'IBM Plex Mono'" };

export default function Page1({ mi, mc, filters, setFilters }) {
  const { pc, cm } = useMemo(() => runModel(9, mi), [mi]);

  // Apply ALL filters to raw data
  const fRaw = useMemo(() =>
    RAW_DATA.filter((r) =>
      filters.branches.includes(r[2]) &&
      filters.custType.includes(r[3]) &&
      filters.gender.includes(r[4]) &&
      filters.products.includes(r[5]) &&
      filters.payments.includes(r[6]) &&
      r[0] >= filters.yr[0] && r[0] <= filters.yr[1]
    ), [filters]);

  const fTest = useMemo(() =>
    TEST.filter((r) =>
      filters.branches.includes(r[2]) &&
      filters.custType.includes(r[3]) &&
      filters.gender.includes(r[4]) &&
      filters.products.includes(r[5]) &&
      filters.payments.includes(r[6]) &&
      r[0] >= filters.yr[0] && r[0] <= filters.yr[1]
    ), [filters]);

  // KPIs
  const totalActual = fRaw.reduce((s,r) => s + r[9], 0);
  const totalPred   = fTest.reduce((s,r) => {
    const oi = TEST.indexOf(r);
    return s + (oi >= 0 ? cm[pc[oi]] : 0);
  }, 0);
  const avgRating = mean(fRaw.map((r) => r[11])) || 0;
  const rec  = fRaw.filter((r) => r[0] >= 2023);
  const prev = fRaw.filter((r) => r[0] >= 2021 && r[0] < 2023);
  const chg  = (a, b, k) => a.length && b.length ? (mean(b.map((r) => r[k])) / mean(a.map((r) => r[k])) - 1) * 100 : 0;

  const kpis = [
    { label:'Total Actual Sales',                                    value:'$'+Math.round(totalActual).toLocaleString(), delta:chg(prev,rec,9),     color:'#38bdf8' },
    { label:'Predicted Sales ('+MODEL_NAMES[mi].split(' ')[0]+')',   value:'$'+Math.round(totalPred).toLocaleString(),  delta:chg(prev,fTest,9)*0.8, color:mc },
    { label:'Avg Customer Rating',                                   value:avgRating.toFixed(2)+' / 10',                delta:chg(prev,rec,11)*12,   color:'#fbbf24' },
  ];

  // Monthly area+prediction line
  const monthlyChart = useMemo(() => {
    const map = {};
    fRaw.forEach((r) => {
      const k = r[0]+'-'+String(r[1]).padStart(2,'0');
      if (!map[k]) map[k] = { label:MONTHS[r[1]-1]+"'"+String(r[0]).slice(2), actual:0, pred:null };
      map[k].actual += r[9];
    });
    fTest.forEach((r) => {
      const oi = TEST.indexOf(r);
      const k  = r[0]+'-'+String(r[1]).padStart(2,'0');
      if (!map[k]) return;
      if (map[k].pred === null) map[k].pred = 0;
      map[k].pred += oi >= 0 ? cm[pc[oi]] : 0;
    });
    return Object.entries(map).sort(([a],[b]) => a.localeCompare(b)).map(([,v]) => v);
  }, [fRaw, fTest, pc, cm]);

  // All-model comparison
  const modelCmp = useMemo(() => {
    const testSubset = TEST.filter((r) =>
      filters.branches.includes(r[2]) &&
      filters.custType.includes(r[3]) &&
      filters.gender.includes(r[4]) &&
      filters.products.includes(r[5]) &&
      filters.payments.includes(r[6])
    );
    const map = {};
    testSubset.forEach((r) => {
      const k = r[0]+'-'+String(r[1]).padStart(2,'0');
      if (!map[k]) map[k] = { label:MONTHS[r[1]-1]+"'"+String(r[0]).slice(2), actual:0 };
      MODEL_NAMES.forEach((_,mi2) => {
        if (!map[k]['m'+mi2]) map[k]['m'+mi2] = 0;
        const oi = TEST.indexOf(r);
        map[k]['m'+mi2] += oi >= 0 ? ALL_SALES[mi2].cm[ALL_SALES[mi2].pc[oi]] : 0;
      });
      map[k].actual += r[9];
    });
    return Object.entries(map)
      .filter(([k]) => { const y=+k.split('-')[0]; return y>=filters.yr[0]&&y<=filters.yr[1]; })
      .sort(([a],[b]) => a.localeCompare(b)).map(([,v]) => v);
  }, [filters]);

  // Value comparison table
  const tableData = useMemo(() => {
    return [2021,2022,2023,2024,2025].map((y) => {
      const rows = TEST.filter((r) =>
        r[0]===y &&
        filters.branches.includes(r[2]) &&
        filters.custType.includes(r[3]) &&
        filters.gender.includes(r[4]) &&
        filters.products.includes(r[5]) &&
        filters.payments.includes(r[6])
      );
      if (!rows.length) return { year:y, actual:0, preds:MODEL_NAMES.map(()=>0), best:0 };
      const actual = Math.round(mean(rows.map((r) => r[9])));
      const preds  = MODEL_NAMES.map((_,mi2) =>
        Math.round(mean(rows.map((r) => {
          const oi = TEST.indexOf(r);
          return oi >= 0 ? ALL_SALES[mi2].cm[ALL_SALES[mi2].pc[oi]] : 0;
        })))
      );
      const best = preds.reduce((bi,v,i) => Math.abs(v-actual)<Math.abs(preds[bi]-actual)?i:bi, 0);
      return { year:y, actual, preds, best };
    });
  }, [filters]);

  // Inline filter toggle helpers for green chips in KPI row
  const togArr = (key, idx, maxLen) =>
    setFilters((prev) => {
      if (idx === '__all__') {
        const full = Array.from({length:maxLen},(_,i)=>i);
        return { ...prev, [key]: prev[key].length===maxLen?[0]:full };
      }
      const cur  = prev[key];
      const next = cur.includes(idx) ? cur.filter((x)=>x!==idx) : [...cur,idx];
      return next.length ? { ...prev, [key]:next } : prev;
    });

  return (
    <div className="page-anim">

      {/* Row 1: KPIs + green filter chips */}
      <div className="kpi-row">
        {kpis.map((k,i) => <KPICard key={i} {...k} />)}
        <DropFilter label="Branch"       items={BRANCH_NAMES}  selected={filters.branches} toggle={(i)=>togArr('branches',i,3)} />
        <DropFilter label="Product Line" items={PRODUCT_NAMES} selected={filters.products} toggle={(i)=>togArr('products',i,6)} />
        <DropFilter label="Payment"      items={PAYMENT_NAMES} selected={filters.payments} toggle={(i)=>togArr('payments',i,3)} />
      </div>

      {/* Row 2: Full-width sales prediction chart */}
      <div className="card" style={{ marginBottom:10 }}>
        <div className="card-title">Sales Prediction and Actual Graph</div>
        <div className="card-sub">
          <span style={{color:'#4d8a8a'}}>2019-2020: Training zone (actual)</span>
          &nbsp;|&nbsp;
          <span style={{color:mc}}>2021-2025: {MODEL_NAMES[mi]} prediction (dashed)</span>
          &nbsp;|&nbsp;
          <span style={{color:'var(--muted)',fontSize:9}}>
            Showing {fRaw.length} rows after filters
          </span>
        </div>
        <ResponsiveContainer width="100%" height={222}>
          <ComposedChart data={monthlyChart} margin={{top:4,right:12,left:0,bottom:0}}>
            <defs>
              <linearGradient id="aGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#6ea8fe" stopOpacity={0.35}/>
                <stop offset="95%" stopColor="#6ea8fe" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="pGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={mc} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={mc} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false}/>
            <XAxis dataKey="label" tick={monoTick} interval={5} axisLine={false} tickLine={false}/>
            <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={fmt$} width={54}/>
            <Tooltip {...TT} formatter={(v,n)=>[fmt$(v),n]}/>
            <Legend wrapperStyle={{fontSize:10,paddingTop:8}}/>
            <ReferenceLine x={"Jan'21"} stroke="var(--border2)" strokeDasharray="5 3"
              label={{value:'Prediction Zone',position:'insideTopLeft',fill:'var(--muted)',fontSize:8}}/>
            {filters.showActual && (
              <Area type="monotone" dataKey="actual" name="Actual Sales"
                stroke="#6ea8fe" strokeWidth={2} fill="url(#aGrad)" dot={false}/>
            )}
            {filters.showPred && (
              <Line type="monotone" dataKey="pred" name={MODEL_NAMES[mi]+' Pred'}
                stroke={mc} strokeWidth={2.5} strokeDasharray="8 4" dot={false} connectNulls/>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Row 3: Model comparison + value table */}
      <div className="g-main-tb">
        <div className="card">
          <div className="card-title">Model Comparison Graph</div>
          <div className="card-sub">All 4 classifiers vs actual  -  active model highlighted in full opacity</div>
          <ResponsiveContainer width="100%" height={215}>
            <LineChart data={modelCmp} margin={{top:4,right:8,left:0,bottom:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="label" tick={monoTick} interval={5} axisLine={false} tickLine={false}/>
              <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={fmt$} width={54}/>
              <Tooltip {...TT} formatter={(v)=>fmt$(v)}/>
              <Legend wrapperStyle={{fontSize:9.5,paddingTop:8}}/>
              <Line type="monotone" dataKey="actual" name="Actual"
                stroke="var(--actual)" strokeWidth={2.5} dot={false}/>
              {MODEL_NAMES.map((name,mi2) => (
                <Line key={name} type="monotone" dataKey={'m'+mi2} name={name}
                  stroke={MODEL_COLORS[mi2]}
                  strokeWidth={mi2===mi?2.5:1.2}
                  strokeOpacity={mi2===mi?1:0.45}
                  strokeDasharray={mi2===mi?'0':'5 3'}
                  dot={false} connectNulls/>
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">Table for Value Comparison</div>
          <div className="card-sub">Avg predicted sale per year  -  WIN = closest to actual</div>
          <div style={{overflowY:'auto',maxHeight:220}}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Actual</th>
                  {MODEL_NAMES.map((n,i) => <th key={i} style={{color:MODEL_COLORS[i]}}>{n.split(' ')[0]}</th>)}
                  <th>Best</th>
                </tr>
              </thead>
              <tbody>
                {tableData.map((row) => (
                  <tr key={row.year}>
                    <td style={{fontWeight:700,color:'var(--text)'}}>{row.year}</td>
                    <td className="num">${row.actual}</td>
                    {row.preds.map((p,i) => (
                      <td key={i} className={'num'+(i===row.best?' best':'')}>
                        ${p}{i===row.best && <span className="win-badge">WIN</span>}
                      </td>
                    ))}
                    <td style={{color:MODEL_COLORS[row.best],fontSize:10,fontWeight:700}}>
                      {MODEL_NAMES[row.best].split(' ')[0]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{marginTop:8,padding:'6px 8px',background:'var(--bg)',borderRadius:6,display:'flex',gap:12,flexWrap:'wrap'}}>
            {MODEL_NAMES.map((name,i) => {
              const wins = tableData.filter((r)=>r.best===i).length;
              return (
                <div key={i} style={{display:'flex',alignItems:'center',gap:5,fontSize:9}}>
                  <div style={{width:6,height:6,borderRadius:'50%',background:MODEL_COLORS[i]}}/>
                  <span style={{color:'var(--muted)'}}>{name.split(' ')[0]}</span>
                  <span className="mono" style={{color:wins>0?MODEL_COLORS[i]:'var(--muted)'}}>{wins} wins</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
