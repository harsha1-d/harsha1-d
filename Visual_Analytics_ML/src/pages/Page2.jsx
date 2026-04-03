import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ComposedChart, Bar, Line, Area,
  BarChart,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts';

import { MODEL_NAMES, MODEL_COLORS } from '../components/ModelBar';
import { runModel } from '../ml/engine';
import { mean } from '../ml/models';
import { RAW_DATA, TEST, BRANCH_NAMES, PRODUCT_NAMES, PAYMENT_NAMES, MONTHS } from '../data/dataset';

const TT = {
  contentStyle: { background:'#131c30', border:'1px solid #253657', borderRadius:8, fontFamily:"'IBM Plex Mono',monospace", fontSize:10, padding:'8px 12px' },
  labelStyle:   { color:'#e4eeff', fontWeight:700, marginBottom:4 },
};
const fmt$  = (v) => '$' + Number(v).toFixed(0);
const tick  = { fontSize: 8, fill: 'var(--muted)' };
const monoTick = { fontSize: 8, fill:'var(--muted)', fontFamily:"'IBM Plex Mono'" };

const SubLabel = ({ mc, modelName }) => (
  <div className="card-sub">
    <span style={{color:'#4d8a8a',fontSize:9}}>Solid = Actual</span>
    &nbsp;|&nbsp;
    <span style={{color:mc,fontSize:9}}>Dashed/Lighter = {modelName} Predicted</span>
  </div>
);

export default function Page2({ mi, mc, filters }) {
  const { pc, cm } = useMemo(() => runModel(9, mi), [mi]);
  const modelShort = MODEL_NAMES[mi].split(' ')[0];

  // Filtered subsets  -  apply ALL shared filters
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

  // Predicted value helper
  const predVal = React.useCallback((r) => {
    const oi = TEST.indexOf(r);
    return oi >= 0 ? cm[pc[oi]] : 0;
  }, [pc, cm]);

  // ==================================================================
  // CHART 1  -  RADAR CHART
  // Customer Type x Payment x Sales  -  multi-axis view
  // Each axis = one payment method; shapes = Member vs Normal (actual + pred)
  // ==================================================================
  const radarData = useMemo(() => {
    return PAYMENT_NAMES.map((pay, pi) => {
      if (!filters.payments.includes(pi)) return null;
      const actMember = mean(fRaw.filter((r) => r[3]===0 && r[6]===pi).map((r) => r[9])) || 0;
      const actNormal = mean(fRaw.filter((r) => r[3]===1 && r[6]===pi).map((r) => r[9])) || 0;
      const predMember = mean(fTest.filter((r) => r[3]===0 && r[6]===pi).map(predVal)) || 0;
      const predNormal = mean(fTest.filter((r) => r[3]===1 && r[6]===pi).map(predVal)) || 0;
      return {
        axis: pay,
        'Member Actual':  +actMember.toFixed(1),
        'Normal Actual':  +actNormal.toFixed(1),
        'Member Pred':    +predMember.toFixed(1),
        'Normal Pred':    +predNormal.toFixed(1),
      };
    }).filter(Boolean);
  }, [fRaw, fTest, predVal, filters.payments]);

  // ==================================================================
  // CHART 2  -  COMPOSED CHART (Bar + Line overlay)
  // Product Line + Customer Type + Sales
  // Grouped bars = actual (Member / Normal), Lines = predicted per customer type
  // ==================================================================
  const plCustData = useMemo(() => {
    return PRODUCT_NAMES.map((pl, pli) => {
      if (!filters.products.includes(pli)) return null;
      const actMember = Math.round(mean(fRaw.filter((r) => r[5]===pli && r[3]===0).map((r) => r[9])) || 0);
      const actNormal = Math.round(mean(fRaw.filter((r) => r[5]===pli && r[3]===1).map((r) => r[9])) || 0);
      const predMember = Math.round(mean(fTest.filter((r) => r[5]===pli && r[3]===0).map(predVal)) || 0);
      const predNormal = Math.round(mean(fTest.filter((r) => r[5]===pli && r[3]===1).map(predVal)) || 0);
      return { pl, actMember, actNormal, predMember, predNormal };
    }).filter(Boolean);
  }, [fRaw, fTest, predVal, filters.products]);

  // ==================================================================
  // CHART 3  -  STACKED AREA CHART
  // Product Line + Gender + Sales  -  shows composition + predicted overlay
  // X = product line index, stacked areas = Female/Male actual
  // Lines = Female/Male predicted
  // ==================================================================
  const plGenderArea = useMemo(() => {
    return PRODUCT_NAMES.map((pl, pli) => {
      if (!filters.products.includes(pli)) return null;
      const femaleAct  = Math.round(mean(fRaw.filter((r) => r[5]===pli && r[4]===0).map((r) => r[9])) || 0);
      const maleAct    = Math.round(mean(fRaw.filter((r) => r[5]===pli && r[4]===1).map((r) => r[9])) || 0);
      const femalePred = Math.round(mean(fTest.filter((r) => r[5]===pli && r[4]===0).map(predVal)) || 0);
      const malePred   = Math.round(mean(fTest.filter((r) => r[5]===pli && r[4]===1).map(predVal)) || 0);
      return { pl, femaleAct, maleAct, femalePred, malePred };
    }).filter(Boolean);
  }, [fRaw, fTest, predVal, filters.products]);

  // ==================================================================
  // CHART 4  -  LINE + BAR COMPOSED (Branch / City)
  // Gender x Sales x City  -  bars show actual Female/Male per branch,
  // lines show the predicted values per branch for each gender
  // ==================================================================
  const genderCityData = useMemo(() => {
    return BRANCH_NAMES.map((br, bi) => {
      if (!filters.branches.includes(bi)) return null;
      const femaleAct  = Math.round(mean(fRaw.filter((r) => r[2]===bi && r[4]===0).map((r) => r[9])) || 0);
      const maleAct    = Math.round(mean(fRaw.filter((r) => r[2]===bi && r[4]===1).map((r) => r[9])) || 0);
      const femalePred = Math.round(mean(fTest.filter((r) => r[2]===bi && r[4]===0).map(predVal)) || 0);
      const malePred   = Math.round(mean(fTest.filter((r) => r[2]===bi && r[4]===1).map(predVal)) || 0);
      return { branch: br, femaleAct, maleAct, femalePred, malePred };
    }).filter(Boolean);
  }, [fRaw, fTest, predVal, filters.branches]);

  // ==================================================================
  // CHART 5  -  STACKED BAR over years
  // Sales + Branch & City + Customer  -  two side-by-side stacked bars per year
  // Stack A = actual, Stack B = predicted (2021-2025 only)
  // ==================================================================
  const STACK_COLORS = ['#38bdf8','#0ea5e9','#34d399','#059669','#a78bfa','#7c3aed'];

  const branchCustYear = useMemo(() => {
    const out = [];
    for (let y = 2019; y <= 2025; y++) {
      if (y < filters.yr[0] || y > filters.yr[1]) continue;
      const row = { year: y };
      BRANCH_NAMES.forEach((br, bi) => {
        ['Member', 'Normal'].forEach((ct, ci) => {
          const key = br + ' ' + ct;
          const actRows  = fRaw.filter((r) => r[0]===y && r[2]===bi && r[3]===ci);
          const predRows = fTest.filter((r) => r[0]===y && r[2]===bi && r[3]===ci);
          row[key] = Math.round(actRows.reduce((s, r) => s + r[9], 0));
          if (y >= 2021) {
            row[key + ' Pred'] = Math.round(predRows.reduce((s, r) => s + predVal(r), 0));
          }
        });
      });
      out.push(row);
    }
    return out;
  }, [fRaw, fTest, predVal, filters.yr]);

  return (
    <div className="page-anim">

      {/* ===== ROW 1: 3 charts ===== */}
      <div className="g3">

        {/* CHART 1  -  RADAR */}
        <div className="card">
          <div className="card-title">Customer Type x Payment  -  Radar</div>
          <SubLabel mc={mc} modelName={modelShort} />
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData} outerRadius={75} margin={{top:0,right:20,left:20,bottom:0}}>
              <PolarGrid stroke="var(--border2)" gridType="polygon"/>
              <PolarAngleAxis dataKey="axis" tick={{fontSize:9,fill:'var(--muted)',fontFamily:"'IBM Plex Mono'"}}/>
              <PolarRadiusAxis angle={30} tick={{fontSize:7,fill:'var(--dim)'}} axisLine={false} tickCount={4} tickFormatter={fmt$}/>
              <Radar name="Member Actual" dataKey="Member Actual" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.2} strokeWidth={2}/>
              <Radar name="Normal Actual" dataKey="Normal Actual" stroke="#34d399" fill="#34d399" fillOpacity={0.15} strokeWidth={2}/>
              <Radar name="Member Pred"   dataKey="Member Pred"   stroke={mc}      fill={mc}      fillOpacity={0.1}  strokeWidth={1.5} strokeDasharray="5 3"/>
              <Radar name="Normal Pred"   dataKey="Normal Pred"   stroke="#fb24ed" fill="#fbbf24" fillOpacity={0.08} strokeWidth={1.5} strokeDasharray="5 3"/>
              <Legend wrapperStyle={{fontSize:9,paddingTop:4}}/>
              <Tooltip {...TT} formatter={(v)=>fmt$(v)}/>
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* CHART 2  -  COMPOSED: Bar (actual) + Line (predicted) */}
        <div className="card">
          <div className="card-title">Product Line x Customer Type</div>
          <SubLabel mc={mc} modelName={modelShort} />
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={plCustData} margin={{top:4,right:4,left:-12,bottom:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="pl" tick={{...tick,fontSize:7.5}} axisLine={false} tickLine={false}/>
              <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={fmt$}/>
              <Tooltip {...TT} formatter={(v)=>fmt$(v)}/>
              <Legend wrapperStyle={{fontSize:9,paddingTop:4}}/>
              <Bar dataKey="actMember"  name="Member Act"  fill="#38bdf8" radius={[3,3,0,0]} maxBarSize={18}/>
              <Bar dataKey="actNormal"  name="Normal Act"  fill="#6ea8fe" radius={[3,3,0,0]} maxBarSize={18}/>
              <Line type="monotone" dataKey="predMember" name="Member Pred" stroke={mc}      strokeWidth={2} strokeDasharray="6 3" dot={{r:3,fill:mc}}/>
              <Line type="monotone" dataKey="predNormal" name="Normal Pred" stroke="#fb12cc" strokeWidth={2} strokeDasharray="6 3" dot={{r:3,fill:'#fb12cc'}}/>
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* CHART 3  -  STACKED AREA: Product Line x Gender */}
        <div className="card">
          <div className="card-title">Product Line x Gender  -  Area</div>
          <SubLabel mc={mc} modelName={modelShort} />
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={plGenderArea} margin={{top:4,right:4,left:-12,bottom:0}}>
              <defs>
                <linearGradient id="fGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#f472b6" stopOpacity={0.5}/>
                  <stop offset="95%" stopColor="#f472b6" stopOpacity={0.05}/>
                </linearGradient>
                <linearGradient id="mGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#818cf8" stopOpacity={0.5}/>
                  <stop offset="95%" stopColor="#818cf8" stopOpacity={0.05}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="pl" tick={{...tick,fontSize:7.5}} axisLine={false} tickLine={false}/>
              <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={fmt$}/>
              <Tooltip {...TT} formatter={(v)=>fmt$(v)}/>
              <Legend wrapperStyle={{fontSize:9,paddingTop:4}}/>
              <Area type="monotone" dataKey="femaleAct"  name="Female Act"  stroke="#f472b6" fill="url(#fGrad)" strokeWidth={2}/>
              <Area type="monotone" dataKey="maleAct"    name="Male Act"    stroke="#818cf8" fill="url(#mGrad)" strokeWidth={2}/>
              <Line type="monotone" dataKey="femalePred" name="Female Pred" stroke={mc}      strokeWidth={2} strokeDasharray="7 3" dot={false}/>
              <Line type="monotone" dataKey="malePred"   name="Male Pred"   stroke="#149eb7" strokeWidth={2} strokeDasharray="7 3" dot={false}/>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ===== ROW 2: small + large ===== */}
      <div className="g-page2-bot">

        {/* CHART 4  -  COMPOSED Bar + Line: Gender x Branch x Sales */}
        <div className="card">
          <div className="card-title">Gender x Sales x Branch</div>
          <div className="card-sub" style={{fontSize:9}}>
            Bars = actual Female/Male per branch &nbsp;|&nbsp;
            <span style={{color:mc}}>Lines = {modelShort} predicted</span>
          </div>
          <ResponsiveContainer width="100%" height={210}>
            <ComposedChart data={genderCityData} margin={{top:4,right:8,left:-12,bottom:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="branch" tick={{...tick,fontSize:10}} axisLine={false} tickLine={false}/>
              <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={fmt$}/>
              <Tooltip {...TT} formatter={(v)=>fmt$(v)}/>
              <Legend wrapperStyle={{fontSize:9,paddingTop:4}}/>
              <Bar dataKey="femaleAct"  name="Female Act"  fill="#f472b6" radius={[3,3,0,0]} maxBarSize={28}/>
              <Bar dataKey="maleAct"    name="Male Act"    fill="#818cf8" radius={[3,3,0,0]} maxBarSize={28}/>
              <Line type="monotone" dataKey="femalePred" name="Female Pred" stroke={mc}      strokeWidth={2.5} strokeDasharray="7 3" dot={{r:4,fill:mc}}/>
              <Line type="monotone" dataKey="malePred"   name="Male Pred"   stroke="#ff0fb7" strokeWidth={2.5} strokeDasharray="7 3" dot={{r:4,fill:'#ff0fb7'}}/>
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* CHART 5 */}
        <div className="card">
          <div className="card-title">Sales, Branch &amp; City, Customer - Annual View</div>
          <div className="card-sub">
            <span style={{ color: '#4d8a8a', fontSize: 9 }}>2019-20: Actual stacked</span>
            &nbsp;|&nbsp;
            <span style={{ color: mc, fontSize: 9 }}>
              2021-25: Actual stack A + {MODEL_NAMES[mi].split(' ')[0]} prediction stack B
            </span>
          </div>
          <ResponsiveContainer width="100%" height={195}>
            <BarChart data={branchCustYear} margin={{ top: 4, right: 8, left: 0, bottom: 0 }} barCategoryGap="20%">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="year" tick={{ ...tick, fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={tick} axisLine={false} tickLine={false} tickFormatter={(v) => '$' + v.toLocaleString()} width={62} />
              <Tooltip {...TT} formatter={(v) => '$' + v.toLocaleString()} />
              <Legend wrapperStyle={{ fontSize: 8.5, paddingTop: 6 }} />
              {BRANCH_NAMES.flatMap((br, bi) =>
                ['Member', 'Normal'].map((ct, ci) => (
                  <Bar
                    key={br + ct + 'a'}
                    dataKey={`${br} ${ct}`}
                    name={`${br} ${ct}`}
                    stackId="a"
                    fill={STACK_COLORS[bi * 2 + ci]}
                    fillOpacity={0.85}
                    maxBarSize={36}
                  />
                ))
              )}
              {BRANCH_NAMES.flatMap((br, bi) =>
                ['Member', 'Normal'].map((ct, ci) => (
                  <Bar
                    key={br + ct + 'p'}
                    dataKey={`${br} ${ct} Pred`}
                    name={`${br} ${ct} (pred)`}
                    stackId="b"
                    fill={STACK_COLORS[bi * 2 + ci]}
                    fillOpacity={0.3}
                    maxBarSize={24}
                  />
                ))
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
}
