import React, { useState } from 'react';
import Toggle from './Toggle';
import { MODEL_NAMES, MODEL_COLORS } from './ModelBar';
import { MODEL_ACC } from '../ml/engine';
import { BRANCH_NAMES, PRODUCT_NAMES, PAYMENT_NAMES } from '../data/dataset';

/**
 * Right-side filter panel.
 * Uses ONE shared filter state - every change updates ALL charts on BOTH pages.
 */
export default function FilterPanel({ page, mi, mc, filters, setFilters }) {
  const [openSec, setOpenSec] = useState(null);
  const tog = (sec) => setOpenSec((s) => (s === sec ? null : sec));

  const togChip = (key, val, maxLen) =>
    setFilters((prev) => {
      if (val === '__all__') {
        const full = Array.from({ length: maxLen }, (_, i) => i);
        return { ...prev, [key]: prev[key].length === maxLen ? [0] : full };
      }
      const cur  = prev[key];
      const next = cur.includes(val) ? cur.filter((x) => x !== val) : [...cur, val];
      return next.length ? { ...prev, [key]: next } : prev;
    });

  const SHORTCUTS = [
    ['Full',   2019, 2025],
    ['Train',  2019, 2020],
    ['Pred',   2021, 2025],
    ['Recent', 2023, 2025],
  ];

  const ChipGroup = ({ label, items, filterKey, maxLen }) => (
    <div className="fp-sec">
      <button
        className={'fp-green-btn' + (openSec === filterKey ? ' open' : '')}
        onClick={() => tog(filterKey)}
      >
        {label}
        <div className="fp-green-btn-sub">
          {filters[filterKey].length === maxLen
            ? 'All selected'
            : filters[filterKey].length + '/' + maxLen + ' selected'}
        </div>
      </button>
      {openSec === filterKey && (
        <div style={{ paddingTop: 4 }}>
          <div className="chip-wrap" style={{ marginBottom: 4 }}>
            <button
              className={'chip2' + (filters[filterKey].length === maxLen ? ' on' : '')}
              onClick={() => togChip(filterKey, '__all__', maxLen)}
              style={{ fontSize: 8, padding: '2px 7px' }}
            >
              All
            </button>
          </div>
          <div className="chip-wrap">
            {items.map((name, i) => (
              <button
                key={i}
                className={'chip2' + (filters[filterKey].includes(i) ? ' on' : '')}
                onClick={() => togChip(filterKey, i, maxLen)}
              >
                {name}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <aside className="fp">
      <div className="fp-title">Filters</div>
      <div style={{ fontSize: 9, color: 'var(--muted)', marginBottom: 12, lineHeight: 1.4 }}>
        All filters apply to both pages instantly
      </div>

      {/* Active model card */}
      <div className="fp-sec">
        <div className="fp-sec-title">Active Model</div>
        <div style={{
          background: 'color-mix(in srgb,' + mc + ' 10%, var(--bg))',
          border: '1px solid color-mix(in srgb,' + mc + ' 35%, transparent)',
          borderRadius: 8, padding: '8px 10px', marginBottom: 8,
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: mc, marginBottom: 4 }}>
            {MODEL_NAMES[mi]}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: 'var(--muted)', marginBottom: 4 }}>
            <span>Val Accuracy</span>
            <span className="mono" style={{ color: mc }}>{MODEL_ACC[mi]}%</span>
          </div>
          <div style={{ height: 2, background: 'var(--border)', borderRadius: 1 }}>
            <div style={{ height: '100%', width: MODEL_ACC[mi] + '%', background: mc, borderRadius: 1, transition: 'width .5s' }} />
          </div>
        </div>
        {MODEL_NAMES.map((name, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 3, fontSize: 9 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: MODEL_COLORS[i] }} />
              <span style={{ color: i === mi ? MODEL_COLORS[i] : 'var(--muted)' }}>{name.split(' ')[0]}</span>
            </div>
            <span className="mono" style={{ color: MODEL_COLORS[i] }}>{MODEL_ACC[i]}%</span>
          </div>
        ))}
      </div>

      {/* Date & Time */}
      <div className="fp-sec">
        <button
          className={'fp-green-btn' + (openSec === 'date' ? ' open' : '')}
          onClick={() => tog('date')}
        >
          Date &amp; Time
          <div className="fp-green-btn-sub">{filters.yr[0]} - {filters.yr[1]}</div>
        </button>
        {openSec === 'date' && (
          <div style={{ padding: '6px 0' }}>
            <div style={{ fontSize: 9, color: 'var(--muted)', marginBottom: 4 }}>From</div>
            <select className="fp-select" value={filters.yr[0]}
              onChange={(e) => setFilters((p) => ({ ...p, yr: [+e.target.value, p.yr[1]] }))}>
              {[2019,2020,2021,2022,2023,2024,2025].map((y) => <option key={y}>{y}</option>)}
            </select>
            <div style={{ fontSize: 9, color: 'var(--muted)', marginBottom: 4 }}>To</div>
            <select className="fp-select" value={filters.yr[1]}
              onChange={(e) => setFilters((p) => ({ ...p, yr: [p.yr[0], +e.target.value] }))}>
              {[2019,2020,2021,2022,2023,2024,2025].map((y) => <option key={y}>{y}</option>)}
            </select>
            <div className="chip-wrap">
              {SHORTCUTS.map(([lbl, fr, to]) => (
                <button key={lbl} className="chip2"
                  onClick={() => setFilters((p) => ({ ...p, yr: [fr, to] }))}>
                  {lbl}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Customer Type */}
      <ChipGroup label="Customer Type" items={['Member','Normal']} filterKey="custType" maxLen={2} />

      {/* Gender */}
      <ChipGroup label="Gender" items={['Female','Male']} filterKey="gender" maxLen={2} />

      {/* Branch */}
      <ChipGroup label="Branch / City" items={BRANCH_NAMES} filterKey="branches" maxLen={3} />

      {/* Product Line */}
      <ChipGroup label="Product Line" items={PRODUCT_NAMES} filterKey="products" maxLen={6} />

      {/* Payment */}
      <ChipGroup label="Payment Method" items={PAYMENT_NAMES} filterKey="payments" maxLen={3} />

      {/* Display toggles */}
      <div className="fp-sec">
        <button
          className={'fp-green-btn' + (openSec === 'disp' ? ' open' : '')}
          onClick={() => tog('disp')}
        >
          Display Options
          <div className="fp-green-btn-sub">Show/hide actual & predicted lines</div>
        </button>
        {openSec === 'disp' && (
          <div style={{ paddingTop: 6 }}>
            <div className="tog-row">
              <span className="tog-lbl">Show Actual</span>
              <Toggle on={filters.showActual} setOn={(v) => setFilters((p) => ({ ...p, showActual: v }))} color="var(--actual)" />
            </div>
            <div className="tog-row">
              <span className="tog-lbl">Show Predicted</span>
              <Toggle on={filters.showPred} setOn={(v) => setFilters((p) => ({ ...p, showPred: v }))} color={mc} />
            </div>
          </div>
        )}
      </div>

      {/* Dataset info */}
      <div className="fp-sec">
        <div className="fp-sec-title">Dataset</div>
        {[['Total rows','3,000'],['Train 2019-20','865'],['Test 2021-25','2,135'],['Features','9'],['Classes','Low/Med/High']].map(([k,v]) => (
          <div key={k} className="stat-item">
            <span style={{ color: 'var(--muted)' }}>{k}</span>
            <span className="mono" style={{ color: 'var(--text)', fontSize: 8 }}>{v}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}
