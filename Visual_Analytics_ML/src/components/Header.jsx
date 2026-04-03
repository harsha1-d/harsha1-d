import React from 'react';

/**
 * Top navigation header with logo, page nav pills, and train/predict badges.
 *
 * Props:
 *   page     -  0 | 1
 *   setPage  -  fn
 *   mc       -  active model color
 */
export default function Header({ page, setPage, mc }) {
  return (
    <header className="hdr">
      <div className="hdr-logo">
        Marketing <span>Analytics</span> Dashboard
      </div>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Page navigation */}
      <nav className="nav-wrap">
        <button
          className={`nav-btn ${page === 0 ? 'active' : ''}`}
          onClick={() => setPage(0)}
        >
          Sales Overview
        </button>
        <button
          className={`nav-btn ${page === 1 ? 'active' : ''}`}
          onClick={() => setPage(1)}
        >
          Segment Analysis
        </button>
      </nav>

      {/* Train / Predict badges */}
      <div className="hdr-badges">
        <span className="badge">Train: 2019-20</span>
        <span
          className="badge-pred"
          style={{
            background: `color-mix(in srgb, ${mc} 12%, var(--bg))`,
            borderColor: `color-mix(in srgb, ${mc} 40%, transparent)`,
            color: mc,
          }}
        >
          Pred: 2021-25
        </span>
      </div>
    </header>
  );
}
