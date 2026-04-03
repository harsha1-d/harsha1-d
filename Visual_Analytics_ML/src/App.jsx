import React, { useState } from 'react';

import Header      from './components/Header';
import ModelBar    from './components/ModelBar';
import FilterPanel from './components/FilterPanel';
import Page1       from './pages/Page1';
import Page2       from './pages/Page2';
import { MODEL_COLORS } from './components/ModelBar';

export default function App() {
  const [page, setPage] = useState(0);
  const [mi,   setMi]   = useState(0);
  const mc = MODEL_COLORS[mi];

  // ONE unified filter state - drives ALL charts on BOTH pages
  const [filters, setFilters] = useState({
    yr:         [2019, 2025],
    branches:   [0, 1, 2],
    products:   [0, 1, 2, 3, 4, 5],
    payments:   [0, 1, 2],
    custType:   [0, 1],
    gender:     [0, 1],
    showActual: true,
    showPred:   true,
  });

  return (
    <div className="app-shell">
      <Header page={page} setPage={setPage} mc={mc} />
      <ModelBar mi={mi} setMi={setMi} />
      <div className="body-wrap">
        <main className="content">
          {page === 0 ? (
            <Page1 key={'p1-' + mi} mi={mi} mc={mc} filters={filters} setFilters={setFilters} />
          ) : (
            <Page2 key={'p2-' + mi} mi={mi} mc={mc} filters={filters} setFilters={setFilters} />
          )}
        </main>
        <FilterPanel page={page} mi={mi} mc={mc} filters={filters} setFilters={setFilters} />
      </div>
    </div>
  );
}
