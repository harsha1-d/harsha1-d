import React, { useState, useRef, useEffect } from 'react';

/**
 * Green dropdown filter chip matching the wireframe design.
 *
 * Props:
 *   label     -  display label
 *   items     -  string[] of option names
 *   selected  -  int[]  of selected indices
 *   toggle    -  fn(index | '__all__')  called when a row is clicked
 *   color     -  accent color (default green)
 */
export default function DropFilter({
  label,
  items,
  selected,
  toggle,
  color = 'var(--green)',
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const allSelected = selected.length === items.length;

  return (
    <div ref={wrapRef} className="dropdown-wrap">
      {/* Chip button */}
      <div className="fchip" onClick={() => setOpen((o) => !o)}>
        <div className="fchip-label" style={{ color }}>
          {label}
        </div>
        <div className="fchip-sub">
          {allSelected ? 'All' : `${selected.length}/${items.length} sel`}{' '}
          {open ? '\u25b4' : '\u25be'}
        </div>
      </div>

      {/* Dropdown */}
      {open && (
        <div className="dropdown">
          {/* Select All row */}
          <div
            className={`dd-item ${allSelected ? 'selected' : ''}`}
            onClick={() => toggle('__all__')}
          >
            <div className={`check-box ${allSelected ? 'on' : ''}`}>
              {allSelected && 'ok'}
            </div>
            <span style={{ fontWeight: 700 }}>Select All</span>
          </div>

          {/* Individual options */}
          {items.map((item, i) => (
            <div
              key={i}
              className={`dd-item ${selected.includes(i) ? 'selected' : ''}`}
              onClick={() => toggle(i)}
            >
              <div className={`check-box ${selected.includes(i) ? 'on' : ''}`}>
                {selected.includes(i) && 'ok'}
              </div>
              {item}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
