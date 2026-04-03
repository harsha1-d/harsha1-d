import React from 'react';

/**
 * Simple animated toggle switch.
 * Props: on (bool), setOn (fn), color (css string)
 */
export default function Toggle({ on, setOn, color = '#34d399' }) {
  return (
    <div
      onClick={() => setOn(!on)}
      style={{
        width: 32, height: 17,
        borderRadius: 9,
        background: on ? color : '#1e2d47',
        cursor: 'pointer',
        position: 'relative',
        flexShrink: 0,
        transition: 'background .2s',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: 2,
          left: on ? 17 : 2,
          width: 13, height: 13,
          borderRadius: 7,
          background: 'white',
          boxShadow: '0 1px 4px #0006',
          transition: 'left .2s',
        }}
      />
    </div>
  );
}
