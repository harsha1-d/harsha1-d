import React from 'react';
import { MODEL_ACC } from '../ml/engine';

export const MODEL_NAMES  = ['Naive Bayes', 'Decision Tree', 'KNN (k=5)', 'Random Forest'];
export const MODEL_COLORS = ['#38bdf8', '#34d399', '#fbbf24', '#f87171'];

/**
 * Horizontal bar below the header for selecting the active classifier.
 * Clicking a card instantly re-predicts all charts on both pages.
 *
 * Props:
 *   mi     -  active model index (0-3)
 *   setMi  -  fn
 */
export default function ModelBar({ mi, setMi }) {
  return (
    <div className="model-bar">
      <span className="model-bar-label">Classifier</span>

      {MODEL_NAMES.map((name, i) => (
        <div
          key={i}
          className={`mcard ${i === mi ? 'active' : ''}`}
          style={{ '--mc': MODEL_COLORS[i] }}
          onClick={() => setMi(i)}
        >
          <div className="mcard-dot" style={{ background: MODEL_COLORS[i] }} />
          <div>
            <div className="mcard-name">{name}</div>
            <div className="acc-track">
              <div
                className="acc-fill"
                style={{ width: `${MODEL_ACC[i]}%`, background: MODEL_COLORS[i] }}
              />
            </div>
          </div>
          <div className="mcard-acc">{MODEL_ACC[i]}%</div>
        </div>
      ))}

      <div className="model-bar-hint">
        Select model to update all predictions instantly
      </div>
    </div>
  );
}
