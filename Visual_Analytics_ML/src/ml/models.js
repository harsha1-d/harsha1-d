// ---------------------------------------------
//  ML Utility helpers
// ---------------------------------------------
export const mean = (arr) =>
  arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;

export const percentile = (arr, p) => {
  const s = [...arr].sort((a, b) => a - b);
  const i = (p / 100) * (s.length - 1);
  const lo = Math.floor(i);
  return s[lo] + (s[lo + 1] !== undefined ? s[lo + 1] - s[lo] : 0) * (i - lo);
};

export const binarize = (vals, q1, q2) =>
  vals.map((v) => (v <= q1 ? 0 : v <= q2 ? 1 : 2));

/** Extract 10 numeric features from a raw data row */
export const getFeatures = (row) => {
  const angle = ((row[1] - 1) / 12) * 2 * Math.PI;
  return [
    Math.sin(angle),          // month sin encoding
    Math.cos(angle),          // month cos encoding
    row[2] / 2,               // branch  (0-1)
    row[3],                   // custType (0/1)
    row[4],                   // gender  (0/1)
    row[5] / 5,               // productLine (0-1)
    row[6] / 2,               // payment (0-1)
    row[7] / 100,             // unitPrice (0-1)
    row[8] / 10,              // qty (0-1)
    (row[0] - 2019) / 6,      // year normalised (0-1)
  ];
};

// ---------------------------------------------
//  Seeded random number generator (for RF)
// ---------------------------------------------
function seededRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

// ---------------------------------------------
//  1. Naive Bayes (Gaussian)
// ---------------------------------------------
export class NaiveBayes {
  fit(X, y) {
    this.classes = [0, 1, 2];
    this.prior = {};
    this.mu = {};
    this.sd = {};

    this.classes.forEach((c) => {
      const idx = y.flatMap((yi, i) => (yi === c ? [i] : []));
      this.prior[c] = idx.length / y.length || 1e-7;

      this.mu[c] = X[0].map((_, j) =>
        idx.length ? mean(idx.map((i) => X[i][j])) : 0
      );
      this.sd[c] = X[0].map((_, j) => {
        if (!idx.length) return 1;
        const m = this.mu[c][j];
        return Math.sqrt(mean(idx.map((i) => (X[i][j] - m) ** 2)) + 1e-8);
      });
    });
  }

  _logProb(x, c) {
    let lp = Math.log(this.prior[c]);
    x.forEach((xi, j) => {
      const d = (xi - this.mu[c][j]) / this.sd[c][j];
      lp -= 0.5 * d * d + Math.log(this.sd[c][j] * 2.5066 + 1e-9);
    });
    return lp;
  }

  predict(X) {
    return X.map((x) => {
      const scores = this.classes.map((c) => this._logProb(x, c));
      return scores.indexOf(Math.max(...scores));
    });
  }
}

// ---------------------------------------------
//  2. Decision Tree (CART  -  information gain)
// ---------------------------------------------
export class DecisionTree {
  _entropy(y) {
    const cnt = {};
    y.forEach((v) => (cnt[v] = (cnt[v] || 0) + 1));
    return -Object.values(cnt).reduce((s, n) => {
      const p = n / y.length;
      return s + p * Math.log2(p + 1e-12);
    }, 0);
  }

  _majority(y) {
    const cnt = {};
    y.forEach((v) => (cnt[v] = (cnt[v] || 0) + 1));
    return +Object.entries(cnt).sort((a, b) => b[1] - a[1])[0][0];
  }

  _buildNode(X, y, depth) {
    if (depth >= this.maxDepth || new Set(y).size <= 1 || y.length < 4) {
      return { leaf: true, val: this._majority(y) };
    }

    let bestGain = -1, bestFeat = 0, bestThresh = 0;
    const baseE = this._entropy(y);

    for (let j = 0; j < X[0].length; j++) {
      const vals = [...new Set(X.map((r) => r[j]))].sort((a, b) => a - b);
      for (let t = 0; t < vals.length - 1; t++) {
        const th = (vals[t] + vals[t + 1]) / 2;
        const li = [], ri = [];
        X.forEach((r, i) => (r[j] <= th ? li : ri).push(i));
        if (!li.length || !ri.length) continue;

        const g =
          baseE -
          (li.length / y.length) * this._entropy(li.map((i) => y[i])) -
          (ri.length / y.length) * this._entropy(ri.map((i) => y[i]));

        if (g > bestGain) { bestGain = g; bestFeat = j; bestThresh = th; }
      }
    }

    if (bestGain <= 0) return { leaf: true, val: this._majority(y) };

    const li = X.map((_, i) => i).filter((i) => X[i][bestFeat] <= bestThresh);
    const ri = X.map((_, i) => i).filter((i) => X[i][bestFeat] > bestThresh);

    return {
      leaf: false,
      feat: bestFeat,
      thresh: bestThresh,
      left:  this._buildNode(li.map((i) => X[i]), li.map((i) => y[i]), depth + 1),
      right: this._buildNode(ri.map((i) => X[i]), ri.map((i) => y[i]), depth + 1),
    };
  }

  fit(X, y, maxDepth = 5) {
    this.maxDepth = maxDepth;
    this.root = this._buildNode(X, y, 0);
  }

  _predictOne(x, node) {
    return node.leaf
      ? node.val
      : x[node.feat] <= node.thresh
      ? this._predictOne(x, node.left)
      : this._predictOne(x, node.right);
  }

  predict(X) {
    return X.map((x) => this._predictOne(x, this.root));
  }
}

// ---------------------------------------------
//  3. K-Nearest Neighbours
// ---------------------------------------------
export class KNN {
  fit(X, y, k = 5) {
    this.X = X;
    this.y = y;
    this.k = k;
  }

  _dist(a, b) {
    return Math.sqrt(a.reduce((s, ai, i) => s + (ai - b[i]) ** 2, 0));
  }

  predict(X) {
    return X.map((x) => {
      const dists = this.X.map((xi, i) => ({ d: this._dist(x, xi), l: this.y[i] }));
      dists.sort((a, b) => a.d - b.d);
      const knn = dists.slice(0, this.k);
      const cnt = {};
      knn.forEach(({ l }) => (cnt[l] = (cnt[l] || 0) + 1));
      return +Object.entries(cnt).sort((a, b) => b[1] - a[1])[0][0];
    });
  }
}

// ---------------------------------------------
//  4. Random Forest
// ---------------------------------------------
export class RandomForest {
  fit(X, y, nTrees = 12, maxDepth = 4) {
    const rng = seededRng(42);
    this.trees = [];

    for (let t = 0; t < nTrees; t++) {
      // Bootstrap sample
      const n = X.length;
      const idx = Array.from({ length: n }, () => Math.floor(rng() * n));

      // Random feature subset (sqrt)
      const allFeats = Array.from({ length: X[0].length }, (_, i) => i);
      for (let i = allFeats.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [allFeats[i], allFeats[j]] = [allFeats[j], allFeats[i]];
      }
      const fi = allFeats.slice(0, Math.max(2, Math.ceil(Math.sqrt(X[0].length))));

      const Xs = idx.map((i) => fi.map((j) => X[i][j]));
      const ys = idx.map((i) => y[i]);

      const tree = new DecisionTree();
      tree.fit(Xs, ys, maxDepth);
      this.trees.push({ tree, fi });
    }
  }

  predict(X) {
    const allPreds = this.trees.map(({ tree, fi }) =>
      tree.predict(X.map((x) => fi.map((j) => x[j])))
    );
    return X.map((_, i) => {
      const cnt = {};
      allPreds.forEach((p) => (cnt[p[i]] = (cnt[p[i]] || 0) + 1));
      return +Object.entries(cnt).sort((a, b) => b[1] - a[1])[0][0];
    });
  }
}
