import { DecisionTree, getFeatures, mean, percentile, businessQuestion } from './models';
import { RAW_DATA, TRAIN, TEST } from '../data/dataset';

export const TRAIN_X = TRAIN.map(getFeatures);
export const TEST_X  = TEST.map(getFeatures);

// -------------------------------------------------------------
//  Comprehensive metrics: precision, recall, F1, CI, Kappa
// -------------------------------------------------------------
function computeFullMetrics(confMatrix, testY, testPreds) {
  const n = testY.length;
  const C = [0, 1, 2];

  const perClass = C.map((c) => {
    const TP = confMatrix[c][c];
    const FP = C.reduce((s, r) => r !== c ? s + confMatrix[r][c] : s, 0);
    const FN = confMatrix[c].reduce((s, v, i) => i !== c ? s + v : s, 0);
    const TN = n - TP - FP - FN;
    const prec  = TP + FP > 0 ? TP / (TP + FP) : 0;
    const rec   = TP + FN > 0 ? TP / (TP + FN) : 0;
    const f1    = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
    const spec  = TN + FP > 0 ? TN / (TN + FP) : 0;
    return { precision:prec, recall:rec, f1, specificity:spec,
             support:TP+FN, TP, FP, FN, TN };
  });

  const correct  = testPreds.filter((p, i) => p === testY[i]).length;
  const accuracy = correct / n;

  // 95% Confidence Interval (Wilson score)
  const z = 1.96;
  const wDen = 2 * (n + z * z);
  const wBase= 2 * correct + z * z;
  const wSqrt= z * Math.sqrt(z * z + 4 * correct * (1 - accuracy));
  const ciLo = (wBase - wSqrt) / wDen * 100;
  const ciHi = (wBase + wSqrt) / wDen * 100;

  // Cohen's Kappa
  const Pe = C.reduce((s, c) => {
    const pAct  = confMatrix[c].reduce((a, b) => a + b, 0) / n;
    const pPred = C.reduce((a, r) => a + confMatrix[r][c], 0) / n;
    return s + pAct * pPred;
  }, 0);
  const kappa = (1 - Pe) > 0 ? (accuracy - Pe) / (1 - Pe) : 0;

  const kappaLabel =
    kappa < 0    ? 'Worse than chance' :
    kappa < 0.20 ? 'Slight agreement'  :
    kappa < 0.40 ? 'Fair agreement'    :
    kappa < 0.60 ? 'Moderate agreement':
    kappa < 0.80 ? 'Substantial agreement' : 'Almost perfect agreement';

  const totalSupport = n;
  const macro = {
    precision: mean(perClass.map((m) => m.precision)),
    recall:    mean(perClass.map((m) => m.recall)),
    f1:        mean(perClass.map((m) => m.f1)),
  };
  const weighted = {
    precision: perClass.reduce((s, m) => s + m.precision * m.support, 0) / totalSupport,
    recall:    perClass.reduce((s, m) => s + m.recall    * m.support, 0) / totalSupport,
    f1:        perClass.reduce((s, m) => s + m.f1        * m.support, 0) / totalSupport,
  };

  return { accuracy, correct, n, ciLo, ciHi, kappa, kappaLabel, perClass, macro, weighted };
}

// -------------------------------------------------------------
//  Pearson r & eta coefficient for correlation analysis
// -------------------------------------------------------------
function pearsonR(x, y) {
  const n = x.length;
  const mx = mean(x), my = mean(y);
  const num = x.reduce((s, xi, i) => s + (xi - mx) * (y[i] - my), 0);
  const dx  = Math.sqrt(x.reduce((s, xi) => s + (xi - mx) ** 2, 0));
  const dy  = Math.sqrt(y.reduce((s, yi) => s + (yi - my) ** 2, 0));
  return dx * dy > 0 ? num / (dx * dy) : 0;
}

function etaCoeff(groupCol, sales) {
  const grandMean = mean(sales);
  const ssTot     = sales.reduce((s, v) => s + (v - grandMean) ** 2, 0);
  const groups    = [...new Set(groupCol)];
  const ssBetween = groups.reduce((s, g) => {
    const grp = sales.filter((_, i) => groupCol[i] === g);
    return s + grp.length * (mean(grp) - grandMean) ** 2;
  }, 0);
  return ssTot > 0 ? Math.sqrt(ssBetween / ssTot) : 0;
}

export function computeCorrelations() {
  const sales = RAW_DATA.map((r) => r[9]);
  return [
    { name:'Unit Price',    col:'Unit price',    r: pearsonR(RAW_DATA.map((r)=>r[7]),  sales),  type:'numeric',      interpretation:'Each $1 increase in unit price tends to increase sales by this proportion' },
    { name:'Quantity',      col:'Quantity',       r: pearsonR(RAW_DATA.map((r)=>r[8]),  sales),  type:'numeric',      interpretation:'More items in one transaction directly multiplies total sales' },
    { name:'Customer Rating',col:'Rating',        r: pearsonR(RAW_DATA.map((r)=>r[11]), sales),  type:'numeric',      interpretation:'Customer satisfaction has almost no linear relationship with sale size' },
    { name:'Product Line',  col:'Product line',   r: etaCoeff(RAW_DATA.map((r)=>r[5]),  sales),  type:'categorical',  interpretation:'Different product lines have slightly different average unit prices' },
    { name:'Payment Method',col:'Payment',        r: etaCoeff(RAW_DATA.map((r)=>r[6]),  sales),  type:'categorical',  interpretation:'Payment method has very weak association with sale amount' },
    { name:'Branch / City', col:'Branch',         r: etaCoeff(RAW_DATA.map((r)=>r[2]),  sales),  type:'categorical',  interpretation:'All three branches have very similar average sales amounts' },
    { name:'Customer Type', col:'Customer type',  r: etaCoeff(RAW_DATA.map((r)=>r[3]),  sales),  type:'categorical',  interpretation:'Member vs Normal customers spend nearly identical amounts per transaction' },
    { name:'Gender',        col:'Gender',         r: etaCoeff(RAW_DATA.map((r)=>r[4]),  sales),  type:'categorical',  interpretation:'Male and female customers have near-identical average transaction values' },
    { name:'Month/Season',  col:'Date',           r: Math.abs(pearsonR(RAW_DATA.map((r)=>r[1]), sales)), type:'numeric', interpretation:'Month has very little linear correlation with transaction amount' },
  ].sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
}

// -------------------------------------------------------------
//  Extract top decision paths for the story panel
// -------------------------------------------------------------
function extractPaths(node, path = []) {
  if (!node) return [];
  if (node.isLeaf) return [{ path, leaf: node }];
  const bq = businessQuestion(node.feat, node.thresh);
  return [
    ...extractPaths(node.left,  [...path, { bq, dir:'yes', nSamples: node.left?.nSamples  || 0 }]),
    ...extractPaths(node.right, [...path, { bq, dir:'no',  nSamples: node.right?.nSamples || 0 }]),
  ];
}

export function getTopPaths(root, topN = 3) {
  const allPaths = extractPaths(root);
  return allPaths
    .sort((a, b) => b.leaf.nSamples - a.leaf.nSamples)
    .slice(0, topN)
    .map((p) => {
      const total    = root.nSamples;
      const pct      = Math.round(p.leaf.nSamples / total * 100);
      const dist     = p.leaf.classDist;
      const distTot  = dist.reduce((s, v) => s + v, 0) || 1;
      const purity   = Math.round(dist[p.leaf.majority] / distTot * 100);
      return { path: p.path, leaf: p.leaf, pct, purity };
    });
}

// -------------------------------------------------------------
//  Main: build + evaluate Decision Tree
// -------------------------------------------------------------
export function buildAndEval(params) {
  const {
    maxDepth         = 4,
    minSamplesSplit  = 10,
    classWeights     = [1, 1, 1],
    lowBoundaryPct   = 33,
    highBoundaryPct  = 66,
    activeFeatures   = [0,1,2,3,4,5,6,7,8,9,10],
  } = params;

  // 1. Class boundaries
  const trainSales = TRAIN.map((r) => r[9]);
  const q1         = percentile(trainSales, lowBoundaryPct);
  const q2         = percentile(trainSales, highBoundaryPct);
  const bin        = (v) => v <= q1 ? 0 : v <= q2 ? 1 : 2;
  const classMeans = [0,1,2].map((c) => {
    const vs = trainSales.filter((_, i) => bin(trainSales[i]) === c);
    return vs.length ? mean(vs) : mean(trainSales);
  });

  const trainY     = trainSales.map(bin);
  const testSales  = TEST.map((r) => r[9]);
  const testY      = testSales.map(bin);

  // 2. Train
  const model = new DecisionTree();
  model.fit(TRAIN_X, trainY, { maxDepth, minSamplesSplit, classWeights:classWeights.map(Number), activeFeatures });

  // 3. Predict
  const trainPreds = model.predict(TRAIN_X);
  const testPreds  = model.predict(TEST_X);

  // 4. Confusion matrix
  const confMatrix = [[0,0,0],[0,0,0],[0,0,0]];
  testPreds.forEach((p, i) => confMatrix[testY[i]][p]++);

  // 5. Full metrics
  const trainCorrect = trainPreds.filter((p, i) => p === trainY[i]).length;
  const trainAcc = +(trainCorrect / trainY.length * 100).toFixed(1);
  const metrics  = computeFullMetrics(confMatrix, testY, testPreds);
  const testAcc  = +(metrics.accuracy * 100).toFixed(1);
  const perClassAcc = metrics.perClass.map((m) => +(m.recall * 100).toFixed(1));

  // 6. Feature importance
  const fi = model.getFeatureImportance();

  // 7. Tree stats
  let nNodes = 0, nLeaves = 0, maxActualDepth = 0;
  const walkTree = (node) => {
    if (!node) return;
    nNodes++;
    if (node.depth > maxActualDepth) maxActualDepth = node.depth;
    if (node.isLeaf) nLeaves++;
    else { walkTree(node.left); walkTree(node.right); }
  };
  walkTree(model.root);

  // 8. Top decision paths (for story panel)
  const topPaths = getTopPaths(model.root, 4);

  // 9. Monthly prediction series
  const monthlyMap = {};
  RAW_DATA.forEach((r) => {
    const k = r[0]+'-'+String(r[1]).padStart(2,'0');
    if (!monthlyMap[k]) monthlyMap[k] = {
      label: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][r[1]-1]+"'"+String(r[0]).slice(2),
      actual:0, pred:null, year:r[0],
    };
    monthlyMap[k].actual += r[9];
  });
  TEST.forEach((r, i) => {
    const k = r[0]+'-'+String(r[1]).padStart(2,'0');
    if (!monthlyMap[k]) return;
    if (monthlyMap[k].pred === null) monthlyMap[k].pred = 0;
    monthlyMap[k].pred += classMeans[testPreds[i]];
  });
  const monthlySeries = Object.entries(monthlyMap)
    .sort(([a],[b]) => a.localeCompare(b)).map(([,v]) => v);

  return {
    root:model.root, fi, classMeans,
    trainAcc, testAcc, perClassAcc,
    confMatrix, trainY, testY, testPreds,
    q1, q2, nNodes, nLeaves, maxActualDepth,
    metrics, topPaths, monthlySeries,
  };
}

export function computeDepthCurve(baseParams) {
  return Array.from({ length:8 }, (_, i) => {
    const r = buildAndEval({ ...baseParams, maxDepth:i+1 });
    return { depth:i+1, trainAcc:r.trainAcc, testAcc:r.testAcc };
  });
}
