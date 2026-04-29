// -------------------------------------------------------------
//  Shared utilities
// -------------------------------------------------------------
export const mean = (arr) =>
  arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0;

export const percentile = (arr, p) => {
  const s = [...arr].sort((a, b) => a - b);
  const i = (p / 100) * (s.length - 1);
  const lo = Math.floor(i);
  return s[lo] + (s[lo + 1] !== undefined ? s[lo + 1] - s[lo] : 0) * (i - lo);
};

// -------------------------------------------------------------
//  Feature metadata  -  maps to actual dataset column names
//  idx 0: sin(month), 1: cos(month), 2: branch/2,
//      3: custType, 4: gender, 5: productLine/5,
//      6: payment/2, 7: unitPrice/100, 8: qty/10,
//      9: (year-2019)/6, 10: rating/10
// -------------------------------------------------------------
export const FEATURE_META = [
  { name:'Month (sin)',   col:'Date',         short:'Month',       fmt:(t)=> t >= 0 ? 'Jan-Jun period' : 'Jul-Dec period' },
  { name:'Month (cos)',   col:'Date',         short:'Season',      fmt:(t)=> t.toFixed(2) },
  { name:'Branch',        col:'Branch',       short:'Branch',      fmt:(t)=> ['Alex','Cairo','Giza'][Math.min(Math.round(t*2),2)] },
  { name:'Customer Type', col:'Customer type',short:'Cust. Type',  fmt:(t)=> t < 0.5 ? 'Member' : 'Normal' },
  { name:'Gender',        col:'Gender',       short:'Gender',      fmt:(t)=> t < 0.5 ? 'Female' : 'Male' },
  { name:'Product Line',  col:'Product line', short:'Product',     fmt:(t)=> ['Electronic accessories','Fashion accessories','Food and beverages','Health and beauty','Home and lifestyle','Sports and travel'][Math.min(Math.round(t*5),5)] },
  { name:'Payment Method',col:'Payment',      short:'Payment',     fmt:(t)=> t < 0.4 ? 'Cash' : t < 0.7 ? 'Credit Card' : 'E-Wallet' },
  { name:'Unit Price',    col:'Unit price',   short:'Unit Price',  fmt:(t)=> '$'+Math.round(t*100) },
  { name:'Quantity',      col:'Quantity',     short:'Quantity',    fmt:(t)=> Math.round(t*10)+' items' },
  { name:'Year',          col:'Date (year)',  short:'Year',        fmt:(t)=> String(Math.round(t*6+2019)) },
  { name:'Customer Rating',col:'Rating',      short:'Rating',      fmt:(t)=> (t*10).toFixed(1)+'/10' },
];

export const CLASS_NAMES  = ['Low Sale',    'Medium Sale',   'High Sale'];
export const CLASS_COLORS = ['#38bdf8',     '#f59e0b',       '#34d399'];
export const CLASS_ICONS  = ['🔵',          '🟡',            '🟢'];

// -------------------------------------------------------------
//  Feature extraction  -  11 features from a raw row
// -------------------------------------------------------------
export const getFeatures = (row) => {
  const angle = ((row[1] - 1) / 12) * 2 * Math.PI;
  return [
    Math.sin(angle),           // 0: month sin
    Math.cos(angle),           // 1: month cos
    row[2] / 2,                // 2: branch (0-1)
    row[3],                    // 3: customer type
    row[4],                    // 4: gender
    row[5] / 5,                // 5: product line (0-1)
    row[6] / 2,                // 6: payment (0-1)
    row[7] / 100,              // 7: unit price (normalised)
    row[8] / 10,               // 8: quantity (normalised)
    (row[0] - 2019) / 6,       // 9: year trend (0-1)
    row[11] / 10,              // 10: customer rating (normalised)
  ];
};

// -------------------------------------------------------------
//  Plain-English decision node questions (uses actual col names)
// -------------------------------------------------------------
export function businessQuestion(feat, thresh) {
  const price   = Math.round(thresh * 100);
  const qty     = (thresh * 10).toFixed(1);
  const year    = Math.round(thresh * 6 + 2019);
  const rating  = (thresh * 10).toFixed(1);
  const PL      = ['Electronic acc.','Fashion acc.','Food & Bev.','Health & Beauty','Home & Lifestyle','Sports & Travel'];

  switch (feat) {
    case 0: return {
      question: 'Is the sale in the first half of the year? (Jan-Jun)',
      yesLabel: 'YES - Jan to Jun',  noLabel: 'NO - Jul to Dec',
      col:'Date', icon:'📅',
      explanation:'Seasonal sales pattern check  -  first half vs second half of year',
    };
    case 1: return {
      question: 'Is it a year-end or year-start period?',
      yesLabel: 'YES - Q4 / Q1',    noLabel: 'NO - Mid-year',
      col:'Date', icon:'🗓',
      explanation:'Second seasonality check to capture holiday and new-year sales spikes',
    };
    case 2:
      if (thresh < 0.4)  return { question:'Is this the Alex branch?',         yesLabel:'YES - Alex',          noLabel:'NO - Cairo or Giza', col:'Branch',  icon:'🏪', explanation:'Location-based split  -  Alex branch vs other branches' };
      if (thresh < 0.85) return { question:'Is this Alex or Cairo branch?',    yesLabel:'YES - Alex / Cairo',  noLabel:'NO - Giza',          col:'Branch',  icon:'🏪', explanation:'Location-based split  -  Alex or Cairo vs Giza' };
      return                   { question:'Is this NOT the Giza branch?',       yesLabel:'YES - Alex / Cairo',  noLabel:'NO - Giza',          col:'Branch',  icon:'🏪', explanation:'Location-based split' };
    case 3: return {
      question: 'Is this customer a Loyalty Member?',
      yesLabel: 'YES - Member',     noLabel: 'NO - Normal customer',
      col:'Customer type', icon:'💳',
      explanation:'Customer Type column  -  Members have loyalty cards, Normal are walk-in',
    };
    case 4: return {
      question: 'Is the customer Female?',
      yesLabel: 'YES - Female',     noLabel: 'NO - Male',
      col:'Gender', icon:'👥',
      explanation:'Gender column  -  Female vs Male customer spending patterns',
    };
    case 5: {
      const idx     = Math.min(Math.round(thresh * 5), 4);
      const yesGrp  = PL.slice(0, idx + 1).join(', ');
      const noGrp   = PL.slice(idx + 1).join(', ') || 'Other';
      return { question:`Product in: ${yesGrp}?`,   yesLabel:'YES - '+yesGrp, noLabel:'NO - '+noGrp, col:'Product line', icon:'🛒', explanation:'Product Line column  -  different product categories have different average prices' };
    }
    case 6:
      if (thresh < 0.4)  return { question:'Paid by Cash?',                    yesLabel:'YES - Cash',          noLabel:'NO - Card / Ewallet', col:'Payment', icon:'💵', explanation:'Payment method  -  Cash payments may correlate with spending habits' };
      if (thresh < 0.85) return { question:'Paid by Cash or Credit Card?',     yesLabel:'YES - Cash/Card',     noLabel:'NO - E-Wallet',        col:'Payment', icon:'💳', explanation:'Payment method split  -  Cash or Card vs E-Wallet' };
      return                   { question:'Not paying by E-Wallet?',            yesLabel:'YES - Cash/Card',     noLabel:'NO - E-Wallet',        col:'Payment', icon:'📱', explanation:'Payment method  -  E-Wallet users vs traditional payment' };
    case 7: return {
      question: `Is Unit Price above $${price}?`,
      yesLabel: `YES - Unit Price > $${price}`,  noLabel: `NO - Unit Price <= $${price}`,
      col:'Unit price', icon:'💰',
      explanation:`Unit Price column: Items priced over $${price} per unit tend to generate higher total sales`,
      isKeyDriver: true,
    };
    case 8: return {
      question: `Did the customer buy more than ${qty} items?`,
      yesLabel: `YES - Qty > ${qty}`,            noLabel: `NO - Qty <= ${qty}`,
      col:'Quantity', icon:'📦',
      explanation:`Quantity column: Larger orders naturally result in higher total sales (Sales = Unit Price x Qty x 1.05)`,
      isKeyDriver: true,
    };
    case 9: return {
      question: `Is this transaction before year ${year}?`,
      yesLabel: `YES - Before ${year}`,          noLabel: `NO - ${year} or later`,
      col:'Date (year)', icon:'📆',
      explanation:'Year trend  -  sales patterns may shift over the 2019-2025 period',
    };
    case 10: return {
      question: `Is Customer Rating above ${rating}?`,
      yesLabel: `YES - Rating > ${rating}`,      noLabel: `NO - Rating <= ${rating}`,
      col:'Rating', icon:'*',
      explanation:'Rating column  -  Customer satisfaction score (4-10); low correlation with sales amount',
    };
    default: return {
      question: `Feature ${feat} <= ${thresh.toFixed(2)}?`,
      yesLabel: 'YES', noLabel: 'NO',
      col:'Unknown', icon:'?',
      explanation:'',
    };
  }
}

// -------------------------------------------------------------
//  Weighted entropy (class-weight bias control)
// -------------------------------------------------------------
function weightedEntropy(y, classWeights) {
  const n = y.length;
  if (!n) return 0;
  const cnt = [0, 0, 0];
  y.forEach((v) => cnt[v]++);
  let H = 0;
  for (let c = 0; c < 3; c++) {
    if (!cnt[c]) continue;
    const pw = (cnt[c] * classWeights[c]) / n;
    H -= pw * Math.log2(pw + 1e-12);
  }
  return H;
}

// -------------------------------------------------------------
//  Decision Tree  -  full CART implementation
// -------------------------------------------------------------
export class DecisionTree {
  fit(X, y, params = {}) {
    const {
      maxDepth            = 4,
      minSamplesSplit     = 10,
      classWeights        = [1, 1, 1],
      minImpurityDecrease = 0,
      activeFeatures      = null,
    } = params;

    this.params           = { maxDepth, minSamplesSplit, classWeights, minImpurityDecrease };
    this.nSamples         = y.length;
    this.nFeatures        = X[0].length;
    this.featureImportance = new Array(this.nFeatures).fill(0);

    const feats = activeFeatures || Array.from({ length: this.nFeatures }, (_, i) => i);
    this.root = this._build(X, y, 0, feats, {});
  }

  _classDist(y) { const c=[0,0,0]; y.forEach((v)=>c[v]++); return c; }
  _majority(y)  { const c=this._classDist(y); return c.indexOf(Math.max(...c)); }

  /**
   * Depth-constrained CART build with monotonic threshold enforcement.
   *  - Even depth (0,2,4): splits ONLY on Quantity (feat 8) if active
   *  - Odd  depth (1,3,5): splits ONLY on Unit Price (feat 7) if active
   *  - For each key feature, every child's threshold must EXCEED its ancestor's
   *    threshold on that same feature (parent always shows lesser value, child higher)
   *  minThreshByFeat: running map of { featIndex -> minThresholdRequired }
   */
  _build(X, y, depth, feats, minThreshByFeat) {
    const QTY_FEAT   = 8;
    const PRICE_FEAT = 7;

    const dist     = this._classDist(y);
    const majority = dist.indexOf(Math.max(...dist));
    const impurity = weightedEntropy(y, this.params.classWeights);
    const node     = { depth, nSamples:y.length, classDist:dist, majority, impurity, infoGain:0,
                       isLeaf:true, feat:null, thresh:null, left:null, right:null };

    if (depth >= this.params.maxDepth || y.length < this.params.minSamplesSplit || new Set(y).size <= 1)
      return node;

    // Which key feature is designated for this depth level
    const primaryFeat = depth % 2 === 0 ? QTY_FEAT   : PRICE_FEAT;
    const altFeat     = depth % 2 === 0 ? PRICE_FEAT : QTY_FEAT;

    // Candidate features: prefer the designated one exclusively;
    // fall back to all active features except the alternate key feat
    let depthFeats;
    if (feats.includes(primaryFeat)) {
      depthFeats = [primaryFeat];
    } else {
      depthFeats = feats.filter((f) => f !== altFeat);
      if (depthFeats.length === 0) depthFeats = feats;
    }

    // Minimum threshold for the designated feature (monotonic increase)
    const minThreshForPrimary =
      minThreshByFeat[primaryFeat] !== undefined ? minThreshByFeat[primaryFeat] : -Infinity;

    let bestGain = this.params.minImpurityDecrease, bestFeat = -1, bestThresh = 0;

    for (const j of depthFeats) {
      const vals = [...new Set(X.map((r)=>r[j]))].sort((a,b)=>a-b);
      for (let t = 0; t < vals.length-1; t++) {
        const th = (vals[t]+vals[t+1])/2;
        // Skip threshold if it violates the monotonic-increase constraint
        if (j === primaryFeat && th <= minThreshForPrimary) continue;
        const li=[], ri=[];
        X.forEach((r,i)=>(r[j]<=th?li:ri).push(i));
        if (!li.length||!ri.length) continue;
        const g = impurity
          - (li.length/y.length)*weightedEntropy(li.map((i)=>y[i]), this.params.classWeights)
          - (ri.length/y.length)*weightedEntropy(ri.map((i)=>y[i]), this.params.classWeights);
        if (g > bestGain) { bestGain=g; bestFeat=j; bestThresh=th; }
      }
    }

    if (bestFeat === -1) return node;

    this.featureImportance[bestFeat] += (bestGain * y.length) / this.nSamples;
    const li = X.map((_,i)=>i).filter((i)=>X[i][bestFeat]<=bestThresh);
    const ri = X.map((_,i)=>i).filter((i)=>X[i][bestFeat]>bestThresh);

    // Both subtrees inherit the updated minimum threshold for the key feature
    const childMinThresh = bestFeat === primaryFeat
      ? { ...minThreshByFeat, [primaryFeat]: bestThresh }
      : minThreshByFeat;

    node.isLeaf   = false;
    node.feat     = bestFeat;
    node.thresh   = bestThresh;
    node.infoGain = bestGain;
    node.left  = this._build(li.map((i)=>X[i]), li.map((i)=>y[i]), depth+1, feats, childMinThresh);
    node.right = this._build(ri.map((i)=>X[i]), ri.map((i)=>y[i]), depth+1, feats, childMinThresh);
    return node;
  }

  _predictOne(x, node) {
    return node.isLeaf ? node.majority
      : x[node.feat]<=node.thresh ? this._predictOne(x,node.left) : this._predictOne(x,node.right);
  }

  predict(X) { return X.map((x)=>this._predictOne(x,this.root)); }

  getFeatureImportance() {
    const total = this.featureImportance.reduce((s,v)=>s+v,0)||1;
    return this.featureImportance.map((v)=>+(v/total*100).toFixed(2));
  }
}
