import {
  NaiveBayes, DecisionTree, KNN, RandomForest,
  mean, percentile, binarize, getFeatures,
} from './models';
import { TRAIN, TEST } from '../data/dataset';

// Pre-compute feature matrices once
export const TRAIN_X = TRAIN.map(getFeatures);
export const TEST_X  = TEST.map(getFeatures);

/**
 * Build + run a model for a given metric column index and model index.
 * Returns { pc: predictedClasses[], cm: classMeans[3] }
 */
export function runModel(metricIdx, modelIdx) {
  const trainVals = TRAIN.map((r) => r[metricIdx]);
  const q1 = percentile(trainVals, 33);
  const q2 = percentile(trainVals, 66);
  const trainBin = binarize(trainVals, q1, q2);

  // Class means  -  used to map class (0/1/2) back to a numeric value
  const classMeans = [0, 1, 2].map((c) => {
    const vs = trainVals.filter((_, i) => trainBin[i] === c);
    return vs.length ? mean(vs) : mean(trainVals);
  });

  let model;
  switch (modelIdx) {
    case 0: model = new NaiveBayes();  model.fit(TRAIN_X, trainBin);         break;
    case 1: model = new DecisionTree(); model.fit(TRAIN_X, trainBin, 5);     break;
    case 2: model = new KNN();          model.fit(TRAIN_X, trainBin, 5);     break;
    default: model = new RandomForest(); model.fit(TRAIN_X, trainBin, 12, 4); break;
  }

  const pc = model.predict(TEST_X);
  return { pc, cm: classMeans };
}

/**
 * Compute validation accuracy: train on first 70% of TRAIN, test on last 30%.
 */
export function validationAccuracy(metricIdx, modelIdx) {
  const trainVals = TRAIN.map((r) => r[metricIdx]);
  const q1 = percentile(trainVals, 33);
  const q2 = percentile(trainVals, 66);
  const trainBin = binarize(trainVals, q1, q2);

  const split = Math.floor(TRAIN.length * 0.7);
  const trX = TRAIN_X.slice(0, split);
  const trY = trainBin.slice(0, split);
  const vaX = TRAIN_X.slice(split);
  const vaY = trainBin.slice(split);

  let model;
  switch (modelIdx) {
    case 0: model = new NaiveBayes();  model.fit(trX, trY);         break;
    case 1: model = new DecisionTree(); model.fit(trX, trY, 5);     break;
    case 2: model = new KNN();          model.fit(trX, trY, 5);     break;
    default: model = new RandomForest(); model.fit(trX, trY, 12, 4); break;
  }

  const preds = model.predict(vaX);
  const correct = preds.filter((p, i) => p === vaY[i]).length;
  return parseFloat((correct / vaY.length * 100).toFixed(1));
}

/** Pre-compute accuracy for all 4 models on the sales column (index 9) */
export const MODEL_ACC = [0, 1, 2, 3].map((mi) => validationAccuracy(9, mi));

/** Pre-run all 4 models on key metrics  -  memoised at module level */
export const ALL_SALES = [0, 1, 2, 3].map((mi) => runModel(9,  mi));
export const ALL_GI    = [0, 1, 2, 3].map((mi) => runModel(10, mi));
export const ALL_RT    = [0, 1, 2, 3].map((mi) => runModel(11, mi));
