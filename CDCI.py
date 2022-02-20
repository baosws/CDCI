#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from cdt.causality.pairwise import ANM, CDS, IGCI, RECI
from collections import Counter
import argparse
import os

EPSILON = 1e-8

def score(y_true, predictions):
    ret = (roc_auc_score(y_true == 1, predictions) + roc_auc_score(y_true == -1, -predictions)) / 2
    return round(ret, 3)

existing_methods = dict(
    ANM=ANM(),
    CDS=CDS(),
    IGCI=IGCI(),
    RECI=RECI()
)

def cond_dist(x, y, max_dev=3):
    vmax =  2 * max_dev
    vmin = -2 * max_dev

    x = (x - x.mean()) / (x.std() + EPSILON)
    t = x[np.abs(x) < max_dev]
    x = (x - t.mean()) / (t.std() + EPSILON)
    xd = np.round(x * 2)
    xd[xd > vmax] = vmax
    xd[xd < vmin] = vmin

    x_count = Counter(xd)
    vrange = range(vmin, vmax + 1)

    pyx = []
    for x in x_count:
        if x_count[x] > 12:
            yx = y[xd == x]
            yx = (yx - np.mean(yx)) / (np.std(yx) + EPSILON)
            yx = np.round(yx * 2)
            yx[yx > vmax] = vmax
            yx[yx < vmin] = vmin
            count_yx = Counter(yx.astype(int))
            pyx_x = np.array([count_yx[i] for i in vrange], dtype=np.float64)
            pyx_x = pyx_x / pyx_x.sum()
            pyx.append(pyx_x)
    return pyx

def CKL(A, B, **kargs):
    '''Causal score via Kullback-Leibler divergence'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return (pyx * np.log((pyx + EPSILON) / (mean_y + EPSILON))).sum(axis=1).mean()

def CKM(A, B, **kargs):
    '''Causal score via Kullback-Leibler divergence'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0).cumsum()
    pyx = pyx.cumsum(axis=1)

    return np.abs(pyx - mean_y).max(axis=1).mean()

def CHD(A, B, **kargs):
    '''Causal score via Hellinger Distance'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return (((pyx ** 0.5 - mean_y ** 0.5) ** 2).sum(axis=1) ** 0.5).mean()

def CCS(A, B, **kargs):
    '''Causal score via Chi-Squared distance'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return ((pyx - mean_y) ** 2 / (mean_y + EPSILON)).sum(axis=1).mean()

def CTV(A, B, **kargs):
    '''Causal score via Total Variation distance'''
    pyx = cond_dist(A, B, **kargs)
    if len(pyx) == 0:
        return 0
    pyx = np.array(pyx) # axis 0: x; axis 1: y
    mean_y = pyx.mean(axis=0)
    return 0.5 * np.abs(pyx - mean_y).sum(axis=1).mean()

def causal_score(variant, A, B, **kargs):
    variant = eval(variant)
    return variant(B, A, **kargs) - variant(A, B, **kargs)

def predict(method, data, **kargs):
    if method in existing_methods:
        return data.apply(lambda row: existing_methods[method].predict_proba((row['A'], row['B'])), axis=1)
    return data.apply(lambda row: causal_score(method, row['A'], row['B'], **kargs), axis=1)

def parse_numeric(df):
    parse_cell = lambda cell: np.fromstring(cell.replace('[', '').replace(']', ''), dtype=np.float64, sep=" ")
    df = df.applymap(parse_cell)
    return df

def write_predictions(directory, method, dataset, index, preds):
    if directory:
        os.makedirs(directory, exist_ok=True)
        df = pd.DataFrame(dict(SampleID=index, Target=preds))
        df.to_csv(f'{directory}/{dataset}_{method}.csv', index=False)

def compare_all():
    all_data = []
    results = []
    for dataset in args.datasets + ['All']:
        print(dataset.center(20, '-'))
        if dataset == 'All':
            data = pd.concat(all_data)
        else:
            data = pd.read_csv(f'data/{dataset}.csv', index_col='SampleID')
            data[['A', 'B']] = parse_numeric(data[['A', 'B']])
            all_data.append(data)
        print(f'{data.shape = }')
        targets = data['Target']

        for method in args.methods:
            predictions = predict(method, data, max_dev=args.maxdev)
            auc = score(targets, predictions)
            print(method.ljust(5, ' '), auc)
            results.append(dict(Method=method, Dataset=dataset, Score=auc))
            # write_predictions(args.out, method, dataset, targets.index, predictions)

    df = pd.DataFrame(results)
    df = pd.pivot(df, index='Method', columns='Dataset', values='Score')
    df.to_csv(args.out)

def test_hyperparameters():
    param_grid = [dict(max_dev=max_dev) for max_dev in range(1, 6)]
    all_data = []
    for dataset in args.datasets:
        data = pd.read_csv(f'data/{dataset}.csv', index_col='SampleID')
        data[['A', 'B']] = parse_numeric(data[['A', 'B']])
        all_data.append(data)
    data = pd.concat(all_data)
    targets = data['Target']

    results = []
    for params in param_grid:
        print(str(params).ljust(40, '-'))
        for method in args.methods:
            predictions = predict(method, data, **params)
            auc = score(targets, predictions)
            print(method.ljust(6, ' '), auc)
            results.append(dict(Method=method, **params, Score=auc))

    df = pd.DataFrame(results)
    df.to_csv(args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', '-m', nargs='+', default='ANM CDS IGCI RECI CCS CHD CKL CKM CTV'.split(), help='Applicable methods: ANM, CDS, IGCI, RECI, CCS, CHD, CKL, CKM, CTV. [Default: All methods]')
    parser.add_argument('--datasets', '-d', nargs='+', default='CE-Cha CE-Gauss CE-Net CE-Multi D4S1 D4S2A D4S2B D4S2C'.split(), help='Available data sets: CE-Cha, CE-Gauss, CE-Net, CE-Multi, D4S1, D4S2A, D4S2B, and D4S2C. [Default: All data sets]')
    parser.add_argument('--maxdev', default=3, type=int, help='Discretization parameter')
    parser.add_argument('--out', '-o', metavar='PATH', default=None, help='Output file')
    parser.add_argument('--hyper', action="store_true", help='Test hyperparameters')
    args = parser.parse_args()

    if args.hyper:
        test_hyperparameters()
    else:
        compare_all()
