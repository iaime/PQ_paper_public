import stan
import argparse
import pandas as pd
import numpy as np
import os, sys
from quantifiers import stochastic_rounder
from sklearn.calibration import CalibratedClassifierCV
from calibration_helper import Predictor

def pq_stan(validation_data, test_data, number_of_samples, n_buckets, fixed_buckets=False):
    with open('../scripts/PQ.stan', 'r') as f:
        stan_code = str(f.read())
    
    #bucketize validation data
    bins, buckets = None, None
    if fixed_buckets:
        bins = [-1e-6]+ [(i+1)/n_buckets for i in range(n_buckets)]
        print(len(bins), bins)
        buckets = pd.cut(validation_data['y_pred'], bins)
    else:
        buckets = pd.qcut(validation_data['y_pred'], n_buckets)
        buckets_intervals = buckets.cat.categories.values.tolist()
        bins = [-1e-6] + [x.left for x in buckets_intervals[1:]] + [1.0]

    validation_data['buckets'] = buckets
    pos_validation_data = validation_data[validation_data['y']==1]
    neg_validation_data = validation_data[validation_data['y']==0]

    pos_validation_dist = pos_validation_data.groupby('buckets').count()
    neg_validation_dist = neg_validation_data.groupby('buckets').count()

    #bucketize test data
    test_buckets = pd.cut(test_data['y_pred'], bins=bins)
    test_data['buckets'] = test_buckets
    test_dist = test_data.groupby('buckets').count()

    stan_data = {
        'n_bucket': n_buckets,
        'train_neg': list(neg_validation_dist['y_pred']),
        'train_pos': list(pos_validation_dist['y_pred']),
        'test': list(test_dist['y_pred']),
        'posterior': 1
    }

    # print(stan_data)

    stan_model = stan.build(stan_code, data=stan_data, random_seed=1)
    fit = stan_model.sample(num_chains=1, num_samples=number_of_samples)

    return fit['prev']


def run(val_size, val_prev, testing_sizes, test_prev, dataset, number_of_samples, n_buckets, confidence, fixed_buckets=False, calibrate=False):
    all_data = pd.read_csv(f'../reviews_datasets/{dataset}_testing_predictions.csv')
    all_data = all_data.sample(frac=1)
    all_data = all_data.rename(columns={'label': 'y', 'predictions': 'y_pred'})
    all_data['ids'] = [i for i in range(len(all_data))]
    all_positive = all_data[all_data['y']==1]
    all_negative = all_data[all_data['y']==0]

    ret = []
    for test_size in testing_sizes:
        print('test size', test_size)
        for fold in range(1,11,1):
            print('fold', fold)
            n_positive = int(stochastic_rounder(val_size * val_prev, 0))
            n_negative = val_size - n_positive
            validation_pos = all_positive.sample(n=n_positive, replace=False)
            validation_neg = all_negative.sample(n=n_negative, replace=False)
            validation_data = pd.concat([validation_pos, validation_neg])
            validation_data = validation_data.sample(frac=1)

            test_data = all_data[~all_data['ids'].isin(list(validation_data['ids']))]
            print('before', 'val data len', len(validation_data), 'test data len', len(test_data))

            n_positive = int(stochastic_rounder(test_size * test_prev, 0))
            n_negative = test_size - n_positive
            positive, negative = None, None
            if len(test_data[test_data['y']==1]) < n_positive:
                positive = test_data[test_data['y']==1].sample(n=n_positive, replace=True)
            else:
                positive = test_data[test_data['y']==1].sample(n=n_positive, replace=False)

            if len(test_data[test_data['y']==0]) < n_negative:
                negative = test_data[test_data['y']==0].sample(n=n_negative, replace=True)
            else:
                negative = test_data[test_data['y']==0].sample(n=n_negative, replace=False)

            test_data = pd.concat([negative, positive])
            test_data = test_data.sample(frac=1)

            print('after', 'val data len', len(validation_data), 'test data len', len(test_data))
            print('val data prev', np.mean(validation_data['y']), 'test data prev', np.mean(test_data['y']))

            #calibrate probabilities if desired
            logistic_model = Predictor()
            logistic_model.fit([[x] for x in validation_data['y_pred']], [y for y in validation_data['y']])
            calibrated_model = CalibratedClassifierCV(logistic_model, cv='prefit')

            if calibrate:
                calibrated_model.fit([[x] for x in validation_data['y_pred']], [y for y in validation_data['y']])
            
                test_data['y_pred'] = calibrated_model.predict_proba([[x] for x in test_data['y_pred']])[:, 1]
                validation_data['y_pred'] = calibrated_model.predict_proba([[x] for x in validation_data['y_pred']])[:, 1]

            prev_distribution = pq_stan(validation_data, test_data, number_of_samples, n_buckets=n_buckets, fixed_buckets=fixed_buckets)[0]
            
            low_bound = (100 - confidence)/2
            low = np.percentile(prev_distribution, low_bound)
            high = np.percentile(prev_distribution, low_bound + confidence)
            ret.append((len(test_data), np.mean(test_data['y']), 'PQ', np.mean(prev_distribution), low, high))

    return ret


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--val_size', action='store', type=int, required=True)
    parser.add_argument('-l', '--val_prev', action='store', type=float, required=True)
    parser.add_argument('-d', '--dataset', action='store', type=str, required=True)
    parser.add_argument('-m', '--number_of_samples', action='store', type=int, required=True)
    parser.add_argument('-n', '--confidence', action='store', type=int, required=True)
    parser.add_argument('-b', '--n_buckets', action='store', type=int, required=True)
    parser.add_argument('-c', '--calibrate', action='store_true')
    parser.add_argument('-f', '--fixed_buckets', action='store_true')


    args = parser.parse_args()
    for test_prev in range(101):
        print('prev', test_prev)
        test_prev = test_prev/100
        testing_sizes = [100, 500, 1000, 5000, 10000]

        calibration_prefix = 'calibrated_' if args.calibrate else ''
        bucket_type = 'fixed_bins_' if args.fixed_buckets else 'adaptive_bins_'
        output_filename = f'../reviews/{calibration_prefix}{bucket_type}pq_{args.n_buckets}buckets_{args.dataset}_val_size{args.val_size}_val_prev{args.val_prev}_test_prev{test_prev}_{args.number_of_samples}samples_confidence{args.confidence}.csv'
    
        if os.path.isfile(output_filename): 
            continue
        print('dataset', args.dataset)
        quantities = run(args.val_size, args.val_prev, testing_sizes, test_prev, args.dataset, args.number_of_samples, args.n_buckets, args.confidence, fixed_buckets=args.fixed_buckets, calibrate=args.calibrate)
        quantities = pd.DataFrame(quantities, columns=['test_size', 'test_prev', 'method', 'mean', f'low {args.confidence}', f'high {args.confidence}'])
        quantities.to_csv(output_filename)

    
