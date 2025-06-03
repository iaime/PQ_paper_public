import stan
import argparse
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from simulations import simulate_data

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

    # print(pos_validation_dist)
    # print(neg_validation_dist)
    # print(test_dist)
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

def run(train_size, train_prev, val_size, val_prev, testing_sizes, test_prev, mean, number_of_samples, n_buckets, confidence, fixed_buckets=False, calibrate=False):
    mean1, mean2 = 0, mean
    sd = 1
    ret = []
    for test_size in testing_sizes:
        print('test size', test_size)
        for fold in range(1,11,1):
            print('fold', fold)
            #simulate data
            training_data = simulate_data(train_size, train_prev, mean1, mean2, sd)
            test_data = simulate_data(test_size, test_prev, mean1, mean2, sd)
            validation_data = simulate_data(val_size, val_prev, mean1, mean2, sd)

            #fit logistic model and calibrated it if desired
            logistic_model = LogisticRegression(random_state=0).fit(np.array(training_data['x']).reshape(-1,1), np.array(training_data['y']))
            if calibrate:
                calibrated_model = CalibratedClassifierCV(logistic_model, cv='prefit')
                calibrated_model.fit([[x] for x in validation_data['x']], [y for y in validation_data['y']])
            
                test_data['y_pred'] = calibrated_model.predict_proba(np.array(test_data['x']).reshape(-1,1))[:, 1]
                validation_data['y_pred'] = calibrated_model.predict_proba(np.array(validation_data['x']).reshape(-1,1))[:, 1]
            else:
                test_data['y_pred'] = logistic_model.predict_proba(np.array(test_data['x']).reshape(-1,1))[:, 1]
                validation_data['y_pred'] = logistic_model.predict_proba(np.array(validation_data['x']).reshape(-1,1))[:, 1]

            prev_distribution = pq_stan(validation_data, test_data, number_of_samples, n_buckets=n_buckets, fixed_buckets=fixed_buckets)[0]
            
            low_bound = (100 - confidence)/2
            low = np.percentile(prev_distribution, low_bound)
            high = np.percentile(prev_distribution, low_bound + confidence)
            ret.append((len(test_data), np.mean(test_data['y']), 'PQ', np.mean(prev_distribution), low, high))

    return ret


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--train_prev', action='store', type=float, required=True)
    parser.add_argument('-l', '--val_prev', action='store', type=float, required=True)
    parser.add_argument('-t', '--train_size', action='store', type=int, required=True)
    parser.add_argument('-v', '--val_size', action='store', type=int, required=True)
    parser.add_argument('-d', '--mean', action='store', type=float, required=True)
    parser.add_argument('-m', '--number_of_samples', action='store', type=int, required=True)
    parser.add_argument('-n', '--confidence', action='store', type=int, required=True)
    parser.add_argument('-b', '--n_buckets', action='store', type=int, required=True)
    parser.add_argument('-c', '--calibrate', action='store_true')
    parser.add_argument('-f', '--fixed_buckets', action='store_true')

    args = parser.parse_args()
    for test_prev in range(101):
        print('prev', test_prev)
        train_prev = args.train_prev
        val_prev = args.val_prev  
        test_prev = test_prev/100
        testing_sizes = [100, 500, 1000, 5000, 10000]
        mean = args.mean

        calibration_prefix = 'calibrated_' if args.calibrate else ''
        bucket_type = 'fixed_bins_' if args.fixed_buckets else 'adaptive_bins_'
        output_filename = f'../simulations/{calibration_prefix}{bucket_type}pq_{args.n_buckets}buckets_mean{mean}_train_size{args.train_size}_train_prev{train_prev}_val_size{args.val_size}_val_prev{val_prev}_test_prev{test_prev}_{args.number_of_samples}samples_confidence{args.confidence}.csv'
    
        if os.path.isfile(output_filename): 
            continue
        print('mean', mean)

        quantities = run(args.train_size, train_prev, args.val_size, val_prev, testing_sizes, test_prev, mean, args.number_of_samples, args.n_buckets, args.confidence, fixed_buckets=args.fixed_buckets, calibrate=args.calibrate)
        quantities = pd.DataFrame(quantities, columns=['test_size', 'test_prev', 'method', 'mean', f'low {args.confidence}', f'high {args.confidence}'])
        quantities.to_csv(output_filename)