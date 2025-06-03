import numpy as np
import pandas as pd
import os, sys, argparse
from quantifiers import run_old_quantifiers, run_bayesianCC, stochastic_rounder
from sklearn.calibration import CalibratedClassifierCV
from calibration_helper import Predictor

def run(val_size, val_prev, testing_sizes, test_prev, dataset, number_of_samples, confidence, bca=False):
    all_data = pd.read_csv(f'../reviews_datasets/{dataset}_testing_predictions.csv')
    all_data = all_data.sample(frac=1)
    all_data = all_data.rename(columns={'label': 'y', 'predictions': 'y_pred'})
    all_data['ids'] = [i for i in range(len(all_data))]
    all_positive = all_data[all_data['y']==1]
    all_negative = all_data[all_data['y']==0]

    all_ret = []
    for test_size in testing_sizes:
        print('test size', test_size)
        for fold in range(1,11,1):
            print('fold', fold)
            for method in ['PCC', 'PACC', 'EMQ', 'HDy', 'BayesianCC']:
                #prepare data
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

                if method == 'EMQ':
                    calibrated_model = CalibratedClassifierCV(logistic_model, cv='prefit')
                    calibrated_model.fit([[x] for x in validation_data['y_pred']], [y for y in validation_data['y']])
                
                    test_data['y_pred'] = calibrated_model.predict_proba(np.array(test_data['y_pred']).reshape(-1,1))[:, 1]
                    validation_data['y_pred'] = calibrated_model.predict_proba(np.array(validation_data['y_pred']).reshape(-1,1))[:, 1]

                #run classical quantifiers
                if method != 'BayesianCC':
                    quant = run_old_quantifiers(validation_data, test_data, method, confidence, bca=bca, bootstrap_validation=True, number_of_samples=number_of_samples)
                    print(quant)
                    all_ret.extend(quant)
                else:
                    #run BayesianCC
                    quant = run_bayesianCC(validation_data, test_data, confidence, number_of_samples)
                    print(quant)
                    all_ret.extend(quant)
      
    return all_ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--test_prev', action='store', type=float, required=True)
    parser.add_argument('-v', '--val_size', action='store', type=int, required=True)
    parser.add_argument('-l', '--val_prev', action='store', type=float, required=True)
    parser.add_argument('-d', '--dataset', action='store', type=str, required=True)
    parser.add_argument('-n', '--confidence', action='store', type=int, required=True)
    parser.add_argument('-m', '--number_of_samples', action='store', type=int, required=True)

    args = parser.parse_args()
 
    test_prev = args.test_prev/100

    testing_sizes = [100, 500, 1000, 5000, 10000]


    output_filename = f'../reviews/{args.dataset}_val_size{args.val_size}_val_prev{args.val_prev}_test_prev{test_prev}_{args.number_of_samples}samples_confidence{args.confidence}.csv'
    
    if os.path.isfile(output_filename): 
        sys.exit()
    print('dataset', args.dataset)

    quantities = run(args.val_size, args.val_prev, testing_sizes, test_prev, args.dataset, args.number_of_samples, args.confidence, bca=args.bca)

    quantities = pd.DataFrame(quantities, columns=['test_size', 'test_prev', 'method', 'mean', f'low {args.confidence}', f'high {args.confidence}'])
    quantities.to_csv(output_filename)
