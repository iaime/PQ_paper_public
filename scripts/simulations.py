import numpy as np
import pandas as pd
import os, argparse, sys
from quantifiers import stochastic_rounder, run_old_quantifiers, run_bayesianCC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def simulate_data(size, prev, mean1, mean2, sd):
    n_positive = int(stochastic_rounder(size * prev, 0))
    n_negative = size - n_positive
    negative_data = np.random.normal(mean1, sd, n_negative)
    positive_data = np.random.normal(mean2, sd, n_positive)
    negative_data = pd.DataFrame([(x, 0) for x in list(negative_data)], columns=['x', 'y'])
    positive_data = pd.DataFrame([(x, 1) for x in list(positive_data)], columns=['x', 'y'])
    data = pd.concat([positive_data, negative_data])
    data = data.sample(frac=1)
    return data

def run(train_size, train_prev, val_size, val_prev, testing_sizes, test_prev, mean, number_of_samples, confidence, bca=False):
    mean1, mean2 = 0, mean
    sd = 1
    all_ret = []
    for test_size in testing_sizes:
        print('test size', test_size)
        for fold in range(1,11,1):
            print('fold', fold)

            for method in ['PCC', 'PACC', 'EMQ', 'HDy', 'BayesianCC']:
                #simulate data
                training_data = simulate_data(train_size, train_prev, mean1, mean2, sd)
                test_data = simulate_data(test_size, test_prev, mean1, mean2, sd)
                validation_data = simulate_data(val_size, val_prev, mean1, mean2, sd)

                #fit logistic model and calibrated it if desired
                logistic_model = LogisticRegression(random_state=0).fit(np.array(training_data['x']).reshape(-1,1), np.array(training_data['y']))
                if method == 'EMQ':
                    calibrated_model = CalibratedClassifierCV(logistic_model, cv='prefit')
                    calibrated_model.fit([[x] for x in validation_data['x']], [y for y in validation_data['y']])
                
                    test_data['y_pred'] = calibrated_model.predict_proba(np.array(test_data['x']).reshape(-1,1))[:, 1]
                    validation_data['y_pred'] = calibrated_model.predict_proba(np.array(validation_data['x']).reshape(-1,1))[:, 1]
                else:
                    test_data['y_pred'] = logistic_model.predict_proba(np.array(test_data['x']).reshape(-1,1))[:, 1]
                    validation_data['y_pred'] = logistic_model.predict_proba(np.array(validation_data['x']).reshape(-1,1))[:, 1]

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
    parser.add_argument('-r', '--train_prev', action='store', type=float, required=True)
    parser.add_argument('-l', '--val_prev', action='store', type=float, required=True)
    parser.add_argument('-p', '--test_prev', action='store', type=float, required=True)
    parser.add_argument('-t', '--train_size', action='store', type=int, required=True)
    parser.add_argument('-v', '--val_size', action='store', type=int, required=True)
    parser.add_argument('-d', '--mean', action='store', type=float, required=True)
    parser.add_argument('-m', '--number_of_samples', action='store', type=int, required=True)
    parser.add_argument('-n', '--confidence', action='store', type=int, required=True)
    parser.add_argument('-b', '--bca', action='store_true')

    args = parser.parse_args()
    val_size = args.val_size
    train_prev = args.train_prev
    val_prev = args.val_prev   
    test_prev = args.test_prev/100
    testing_sizes = [100, 500, 1000, 5000, 10000]
    mean = args.mean
    
    bca_prefix = 'BCa_' if args.bca else ''

    output_filename = f'../simulations/{bca_prefix}mean{mean}_train_size{args.train_size}_train_prev{train_prev}_val_size{val_size}_val_prev{val_prev}_test_prev{test_prev}_{args.number_of_samples}samples_confidence{args.confidence}.csv'
    
    if os.path.isfile(output_filename): 
        sys.exit()
    print('mean', mean)

    quantities = run(args.train_size, train_prev, val_size, val_prev, testing_sizes, test_prev, mean, args.number_of_samples, args.confidence, bca=args.bca)
    quantities = pd.DataFrame(quantities, columns=['test_size', 'test_prev', 'method', 'mean', f'low {args.confidence}', f'high {args.confidence}'])
    quantities.to_csv(output_filename)


    
                    

                
