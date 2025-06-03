import numpy as np
import pandas as pd
import random, sys
from typing import Callable, Union
from scipy.stats import bootstrap
from quapy.method.aggregative import DyS, EMQ, BayesianCC
import quapy.functional as F
from quapy.functional import get_divergence
from quapy.method import _bayesian
from sklearn.metrics import confusion_matrix, matthews_corrcoef

def stochastic_rounder(p, decimal_points):
    d = 10**decimal_points
    p = int(p*d + random.random())
    p = p/d
    return p

def run_bayesianCC(validation_data, test_data, confidence, number_of_samples):
    quant = customBayesianCC(number_of_samples)
    quant.aggregation_fit(validation_data)
    samples = quant.aggregate(test_data)
    samples = [x[1] for x in samples]
    low_bound = (100 - confidence)/2
    low = np.percentile(samples, low_bound)
    high = np.percentile(samples, low_bound + confidence)
    return [[len(test_data), np.mean(test_data['y']), 'BayesianCC', np.mean(samples), low, high]]

def run_old_quantifiers(validation_data, test_data, method, confidence, bca=False, bootstrap_validation=True, number_of_samples=1000):
    classif_posteriors = test_data['y_pred']
    validation_prevalence = np.mean(validation_data['y_pred'])
    results = {}
    estimates = []

    if method == 'HDy':
        HDy_4 = customDyS(n_bins=4)
        HDy_4.fit(validation_data.copy())
        results['HDy'] = HDy_4.aggregate(classif_posteriors.copy())[1]

    if method == 'EMQ':
        emq = customEMQ()
        results['EMQ'] = emq.aggregate(np.array([[1-x, x] for x in classif_posteriors]), [1-validation_prevalence, validation_prevalence])[0][1]

    if method == 'PCC':
        pcc = customPCC()
        results['PCC'] = pcc.aggregate(classif_posteriors)[1]

    if method == 'PACC':
        pacc = customPACC()
        results['PACC'] = pacc.aggregate(classif_posteriors, validation_data.copy())[1]

    ests = {}
    confidence_level = confidence/100
    custom_stat = prevalence_statistic(method, validation_data.copy(), bootstrap_validation=bootstrap_validation)
    boot_res = bootstrap((classif_posteriors,), custom_stat, method='BCa' if bca else 'percentile', alternative='two-sided', n_resamples=number_of_samples, vectorized=False, confidence_level=confidence_level) 
    low = boot_res.confidence_interval.low
    high = boot_res.confidence_interval.high
    ests[confidence_level] = {}
    ests[confidence_level]['low'] = low
    ests[confidence_level]['high'] = high

    estimates.append([  len(test_data), np.mean(test_data['y']),
                        method, 
                        results[method], 
                        ests[confidence_level]['low'], ests[confidence_level]['high'], 
                    ])
    return estimates

def prevalence_statistic(method, validation_data, bootstrap_validation=True):
    def calculate_stat(classif_posteriors): 
        new_validation_data = validation_data.copy()
        if bootstrap_validation:
            positives = validation_data[validation_data['y']==1]
            negatives = validation_data[validation_data['y']==0]
            new_positives = positives.sample(n=len(positives), replace=True)
            new_negatives = negatives.sample(n=len(negatives), replace=True)
            new_validation_data = pd.concat([new_positives, new_negatives]).sample(frac=1)
        assert len(new_validation_data) == len(validation_data)
        validation_prevalence = np.mean(new_validation_data['y_pred'])
        if method == 'PACC':
            pacc = customPACC()
            pacc_results = pacc.aggregate(classif_posteriors.copy(), new_validation_data.copy())[1]
            return pacc_results
        elif method == 'CC':
            cc = CC()
            cc_results = cc.aggregate(classif_posteriors.copy())[1]
            return cc_results
        elif method == 'PCC':
            pcc = customPCC()
            pcc_results = pcc.aggregate(classif_posteriors.copy())[1]
            return pcc_results
        elif method == 'EMQ':
            emq = customEMQ()
            emq_results = emq.aggregate(np.array([[1-x, x] for x in classif_posteriors.copy()]), [1-validation_prevalence, validation_prevalence])[0][1]
            return emq_results
        elif method == 'HDy':
            HDy_4 = customDyS(n_bins=4)
            HDy_4.fit(new_validation_data.copy())
            HDy_4_results = HDy_4.aggregate(classif_posteriors.copy())[1]
            return HDy_4_results
    return calculate_stat

def calculate_corrected_prev_estimate(p_zero, predictions):
    pos_preds = predictions[predictions['y']==1]
    neg_preds = predictions[predictions['y']==0]
    tppa = np.mean(pos_preds['y_pred'])
    fppa = np.mean(neg_preds['y_pred'])
    corrected_p = (p_zero - fppa)/(tppa - fppa)
    if corrected_p >= 0 and corrected_p <= 1.0:
        return corrected_p
    else:
        # print(f'Warning! corrrected estimate out of range: {corrected_p}. Sigmoid will be applied.')
        return 1/(1 + np.exp(-corrected_p))

class customPCC():
    def __init__(self):
        pass

    def aggregate(self, classif_posteriors):
        return  1 - np.mean(classif_posteriors), np.mean(classif_posteriors)    

class CC():
    def __init__(self):
        pass

    def aggregate(self, classif_posteriors):
        pred_labels = [1 if x > 0.5 else 0 for x in classif_posteriors]
        return 1 - np.mean(pred_labels), np.mean(pred_labels)

class customPACC():
    def __init__(self):
        pass

    def aggregate(self, classif_posteriors, validation_data):
        corrected_p = calculate_corrected_prev_estimate(np.mean(classif_posteriors), validation_data)
        return 1-corrected_p, corrected_p

class customEMQ(EMQ):
    def __init__(self):
        pass
    
    def fit(self):
        return self

    def aggregate(self, classif_posteriors, training_prevalence, epsilon=EMQ.EPSILON):
        priors, posteriors = self.EM(training_prevalence, classif_posteriors, epsilon)
        return priors, posteriors

class customDyS(DyS):
    def __init__(self, n_bins=4, divergence: Union[str, Callable]= 'HD', tol=1e-05):
        self.tol = tol
        self.divergence = divergence
        self.n_bins = n_bins
    
    def fit(self, validation_data):
        self.Pxy1 = validation_data[validation_data['y']==1]['y_pred']
        self.Pxy0 = validation_data[validation_data['y']==0]['y_pred']
        self.Pxy1_density = np.histogram(self.Pxy1, bins=self.n_bins, range=(0, 1), density=True)[0]
        self.Pxy0_density = np.histogram(self.Pxy0, bins=self.n_bins, range=(0, 1), density=True)[0]
        return self

    def aggregate(self, classif_posteriors):
        Px = classif_posteriors
        Px_test = np.histogram(Px, bins=self.n_bins, range=(0, 1), density=True)[0]
        divergence = get_divergence(self.divergence)

        def distribution_distance(prev):
            Px_validation = prev * self.Pxy1_density + (1 - prev) * self.Pxy0_density
            return divergence(Px_validation, Px_test)
            
        class1_prev = self._ternary_search(f=distribution_distance, left=0, right=1, tol=self.tol)
        return np.asarray([1 - class1_prev, class1_prev])

class customBayesianCC(BayesianCC):
    def __init__(self, number_of_samples) -> None:
        super().__init__()
        self.num_samples = number_of_samples
        self.thres = None
    
    def aggregation_fit(self, validation_predictions):
        #determine best threshold
        thres = None
        best_mcc = None
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            mcc = matthews_corrcoef(validation_predictions['y'], validation_predictions['y_pred']>=t)
            if best_mcc is None or mcc > best_mcc: 
                best_mcc = mcc
                thres = t
        print('best threshold:', thres, 'mcc:', best_mcc)
        self.thres = thres
        pred_labels = [1 if x > thres else 0 for x in validation_predictions['y_pred']]
        true_labels = list(validation_predictions['y'])
        self._n_and_c_labeled = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=[0,1])
    
    def aggregate(self, classif_predictions):
        pred_labels = [1 if x > self.thres else 0 for x in classif_predictions['y_pred']]
        samples = self.sample_from_posterior(pred_labels)[_bayesian.P_TEST_Y]
        # return np.asarray(samples.mean(axis=0), dtype=float)
        samples = np.array(samples)
        return samples
    
    def sample_from_posterior(self, classif_predictions, labels=[0,1]):
        if self._n_and_c_labeled is None:
            raise ValueError("aggregation_fit must be called before sample_from_posterior")

        n_c_unlabeled = F.counts_from_labels(classif_predictions, labels)

        self._samples = _bayesian.sample_posterior(
            n_c_unlabeled=n_c_unlabeled,
            n_y_and_c_labeled=self._n_and_c_labeled,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            seed=self.mcmc_seed,
        )
        return self._samples