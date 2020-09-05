################
# EXPERIMENTAL #
################


import numpy as np

from covid_19.utils import moving_average


def uncertainty_quantification(
    sensitivity, 
    specificity, 
    confirmed_cases,
    daily_tests,
    ):
    positives = np.diff(confirmed_cases)
    std_sensitivity_err = np.sqrt(np.divide(
        (1 - sensitivity) * sensitivity, 
        positives, 
        out=np.zeros(positives.shape, dtype=float), 
        where=positives!=0,
    ))
    sensitivity_ci = 1.96 * std_sensitivity_err
    lower_bound_sensitivity = np.abs(sensitivity - sensitivity_ci)
    lower_bound_true_positives = lower_bound_sensitivity * positives
    lower_bound_cumulative_cases = np.cumsum(lower_bound_true_positives) \
                                    + confirmed_cases[0]
    
    daily_tests = np.concatenate((
        np.array([daily_tests[0]]), daily_tests[:-1]))
    negatives = daily_tests[1:] - positives
    std_specificity_err = np.sqrt(np.divide(
        (1 - specificity) * specificity,
        negatives,
        out=np.zeros(negatives.shape, dtype=float), 
        where=negatives!=0,
    ))
    specificity_ci = 1.96 * std_specificity_err
    upper_bound_specificity = np.abs(specificity + specificity_ci)
    upper_bound_true_negatives = upper_bound_specificity * negatives
    upper_bound_false_negatives = negatives - upper_bound_true_negatives
    upper_bound_true_positives = upper_bound_false_negatives + positives
    upper_bound_cumulative_cases = np.cumsum(upper_bound_true_positives) \
                                    + confirmed_cases[0]

    lower_bound_cumulative_cases = np.concatenate((
        np.array([confirmed_cases[0]]), lower_bound_cumulative_cases))
    upper_bound_cumulative_cases = np.concatenate((
        np.array([confirmed_cases[0]]), upper_bound_cumulative_cases))

    lower_bound_scaler = lower_bound_cumulative_cases / confirmed_cases
    lower_bound_scaler[lower_bound_scaler>1]=1
    upper_bound_scaler = upper_bound_cumulative_cases / confirmed_cases
    upper_bound_scaler[upper_bound_scaler<1]=1
    return lower_bound_scaler, upper_bound_scaler


def estimate(confirmed_cases, averaging_period=7, testing_delay=5):
    confirmed_cases_bias = np.r_[0, confirmed_cases]
    new_daily_cases = np.diff(confirmed_cases_bias)
    smoothed_new_daily_cases = moving_average(new_daily_cases, averaging_period)
    R = np.divide(
        smoothed_new_daily_cases[testing_delay:], 
        smoothed_new_daily_cases[:-testing_delay],
        out=np.zeros(smoothed_new_daily_cases[:-testing_delay].size, dtype=float),
        where=smoothed_new_daily_cases[:-testing_delay]!=0)
    return R, smoothed_new_daily_cases