import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import resample


def accuracy(df, true_col, pred_prob_col, threshold=0.5):
    """Define function to calculate accuracy"""
    true_positive = df[(df[true_col] == 1) & (df[pred_prob_col] >= threshold)].shape[0]
    false_positive = df[(df[true_col] == 0) & (df[pred_prob_col] > threshold)].shape[0]
    true_negative = df[(df[true_col] == 0) & (df[pred_prob_col] <= threshold)].shape[0]
    false_negative = df[(df[true_col] == 1) & (df[pred_prob_col] < threshold)].shape[0]

    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    

def sensitivity(df, true_col, pred_prob_col, threshold):
    """Define function to calculate sensitivity"""
    true_positive = df[(df[true_col] == 1) & (df[pred_prob_col] >= threshold)].shape[0]
    false_negative = df[(df[true_col] == 1) & (df[pred_prob_col] < threshold)].shape[0]

    return true_positive / (true_positive + false_negative)


def specificity(df, true_col, pred_prob_col, threshold):
    """Define function to calculate specificity"""
    true_negative = df[(df[true_col] == 0) & (df[pred_prob_col] <= threshold)].shape[0]
    false_positive = df[(df[true_col] == 0) & (df[pred_prob_col] > threshold)].shape[0]

    return true_negative / (true_negative + false_positive)


def auc():
    """Define function to calculate auc & plot ROC"""
    # Calculate sensitivity & 1-specificity for each threshold between 0 and 1.
    thresholds = np.linspace(0, 1, 200)
    tpr_values = [sensitivity(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
    fpr_values = [1- specificity(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
    
    # Calculate AUC value
    auc_value = round(metrics.roc_auc_score(pred_df['true_values'], pred_df['pred_probs']), 3)
    
    # Plot ROC curve.
    plt.plot(fpr_values, tpr_values, label='ROC Curve')
    plt.plot(np.linspace(0, 1, 200), np.linspace(0, 1, 200), label='baseline', linestyle='--')
    plt.title(f"ROC Curve with AUC = {auc_value}", fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.show()

    return auc_value

def equal_point(pred_df):
    """Define function to calculate the point where sensitivity equals specificity"""
    thresholds = np.linspace(0, 1, 200)
    tpr_values = [sensitivity(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
    fpr_values = [specificity(pred_df, 'true_values', 'pred_probs', prob) for prob in thresholds]
    min_val = float('inf')
    min_idx = 0
    for i, val in enumerate(tpr_values):
        sub_val = abs(tpr_values[i] - fpr_values[i]) 
        if sub_val < min_val:
            min_val = sub_val
            min_idx = i
    return min_val, min_idx


def bootstrap(n_iterations, pred_df, alpha):
    """Get CI with bootstrap"""
    # configure bootstrap
    n_size = int(pred_df.shape[0] * 0.50)
    values = pred_df.values

    # run bootstrap
    stats = list()
    for i in range(n_iterations):
        resampled_data = resample(values, n_samples=n_size)
        resample_df = pd.DataFrame({'true_values': resampled_data[:, 0],
                        'pred_probs':resampled_data[:, 1]})
        # calculate accuracy 
        score = accuracy(resample_df, 'true_values', 'pred_probs', threshold=0.5)
        stats.append(score)

    # confidence intervals
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))

    # plot scores
    plt.hist(stats)
    plt.title(f"Alpha={alpha}, Lower={round(lower, 3)}, Upper={round(upper, 3)}", fontsize=16)
    plt.show()


    return lower, upper


if __name__ == "__main__":
    # Get the args
    try:
        input_file, output_file = sys.argv[1:3]
        print("input json file: ", input_file)
        print("output json file: ", output_file)
    except Exception as e:
        print(sys.argv)
        print(e)

    # Read and process JSON file
    f = open(input_file)
    data = json.load(f)
    metric_func = data["metirc"]
    model_outputs = data['model_outputs']
    gt_labels = data["gt_labels"]
    threshold = data["threshold"]
    ci = data["ci"]
    num_bootstraps = data["num_bootstraps"]
    alpha = data["alpha"]
    f.close()

    # Prepare a dataframe for metric functions
    pred_df = pd.DataFrame({'true_values': gt_labels, 'pred_probs':model_outputs})

    # Calculate accuracy for the input
    acc_val = accuracy(pred_df, 'true_values', 'pred_probs', threshold)
    print("***** accuracy: ", round(acc_val, 3))
    
    # Calculate AUC and plot ROC
    auc_val = auc()
    print("***** auc value: ", auc_val)

    # Get the equal point
    min_val, min_idx = equal_point(pred_df)
    print("***** the equal point is obtained when threshold: %.1f" % (min_idx/200))

    # Get confidence interval with bootstrap (for accuracy but other metrics can also be added)
    if ci:
        lower, upper = bootstrap(num_bootstraps, pred_df, alpha)
        print("****** %.1f confidence interval %.1f%% and %.1f%%" % (alpha*100, lower*100, upper*100))
    else:
        lower, upper = None, None
   
    # Write to JSON
    dictionary = {
       "value": round(acc_val, 3),
       "lower_bound": round(lower, 3),
       "upper_bound": round(upper, 3)
    }
    with open("output.json", "w") as outfile:
        json.dump(dictionary, outfile)

