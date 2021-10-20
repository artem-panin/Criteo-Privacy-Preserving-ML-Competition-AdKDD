import mlflow
import numpy as np
import pandas as pd

from datetime import datetime
from zipfile import ZipFile

from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.metrics import log_loss
import lightgbm as lgb

from sklearn.calibration import calibration_curve
from sklearn.calibration import IsotonicRegression

import matplotlib.pyplot as plt


Y_HAT_CLICK_FN = 'y_hat_click.txt'
Y_HAT_SALE_FN = 'y_hat_sale.txt'


def log_mlflow(metrics, tags, parameters, artifacts=None, run_name=None, silent=False):
    """
    Логирование метрик в mlflow
    
    :param metrics: ассоц. массив, в котором key - название метрики, value - значение метрики
    :param tags: ассоц. массив произвольных тэгов
    :param parameters: ассоц. массив, в котором key - название параметра, value - его значение
    :param run_name: уникальное имя запуска
    """
    
    if not silent:
        print(run_name, tags, parameters, metrics)
    
    with mlflow.start_run(nested=True, run_name=run_name):
        if parameters is not None:
            mlflow.log_params(parameters)

        if metrics is not None:
            mlflow.log_metrics(metrics)

        if tags is not None:
            mlflow.set_tags(tags)
            
        if artifacts is not None:
            mlflow.log_artifacts(local_dir=artifacts)
            
            
def create_submission(y_hat_click, y_hat_sale=None, filename: str = None, description: str = None):
    """Method to export your solution.

    The zip file
      - must contain at the root (not in a subdirectory) a file \"y_hat_click.txt\" and/or a file \"y_hat_sale.txt\"

    These files
      - shall contain individual predictions (a float in [0;1]), one per line
      - must be of the same length as 'X_test.csv.gz' (in number of lines)
    """
    np.savetxt(Y_HAT_CLICK_FN, y_hat_click, fmt='%1.6f')
    if y_hat_sale is not None:
        np.savetxt(Y_HAT_SALE_FN, y_hat_sale, fmt='%1.6f')
    if filename is None:
        filename = 'submissions/submission-%s.zip' % str(datetime.now()).replace(' ', '_').replace(':', '-')
    with ZipFile(filename, 'w') as zip:
        zip.write(Y_HAT_CLICK_FN)
        if y_hat_sale is not None:
            zip.write(Y_HAT_SALE_FN)
        if description is not None and len(description):
            zip.writestr('description', description)
    print('wrote', filename)
    return filename
        
    
def apply_isotonic_regression(val_pred, val_gt, test_pred):
    isotonic = IsotonicRegression(out_of_bounds='clip',
                              y_min=val_pred.min(),
                              y_max=val_pred.max())
    isotonic.fit(val_pred, val_gt)
    isotonic_test_pred = isotonic.predict(test_pred)
    return isotonic_test_pred


def compare_calibration_curves(prediction_dataframe: pd.DataFrame):
    
    # binning the dataframe, so we can see success rates for bins of probability
    bins = np.arange(0.05, 1.00, 0.05)

    # opening figure
    plt.figure(figsize=(12,7), dpi=150)

    # plotting ideal line
    plt.plot([0,1],[0,1], 'k--', label='ideal')
    
    for model_label in prediction_dataframe.columns:
        if model_label == 'y':
            pass
        
        prediction_dataframe.loc[:, model_label + 'prob_bin'] = np.digitize(prediction_dataframe[model_label], bins)
        prediction_dataframe.loc[:, model_label + 'prob_bin_val'] = prediction_dataframe[model_label + 'prob_bin'].replace(dict(zip(range(len(bins) + 1), list(bins) + [1.00])))

        # plotting calibration
        calibration_y = prediction_dataframe.groupby(model_label + 'prob_bin_val')['y'].mean()
        calibration_x = prediction_dataframe.groupby(model_label + 'prob_bin_val')[model_label].mean()
        plt.plot(calibration_x, calibration_y, marker='o', label=model_label)

    # legend and titles
    plt.title('Calibration plot for LGBM, corrected')
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual fraction of positives')
    plt.legend()
    