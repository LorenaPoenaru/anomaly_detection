# EVALUATION SCRIPT USED IN SR PAPER FROM 
# https://github.com/iopsai/iops/blob/master/evaluation/evaluation.py 
# https://arxiv.org/pdf/1906.03821.pdf

import numpy as np
import pandas as pd
import time
import datetime
from sys import argv
from sklearn.metrics import f1_score, precision_score, recall_score


# consider delay threshold and missing segments
def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict


# set missing = 0
def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)

    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=np.int)
    new_label[idx] = label

    return new_label


def label_evaluation(truth, result, delay=7):

    y_true_list = []
    y_pred_list = []

    time_index = list(range(len(truth)))

    y_true = reconstruct_label(time_index, truth)


    if len(truth) != len(result):
        print('Length of true and predicted labels disagree!!')
        return None

    y_pred = reconstruct_label(time_index, result)

    y_pred = get_range_proba(y_pred, y_true, delay)
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)

    try:
        fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
        precision = precision_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
        recall = recall_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))

    except Exception as e:
        print(e)
        return None

    return fscore, precision, recall
