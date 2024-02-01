import cv2
import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics

# 평가지표
def confusion_matrix(label, predict):
    tn_list = []
    fp_list = []
    fn_list = []
    tp_list = []
    for i in range(len(label)):
        p = predict[i,:,0]
        l = label[i,:,0]
        
        tn, fp, fn, tp = confusion_matrix(p, l).ravel()

        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)
    return tn_list, fp_list, fn_list, tp_list

def confusion_matrix_list(label_list, predict_list):
    total_tn_list = []
    total_fp_list = []
    total_fn_list = []
    total_tp_list = []
    for n in range(len(label_list)):
        label = label_list[n]
        predict = predict_list[n]

        tn_list, fp_list, fn_list, tp_list = confusion_matrix(label, predict)

        total_tn_list.extend(tn_list)
        total_fp_list.extend(fp_list)
        total_fn_list.extend(fn_list)
        total_tp_list.extend(tp_list)
    return total_tn_list, total_fp_list, total_fn_list, total_tp_list

def recall(label, predict):
    recall_list = []
    for i in range(len(label)):
        p = predict[i,:,0]
        l = label[i,:,0]
        
        recall = metrics.recall_score(l, p)

        recall_list.append(recall)
    return recall_list

def recall_list(label_list, predict_list):
    total_recall_list = []
    for n in range(len(label_list)):
        label = label_list[n]
        predict = predict_list[n]
        
        recall_list = recall(label, predict)

        total_recall_list.append(recall_list)
    return total_recall_list

def accuracy(label, predict):
    accuracy_list = []
    for i in range(len(label)):
        p = predict[i,:,0]
        l = label[i,:,0]
        
        accuracy = metrics.accuracy_score(l, p)

        accuracy_list.append(accuracy)
    return accuracy_list

def accuracy_list(label_list, predict_list):
    total_accuracy_list = []
    for n in range(len(label_list)):
        label = label_list[n]
        predict = predict_list[n]
        
        accuracy_list = accuracy(label, predict)

        total_accuracy_list.append(accuracy_list)
    return total_accuracy_list

def precision(label, predict):
    precision_list = []
    for i in range(len(label)):
        p = predict[i,:,0]
        l = label[i,:,0]
        
        precision = metrics.precision_score(l, p)

        precision_list.append(precision)
    return precision_list

def precision_list(label_list, predict_list):
    total_precision_list = []
    for n in range(len(label_list)):
        label = label_list[n]
        predict = predict_list[n]
        
        precision_list = precision(label, predict)

        total_precision_list.append(precision_list)
    return total_precision_list

def f1_score(label, predict):
    f1_score_list = []
    for i in range(len(label)):
        p = predict[i,:,0]
        l = label[i,:,0]
        
        f1_score = metrics.f1_score(l, p)

        f1_score_list.append(f1_score)
    return f1_score_list

def f1_score_list(label_list, predict_list):
    total_f1_score_list = []
    for n in range(len(label_list)):
        label = label_list[n]
        predict = predict_list[n]
        
        f1_score_list = f1_score(label, predict)

        total_f1_score_list.append(f1_score_list)
    return total_f1_score_list

