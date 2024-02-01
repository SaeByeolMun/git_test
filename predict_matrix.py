import keras.backend as K
import numpy as np # 배열처리하는 라이브러리

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix, precision_recall_curve

def dice_cost_1(y_true, y_predicted):

    mask_true = y_true[:, :, :, :, 1]
    mask_pred = y_predicted[:, :, :, :, 1]

    num_sum = 2.0 * K.sum(mask_true * mask_pred) + K.epsilon()
    den_sum = K.sum(mask_true) + K.sum(mask_pred)+ K.epsilon()

    return num_sum/den_sum

def Dice(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def resultConf(test_label, predict_model, weight):
    recall_list = []
    specificity_list = []
    precision_list = []
    acc_list = []
    dice_list = []
    mAP_list = []
    
    all_data_result = []
    dices = []
    
    #for i in range(n_test):
    for i in (range(test_label.shape[0])):
        label = test_label[i, :, :,  :, 1] # ground truth binary mask
        
        if np.sum(label) > 0:
            predi = predict_model[i, :, :, :, 1] > weight# binary prediction
            
            label = label.flatten()
            predi = predi.flatten()
           
            label = label.astype(np.bool)
            predi = predi.astype(np.bool)
            
#             print(label)
#             print('=='*50)
#             print(predi)
            accuracy = accuracy_score(label, predi)
            precision = precision_score(label, predi)
            tn, fp, fn, tp = confusion_matrix(label, predi).ravel()
            specificity = tn / (tn+fp)
#             specificity = specificity_score(label, predi)
            recall = recall_score(label, predi)
            dice = f1_score(label, predi)      
            ap = average_precision_score(label, predi)
            
            recall_list.append(recall)
            specificity_list.append(specificity)
            precision_list.append(precision)
            acc_list.append(accuracy)
            dice_list.append(dice)
            mAP_list.append(ap)
                
    dice_list = np.array(dice_list)
    recall_list = np.array(recall_list)
    specificity_list = np.array(specificity_list)
    precision_list = np.array(precision_list)
    mAP_list = np.array(mAP_list)
    
    dice_list = dice_list[~np.isnan(dice_list)]
    recall_list = recall_list[~np.isnan(recall_list)]
    specificity_list = specificity_list[~np.isnan(specificity_list)]
    precision_list = precision_list[~np.isnan(precision_list)]
    mAP_list = mAP_list[~np.isnan(mAP_list)]
    
    
    return recall_list, specificity_list, precision_list, acc_list, dice_list, mAP_list

def returnTable(test_label, predict_model, weight):
    recall_list, specificity_list, precision_list, acc_list, dice_list, mAP_list = resultConf(test_label, predict_model, weight)

    row_data1 = []
    row_item1 = []
    
    row_item1.append('{0}'.format(weight))
    row_item1.append('{0:.2f}'.format(np.mean(recall_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(specificity_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(precision_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(acc_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(dice_list)*100))
    row_item1.append('{0:.2f}'.format(np.mean(mAP_list)*100))
    row_data1.append(row_item1)
    
    head1 = ["Weight", "Sensitivity", "Specificity", "Precision", "Accuracy", "DSC", "mAP"]
    
    return row_data1, head1

def findOptimizedWeight(test_label, predict_model):
    mean_dices = []
    for j in (range(0, 100, 1)):
        weight = j / 100
        dices = []
        for i in (range(test_label.shape[0])):
            gt = test_label[i,:,:,0] # ground truth binary mask
            
            if np.sum(gt) > 0:
                pr = predict_model[i,:,:,0] > weight# binary prediction
                gt = gt.astype(np.bool)
                pr = pr.astype(np.bool)
                
                # Compute scores
                dices.append(Dice(gt, pr))
        dices = np.array(dices)
        mean_dices.append(np.mean(dices))
    
    optimizedWeight = np.argmax(mean_dices)
    optimizedWeight /= 100
    
    return optimizedWeight, mean_dices
