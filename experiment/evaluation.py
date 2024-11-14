import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch
from holisticai.bias.metrics import multiclass_equality_of_opp
from holisticai.bias.metrics import multiclass_statistical_parity

from torch import nn
from torch.utils import data
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, classification_report
from aif360.sklearn.metrics import statistical_parity_difference, equal_opportunity_difference
warnings.filterwarnings('ignore')

'''
NOTE: 
- EvalMetric2 추가 (for depression evaluation)
    finetune depression에서 증상별로 여러개의 prediction값이 나오기 때문에, 이를 병합하기 위해 활용. 
    daic-woz 논문과 통일하기 위해 mf1 기준으로 변경
    f1 score 수정 
    
- f1_score, classification_report 추가 
'''
class EvalMetric2:
    def __init__(self, multilabel=True):
        self.multilabel = multilabel
        self.pred_list = []
        self.truth_list = []
        self.loss_list = []

    def append_classification_results(self, labels, outputs, loss=None):
        if self.multilabel:  # For multi-label binary classification (e.g., symptoms)
            predictions = (outputs.detach().cpu().numpy() > 0.5).astype(int)  # Use threshold for binary classification
        else:  
            predictions = (outputs.detach().cpu().numpy() > 0.5).astype(int)
        
        if len(predictions.shape) > 1:  # Handling multiclass multi-output scenario (like for symptoms)
            for i in range(predictions.shape[0]):
                self.pred_list.append(predictions[i].tolist())
                self.truth_list.append(labels[i].cpu().numpy().tolist())
                
        else:  # Single-output scenario (like for depression binary)
            self.pred_list.extend(predictions)
            self.truth_list.extend(labels.cpu().numpy())

        if loss is not None:
            self.loss_list.append(loss.item())


    def classification_summary(self):
        result_dict = dict()

        if isinstance(self.truth_list[0], list):  # Handling multi-output predictions
            flatten_truth = [label for sublist in self.truth_list for label in sublist]
            flatten_pred = [pred for sublist in self.pred_list for pred in sublist]
        else:  # Handling single-output predictions
            flatten_truth = self.truth_list
            flatten_pred = self.pred_list

        # Ensure that predictions and truths are in the same format (int for multi-class, float for multi-label binary)
        flatten_truth = np.array(flatten_truth)
        flatten_pred = np.array(flatten_pred)

        if self.multilabel:  # Multi-label binary classification
            
            result_dict['acc'] = accuracy_score(flatten_truth, flatten_pred) * 100  # Exact matches for multi-label
            result_dict['uar'] = recall_score(flatten_truth, flatten_pred, average="macro") * 100
            result_dict['mf1'] = f1_score(flatten_truth, flatten_pred, average="weighted") * 100  # For multi-label, use 'samples'
            
            # Calculate per-label accuracy and F1 score
            truth_np = np.array(self.truth_list)
            pred_np = np.array(self.pred_list)
            
            # sym = ['nointerest', 'depressed', 'sleep', 'tired', 
            #     'appetite', 'failure', 'concentrating', 'moving']
            
            sym = ['nointerest', 'depressed', 'sleep', 'tired', 'failure']
            
            for i in range(truth_np.shape[1]):
                result_dict[f'acc_{sym[i]}'] = accuracy_score(truth_np[:, i], pred_np[:, i]) * 100
                result_dict[f'mf1_{sym[i]}'] = f1_score(truth_np[:, i], pred_np[:, i], average="weighted") * 100
                
                # print(f'[acc_{sym[i]}]: ', result_dict[f'acc_{sym[i]}'] ,
                #       f'[mf1_{sym[i]}]: ', result_dict[f'mf1_{sym[i]}'])
            
        else:  # Multi-class classification (including binary classification)
            result_dict['acc'] = accuracy_score(flatten_truth, flatten_pred) * 100
            result_dict['uar'] = recall_score(flatten_truth, flatten_pred, average="macro") * 100
            result_dict['mf1'] = f1_score(flatten_truth, flatten_pred, average="weighted") * 100

        result_dict['report'] = classification_report(flatten_truth, flatten_pred, output_dict=True)
        # print(f"[report]:{result_dict['report']}")
        result_dict['conf'] = np.round(confusion_matrix(flatten_truth, flatten_pred, normalize='true') * 100, decimals=2)
        result_dict["loss"] = np.mean(self.loss_list)
        result_dict["sample"] = len(self.truth_list)

        return result_dict


class EvalMetric(object):
    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        self.pred_list = list()
        self.truth_list = list()
        self.top_k_list = list()
        self.loss_list = list()
        self.demo_list = list()
        self.speaker_list = list()
        
    def append_classification_results(
        self, 
        labels,
        outputs,
        loss            =None,
        demographics    =None,
        speaker_id      =None
    ):
        predictions = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        for idx in range(len(predictions)):
            self.pred_list.append(predictions[idx])
            self.truth_list.append(labels.detach().cpu().numpy()[idx])
        if loss is not None: self.loss_list.append(loss.item())
        if demographics is not None: 
            self.demo_list.append(demographics)
            # if demographics == "male": self.demo_list.append(1.0)
            # else: self.demo_list.append(0.0)
        if speaker_id is not None: self.speaker_list.append(speaker_id)
        
    def classification_summary(
        self, return_auc: bool=False
    ):
        result_dict = dict()
        result_dict['acc'] = accuracy_score(self.truth_list, self.pred_list)*100
        result_dict['uar'] = recall_score(self.truth_list, self.pred_list, average="macro")*100
        result_dict['mf1'] = f1_score(self.truth_list, self.pred_list, average="weighted")*100
        result_dict['report'] = classification_report(self.truth_list, self.pred_list, output_dict=True)
        result_dict['top5_acc'] = (np.sum(self.top_k_list == np.array(self.truth_list).reshape(len(self.truth_list), 1)) / len(self.truth_list))*100
        result_dict['conf'] = np.round(confusion_matrix(self.truth_list, self.pred_list, normalize='true')*100, decimals=2)
        result_dict["loss"] = np.mean(self.loss_list)
        result_dict["sample"] = len(self.truth_list)
        return result_dict

    def demographic_parity(self):
        """
        Calculate demographic parity metric for multi-class labels.
        Args:
            None.
        Returns:
            demographic_parity (float): Demographic parity metric.
        """
        y_true = np.array(self.truth_list)
        y_pred = np.array(self.pred_list)
        sensitive_feature = np.array(self.demo_list)
        
        # Calculate the number of positive outcomes for each class and group
        classes = np.unique(y_true)
        pos_counts_group1 = np.zeros(len(classes))
        pos_counts_group2 = np.zeros(len(classes))
        for i, c in enumerate(classes):
            group1_mask = np.logical_and(sensitive_feature == "male", y_true == c)
            pos_counts_group1[i] = np.sum(y_pred[group1_mask] == c)
            group2_mask = np.logical_and(sensitive_feature == "female", y_true == c)
            pos_counts_group2[i] = np.sum(y_pred[group2_mask] == c)

        # Calculate the proportion of positive outcomes for each group
        prop_group1 = pos_counts_group1 / np.sum(sensitive_feature == "male")
        prop_group2 = pos_counts_group2 / np.sum(sensitive_feature == "female")

        # Calculate the absolute difference between the two proportions
        demographic_parity = np.max(np.abs(prop_group1 - prop_group2))
        return demographic_parity

    def statistical_parity(self):
        y_pred = np.array(self.pred_list)
        p_attr = np.array(self.demo_list)
        # p_attr = np.array(self.speaker_list)

        statistical_parity = multiclass_statistical_parity(
            p_attr, y_pred, groups=None, classes=None, aggregation_fun="max"
        )
        return statistical_parity

    def equality_of_opp(self):
        y_true = np.array(self.truth_list)
        y_pred = np.array(self.pred_list)
        p_attr = np.array(self.demo_list)
        # p_attr = np.array(self.speaker_list)

        equality_of_opp = multiclass_equality_of_opp(
            p_attr, y_pred, y_true, groups=None, classes=None, aggregation_fun="max"
        )
        return equality_of_opp
