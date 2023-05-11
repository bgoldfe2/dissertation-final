from operator import index
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
from sklearn import svm, metrics
import matplotlib.pyplot as plt

from engine import test_eval_fn
from Model_Config import Model_Config

from utils import set_device, load_models, generate_dataset_for_ensembling, calc_roc_auc

def test_evaluate(test_df, test_data_loader, model, device, args: Model_Config):
    # modified using the Model_Config instance args as the state reference
    pretrained_model = args.pretrained_model
    print(f'\nEvaluating: ---{pretrained_model}---\n')
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, model, device, args)
    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print('Accuracy:', acc)
    print('Mcc Score:', mcc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1_score:', f1)
    print('classification_report: ', classification_report(y_test, y_pred, digits=4))
    test_df['y_pred'] = y_pred
    pred_test = test_df[['text', 'label', 'target', 'y_pred']]
    #pred_test.to_csv(f'{args.output_path}{pretrained_model}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)
    # auc evaluation new for this version
    #ROC Curve

    calc_roc_auc(np.array(y_test), np.array(y_proba), args)

    # Return the test results for saving in train.py
    return pred_test, acc

def evaluate_all_models(args: Model_Config):
    deberta, xlnet, roberta, albert, gpt_neo, gpt_neo13 = load_models()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device()

    deberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="microsoft/deberta-v3-base", df =test_df)
    test_evaluate(test_df, test_data_loader, bert, device, pretrained_model="microsoft/deberta-v3-base")
    del bert, test_data_loader

    xlnet.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="xlnet-base-cased", df=test_df)
    test_evaluate(test_df, test_data_loader, xlnet, device, pretrained_model="xlnet-base-cased")
    del xlnet, test_data_loader

    roberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="roberta-base", df=test_df)
    test_evaluate(test_df, test_data_loader, roberta, device, pretrained_model="roberta-base")
    del roberta, test_data_loader

    albert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="albert-base-v2", df=test_df)
    test_evaluate(test_df, test_data_loader, albert, device, pretrained_model="albert-base-v2")
    del albert, test_data_loader

    gpt_neo.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="EleutherAI/gpt-neo-125m", df=test_df)
    test_evaluate(test_df, test_data_loader, gpt_neo, device, pretrained_model="EleutherAI/gpt-neo-125M")
    del gpt_neo, test_data_loader

    # gpt_neo13.to(device)
    # test_data_loader = generate_dataset_for_ensembling(pretrained_model="EleutherAI/gpt-neo-1.3B", df=test_df)
    # test_evaluate(test_df, test_data_loader, gpt_neo13, device, pretrained_model="EleutherAI/gpt-neo-1.3b")
    # del gpt_neo13, test_data_loader
