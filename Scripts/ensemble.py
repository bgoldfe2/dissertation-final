from os import name
import pandas as pd
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score


from evaluate import test_evaluate
from engine import test_eval_fn_ensemble, test_eval_fn

from utils import sorting_function, evaluate_ensemble, print_stats, load_prediction, set_device, load_models, generate_dataset_for_ensembling, calc_roc_auc
from common import get_parser

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def max_vote():
    print(f'\n---Max voting ensemble---\n')

    bert, xlnet, roberta, albert, gpt2 = load_prediction()

    target = []
    bert_pred = []
    xlnet_pred = []
    roberta_pred = []
    albert_pred = []
    gpt2_pred = []

    for index in range(len(bert)):
       target.append(bert['target'][index])
       bert_pred.append(bert['y_pred'][index])
       xlnet_pred.append(xlnet['y_pred'][index])
       roberta_pred.append(roberta['y_pred'][index])
       albert_pred.append(albert['y_pred'][index])
       gpt2_pred.append(gpt2['y_pred'][index])

    max_vote_df = pd.DataFrame()
    max_vote_df['target'] = target
    max_vote_df['bert'] = bert_pred
    max_vote_df['xlnet'] = xlnet_pred
    max_vote_df['roberta'] = roberta_pred
    max_vote_df['albert'] = albert_pred
    max_vote_df['gpt2'] = gpt2_pred

    # print_stats(max_vote_df, bert, xlnet, roberta, albert)
    # BHG addtional lines into this function until line 88
    preds = []

    for index in range(len(max_vote_df)):
        values = max_vote_df.iloc[index].values[1:]
        sorted_values = sorted(Counter(values).items(), key = sorting_function, reverse=True)
        preds.append(sorted_values[0][0])
        
    max_vote_df['pred'] = preds

    evaluate_ensemble(max_vote_df)
    
def max_vote3():
    print(f'\n---Max voting ensemble for the best three classifiers---\n')

    bert, xlnet, roberta, albert, gpt2 = load_prediction()

    target = []
    
    xlnet_pred = []
    roberta_pred = []
    
    gpt2_pred = []

    for index in range(len(bert)):
       target.append(bert['target'][index])
       
       xlnet_pred.append(xlnet['y_pred'][index])
       roberta_pred.append(roberta['y_pred'][index])
       
       gpt2_pred.append(gpt2['y_pred'][index])

    max_vote_df = pd.DataFrame()
    max_vote_df['target'] = target
    
    max_vote_df['xlnet'] = xlnet_pred
    max_vote_df['roberta'] = roberta_pred
    
    max_vote_df['gpt2'] = gpt2_pred

    # print_stats(max_vote_df, bert, xlnet, roberta, albert)
    # end of additional lines ? what changed?
    preds = []

    for index in range(len(max_vote_df)):
        values = max_vote_df.iloc[index].values[1:]
        sorted_values = sorted(Counter(values).items(), key = sorting_function, reverse=True)
        preds.append(sorted_values[0][0])
        
    max_vote_df['pred'] = preds

    evaluate_ensemble(max_vote_df)
# BHG Added new function
def rocauc():
    bert, xlnet, roberta, albert, gpt2 = load_models()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device()

    bert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="bert-base-uncased", df =test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, bert, device, pretrained_model="bert-base-uncased")
    calc_roc_auc(np.array(y_test), np.array(y_proba), name='BERT')
    del bert, test_data_loader

    xlnet.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="xlnet-base-cased", df=test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, xlnet, device, pretrained_model="xlnet-base-cased")
    calc_roc_auc(np.array(y_test), np.array(y_proba), name='XLNet')
    del xlnet, test_data_loader

    roberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="roberta-base", df=test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, roberta, device, pretrained_model="roberta-base")
    calc_roc_auc(np.array(y_test), np.array(y_proba), name='RoBERTa')
    del roberta, test_data_loader

    albert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="albert-base-v2", df=test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, albert, device, pretrained_model="albert-base-v2")
    calc_roc_auc(np.array(y_test), np.array(y_proba), name='albert')
    del albert, test_data_loader

    gpt2.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="gpt2", df=test_df)
    y_pred, y_test, y_proba = test_eval_fn(test_data_loader, gpt2, device, pretrained_model="gpt2")
    calc_roc_auc(np.array(y_test), np.array(y_proba), name='GPT2')
    del gpt2, test_data_loader
    
    

def averaging():
    bert, xlnet, roberta, albert, gpt2 = load_models()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device()

    bert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="bert-base-uncased", df =test_df)
    bert_output, target = test_eval_fn_ensemble(test_data_loader, bert, device, pretrained_model="bert-base-uncased")
    del bert, test_data_loader

    xlnet.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="xlnet-base-cased", df=test_df)
    xlnet_output, target = test_eval_fn_ensemble(test_data_loader, xlnet, device, pretrained_model="xlnet-base-cased")
    del xlnet, test_data_loader

    roberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="roberta-base", df=test_df)
    roberta_output, target = test_eval_fn_ensemble(test_data_loader, roberta, device, pretrained_model="roberta-base")
    del roberta, test_data_loader

    albert.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="albert-base-v2", df=test_df)
    albert_output, target = test_eval_fn_ensemble(test_data_loader, albert, device, pretrained_model="albert-base-v2")
    del albert, test_data_loader
    # BHG a lot of extra code in here?
    gpt2.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="gpt2", df=test_df)
    gpt2_output, target = test_eval_fn_ensemble(test_data_loader, gpt2, device, pretrained_model="gpt2")
    del gpt2, test_data_loader
    
    print(bert_output)
    print(gpt2_output)
    print('------------------------------')
    output1 = np.add(bert_output, xlnet_output)
    output2 = np.add(roberta_output, albert_output)
    output = np.add(output1, output2)
    output = np.add(output, gpt2_output)
    output = (np.divide(output,5.0))
    output = np.argmax(output, axis=1)

    y_test = target
    y_pred = output
    
    print(f'\n---Probability averaging ensemble---\n')
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
    


    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)
    

def averaging3():
    xlnet, roberta, gpt2 = load_models()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()
    device = set_device()

    

    xlnet.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="xlnet-base-cased", df=test_df)
    xlnet_output, target = test_eval_fn_ensemble(test_data_loader, xlnet, device, pretrained_model="xlnet-base-cased")
    del xlnet, test_data_loader

    roberta.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="roberta-base", df=test_df)
    roberta_output, target = test_eval_fn_ensemble(test_data_loader, roberta, device, pretrained_model="roberta-base")
    del roberta, test_data_loader

    

    gpt2.to(device)
    test_data_loader = generate_dataset_for_ensembling(pretrained_model="gpt2", df=test_df)
    gpt2_output, target = test_eval_fn_ensemble(test_data_loader, gpt2, device, pretrained_model="gpt2")
    del gpt2, test_data_loader
    
    output1 = np.add(gpt2_output, xlnet_output)
    output2 = np.add(roberta_output, output1)
    
    output = (np.divide(output2,3.0))
    output = np.argmax(output, axis=1)

    y_test = target
    y_pred = output
    
    print(f'\n---Probability averaging ensemble---\n')
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
    


    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)






if __name__=="__main__":
    if args.ensemble_type == "max-voting":
        max_vote()
    elif args.ensemble_type == "rocauc":
        rocauc()
    elif args.ensemble_type == "mv3":
        max_vote3()
    elif args.ensemble_type == "pa3":
        averaging3()
    else:
        averaging()

