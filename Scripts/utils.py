import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from common import get_parser
from model import DeBertaFGBC, RobertaFGBC, XLNetFGBC, AlbertFGBC, GPT_NeoFGBC
from dataset import DatasetDeberta, DatasetRoberta, DatasetXLNet, DatasetAlbert, DatasetGPT_Neo

parser = get_parser()
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_device():
    device = ""
    if(args.device=="cpu"):
        device = "cpu"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(device=="cpu"):
            print("GPU not available.")
    return device

def sorting_function(val):
    return val[1]    

def load_prediction():
    deberta_path = (f'{args.output_path}microsoft/deberta_v3_base.csv')
    xlnet_path = (f'{args.output_path}xlnet-base-cased.csv')
    roberta_path = (f'{args.output_path}roberta-base.csv')
    albert_path = (f'{args.output_path}albert-xxlarge-v2.csv')
    gpt_neo_path = (f'{args.output_path}EleutherAI/gpt-neo-125M.csv')

    deberta = pd.read_csv(deberta_path)
    xlnet = pd.read_csv(xlnet_path)
    roberta = pd.read_csv(roberta_path)
    albert = pd.read_csv(albert_path)
    gpt_neo = pd.read_csv(gpt_neo_path)

    return deberta, xlnet, roberta, albert, gpt_neo

def print_stats(max_vote_df, deberta, xlnet, roberta, albert):
    print(max_vote_df.head())
    print(f'---Ground Truth---\n{deberta.target.value_counts()}')
    print(f'---Bert---\n{deberta.y_pred.value_counts()}')
    print(f'---XLNet---\n{xlnet.y_pred.value_counts()}')
    print(f'---Roberta---\n{roberta.y_pred.value_counts()}')
    print(f'---albert---\n{albert.y_pred.value_counts()}')

def evaluate_ensemble(max_vote_df):
    y_test = max_vote_df['target'].values
    y_pred = max_vote_df['pred'].values
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
    
    max_vote_df.to_csv(f'{args.output_path}Ensemble-{args.ensemble_type}---test_acc---{acc}.csv', index = False)

    conf_mat = confusion_matrix(y_test,y_pred)
    print(conf_mat)

def generate_dataset_for_ensembling(pretrained_model, df):
    if(pretrained_model == "microsoft/deberta_v3_base.csv"):
        dataset = DatasetDeberta(text=df.text.values, target=df.target.values, pretrained_model="microsoft/deberta_v3_base.csv")
    elif(pretrained_model== "roberta-base"):
        dataset = DatasetRoberta(text=df.text.values, target=df.target.values, pretrained_model="roberta-base")
    elif(pretrained_model== "xlnet-base-cased"):
        dataset = DatasetXLNet(text=df.text.values, target=df.target.values, pretrained_model="xlnet-base-cased")
    elif(pretrained_model == "albert-xxlarge-v2"):
        dataset = DatasetAlbert(text=df.text.values, target=df.target.values, pretrained_model="albert-xxlarge-v2")
    elif(pretrained_model == "EleutherAI/gpt-neo-125M"):
        dataset = DatasetGPT_Neo(text=df.text.values, target=df.target.values, pretrained_model="EleutherAI/gpt-neo-125M")

    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = args.test_batch_size,
        shuffle = False
    )

    return data_loader

def load_models():
    deberta_path = (f'{args.model_path}bert-base-uncased_Best_Val_Acc.bin')
    xlnet_path = (f'{args.model_path}xlnet-base-cased_Best_Val_Acc.bin')
    roberta_path = (f'{args.model_path}roberta-base_Best_Val_Acc.bin')
    albert_path = (f'{args.model_path}albert-xxlarge-v2_Best_Val_Acc.bin')
    gpt_neo_path = (f'{args.model_path}EleutherAI/gpt-neo-125M_Best_Val_Acc.bin')

    deberta = DeBertaFGBC(pretrained_model="bert-base-uncased")
    xlnet = XLNetFGBC(pretrained_model="xlnet-base-cased")
    roberta = RobertaFGBC(pretrained_model="roberta-base")
    albert = AlbertFGBC(pretrained_model="albert-xxlarge-v2")
    gpt_neo = GPT_NeoFGBC(pretrained_model="EleutherAI/gpt-neo-125M")

    deberta.load_state_dict(torch.load(deberta_path))
    xlnet.load_state_dict(torch.load(xlnet_path))
    roberta.load_state_dict(torch.load(roberta_path))
    albert.load_state_dict(torch.load(albert_path))
    gpt_neo.load_state_dict(torch.load(gpt_neo_path))

    return deberta, xlnet, roberta, albert, gpt_neo

def oneHot(arr):
    b = np.zeros((arr.size, arr.max()+1))
    b[np.arange(arr.size),arr] = 1
    return b

def calc_roc_auc(all_labels, all_logits, name=None):
    attributes = []
    if(args.classes==6):
       attributes = ['Age', 'Ethnicity', 'Gender', 'Notcb', 'Others', 'Religion']
    elif(args.classes==5):
        attributes = ['Age', 'Ethnicity', 'Gender', 'Religion', 'Others',]
    elif(args.classes==3):
        attributes = ['None', 'Sexism', 'Racism']
    all_labels = oneHot(all_labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(0,len(attributes)):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='%s %g' % (attributes[i], roc_auc[i]))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    if (name!=None):
        plt.savefig(f"{args.figure_path}{name}---roc_auc_curve---.pdf")
    else:
        plt.savefig(f"{args.figure_path}{args.pretrained_model}---roc_auc_curve---.pdf")
    plt.clf()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f'ROC-AUC Score: {roc_auc["micro"]}')