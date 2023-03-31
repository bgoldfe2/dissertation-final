import torch
import torch.nn as nn
#from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
import warnings
import pandas as pd

# import utils
import engine
from model import RobertaFGBC, XLNetFGBC, AlbertFGBC, GPT_NeoFGBC, DeBertaFGBC, GPT_Neo13FGBC
from dataset import DatasetRoberta, DatasetXLNet, DatasetAlbert, DatasetGPT_Neo, DatasetDeberta, DatasetGPT_Neo13
from common import get_parser
from evaluate import test_evaluate
from train import generate_dataset, set_model, count_model_parameters
import utils

from visualize import save_acc_curves, save_loss_curves
from dataset import train_validate_test_split
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List
from tqdm.auto import tqdm

parser = get_parser()
args = parser.parse_args()
warnings.filterwarnings("ignore")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def test_board(data,model, device):
    
    tb = SummaryWriter()    

   
    tb.add_graph(model, input_to_model=data, verbose=True)  

def get_data_loaders():
    train_df = pd.read_csv(f'{args.dataset_path}train.csv').dropna()
    valid_df = pd.read_csv(f'{args.dataset_path}valid.csv').dropna()
    test_df = pd.read_csv(f'{args.dataset_path}test.csv').dropna()

    print(set(train_df.label.values))
    print("train len - {}, valid len - {}, test len - {}".format(len(train_df),\
     len(valid_df),len(test_df)))
    for col in train_df.columns:
        print(col)
    print("train example text -- ",train_df.text[1],"\nwith target -- ",\
     train_df.label[1])

    # BHG Text encoding occurs at model instantiation  
    train_dataset = generate_dataset(train_df)
    print("train_dataset object is of type -- ",type(train_dataset))
    print("Print Encoded Token Byte tensor at location 1 -- ", train_dataset[1]['input_ids'])
    
    encoding = train_dataset[1]['input_ids']
    print("The Decoded Token Text tensor is -- ",train_dataset.tokenizer.convert_ids_to_tokens(encoding))
    
    valid_dataset = generate_dataset(valid_df)
    valid_data_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = args.valid_batch_size,
        shuffle = True
    )
    
    device = utils.set_device()


    model = set_model()
    # BHG model type and number of parameters initial instantiation
    print("Model Class: ", type(model), "Num Params: ",count_model_parameters(model))
    model = model.to(device)

    return valid_data_loader, device, model

if __name__=="__main__":
    
    data_loader, device, model = get_data_loaders()
    test_board(data_loader, device, model)