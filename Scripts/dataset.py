import torch
import pandas as pd
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer, DistilBertTokenizer, GPT2Tokenizer, DebertaV2Tokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from common import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class DatasetDeberta:
    def __init__(self, text, target, pretrained_model = args.pretrained_model):
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "token_type_ids":torch.tensor(token_type_ids, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }


class DatasetGPT_Neo:
    def __init__(self, text, target, pretrained_model = args.pretrained_model):
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.tokenizer.pad_token = "[PAD]"
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }

class DatasetRoberta:
    def __init__(self, text, target, pretrained_model = args.pretrained_model):
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }

class DatasetDistilBert:
    def __init__(self, text, target, pretrained_model = args.pretrained_model):
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }

class DatasetXLNet:
    def __init__(self, text, target, pretrained_model = args.pretrained_model):
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_length = args.max_length
        self.target = target

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = "".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text = text,
            padding = "max_length",
            truncation = True,
            max_length = self.max_length
        )

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]

        return{
            "input_ids":torch.tensor(input_ids,dtype = torch.long),
            "attention_mask":torch.tensor(attention_mask, dtype = torch.long),
            "token_type_ids":torch.tensor(token_type_ids, dtype = torch.long),
            "target":torch.tensor(self.target[item], dtype = torch.long)
        }
# new version has seed set to 7 not 42 and not to None which is in earlier version
def train_validate_test_split(df, train_percent=0.6, validate_percent=.2, seed=7):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(tokenizer("Hello world"))
    
    df = pd.read_csv(args.dataset_path+"data.csv").dropna()
    
    #Splitting the dataset
    train_df, valid_df, test_df = train_validate_test_split(df)
    train_df.to_csv(args.dataset_path+'train.csv')
    valid_df.to_csv(args.dataset_path+'valid.csv')
    test_df.to_csv(args.dataset_path+'test.csv')

    print(set(df['label'].values))
    dataset = DatasetBert(text=df.text.values, target=df.target.values)
    print(df.iloc[1]['text'])
    print(dataset[1])
