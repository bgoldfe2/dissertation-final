import argparse

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--max_length", default=128, type=int,  help='Maximum number of words in a sample')
    parser.add_argument("--train_batch_size", default=16, type=int,  help='Training batch size')
    parser.add_argument("--valid_batch_size", default=32, type=int,  help='Validation batch size')
    parser.add_argument("--test_batch_size", default=32, type=int,  help='Test batch size')
    parser.add_argument("--epochs", default=4, type=int,  help='Number of training epochs')
    parser.add_argument("-lr","--learning_rate", default=2e-5, type=float,  help='The learning rate to use')
    parser.add_argument("-wd","--weight_decay", default=1e-4, type=float,  help=' Decoupled weight decay to apply')
    parser.add_argument("--adamw_epsilon", default=1e-8, type=float,  help='Adam’s epsilon for numerical stability')
    parser.add_argument("--warmup_steps", default=0, type=int,  help='The number of steps for the warmup phase.')
    parser.add_argument("--classes", default=6, type=int, help='Number of output classes')
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--device", type=str, default="gpu", help="Training device - cpu/gpu")
    parser.add_argument("--dataset", type=str, default="FGBC", help="Select Dataset - FGBC/Twitter")

    parser.add_argument("--pretrained_model", default="microsoft/deberta-v3-base", type=str, help='Name of the pretrained model')  
    parser.add_argument("--deberta_hidden", default=768, type=int, help='Number of hidden states for DeBerta')
    parser.add_argument("--gpt_neo_hidden", default=768, type=int, help='Number of hidden states for GPT_Neo')
    parser.add_argument("--gpt_neo13_hidden", default=2048, type=int, help='Number of hidden states for Albert')
    parser.add_argument("--roberta_hidden", default=768, type=int, help='Number of hidden states for Roberta')
    parser.add_argument("--xlnet_hidden", default=768, type=int, help='Number of hidden states for XLNet')
    parser.add_argument("--albert_hidden", default=768, type=int, help='Number of hidden states for Albert')
    parser.add_argument("--ensemble_type", type=str, default="max-voting", help="Ensemble type - max-voting or averaging")

    parser.add_argument("--run_path", default="../Runs/", type=str, help='Path to Run logs')
    parser.add_argument("--dataset_path", default="../Dataset/SixClass/", type=str, help='Path to dataset file')
    parser.add_argument("--model_path", default="../Models/", type=str, help='Save best model')
    parser.add_argument("--output_path", default="../Output/", type=str, help='Get predicted labels for test data')
    parser.add_argument("--figure_path", default="../Figures/", type=str, help='Directory for accuracy and loss plots')
    parser.add_argument("--split", default="no", type=str, help='If base file needs to be splitted into Train, Val, Test')

    return parser

class Model_Config:

    def __init__(self):
        parser = get_parser()
        args = parser.parse_args()

        self.max_length=args.max_length         
        self.train_batch_size=args.train_batch_size    
        self.valid_batch_size=args.valid_batch_size     
        self.test_batch_size=args.test_batch_size      
        self.epochs=args.epochs
        self.learning_rate=args.learning_rate
        self.weight_decay=args.weight_decay
        self.adamw_epsilon=args.adamw_epsilon
        self.warmup_steps=args.warmup_steps
        self.classes=args.classes
        self.dropout=args.dropout
        self.seed=args.seed
        self.device=args.device
        self.dataset=args.dataset

        self.pretrained_model=args.pretrained_model
        self.deberta_hidden=args.deberta_hidden
        self.gpt_neo_hidden=args.gpt_neo_hidden
        self.gpt_neo13_hidden=args.gpt_neo13_hidden
        self.roberta_hidden=args.roberta_hidden
        self.xlnet_hidden=args.xlnet_hidden
        self.albert_hidden=args.albert_hidden
        self.ensemble_type=args.ensemble_type

        self.run_path=args.run_path
        self.dataset_path=args.dataset_path
        self.model_path=args.model_path
        self.output_path=args.output_path
        self.figure_path=args.figure_path
        self.split=args.split