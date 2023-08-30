# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

from Model_Config import Model_Config

from evaluate import evaluate_all_models
from driver import get_parser
from ensemble import averaging, max_vote
import json

def test_eval(args: Model_Config):

    evaluate_all_models(args)

def test_average_ensemble(args):

    return averaging(args)

def test_max_vote(args):

    max_vote(args)

if __name__=="__main__":
    
    parser = get_parser()
    raw_args = parser.parse_args()

    # Declare the model list
    model_list = ['microsoft/deberta-v3-large', 'EleutherAI/gpt-neo-1.3B', 'roberta-large',\
                    'xlnet-large-cased', 'albert-xxlarge-v2']
    
    #model_list = ['microsoft/deberta-v3-large']
    
    
    args = Model_Config(raw_args)
    args.model_list = model_list

    # TODO This is hardcoded to number of models needs to be made dynamic
    #      This will only work with all current five model runs
    # TODO currently hardcode this test run folder
    run2test =  "2023-08-25_20_03_38--deberta-v3-large" #2023-07-03_14_53_05--deberta-v3-large"
    folder_name = "../Runs/" + run2test 

    # High level folders defined
    args.run_path=folder_name
    args.model_path = folder_name + "/Models/"
    args.output_path = folder_name + "/Output/"
    args.figure_path = folder_name  + "/Figures/"
    print('args.model_path in eval_test are\n',args.model_path)


    # Test the evaluate.py - evaluate_all_models() function
    #test_eval(args)

    # Test the averaging() function in ensembles.py
    avg_rst = test_average_ensemble(args)
    #print(type(avg_rst))
    print(avg_rst)

    # currently hard coded - can adjust to using current run if needed
    with open("../Runs/2023-08-25_20_03_38--deberta-v3-large/Ensemble/avg_results.json", "w") as fp:
        json.dump(avg_rst, fp)  # encode dict into JSON
    
    

    # Test the max_vote() function in ensembles.py
    #test_max_vote(args)