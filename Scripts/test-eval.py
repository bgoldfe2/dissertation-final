from Model_Config import Model_Config

from evaluate import evaluate_all_models
from driver import get_parser
from ensemble import averaging, max_vote

def test_eval(args: Model_Config):

    evaluate_all_models(args)

def test_average_ensemble(args):

    averaging(args)

def test_max_vote(args):

    max_vote(args)

if __name__=="__main__":
    
    parser = get_parser()
    raw_args = parser.parse_args()

    # Declare the model list
    model_list = ['microsoft/deberta-v3-base', 'EleutherAI/gpt-neo-125m', 'roberta-base',\
                    'xlnet-base-cased', 'albert-base-v2']
    
    #model_list = ['microsoft/deberta-v3-base']
    
    
    args = Model_Config(raw_args)
    args.model_list = model_list

    # TODO This is hardcoded to number of models needs to be made dynamic
    #      This will only work with all current five model runs
    # TODO currently hardcode this test run folder
    run2test =  "2023-07-16_11_45_47--deberta-v3-base" #2023-07-03_14_53_05--deberta-v3-base"
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
    #test_average_ensemble(args)

    # Test the max_vote() function in ensembles.py
    test_max_vote(args)