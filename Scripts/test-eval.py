from Model_Config import Model_Config

from evaluate import evaluate_all_models
from driver import get_parser

def eval_test(run_fld_nm, args: Model_Config):
    # TODO This is hardcoded to number of models needs to be made dynamic
    #      This will only work with all current five model runs
    folder_name = "../Runs/" + run_fld_nm

    # High level folders defined
    #fld = ['/Models/', '/Output/', '/Figures/']
    args.run_path=folder_name
    args.model_path = folder_name + "/Models/"
    args.output_path = folder_name + "/Output/"
    args.figure_path = folder_name  + "/Figures/"
    print('args.model_path in eval_test are\n',args.model_path)

    evaluate_all_models(args)


if __name__=="__main__":
    # TODO currently hardcode this test run folder
    run2test = "2023-07-03_14_53_05--deberta-v3-base"
    parser = get_parser()
    raw_args = parser.parse_args()

    # Declare the model list
    #model_list = ['microsoft/deberta-v3-base', 'EleutherAI/gpt-neo-125m', 'roberta-base',\
    #                'xlnet-base-cased', 'albert-base-v2']
    
    model_list = ['microsoft/deberta-v3-base']
    
    
    args = Model_Config(raw_args)
    args.model_list = model_list


    # Test the evaluate.py - evaluate_all_models() function
    eval_test(run2test, args)