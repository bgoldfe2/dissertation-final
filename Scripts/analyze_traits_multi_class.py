# name: Bruce Goldfeder
# class: CSI 999
# university: George Mason University
# date: July 23, 2023

import pandas as pd
import numpy as np
from Model_Config import traits
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os


def parse_tvn(run_folder):
    #folder = 'Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output'
    file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Age-test_acc-0.8619641547007652.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Ethnicity-test_acc-0.7924745833770045.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Gender-test_acc-0.8691961010376271.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Others-test_acc-0.6660727387066345.csv'
    #file = '../Runs/2023-08-14_16_20_29--roberta-base/Ensemble/Output/ensemble-Religion-test_acc-0.9057750759878419.csv'

    ################# Outer Loop to read in all the files aka traits in Ensemble/Output and loop #################################

    # TODO loop through the folder and get each of the files starting with acc in the name or not metrics in the name?
    # iterate over files in that directory

    # Flag that identifies output results
    results_flag = '_acc-'
    results_file_list = []
    output_folder = ''.join([run_folder,'Output/'])
    print(output_folder)
    for filename in os.listdir(output_folder):
        f = os.path.join(output_folder, filename)
        # checking if it is a file
        if os.path.isfile(f):
            if results_flag in filename:
                results_file_list.append(filename)

    print(results_file_list)
    
    for results in results_file_list:    
        # file trait identifier during loop
        file_trt = results.split('---')[0]
        print('The file trait this iteration is ',file_trt)

        # Set the file location for one v rest runs        
        file = ''.join([run_folder,'Output/',results])

        df = pd.read_csv(file)
        total_all = len(df)
        print(total_all)
        print(df.columns)
        # Number in per label that are correct
        df['match'] = df['target']==df['y_pred']
        
        # Number in per label that are correct
        count = df.groupby('label').size()
        print("count is of type ", type(count))
        print("keys are ", count.axes)
        print("get value for Age ", count.get('Age'))
        print("Size of each trait \n", count)

        # Aggregate Confusion Matrix generation for the model        
        #print("confusion matrix for ","religion"," versus Not Cyberbullying")
        cm = confusion_matrix(df['target'], df['y_pred'])
        #print("the type of the confusion matrix is ", type(cm))
        # Print the confusion matrix
        print(cm)
        fig1, ax = plt.subplots()
        plt.title(''.join(["Single Trait ", file_trt, " vs All the Rest"]))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        trt_labels = [file_trt,'All-the-Rest']
        ax.set_yticklabels(trt_labels)
        ax.set_xticklabels(trt_labels)
        #plt.show()  # Show all plots at the end - can be same for saving?
        fig1.savefig(''.join([run_folder, 'Ensemble/Figures/','ensemble-bin-',file_trt,'-vs-All-the-Rest-conf-mat.pdf']))
        
        # Check to see the numbers add to 9541 = size of the test set
        cm_sum =  cm.sum()
        print("sum of numbers in cm is ", cm_sum) 
        
        df['false_pos'] = np.where(df['target']==0, 1, 0) & np.where(df['y_pred']==1, 1, 0)
        df['false_neg'] = np.where(df['target']==1, 1, 0) & np.where(df['y_pred']==0, 1, 0)
        df_cnt_fp = df.groupby('label')['false_pos'].apply(lambda x: (x==True).sum()).reset_index(name='count')
        df_cnt_fn = df.groupby('label')['false_neg'].apply(lambda x: (x==True).sum()).reset_index(name='count')
        # print('this is the df_cnt_fn ', df_cnt_fn)
        # print('false positives')
        # print(df_cnt_fp)
        # print(type(df_cnt_fp))
        # print(df_cnt_fp.axes)

        fp = df_cnt_fp.loc[df_cnt_fp['label']==file_trt, 'count'].values[0]
        fn = df_cnt_fn.loc[df_cnt_fn['label']==file_trt, 'count'].values[0]
        print(type(fp))
        print(fp)
        
        print('false negatives')
        print(df_cnt_fn)

        # Create each sub-confusion matrix of 2 x 2 for the five traits
        # Test hard coded for religion
        total_in_trt = count.get(file_trt)  # 1575
        print('total in ', file_trt, ' religion is ', total_in_trt)

        total_true_religion = cm[0][0]
        total_true_notcb = cm[1][1]
        
        cm_trt = np.array([[total_true_religion, fp], [fn, total_true_notcb]])
        
        print(cm_trt)

        # Show the distribution of false cyberbullying inferences that should have been Notcb
        
        fig2, ax = plt.subplots()
        # Create a barplot
        x = df_cnt_fp.iloc[:, 0].to_list()
        y = df_cnt_fp.iloc[:, 1].to_list()
        #y = cm_religion.T[1]
        print(" x ", x, " y ", y)
        plt.title("".join([file_trt, " vs All the Rest Model, ", file_trt, " that were labelled All the Rest"]))
        plt.bar(x, y)
        fig2.savefig(''.join([run_folder, 'Ensemble/Figures/','ensemble-bin-',file_trt,'-False-All-the-Rest-bar-plot.pdf']))


        # Show the distribution of the Notcb inferences that should have been cyberbullying
        fig3, ax = plt.subplots()
        # Create a barplot
        x = df_cnt_fn.iloc[:, 0].to_list()
        y = df_cnt_fn.iloc[:, 1].to_list()
        #y = cm_religion.T[1]
        print(" x ", x, " y ", y)
        plt.title("".join([file_trt, " vs All the Rest Model, All the Rest that were labelled ", file_trt]))
        plt.bar(x, y)
        fig3.savefig(''.join([run_folder, 'Ensemble/Figures/','ensemble-bin-All-the-Rest-False-',file_trt,'.pdf']))

        # Show the plot TURN ON FOR DEBUG
        plt.show()
        
def plot_conf_mat_from_sk_cm(run_folder, cm):
    # Print the confusion matrix
    print(cm)
    fig1, ax = plt.subplots()
    plt.title(''.join(["Multi-Class Confusion Matrix - Six Traits"]))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    trt_labels = list(traits.values())
    print(trt_labels)
    
    ax.set_yticklabels(trt_labels)
    ax.set_xticklabels(trt_labels)
    plt.show()  # Show all plots at the end - can be same for saving?
    fig1.savefig(''.join([run_folder, 'Ensemble/','ensemble-multiclass-six-traits-conf-mat.pdf']))
        



if __name__=="__main__":
    # Current best largest model run
    test_run = '../Runs/2023-08-25_20_03_38--deberta-v3-large/'
    
    # Currently hard coded from largest models run and average ensemble
    cm = np.array( [[1538, 3, 4, 16, 10, 0], 
                [0, 1571, 2, 7, 3, 3], 
                [1, 3, 1455, 84, 68, 2], 
                [13, 4, 71, 968, 466, 62], 
                [11, 2, 85, 214, 1271, 2], 
                [0, 0, 2, 30, 4, 1566]])
    
    plot_conf_mat_from_sk_cm(test_run,cm)
