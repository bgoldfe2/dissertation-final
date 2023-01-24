# Utility to run through the different models
# Bruce Goldfeder
# CSI 999, George Mason University
# Dec 27, 2022

import os 
 
list_scripts = ["--pretrained_model bert-base-uncased", \
                "--pretrained_model gpt2", \
                "--pretrained_model roberta-base", \
                "--pretrained_model xlnet-base-cased", \
                "--pretrained_model distilbert-base-uncased"] 
 
for script in list_scripts: 
	__ = os.system("python train.py --split 'no' " + script) 