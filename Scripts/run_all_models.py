# Utility to run through the different models
# Bruce Goldfeder
# CSI 999, George Mason University
# Dec 27, 2022

import os 
 
list_scripts = ["--pretrained_model microsoft/deberta-v3-base", \
                "--pretrained_model EleutherAI/gpt-neo-125M", \
                "--pretrained_model roberta-base", \
                "--pretrained_model xlnet-base-cased", \
                "--pretrained_model albert-base-v2"] 
 
for script in list_scripts: 
	__ = os.system("python train.py --split 'no' " + script) 