# e2e-masked-graphcrf
1,most code is from https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/, and thanks to them

2,we implement the e2e masked graph-based CRF module in slot-tagger.py, and the class name is 'graphCRFmodel', note that it is easy and transferable

3,to produce the results, please follow our article, the hyperparams uses our default set, note that it can be changed in params.py.

4,for joint NLU, please save and use two models, one for best F1(slot) on dev, another one for best acc(intent) on dev. 

5,other small features are implemented too, please refer to this code and check.

6,you can obtain data from https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/

7,use finetune_on_pregenerated.py to pretrain lm first for our really used BERT parameters (use all train-sets of all datasets)
 
