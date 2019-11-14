# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:54:07 2019

@author: tomson
"""
def change(config):
    config.task_st='slot_tagger' #slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus
    config.task_sc='hiddenAttention' # none, hiddenAttention, hiddenCNN, maxPooling, 2tails
    config.st_weight=0.5
    
    config.pretrained_model_type='bert'
    config.pretrained_model_name='bert-base-uncased' #bert-base-cased #bert-large-uncased-whole-word-masking #bert-base-uncased
    config.pretrained_model_name='bert-base-cased'
    config.dataroot='data/atis-2'
    config.dataset='atis-2'
    config.dataroot='data/snips'
    config.dataset='snips'
    config.hidden_size=200 # 100, 200
    config.num_layers=1
    config.tag_emb_size=100  ## for slot_tagger_with_focus
    config.batchSize=32 # 16, 32
    
    config.optim='bertadam' # bertadam, adamw
    config.lr=5e-5 # 1e-5, 5e-5, 1e-4, 1e-3
    config.max_norm=1 # working for adamw
    config.dropout=0.1 # 0.1, 0.5
    
    config.max_epoch=50
    
    config.deviceId=0
    # device=0 means auto-choosing a GPU
    # Set deviceId=-1 if you are going to use cpu for training.
    config.experiment='exp'
