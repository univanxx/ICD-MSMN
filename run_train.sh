#!/bin/bash

word_embedding_path="/mnt/s3-data/datasets/mimic4/mimic_iv/mimic3_embeds/word2vec_sg0_100.model"


clear & CUDA_VISIBLE_DEVICES=0 python main.py --combiner lstm --rnn_dim 512 --num_layers 1 --decoder MultiLabelMultiHeadLAATV2 --attention_head 8 --attention_dim 512 --learning_rate 5e-4 --train_epoch 20 --batch_size 16 --gradient_accumulation_steps 1 --xavier --main_code_loss_weight 0.0 --rdrop_alpha 5.0 --est_cls 1 --term_count 8 --word_embedding_path $word_embedding_path