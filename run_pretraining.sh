#!/usr/bin/env bash

input="$HOME/workspace/experiments/tf_examples.tfrecord"
# BERT_BASE_DIR="./pretrained_models/uncased_L-12_H-768_A-12"
BERT_BASE_DIR="$HOME/workspace/experiments/pretrained_models/uncased_L-12_H-768_A-12"
output="$HOME/workspace/experiments/pretraining_output/"

python run_pretraining.py \
  --input_file=$input \
  --output_dir=$output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
