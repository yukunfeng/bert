output="$HOME/workspace/experiments/tf_examples.tfrecord"
BERT_BASE_DIR="./pretrained_models/uncased_L-12_H-768_A-12/"
input="$HOME/workspace/experiments/toy_data/books.txt-00230-of-00500"

python create_pretraining_data.py \
  --input_file=$input \
  --output_file=$output \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
