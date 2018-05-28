#!/bin/bash

exp_name=film

python scripts/train_film_model.py \
  --train_question_h5 "$data_dir/train_questions.h5" \
  --val_question_h5 "$data_dir/val_questions.h5" \
  --train_features_h5 "$data_dir/train_features.h5" \
  --val_features_h5 "$data_dir/val_features.h5" \
  --vocab_json "$data_dir/vocab.json"