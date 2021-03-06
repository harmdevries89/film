#!/bin/bash

exp_name=film

python scripts/train_model.py \
  --train_question_h5 "$data_dir/train_questions.h5" \
  --val_question_h5 "$data_dir/val_questions.h5" \
  --train_features_h5 "$data_dir/train_features.h5" \
  --val_features_h5 "$data_dir/val_features.h5" \
  --vocab_json "$data_dir/vocab.json" \
  --checkpoint_path "$exp_dir"/"$exp_name".pt \
  --shuffle_train_data 0 \
  --model_type FiLM \
  --loader_num_workers 2 \
  --num_iterations 20000000 \
  --print_verbose_every 20000000 \
  --checkpoint_every 11000 \
  --record_loss_every 100 \
  --num_val_samples 149991 \
  --optimizer Adam \
  --learning_rate 3e-4 \
  --batch_size 64 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 1 \
  --module_batchnorm 1 \
  --classifier_batchnorm 1 \
  --bidirectional 0 \
  --decoder_type linear \
  --encoder_type gru \
  --weight_decay 1e-5 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 200 \
  --rnn_hidden_dim 4096 \
  --rnn_output_batchnorm 0 \
  --classifier_downsample maxpoolfull \
  --classifier_proj_dim 512 \
  --classifier_fc_dims 1024 \
  --module_input_proj 1 \
  --module_residual 1 \
  --module_dim 128 \
  --module_dropout 0e-2 \
  --module_stem_kernel_size 3 \
  --module_kernel_size 3 \
  --module_batchnorm_affine 0 \
  --module_num_layers 1 \
  --num_modules 4 \
  --condition_pattern 1,1,1,1 \
  --gamma_option linear \
  --gamma_baseline 1 \
  --use_gamma 1 \
  --use_beta 1 \
  --condition_method bn-film \
  --program_generator_parameter_efficient 1 \
  | tee "$exp_dir"/"$exp_name".log
