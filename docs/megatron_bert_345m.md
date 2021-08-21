

# Performance for vanilla Megatron 

## Single GPU

./scripts/pretrain_megatron_bert_cased_345m.sh 

/opt/conda/bin/python3 /gpfs/mira-home/xduan7/projects/scalable-transformer/third-party/megatron/pretrain_bert.py --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 512 --max-position-embeddings 512 --lr 0.0001 --lr-decay-iters 990000 --train-iters 2000000 --min-lr 0.00001 --lr-warmup-fraction 0.01 --vocab-file /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt --split 949,50,1 --fp16 --micro-batch-size 4 --global-batch-size 8 --log-interval 10 --save-interval 500 --eval-interval 100 --eval-iters 10 --checkpoint-activations --save /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m --load /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/megatron/bert_cased_345m --data-path /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence

------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... True
  checkpoint_num_layers ........................... 1
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_impl ....................................... infer
  data_parallel_size .............................. 1
  data_path ....................................... ['/gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence']
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_seq_length .............................. None
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  embedding_path .................................. None
  encoder_seq_length .............................. 512
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 8
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/megatron/bert_cased_345m
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0001
  lr_decay_iters .................................. 990000
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 512
  merge_file ...................................... None
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 1
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................................... 500
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 512
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 949,50,1
  tensor_model_parallel_size ...................... 1
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  titles_data_path ................................ None
  tokenizer_type .................................. BertWordPieceLowerCase
  train_iters ..................................... 2000000
  train_samples ................................... None
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_ddp ................... False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  weight_decay .................................... 0.01
  world_size ...................................... 1
-------------------- end of arguments ---------------------

GPU util: ~85%
GPU mem: 8430MiB / 40537MiB

(over 100 iterations = 800 samples)
22.39 ms per sample 

time (ms) | forward-compute: 43.42 | backward-compute: 90.74 | backward-params-all-reduce: 10.86 | backward-embedding-all-reduce: 0.02 | optimizer-copy-to-main-grad: 3.32 | optimizer-unscale-and-check-inf: 3.17 | optimizer-clip-main-grad: 5.56 | optimizer-copy-main-to-model-params: 3.63 | optimizer: 24.05 | batch-generator: 1.62

---


## Single-node (8 GPUs)

./scripts/pretrain_megatron_bert_cased_345m.sh -d

/opt/conda/bin/python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr thetagpu13 --master_port 6002 /gpfs/mira-home/xduan7/projects/scalable-transformer/third-party/megatron/pretrain_bert.py --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 512 --max-position-embeddings 512 --lr 0.0001 --lr-decay-iters 990000 --train-iters 2000000 --min-lr 0.00001 --lr-warmup-fraction 0.01 --vocab-file /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt --split 949,50,1 --fp16 --micro-batch-size 4 --global-batch-size 32 --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --log-interval 10 --save-interval 500 --eval-interval 100 --eval-iters 10 --checkpoint-activations --save /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m --data-path /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence

------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... True
  checkpoint_num_layers ........................... 1
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_impl ....................................... infer
  data_parallel_size .............................. 2
  data_path ....................................... ['/gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence']
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_seq_length .............................. None
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  embedding_path .................................. None
  encoder_seq_length .............................. 512
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 32
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ None
  local_rank ...................................... 0
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0001
  lr_decay_iters .................................. 990000
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 512
  merge_file ...................................... None
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 2
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................................... 500
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 512
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 949,50,1
  tensor_model_parallel_size ...................... 2
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  titles_data_path ................................ None
  tokenizer_type .................................. BertWordPieceLowerCase
  train_iters ..................................... 2000000
  train_samples ................................... None
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_ddp ................... False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  weight_decay .................................... 0.01
  world_size ...................................... 8
-------------------- end of arguments ---------------------

GPU util: ~83%
GPU mem: 5624MiB - 6176MiB / 40537MiB

(over 100 iterations = 3200 samples)
7.25 ms per sample 

time (ms) | forward-compute: 53.94 | forward-recv: 11.05 | backward-compute: 99.20 | backward-send: 0.20 | backward-send-forward-recv: 1.98 | backward-params-all-reduce: 6.46 | backward-embedding-all-reduce: 24.96 | optimizer-copy-to-main-grad: 1.82 | optimizer-unscale-and-check-inf: 2.60 | optimizer-clip-main-grad: 2.14 | optimizer-copy-main-to-model-params: 1.48 | optimizer: 10.59 | batch-generator: 2.82

---


## Multi-node (2*8 GPUs)

mpirun --hostfile ${COBALT_NODEFILE} --map-by ppr:1:node ./scripts/pretrain_megatron_bert_cased_345m.sh -d                                  

/opt/conda/bin/python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr thetagpu13 --master_port 6002 /gpfs/mira-home/xduan7/projects/scalable-transformer/src/megatron/pretrain_bert.py --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 512 --max-position-embeddings 512 --lr 0.0001 --lr-decay-iters 990000 --train-iters 2000000 --min-lr 0.00001 --lr-warmup-fraction 0.01 --vocab-file /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt --split 949,50,1 --fp16 --micro-batch-size 4 --global-batch-size 64 --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --log-interval 10 --save-interval 500 --eval-interval 100 --eval-iters 10 --checkpoint-activations --save /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m --data-path /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence
/opt/conda/bin/python3 -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr thetagpu13 --master_port 6002 /gpfs/mira-home/xduan7/projects/scalable-transformer/src/megatron/pretrain_bert.py --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 512 --max-position-embeddings 512 --lr 0.0001 --lr-decay-iters 990000 --train-iters 2000000 --min-lr 0.00001 --lr-warmup-fraction 0.01 --vocab-file /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt --split 949,50,1 --fp16 --micro-batch-size 4 --global-batch-size 64 --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 --log-interval 10 --save-interval 500 --eval-interval 100 --eval-iters 10 --checkpoint-activations --save /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m --data-path /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence

------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... True
  checkpoint_num_layers ........................... 1
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_impl ....................................... infer
  data_parallel_size .............................. 4
  data_path ....................................... ['/gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence']
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_seq_length .............................. None
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  embedding_path .................................. None
  encoder_seq_length .............................. 512
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 64
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ None
  local_rank ...................................... 0
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0001
  lr_decay_iters .................................. 990000
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 512
  merge_file ...................................... None
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 2
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................................... 500
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 512
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 949,50,1
  tensor_model_parallel_size ...................... 2
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  titles_data_path ................................ None
  tokenizer_type .................................. BertWordPieceLowerCase
  train_iters ..................................... 2000000
  train_samples ................................... None
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_ddp ................... False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  weight_decay .................................... 0.01
  world_size ...................................... 16
-------------------- end of arguments ---------------------

GPU util: ~66%
GPU mem: 4788MiB - 5188MiB / 40537MiB

(over 100 iterations = 6400 samples)
44.9 ms per sample 

time (ms) | forward-compute: 909.54 | forward-recv: 222.47 | backward-compute: 925.70 | backward-send: 11.27 | backward-send-forward-recv: 205.57 | backward-params-all-reduce: 100.21 | backward-embedding-all-reduce: 230.54 | optimizer-copy-to-main-grad: 6.12 | optimizer-unscale-and-check-inf: 57.47 | optimizer-clip-main-grad: 54.98 | optimizer-copy-main-to-model-params: 5.90 | optimizer: 128.60 | batch-generator: 90.27

---

## Multi-node (4*8 GPUs)

------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... True
  checkpoint_num_layers ........................... 1
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_impl ....................................... infer
  data_parallel_size .............................. 8
  data_path ....................................... ['/gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence']
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_seq_length .............................. None
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  embedding_path .................................. None
  encoder_seq_length .............................. 512
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 128
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ None
  local_rank ...................................... 0
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0001
  lr_decay_iters .................................. 990000
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 512
  merge_file ...................................... None
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 2
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................................... 500
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 512
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 949,50,1
  tensor_model_parallel_size ...................... 2
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  titles_data_path ................................ None
  tokenizer_type .................................. BertWordPieceLowerCase
  train_iters ..................................... 2000000
  train_samples ................................... None
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_ddp ................... False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  weight_decay .................................... 0.01
  world_size ...................................... 32
-------------------- end of arguments ---------------------

GPU util: 82~%
GPU mem: 4548MiB - 4876MiB / 40537MiB

(over 100 iterations = 1280 samples)
2.95 ms per sample 

time (ms) | forward-compute: 89.49 | forward-recv: 12.34 | backward-compute: 113.45 | backward-send: 0.25 | backward-send-forward-recv: 2.62 | backward-params-all-reduce: 47.78 | backward-embedding-all-reduce: 32.91 | optimizer-copy-to-main-grad: 1.74 | optimizer-unscale-and-check-inf: 40.61 | optimizer-clip-main-grad: 2.45 | optimizer-copy-main-to-model-params: 1.47 | optimizer: 48.84 | batch-generator: 3.21


---


## Multi-node (8*8 GPUs)

------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... True
  checkpoint_num_layers ........................... 1
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_impl ....................................... infer
  data_parallel_size .............................. 16
  data_path ....................................... ['/gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence']
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_seq_length .............................. None
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  embedding_path .................................. None
  encoder_seq_length .............................. 512
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 256
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ None
  local_rank ...................................... 0
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0001
  lr_decay_iters .................................. 990000
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 512
  merge_file ...................................... None
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 2
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................................... 500
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 512
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 949,50,1
  tensor_model_parallel_size ...................... 2
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  titles_data_path ................................ None
  tokenizer_type .................................. BertWordPieceLowerCase
  train_iters ..................................... 2000000
  train_samples ................................... None
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_ddp ................... False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  weight_decay .................................... 0.01
  world_size ...................................... 64
-------------------- end of arguments ---------------------

GPU util: ~88%
GPU mem: 4584MiB - 4682MiB / 40537MiB

(over 100 iterations = 25600 samples)
1.44 ms per sample 

time (ms) | forward-compute: 120.38 | forward-recv: 13.24 | backward-compute: 117.51 | backward-send: 0.29 | backward-send-forward-recv: 2.82 | backward-params-all-reduce: 51.27 | backward-embedding-all-reduce: 33.24 | optimizer-copy-to-main-grad: 1.76 | optimizer-unscale-and-check-inf: 7.81 | optimizer-clip-main-grad: 2.77 | optimizer-copy-main-to-model-params: 1.52 | optimizer: 16.50 | batch-generator: 3.70


---

## Multi-node (16*8 GPUs)

------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. False
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.999
  adam_eps ........................................ 1e-08
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_query_key_layer_scaling ................... True
  apply_residual_connection_post_layernorm ........ False
  attention_dropout ............................... 0.1
  attention_softmax_in_fp32 ....................... False
  bert_binary_head ................................ True
  bert_load ....................................... None
  bf16 ............................................ False
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  checkpoint_activations .......................... True
  checkpoint_num_layers ........................... 1
  clip_grad ....................................... 1.0
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  data_impl ....................................... infer
  data_parallel_size .............................. 32
  data_path ....................................... ['/gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence']
  dataloader_type ................................. single
  DDP_impl ........................................ local
  decoder_seq_length .............................. None
  distribute_checkpointed_activations ............. False
  distributed_backend ............................. nccl
  embedding_path .................................. None
  encoder_seq_length .............................. 512
  eod_mask_loss ................................... False
  eval_interval ................................... 100
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  ffn_hidden_size ................................. 4096
  finetune ........................................ False
  fp16 ............................................ True
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  global_batch_size ............................... 512
  hidden_dropout .................................. 0.1
  hidden_size ..................................... 1024
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_dim ......................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  kv_channels ..................................... 64
  layernorm_epsilon ............................... 1e-05
  lazy_mpu_init ................................... None
  load ............................................ None
  local_rank ...................................... 0
  log_batch_size_to_tensorboard ................... False
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0001
  lr_decay_iters .................................. 990000
  lr_decay_samples ................................ None
  lr_decay_style .................................. linear
  lr_warmup_fraction .............................. 0.01
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  mask_prob ....................................... 0.15
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 512
  merge_file ...................................... None
  micro_batch_size ................................ 4
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  mmap_warmup ..................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  num_attention_heads ............................. 16
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_workers ..................................... 2
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  override_lr_scheduler ........................... False
  params_dtype .................................... torch.float16
  patch_dim ....................................... 16
  pipeline_model_parallel_size .................... 2
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  sample_rate ..................................... 1.0
  save ............................................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................................... 500
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 512
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  split ........................................... 949,50,1
  tensor_model_parallel_size ...................... 2
  tensorboard_dir ................................. None
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  titles_data_path ................................ None
  tokenizer_type .................................. BertWordPieceLowerCase
  train_iters ..................................... 2000000
  train_samples ................................... None
  use_checkpoint_lr_scheduler ..................... False
  use_contiguous_buffers_in_ddp ................... False
  use_cpu_initialization .......................... None
  use_one_sent_docs ............................... False
  virtual_pipeline_model_parallel_size ............ None
  vocab_extra_ids ................................. 0
  vocab_file ...................................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  weight_decay .................................... 0.01
  world_size ...................................... 128
-------------------- end of arguments ---------------------

GPU util: ~79%
GPU mem: 4564MiB - 4840MiB / 40537MiB

(over 100 iterations = 51200 samples)
0.87 ms per sample 

time (ms) | forward-compute: 116.02 | forward-recv: 12.54 | backward-compute: 107.98 | backward-send: 0.25 | backward-send-forward-recv: 2.29 | backward-params-all-reduce: 59.04 | backward-embedding-all-reduce: 32.05 | optimizer-copy-to-main-grad: 1.74 | optimizer-unscale-and-check-inf: 113.75 | optimizer-clip-main-grad: 2.37 | optimizer-copy-main-to-model-params: 1.52 | optimizer: 121.96 | batch-generator: 3.88


---

# Note

- Fixate on the version of both Megatron and DeepSpeed
- Different configuration for parallelization
- Better profiling tools (for better comparison over different config)


