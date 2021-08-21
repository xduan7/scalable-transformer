# Performance for Megatron with DeepSpeed


## Single GPU (stage 0)

-------------------- arguments --------------------
  adam_beta1 ...................... 0.9
  adam_beta2 ...................... 0.999
  adam_eps ........................ 1e-08
  adlr_autoresume ................. False
  adlr_autoresume_interval ........ 1000
  apply_query_key_layer_scaling ... False
  apply_residual_connection_post_layernorm  False
  attention_dropout ............... 0.1
  attention_softmax_in_fp32 ....... False
  batch_size ...................... 4
  bert_load ....................... None
  bias_dropout_fusion ............. False
  bias_gelu_fusion ................ False
  block_data_path ................. None
  checkpoint_activations .......... True
  checkpoint_in_cpu ............... False
  checkpoint_num_layers ........... 1
  clip_grad ....................... 1.0
  contigious_checkpointing ........ False
  cpu_optimizer ................... False
  cpu_torch_adam .................. False
  data_impl ....................... infer
  data_path ....................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence
  DDP_impl ........................ local
  deepscale ....................... False
  deepscale_config ................ None
  deepspeed ....................... True
  deepspeed_activation_checkpointing  False
  deepspeed_config ................ /gpfs/mira-home/xduan7/projects/scalable-transformer/scripts/config/ds_zero0_offload_minimal.json
  deepspeed_mpi ................... False
  distribute_checkpointed_activations  False
  distributed_backend ............. nccl
  dynamic_loss_scale .............. True
  eod_mask_loss ................... False
  eval_interval ................... 100
  eval_iters ...................... 10
  exit_interval ................... None
  faiss_use_gpu ................... False
  finetune ........................ False
  fp16 ............................ True
  fp16_lm_cross_entropy ........... False
  fp32_allreduce .................. False
  hidden_dropout .................. 0.1
  hidden_size ..................... 1024
  hysteresis ...................... 2
  ict_head_size ................... None
  ict_load ........................ None
  indexer_batch_size .............. 128
  indexer_log_interval ............ 1000
  init_method_std ................. 0.02
  layernorm_epsilon ............... 1e-05
  lazy_mpu_init ................... None
  load ............................ None
  local_rank ...................... None
  log_interval .................... 10
  loss_scale ...................... None
  loss_scale_window ............... 1000
  lr .............................. 0.0001
  lr_decay_iters .................. 990000
  lr_decay_style .................. linear
  make_vocab_size_divisible_by .... 128
  mask_prob ....................... 0.15
  max_position_embeddings ......... 512
  memory_centric_tiled_linear ..... False
  merge_file ...................... None
  min_lr .......................... 1e-05
  min_scale ....................... 1
  mmap_warmup ..................... False
  model_parallel_size ............. 1
  no_load_optim ................... False
  no_load_rng ..................... False
  no_save_optim ................... False
  no_save_rng ..................... False
  num_attention_heads ............. 16
  num_layers ...................... 24
  num_unique_layers ............... None
  num_workers ..................... 2
  onnx_safe ....................... None
  openai_gelu ..................... False
  override_lr_scheduler ........... False
  param_sharing_style ............. grouped
  params_dtype .................... torch.float16
  partition_activations ........... False
  profile_backward ................ False
  query_in_block_prob ............. 0.1
  rank ............................ 0
  remote_device ................... none
  report_topk_accuracies .......... []
  reset_attention_mask ............ False
  reset_position_ids .............. False
  save ............................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................... 500
  scaled_masked_softmax_fusion .... False
  scaled_upper_triang_masked_softmax_fusion  False
  scattered_embeddings ............ False
  seed ............................ 1234
  seq_length ...................... 512
  short_seq_prob .................. 0.1
  split ........................... 949,50,1
  split_transformers .............. False
  synchronize_each_layer .......... False
  tensorboard_dir ................. None
  tile_factor ..................... 1
  titles_data_path ................ None
  tokenizer_type .................. BertWordPieceLowerCase
  train_iters ..................... 2000000
  use_checkpoint_lr_scheduler ..... False
  use_cpu_initialization .......... False
  use_one_sent_docs ............... False
  use_pin_memory .................. False
  vocab_file ...................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  warmup .......................... 0.01
  weight_decay .................... 0.01
  world_size ...................... 1
  zero_allgather_bucket_size ...... 0.0
  zero_contigious_gradients ....... False
  zero_reduce_bucket_size ......... 0.0
  zero_reduce_scatter ............. False
  zero_stage ...................... 1.0
---------------- end of arguments ----------------

GPU Util ~50%
GPU mem: 8654MiB / 40537Mi

56.43 ms per sample 

time (ms) | forward: 24.33 | backward: 71.13 | backward-backward: 71.11 | backward-allreduce: 0.00 | optimizer: 67.64 | batch generator: 1.04
Effective Tera Flops per GPU: 2.14 and total parameters 0.335 B


---

## Single-node (8 GPUs) (stage 0)

-------------------- arguments --------------------
  adam_beta1 ...................... 0.9
  adam_beta2 ...................... 0.999
  adam_eps ........................ 1e-08
  adlr_autoresume ................. False
  adlr_autoresume_interval ........ 1000
  apply_query_key_layer_scaling ... False
  apply_residual_connection_post_layernorm  False
  attention_dropout ............... 0.1
  attention_softmax_in_fp32 ....... False
  batch_size ...................... 4
  bert_load ....................... None
  bias_dropout_fusion ............. False
  bias_gelu_fusion ................ False
  block_data_path ................. None
  checkpoint_activations .......... True
  checkpoint_in_cpu ............... False
  checkpoint_num_layers ........... 1
  clip_grad ....................... 1.0
  contigious_checkpointing ........ False
  cpu_optimizer ................... False
  cpu_torch_adam .................. False
  data_impl ....................... infer
  data_path ....................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence
  DDP_impl ........................ local
  deepscale ....................... False
  deepscale_config ................ None
  deepspeed ....................... True
  deepspeed_activation_checkpointing  False
  deepspeed_config ................ /gpfs/mira-home/xduan7/projects/scalable-transformer/scripts/config/ds_zero0_offload_minimal.json
  deepspeed_mpi ................... False
  distribute_checkpointed_activations  False
  distributed_backend ............. nccl
  dynamic_loss_scale .............. True
  eod_mask_loss ................... False
  eval_interval ................... 100
  eval_iters ...................... 10
  exit_interval ................... None
  faiss_use_gpu ................... False
  finetune ........................ False
  fp16 ............................ True
  fp16_lm_cross_entropy ........... False
  fp32_allreduce .................. False
  hidden_dropout .................. 0.1
  hidden_size ..................... 1024
  hysteresis ...................... 2
  ict_head_size ................... None
  ict_load ........................ None
  indexer_batch_size .............. 128
  indexer_log_interval ............ 1000
  init_method_std ................. 0.02
  layernorm_epsilon ............... 1e-05
  lazy_mpu_init ................... None
  load ............................ None
  local_rank ...................... 0
  log_interval .................... 10
  loss_scale ...................... None
  loss_scale_window ............... 1000
  lr .............................. 0.0001
  lr_decay_iters .................. 990000
  lr_decay_style .................. linear
  make_vocab_size_divisible_by .... 128
  mask_prob ....................... 0.15
  max_position_embeddings ......... 512
  memory_centric_tiled_linear ..... False
  merge_file ...................... None
  min_lr .......................... 1e-05
  min_scale ....................... 1
  mmap_warmup ..................... False
  model_parallel_size ............. 2
  no_load_optim ................... False
  no_load_rng ..................... False
  no_save_optim ................... False
  no_save_rng ..................... False
  num_attention_heads ............. 16
  num_layers ...................... 24
  num_unique_layers ............... None
  num_workers ..................... 2
  onnx_safe ....................... None
  openai_gelu ..................... False
  override_lr_scheduler ........... False
  param_sharing_style ............. grouped
  params_dtype .................... torch.float16
  partition_activations ........... False
  profile_backward ................ False
  query_in_block_prob ............. 0.1
  rank ............................ 0
  remote_device ................... none
  report_topk_accuracies .......... []
  reset_attention_mask ............ False
  reset_position_ids .............. False
  save ............................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................... 500
  scaled_masked_softmax_fusion .... False
  scaled_upper_triang_masked_softmax_fusion  False
  scattered_embeddings ............ False
  seed ............................ 1234
  seq_length ...................... 512
  short_seq_prob .................. 0.1
  split ........................... 949,50,1
  split_transformers .............. False
  synchronize_each_layer .......... False
  tensorboard_dir ................. None
  tile_factor ..................... 1
  titles_data_path ................ None
  tokenizer_type .................. BertWordPieceLowerCase
  train_iters ..................... 2000000
  use_checkpoint_lr_scheduler ..... False
  use_cpu_initialization .......... False
  use_one_sent_docs ............... False
  use_pin_memory .................. False
  vocab_file ...................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  warmup .......................... 0.01
  weight_decay .................... 0.01
  world_size ...................... 8
  zero_allgather_bucket_size ...... 0.0
  zero_contigious_gradients ....... False
  zero_reduce_bucket_size ......... 0.0
  zero_reduce_scatter ............. False
  zero_stage ...................... 1.0
---------------- end of arguments ----------------

GPU Util ~74%
GPU mem: 6256MiB - 6544MiB / 40537Mi

13.51 ms per sample 

time (ms) | forward: 25.01 | backward: 60.29 | backward-backward: 60.27 | backward-allreduce: 0.00 | optimizer: 66.85 | batch generator: 1.01
Effective Tera Flops per GPU: 1.11 and total parameters 0.338 B


--- 

## Multi-node (4*8 GPUs) (stage 0)

-------------------- arguments --------------------
  adam_beta1 ...................... 0.9
  adam_beta2 ...................... 0.999
  adam_eps ........................ 1e-08
  adlr_autoresume ................. False
  adlr_autoresume_interval ........ 1000
  apply_query_key_layer_scaling ... False
  apply_residual_connection_post_layernorm  False
  attention_dropout ............... 0.1
  attention_softmax_in_fp32 ....... False
  batch_size ...................... 4
  bert_load ....................... None
  bias_dropout_fusion ............. False
  bias_gelu_fusion ................ False
  block_data_path ................. None
  checkpoint_activations .......... True
  checkpoint_in_cpu ............... False
  checkpoint_num_layers ........... 1
  clip_grad ....................... 1.0
  contigious_checkpointing ........ False
  cpu_optimizer ................... False
  cpu_torch_adam .................. False
  data_impl ....................... infer
  data_path ....................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence
  DDP_impl ........................ local
  deepscale ....................... False
  deepscale_config ................ None
  deepspeed ....................... True
  deepspeed_activation_checkpointing  False
  deepspeed_config ................ /gpfs/mira-home/xduan7/projects/scalable-transformer/scripts/config/ds_zero0_offload_minimal.json
  deepspeed_mpi ................... False
  distribute_checkpointed_activations  False
  distributed_backend ............. nccl
  dynamic_loss_scale .............. True
  eod_mask_loss ................... False
  eval_interval ................... 100
  eval_iters ...................... 10
  exit_interval ................... None
  faiss_use_gpu ................... False
  finetune ........................ False
  fp16 ............................ True
  fp16_lm_cross_entropy ........... False
  fp32_allreduce .................. False
  hidden_dropout .................. 0.1
  hidden_size ..................... 1024
  hysteresis ...................... 2
  ict_head_size ................... None
  ict_load ........................ None
  indexer_batch_size .............. 128
  indexer_log_interval ............ 1000
  init_method_std ................. 0.02
  layernorm_epsilon ............... 1e-05
  lazy_mpu_init ................... None
  load ............................ None
  local_rank ...................... 0
  log_interval .................... 10
  loss_scale ...................... None
  loss_scale_window ............... 1000
  lr .............................. 0.0001
  lr_decay_iters .................. 990000
  lr_decay_style .................. linear
  make_vocab_size_divisible_by .... 128
  mask_prob ....................... 0.15
  max_position_embeddings ......... 512
  memory_centric_tiled_linear ..... False
  merge_file ...................... None
  min_lr .......................... 1e-05
  min_scale ....................... 1
  mmap_warmup ..................... False
  model_parallel_size ............. 2
  no_load_optim ................... False
  no_load_rng ..................... False
  no_save_optim ................... False
  no_save_rng ..................... False
  num_attention_heads ............. 16
  num_layers ...................... 24
  num_unique_layers ............... None
  num_workers ..................... 2
  onnx_safe ....................... None
  openai_gelu ..................... False
  override_lr_scheduler ........... False
  param_sharing_style ............. grouped
  params_dtype .................... torch.float16
  partition_activations ........... False
  profile_backward ................ False
  query_in_block_prob ............. 0.1
  rank ............................ 0
  remote_device ................... none
  report_topk_accuracies .......... []
  reset_attention_mask ............ False
  reset_position_ids .............. False
  save ............................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................... 500
  scaled_masked_softmax_fusion .... False
  scaled_upper_triang_masked_softmax_fusion  False
  scattered_embeddings ............ False
  seed ............................ 1234
  seq_length ...................... 512
  short_seq_prob .................. 0.1
  split ........................... 949,50,1
  split_transformers .............. False
  synchronize_each_layer .......... False
  tensorboard_dir ................. None
  tile_factor ..................... 1
  titles_data_path ................ None
  tokenizer_type .................. BertWordPieceLowerCase
  train_iters ..................... 2000000
  use_checkpoint_lr_scheduler ..... False
  use_cpu_initialization .......... False
  use_one_sent_docs ............... False
  use_pin_memory .................. False
  vocab_file ...................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  warmup .......................... 0.01
  weight_decay .................... 0.01
  world_size ...................... 32
  zero_allgather_bucket_size ...... 0.0
  zero_contigious_gradients ....... False
  zero_reduce_bucket_size ......... 0.0
  zero_reduce_scatter ............. False
  zero_stage ...................... 1.0
---------------- end of arguments ----------------

GPU Util ~86%
GPU mem: 6040MiB - 6072MiB / 40537MiB

5.24 ms per sample 

time (ms) | forward: 24.89 | backward: 166.67 | backward-backward: 166.64 | backward-allreduce: 0.00 | optimizer: 66.19 | batch generator: 0.95
Effective Tera Flops per GPU: 0.77 and total parameters 0.338 B


---

## Single GPU (stage 3 & offload)

-------------------- arguments --------------------
  adam_beta1 ...................... 0.9
  adam_beta2 ...................... 0.999
  adam_eps ........................ 1e-08
  adlr_autoresume ................. False
  adlr_autoresume_interval ........ 1000
  apply_query_key_layer_scaling ... False
  apply_residual_connection_post_layernorm  False
  attention_dropout ............... 0.1
  attention_softmax_in_fp32 ....... False
  batch_size ...................... 4
  bert_load ....................... None
  bias_dropout_fusion ............. False
  bias_gelu_fusion ................ False
  block_data_path ................. None
  checkpoint_activations .......... True
  checkpoint_in_cpu ............... False
  checkpoint_num_layers ........... 1
  clip_grad ....................... 1.0
  contigious_checkpointing ........ False
  cpu_optimizer ................... False
  cpu_torch_adam .................. False
  data_impl ....................... infer
  data_path ....................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence
  DDP_impl ........................ local
  deepscale ....................... False
  deepscale_config ................ None
  deepspeed ....................... True
  deepspeed_activation_checkpointing  False
  deepspeed_config ................ /gpfs/mira-home/xduan7/projects/scalable-transformer/scripts/config/ds_zero3_offload_minimal.json
  deepspeed_mpi ................... False
  distribute_checkpointed_activations  False
  distributed_backend ............. nccl
  dynamic_loss_scale .............. True
  eod_mask_loss ................... False
  eval_interval ................... 100
  eval_iters ...................... 10
  exit_interval ................... None
  faiss_use_gpu ................... False
  finetune ........................ False
  fp16 ............................ True
  fp16_lm_cross_entropy ........... False
  fp32_allreduce .................. False
  hidden_dropout .................. 0.1
  hidden_size ..................... 1024
  hysteresis ...................... 2
  ict_head_size ................... None
  ict_load ........................ None
  indexer_batch_size .............. 128
  indexer_log_interval ............ 1000
  init_method_std ................. 0.02
  layernorm_epsilon ............... 1e-05
  lazy_mpu_init ................... None
  load ............................ None
  local_rank ...................... None
  log_interval .................... 10
  loss_scale ...................... None
  loss_scale_window ............... 1000
  lr .............................. 0.0001
  lr_decay_iters .................. 990000
  lr_decay_style .................. linear
  make_vocab_size_divisible_by .... 128
  mask_prob ....................... 0.15
  max_position_embeddings ......... 512
  memory_centric_tiled_linear ..... False
  merge_file ...................... None
  min_lr .......................... 1e-05
  min_scale ....................... 1
  mmap_warmup ..................... False
  model_parallel_size ............. 1
  no_load_optim ................... False
  no_load_rng ..................... False
  no_save_optim ................... False
  no_save_rng ..................... False
  num_attention_heads ............. 16
  num_layers ...................... 24
  num_unique_layers ............... None
  num_workers ..................... 2
  onnx_safe ....................... None
  openai_gelu ..................... False
  override_lr_scheduler ........... False
  param_sharing_style ............. grouped
  params_dtype .................... torch.float16
  partition_activations ........... False
  profile_backward ................ False
  query_in_block_prob ............. 0.1
  rank ............................ 0
  remote_device ................... none
  report_topk_accuracies .......... []
  reset_attention_mask ............ False
  reset_position_ids .............. False
  save ............................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................... 500
  scaled_masked_softmax_fusion .... False
  scaled_upper_triang_masked_softmax_fusion  False
  scattered_embeddings ............ False
  seed ............................ 1234
  seq_length ...................... 512
  short_seq_prob .................. 0.1
  split ........................... 949,50,1
  split_transformers .............. False
  synchronize_each_layer .......... False
  tensorboard_dir ................. None
  tile_factor ..................... 1
  titles_data_path ................ None
  tokenizer_type .................. BertWordPieceLowerCase
  train_iters ..................... 2000000
  use_checkpoint_lr_scheduler ..... False
  use_cpu_initialization .......... False
  use_one_sent_docs ............... False
  use_pin_memory .................. False
  vocab_file ...................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  warmup .......................... 0.01
  weight_decay .................... 0.01
  world_size ...................... 1
  zero_allgather_bucket_size ...... 0.0
  zero_contigious_gradients ....... False
  zero_reduce_bucket_size ......... 0.0
  zero_reduce_scatter ............. False
  zero_stage ...................... 1.0
---------------- end of arguments ----------------

GPU Util ~20%
GPU mem: 5898MiB / 40537MiB

467.18 ms per sample 

time (ms) | forward: 180.54 | backward: 626.08 | backward-backward: 626.06 | backward-allreduce: 0.00 | optimizer: 1356.72 | batch generator: 0.96
Effective Tera Flops per GPU: 0.24 and total parameters 0.335 B


---

## Single-node (8 GPUs) (stage 3 & offload)

-------------------- arguments --------------------
  adam_beta1 ...................... 0.9
  adam_beta2 ...................... 0.999
  adam_eps ........................ 1e-08
  adlr_autoresume ................. False
  adlr_autoresume_interval ........ 1000
  apply_query_key_layer_scaling ... False
  apply_residual_connection_post_layernorm  False
  attention_dropout ............... 0.1
  attention_softmax_in_fp32 ....... False
  batch_size ...................... 4
  bert_load ....................... None
  bias_dropout_fusion ............. False
  bias_gelu_fusion ................ False
  block_data_path ................. None
  checkpoint_activations .......... True
  checkpoint_in_cpu ............... False
  checkpoint_num_layers ........... 1
  clip_grad ....................... 1.0
  contigious_checkpointing ........ False
  cpu_optimizer ................... False
  cpu_torch_adam .................. False
  data_impl ....................... infer
  data_path ....................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence
  DDP_impl ........................ local
  deepscale ....................... False
  deepscale_config ................ None
  deepspeed ....................... True
  deepspeed_activation_checkpointing  False
  deepspeed_config ................ /gpfs/mira-home/xduan7/projects/scalable-transformer/scripts/config/ds_zero3_offload_minimal.json
  deepspeed_mpi ................... False
  distribute_checkpointed_activations  False
  distributed_backend ............. nccl
  dynamic_loss_scale .............. True
  eod_mask_loss ................... False
  eval_interval ................... 100
  eval_iters ...................... 10
  exit_interval ................... None
  faiss_use_gpu ................... False
  finetune ........................ False
  fp16 ............................ True
  fp16_lm_cross_entropy ........... False
  fp32_allreduce .................. False
  hidden_dropout .................. 0.1
  hidden_size ..................... 1024
  hysteresis ...................... 2
  ict_head_size ................... None
  ict_load ........................ None
  indexer_batch_size .............. 128
  indexer_log_interval ............ 1000
  init_method_std ................. 0.02
  layernorm_epsilon ............... 1e-05
  lazy_mpu_init ................... None
  load ............................ None
  local_rank ...................... 0
  log_interval .................... 10
  loss_scale ...................... None
  loss_scale_window ............... 1000
  lr .............................. 0.0001
  lr_decay_iters .................. 990000
  lr_decay_style .................. linear
  make_vocab_size_divisible_by .... 128
  mask_prob ....................... 0.15
  max_position_embeddings ......... 512
  memory_centric_tiled_linear ..... False
  merge_file ...................... None
  min_lr .......................... 1e-05
  min_scale ....................... 1
  mmap_warmup ..................... False
  model_parallel_size ............. 2
  no_load_optim ................... False
  no_load_rng ..................... False
  no_save_optim ................... False
  no_save_rng ..................... False
  num_attention_heads ............. 16
  num_layers ...................... 24
  num_unique_layers ............... None
  num_workers ..................... 2
  onnx_safe ....................... None
  openai_gelu ..................... False
  override_lr_scheduler ........... False
  param_sharing_style ............. grouped
  params_dtype .................... torch.float16
  partition_activations ........... False
  profile_backward ................ False
  query_in_block_prob ............. 0.1
  rank ............................ 0
  remote_device ................... none
  report_topk_accuracies .......... []
  reset_attention_mask ............ False
  reset_position_ids .............. False
  save ............................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................... 500
  scaled_masked_softmax_fusion .... False
  scaled_upper_triang_masked_softmax_fusion  False
  scattered_embeddings ............ False
  seed ............................ 1234
  seq_length ...................... 512
  short_seq_prob .................. 0.1
  split ........................... 949,50,1
  split_transformers .............. False
  synchronize_each_layer .......... False
  tensorboard_dir ................. None
  tile_factor ..................... 1
  titles_data_path ................ None
  tokenizer_type .................. BertWordPieceLowerCase
  train_iters ..................... 2000000
  use_checkpoint_lr_scheduler ..... False
  use_cpu_initialization .......... False
  use_one_sent_docs ............... False
  use_pin_memory .................. False
  vocab_file ...................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  warmup .......................... 0.01
  weight_decay .................... 0.01
  world_size ...................... 8
  zero_allgather_bucket_size ...... 0.0
  zero_contigious_gradients ....... False
  zero_reduce_bucket_size ......... 0.0
  zero_reduce_scatter ............. False
  zero_stage ...................... 1.0
---------------- end of arguments ----------------

GPU Util ~40%
GPU mem: 5564MiB - 5904MiB / 40537Mi

67.50 ms per sample 

time (ms) | forward: 210.53 | backward: 563.43 | backward-backward: 563.41 | backward-allreduce: 0.00 | optimizer: 307.41 | batch generator: 1.25
Effective Tera Flops per GPU: 0.23 and total parameters 0.338 B


--- 

## Multi-node (4*8 GPUs) (stage 3 & offload)

-------------------- arguments --------------------
  adam_beta1 ...................... 0.9
  adam_beta2 ...................... 0.999
  adam_eps ........................ 1e-08
  adlr_autoresume ................. False
  adlr_autoresume_interval ........ 1000
  apply_query_key_layer_scaling ... False
  apply_residual_connection_post_layernorm  False
  attention_dropout ............... 0.1
  attention_softmax_in_fp32 ....... False
  batch_size ...................... 4
  bert_load ....................... None
  bias_dropout_fusion ............. False
  bias_gelu_fusion ................ False
  block_data_path ................. None
  checkpoint_activations .......... True
  checkpoint_in_cpu ............... False
  checkpoint_num_layers ........... 1
  clip_grad ....................... 1.0
  contigious_checkpointing ........ False
  cpu_optimizer ................... False
  cpu_torch_adam .................. False
  data_impl ....................... infer
  data_path ....................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/processed/enwiki_text/bert_cased/text_sentence
  DDP_impl ........................ local
  deepscale ....................... False
  deepscale_config ................ None
  deepspeed ....................... True
  deepspeed_activation_checkpointing  False
  deepspeed_config ................ /gpfs/mira-home/xduan7/projects/scalable-transformer/scripts/config/ds_zero3_offload_minimal.json
  deepspeed_mpi ................... False
  distribute_checkpointed_activations  False
  distributed_backend ............. nccl
  dynamic_loss_scale .............. True
  eod_mask_loss ................... False
  eval_interval ................... 100
  eval_iters ...................... 10
  exit_interval ................... None
  faiss_use_gpu ................... False
  finetune ........................ False
  fp16 ............................ True
  fp16_lm_cross_entropy ........... False
  fp32_allreduce .................. False
  hidden_dropout .................. 0.1
  hidden_size ..................... 1024
  hysteresis ...................... 2
  ict_head_size ................... None
  ict_load ........................ None
  indexer_batch_size .............. 128
  indexer_log_interval ............ 1000
  init_method_std ................. 0.02
  layernorm_epsilon ............... 1e-05
  lazy_mpu_init ................... None
  load ............................ None
  local_rank ...................... 0
  log_interval .................... 10
  loss_scale ...................... None
  loss_scale_window ............... 1000
  lr .............................. 0.0001
  lr_decay_iters .................. 990000
  lr_decay_style .................. linear
  make_vocab_size_divisible_by .... 128
  mask_prob ....................... 0.15
  max_position_embeddings ......... 512
  memory_centric_tiled_linear ..... False
  merge_file ...................... None
  min_lr .......................... 1e-05
  min_scale ....................... 1
  mmap_warmup ..................... False
  model_parallel_size ............. 2
  no_load_optim ................... False
  no_load_rng ..................... False
  no_save_optim ................... False
  no_save_rng ..................... False
  num_attention_heads ............. 16
  num_layers ...................... 24
  num_unique_layers ............... None
  num_workers ..................... 2
  onnx_safe ....................... None
  openai_gelu ..................... False
  override_lr_scheduler ........... False
  param_sharing_style ............. grouped
  params_dtype .................... torch.float16
  partition_activations ........... False
  profile_backward ................ False
  query_in_block_prob ............. 0.1
  rank ............................ 0
  remote_device ................... none
  report_topk_accuracies .......... []
  reset_attention_mask ............ False
  reset_position_ids .............. False
  save ............................ /gpfs/mira-home/xduan7/projects/scalable-transformer/checkpoints/scalable-transformer/bert_cased_345m
  save_interval ................... 500
  scaled_masked_softmax_fusion .... False
  scaled_upper_triang_masked_softmax_fusion  False
  scattered_embeddings ............ False
  seed ............................ 1234
  seq_length ...................... 512
  short_seq_prob .................. 0.1
  split ........................... 949,50,1
  split_transformers .............. False
  synchronize_each_layer .......... False
  tensorboard_dir ................. None
  tile_factor ..................... 1
  titles_data_path ................ None
  tokenizer_type .................. BertWordPieceLowerCase
  train_iters ..................... 2000000
  use_checkpoint_lr_scheduler ..... False
  use_cpu_initialization .......... False
  use_one_sent_docs ............... False
  use_pin_memory .................. False
  vocab_file ...................... /gpfs/mira-home/xduan7/projects/scalable-transformer/data/raw/enwiki_vocab/bert-large-cased-vocab.txt
  warmup .......................... 0.01
  weight_decay .................... 0.01
  world_size ...................... 32
  zero_allgather_bucket_size ...... 0.0
  zero_contigious_gradients ....... False
  zero_reduce_bucket_size ......... 0.0
  zero_reduce_scatter ............. False
  zero_stage ...................... 1.0
---------------- end of arguments ----------------

GPU Util ~35%
GPU mem: 5282MiB - 5314MiB / 40537MiB

21.4 ms per sample 

time (ms) | forward: 247.32 | backward: 792.31 | backward-backward: 792.28 | backward-allreduce: 0.00 | optimizer: 308.46 | batch generator: 0.97
Effective Tera Flops per GPU: 0.19 and total parameters 0.338 B

