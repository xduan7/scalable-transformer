# Number of samples trained per second

| model \ num_node * num_gpus                     | 1*1  | 1*8   | 4*8   | 8*8   |
|-------------------------------------------------|------|-------|-------|-------|
| megatron bert 345m                              | 44.7 | 137.9 | 339.0 | 694.4 |
| megatron bert 345m with deepspeed zero0         | 17.7 | 72.8  | 173.0 | 324.4 |
| megatron bert 345m with deepspeed zero3+offload | 3.14 | 18.3  | 46.9  | 85.9  |
| megatron bert 175m                              | OOM  | OOM   | OOM   | OOM   |
| megatron bert 175m with deepspeed zero3+offload | OOM  | OOM   | 0.25  | 0.87  |



# Number of samples trained per second for megatron bert 345m with deepspeed zero3+offload

| model \ num_node * num_gpus                             |                  1*8                 |                  4*8                  |                   8*8                  |                  16*8                  |
|---------------------------------------------------------|:------------------------------------:|:-------------------------------------:|:--------------------------------------:|:--------------------------------------:|
| megatron bert 345m (mp=1)                               | 1658.93 samples/sec  2.94 tflops/gpu |  6558.31 samples/sec  2.91 tflops/gpu |  12442.73 samples/sec  2.88 tflops/gpu |  26125.74 samples/sec  2.89 tflops/gpu |
| megatron bert 345m (mp=2)                               |  895.04 samples/sec  1.60 tflops/gpu |  3390.06 samples/sec  1.58 tflops/gpu |   6411.00 samples/sec  1.50 tflops/gpu |  13292.23 samples/sec  1.50 tflops/gpu |
| megatron bert 345m (mp=4)                               |  463.92 samples/sec  0.82 tflops/gpu |  1681.96 samples/sec  0.77 tflops/gpu |   3348.21 samples/sec  0.77 tflops/gpu |   6072.34 samples/sec  0.76 tflops/gpu |
| megatron bert 345m (mp=8)                               |  226.05 samples/sec  0.40 tflops/gpu |   878.17 samples/sec  0.39 tflops/gpu |   1740.99 samples/sec  0.41 tflops/gpu |   3518.14 samples/sec  0.42 tflops/gpu |
| megatron bert 345m (mp=1) with deepspeed zero3          |  349.86 samples/sec  0.72 tflops/gpu |  1001.94 samples/sec  0.52 tflops/gpu |   1450.88 samples/sec  0.39 tflops/gpu |   1943.14 samples/sec  0.26 tflops/gpu |
| megatron bert 345m (mp=2) with deepspeed zero3          |  188.90 samples/sec  0.39 tflops/gpu |   373.29 samples/sec  0.22 tflops/gpu |    415.75 samples/sec  0.14 tflops/gpu |    444.41 samples/sec  0.05 tflops/gpu |
| megatron bert 345m (mp=4) with deepspeed zero3          |   92.49 samples/sec  0.18 tflops/gpu |    148.74 samples/sec  0.1 tflops/gpu |    315.46 samples/sec  0.09 tflops/gpu |    245.39 samples/sec  0.03 tflops/gpu |
| megatron bert 345m (mp=8) with deepspeed zero3          |   36.10 samples/sec  0.07 tflops/gpu |   164.29 samples/sec  0.06 tflops/gpu |    311.47 samples/sec  0.06 tflops/gpu |    561.54 samples/sec  0.07 tflops/gpu |
| megatron bert 345m (mp=1) with deepspeed zero3+offload  |  318.70 samples/sec  0.59 tflops/gpu |               samples/sec  tflops/gpu |                samples/sec  tflops/gpu |                samples/sec  tflops/gpu |
| megatron bert 345m (mp=2) with deepspeed zero3+offload  |  174.48 samples/sec  0.32 tflops/gpu |               samples/sec  tflops/gpu |                samples/sec  tflops/gpu |                samples/sec  tflops/gpu |
| megatron bert 345m (mp=4) with deepspeed zero3+offload  |   92.27 samples/sec  0.17 tflops/gpu |               samples/sec  tflops/gpu |                samples/sec  tflops/gpu |                samples/sec  tflops/gpu |
| megatron bert 345m (mp=8) with deepspeed zero3+offload  |   47.63 samples/sec  0.09 tflops/gpu |               samples/sec  tflops/gpu |                samples/sec  tflops/gpu |                samples/sec  tflops/gpu |



# 

| megatron bert 175b model \ num_node * num_gpus                                                                          |                                                         4*8 |                                                         8*8 |                                                                           16*8 |
|-------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------:|------------------------------------------------------------:|-------------------------------------------------------------------------------:|
| gradient_accumulation_steps=1<br>train_micro_batch_size_per_gpu=1<br>model_parallel_size=(8, 16, 32)<br>deepspeed zero0 |                                                         OOM |                                                         OOM |                                                                            OOM |
| gradient_accumulation_steps=1<br>train_micro_batch_size_per_gpu=1<br>model_parallel_size=32<br>deepspeed zero3          |                                                         OOM |                                                         OOM | 0.61 samples/sec<br> 1.41 tflops/gpu<br>34.6/40.5GB/gpu<br>140/400W/gpu<br>79% |
| gradient_accumulation_steps=1<br>train_micro_batch_size_per_gpu=1<br>model_parallel_size=16<br>deepspeed zero0          |                                                         OOM |                                                         OOM |  0.74 samples/sec<br>1.65 tflops/gpu<br>34.7/40.5GB/gpu<br>125/400W/gpu<br>90% |
| gradient_accumulation_steps=1<br>train_micro_batch_size_per_gpu=1<br>model_parallel_size=8<br>deepspeed zero0           |                                                         OOM |                                                         OOM |                                                                            OOM |
|                                                                                                                         |                                                         OOM |                                                         OOM |  5.97 samples/sec<br>1.67 tflops/gpu<br>34.7/40.5GB/gpu<br>136/400W/gpu<br>88% |
|                                                                                                                         |                                                             |                                                             |                                                                                |
|                                                                                                                         |                                                             |                                                             |                                                                                |
|                                                                                                                         |                                                             |                                                             |                                                                                |
|                                                                                                                         |                                                             |                                                             |                                                                                |
|                                                                                                                         |                                                             |                                                             |                                                                                |
|                                                                                                                         |                                                             |                                                             |                                                                                |
|                                                                                                                         |                                                             |                                                             |                                                                                |
|                                                                                                                         |                                                             |                                                             |                                                                                |
|                                                                                                                         | samples/sec<br> tflops/gpu<br>/40.5GB/gpu<br>/400W/gpu<br>% | samples/sec<br> tflops/gpu<br>/40.5GB/gpu<br>/400W/gpu<br>% |                    samples/sec<br> tflops/gpu<br>/40.5GB/gpu<br>/400W/gpu<br>% |