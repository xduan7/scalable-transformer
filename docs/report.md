# Number of samples trained per second

| model \ num_node * num_gpus                     | 1*1  | 1*8   | 4*8   | 8*8   |
|-------------------------------------------------|------|-------|-------|-------|
| megatron bert 345m                              | 44.7 | 137.9 | 339.0 | 694.4 |
| megatron bert 345m with deepspeed zero0         | 17.7 | 72.8  | 173.0 | 324.4 |
| megatron bert 345m with deepspeed zero3+offload | 3.14 | 18.3  | 46.9  | 85.9  |
| megatron bert 175m                              | OOM  | OOM   | OOM   | OOM   |
| megatron bert 175m with deepspeed zero3+offload | OOM  | OOM   | 0.25  | 0.87  |



# Number of samples trained per second for megatron bert 345m with deepspeed zero3+offload

