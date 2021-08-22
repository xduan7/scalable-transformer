
---

### Configurations

### Performance


## 8*8 GPUs

---

### Configurations

PIPELINE_MP_SIZE=8
{
  "gradient_accumulation_steps": 128,
  "train_micro_batch_size_per_gpu": 4,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": false
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": false
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
OOM


--

### Configurations

PIPELINE_MP_SIZE=8
{
  "gradient_accumulation_steps": 128,
  "train_micro_batch_size_per_gpu": 4,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance

OOM


---

### Configurations

PIPELINE_MP_SIZE=8
{
  "gradient_accumulation_steps": 64,
  "train_micro_batch_size_per_gpu": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
OOM

---

### Configurations
PIPELINE_MP_SIZE=1
{
  "gradient_accumulation_steps": 64,
  "train_micro_batch_size_per_gpu": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}


### Performance
OOM


---

### Configurations
PIPELINE_MP_SIZE=1
{
  "gradient_accumulation_steps": 8,
  "train_micro_batch_size_per_gpu": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
OOM


---

### Configurations
PIPELINE_MP_SIZE=4
{
  "train_micro_batch_size_per_gpu": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
OOM


---

### Configurations

PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 8,
  "train_micro_batch_size_per_gpu": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
SamplesPerSec=0.234
Effective Tera Flops per GPU: 1.06 and total parameters 179.682 B


---

### Configurations

PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
SamplesPerSec=7.35
Effective Tera Flops per GPU: 1.06 and total parameters 179.682 B

---

### Configurations

PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 8,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
SamplesPerSec=7.40
Effective Tera Flops per GPU: 1.05 and total parameters 179.682 B

---

### Configurations
PIPELINE_MP_SIZE=32
{
  "gradient_accumulation_steps": 8,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
SamplesPerSec=4.91
Effective Tera Flops per GPU: 0.69 and total parameters 185.057 B

---

### Configurations
PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": false,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
SamplesPerSec=7.47
Effective Tera Flops per GPU: 1.05 and total parameters 179.682 B


---

### Configurations
PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
SamplesPerSec=7.54
Effective Tera Flops per GPU: 1.06 and total parameters 179.682 B


---

### Configurations
PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
OOM

---

### Configurations
PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
SamplesPerSec=7.34
Effective Tera Flops per GPU: 1.04 and total parameters 179.682 B


---

## 16*8 GPUs

---

### Configurations
PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
SamplesPerSec=16.6
Effective Tera Flops per GPU: 1.17 and total parameters 179.682 B

---

### Configurations
PIPELINE_MP_SIZE=8
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}


### Performance
OOM

---

### Configurations
PIPELINE_MP_SIZE=16
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 0,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
OOM

---

### Configurations
PIPELINE_MP_SIZE=32
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 0,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
OOM

---

### Configurations
PIPELINE_MP_SIZE=32
{
  "gradient_accumulation_steps": 32,
  "train_micro_batch_size_per_gpu": 32,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 1e-05,
      "warmup_max_lr": 1e-04,
      "warmup_num_steps": 1000
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 0,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  }
}

### Performance
OOM

---

### Configurations

### Performance


---

### Configurations

### Performance


---

### Configurations

### Performance


---

### Configurations

### Performance


---

### Configurations

### Performance


---