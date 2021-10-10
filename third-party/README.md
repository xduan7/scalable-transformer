# Third-party Repos

## Megatron-LM (megatron)

```bash
git submodule add https://github.com/NVIDIA/Megatron-LM.git PROJECT_DIR/third-party/megatron
cd PROJECT_DIR/third-party/megatron
git reset --hard 6a6809864a962001a98ab8baab6b031f8dc4a39f 
```

## Megatron-DeepSpeed (magatron-deepspeed)

Uses Megatron-LM v2.4

```bash
git submodule add git@github.com:bigscience-workshop/Megatron-DeepSpeed.git PROJECT_DIR/third-party/magatron-deepspeed    
cd PROJECT_DIR/third-party/magatron-deepspeed
git reset --hard 2f662e841c377f142c7da9c9fbcc2f666d612f78  
```

It's recommended that the big-science branch of deepspeed should be installed:
```bash
git submodule add https://github.com/microsoft/deepspeed PROJECT_DIR/third-party/deepspeed-big-science
cd PROJECT_DIR/third-party/deepspeed-big-science
git checkout big-science
git reset --hard c7f3bc51c27884ad80dcafe4aa60f070c1dfa26e
rm -rf build
TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check
```
Note that CUDA compatibility 8.0 is for A100 GPUs only.
However, it seems that any version of DeepSpeed <= 0.5.0 works.

There is an additional change to be made in the source code: comment out liens 445-448 from `megatron-deepspeed/megatron/training.py` and add `model[0].step()` right after.


##  DeepSpeedExamples (deepseed-examples)

Note that this repo uses relatively old version of Megatron-LM (v1.1.5), which does not have pipeline parallelization and some other features.

```bash
git submodule add git@github.com:microsoft/DeepSpeedExamples.git PROJECT_DIR/third-party/deepseed-examples
cd PROJECT_DIR/third-party/deepseed-examples
git reset --hard 25d73cf73fb3dc66faefa141b7319526555be9fc 
```

## GPT-NeoX (gpt-neox)

```bash
git submodule add https://github.com/EleutherAI/gpt-neox.git PROJECT_DIR/third-party/gpt-neox
cd PROJECT_DIR/third-party/gpt-neox
git reset --hard e1c82509f23d5316c75c441aa9f49c2142d40f3f
```

The singularity image was build using the following command:
```bash
sudo singularity build gpt-neox.simg docker://leogao2/gpt-neox:sha-157d29d
```

However, the image does not work out of the box. 
Make sure that there are only one version of PyTorch installed (1.8) and install fused kernel manually using the following command:
```bash
python PROJECT_DIR/third-party/gpt-neox/megatron/fused_kernels/setup.py install --user
```

Besides, a special edition of DeepSpeed should be installed from the third-party `deeperspeed` directory:
```bash
# pip install git+git://github.com/EleutherAI/DeeperSpeed.git@eb7f5cff36678625d23db8a8fe78b4a93e5d2c75#egg=deepspeed
git submodule add https://github.com/EleutherAI/DeeperSpeed.git PROJECT_DIR/third-party/deeperspeed
cd PROJECT_DIR/third-party/deeperspeed
git reset --hard eb7f5cff36678625d23db8a8fe78b4a93e5d2c75
cd PROJECT_DIR
pip install -e ./third-party/deeperspeed
```

Moreover, there are some changes to be made about this edition of DeepSpeed. 
In `deepspeed/launcher/runner.py`:
```python
# In function `parse_args`
# Comment out the following line and the return
# parser.add_argument('user_args', nargs=argparse.REMAINDER)

# Add the following lines
args, user_args = parser.parse_known_args(args=args)
args.user_args = user_args
return args
```
This would parse the arguments including hostfile correctly.
Another change that may or may not be necessary is that, the function `OpenMPIRunner` in `deepspeed/launcher/multinode_runner` requires `num_nodes` and `num_gpus` to be set to `-1`. 
Change the assertion statement to a warning and then change the values of `num_nodes` and `num_gpus` to `-1` might be a more friendly approach, that prevents any faulty argument parsing.
And finally, to make the configuration easier, add configuration file path as an argument in function `consume_neox_args` in `gpt-neox/neox_arguments` like this:
```python
if os.path.exists(args_parsed.megatron_config):
    with open(args_parsed.megatron_config, 'r') as _f:
        megatron_config = json.load(_f)
```


There are a few changes to be

Remaining issues
- no hostfile so only training with local resources
- incompatibility of NCCL and GPU"NCCL WARN Failed to open libibverbs.so"
- shuffle index length is not equal to sample index length 