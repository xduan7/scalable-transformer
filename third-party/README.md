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
git reset --hard b4cc77fe86585384f57a815ec606d749b5c0874c  
```

It's recommended that the big-science branch of deepspeed should be installed:
```bash
git clone https://github.com/microsoft/deepspeed PROJECT_DIR/third-party/deepspeed-big-science
cd PROJECT_DIR/third-party/deepspeed-big-science
git checkout big-science
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
