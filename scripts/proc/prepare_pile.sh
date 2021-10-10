
# Unpack

# unpack
unzstd 00.jsonl.zst -o  ~/data/pile/interim/train/00.jsonl

# process
python third-party/megatron/tools/preprocess_data.py \
       --input  /home/xduan7/data/scalable-transformer/interim/pile/train/00.jsonl \
       --output-prefix  ~/data/scalable-transformer/processed/pile \
       --vocab  ~/data//scalable-transformer/raw/enwiki_vocab/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ~/data//scalable-transformer/raw/enwiki_vocab/gpt2-merges.txt \
       --append-eod \
       --num_workers 256