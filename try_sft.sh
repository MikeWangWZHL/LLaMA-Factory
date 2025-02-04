export CUDA_VISIBLE_DEVICES=4,5,6,7
# llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
# llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml

llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml
