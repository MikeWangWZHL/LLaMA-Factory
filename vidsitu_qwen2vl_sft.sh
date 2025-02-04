export CUDA_VISIBLE_DEVICES=4,5,6,7
# llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
# llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml

# llamafactory-cli train examples/train_lora_qwen_vidsitu/qwen2vl_lora_sft_video.yaml
# llamafactory-cli export examples/merge_lora_qwen_vidsitu/qwen2vl_lora_sft_video.yaml

# llamafactory-cli train examples/train_lora_qwen_vidsitu/qwen2vl_lora_sft_image.yaml
# llamafactory-cli export examples/merge_lora_qwen_vidsitu/qwen2vl_lora_sft_image.yaml

llamafactory-cli train examples/train_lora_qwen_vidsitu_325k/qwen2vl_lora_sft_image.yaml
llamafactory-cli export examples/merge_lora_qwen_vidsitu_325k/qwen2vl_lora_sft_image.yaml
