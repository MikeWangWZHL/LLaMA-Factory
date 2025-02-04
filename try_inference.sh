# export CUDA_VISIBLE_DEVICES=2,3
export CUDA_VISIBLE_DEVICES=4
# llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml
# llamafactory-cli export examples/merge_lora/qwen2vl_lora_sft.yaml
# llamafactory-cli webchat examples/inference/qwen2_vl.yaml
# llamafactory-cli api examples/inference/qwen2_vl.yaml

# llamafactory-cli webchat examples/inference/qwen2_vl_lora_vidsitu_v3_40k_image.yaml
# llamafactory-cli api examples/inference/qwen2_vl_lora_vidsitu_v3_40k_image.yaml
llamafactory-cli api examples/inference/qwen2_vl_lora_vidsitu_v3_325k_image.yaml