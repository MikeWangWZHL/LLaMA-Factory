from glob import glob
from tqdm import tqdm
import os
import json

REASONING_START= "<think>"
REASONING_END= "</think>"
ANSWER_START= "<answer>"
ANSWER_END= "</answer>"
PERCEPTION_TOKEN = "<percep>"
IMAGE_TOKEN = "<image>"

def get_full_reasoning_text(reasoning_steps, percep_token=PERCEPTION_TOKEN):
    reasoning_thoughts = []
    for step in reasoning_steps:
        reasoning_thoughts.append(step["thought"])
    reasoning_text = f"{percep_token} ".join(reasoning_thoughts)
    return reasoning_text

def prepare_sharegpt_style_data_for_llama_factory_single_process(
        cogcom_dataset_json, 
        output_path="tmp.json", 
        image_root="/shared/nas2/xuehangg/vqa/data/cogcom/images", 
        k = -1
    ):
    
    raw_data = json.load(open(cogcom_dataset_json))
    processed_data = []

    for item in raw_data[:k]:
        try:
            image_paths = []
            for image in item['images']:
                assert os.path.exists(os.path.join(image_root, image))
                image_paths.append(os.path.join(image_root, image))

            target_output = REASONING_START + \
                get_full_reasoning_text(item['reasoning_steps'], percep_token=PERCEPTION_TOKEN) + \
                REASONING_END + \
                f"{ANSWER_START}{item['generated_answer']}{ANSWER_END}"
        
            data_image = {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>"*len(item['images']) + f"{item['question']}"
                    },
                    {
                        "role": "assistant",
                        "content": target_output
                    }
                ],
                "images": image_paths
            }
            processed_data.append(data_image)
        except Exception as e:
            print(f"Error processing item: {item['id']}; Error: {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    prepare_sharegpt_style_data_for_llama_factory_single_process(
        cogcom_dataset_json="/shared/nas2/xuehangg/vqa/data/cogcom_rephrased_positive.json", 
        output_path="/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/cogcom_debug_100.json", 
        k = 100
    )