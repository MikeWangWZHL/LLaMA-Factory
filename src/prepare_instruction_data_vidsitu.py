import json
import os
import sys
import re



PROMPT_TEACHER='''Describe what are the entities in the frames and their actions. Note that the video can contain multiple salient entities. And each entity can have more than one action. Be as detailed as possible. Think step by step and provide your final answer in a JSON format like this:
------
[
    {
        "subject": "person on the left", # who did the action
        "action": "fighting", # action name
        "object": "person on the right", # who did the action to"
    }
    ...
]
'''

def remove_parentheses(text):
    return re.sub(r'\s*\(.*?\)', '', text)

def get_teacher_prompt_and_response(data_json, output_path=None):

    ### expects data.json as input ###
    # Example: {"video_id": "_-4Xy6CjAbw7", "segment_id": "v__-4Xy6CjAbw_seg_25_35", "start_time": 0.0, "end_time": 4.0, "action_list": [{"subject": "man in wife beater", "actions": [{"subject": "man in wife beater (first fighter)", "action": "fight (fight)", "object": "man in tank top with SCC on the back (second fighter, if separate)", "attributes": {}, "start_time": 0.0, "end_time": 2.0}]}, {"subject": "the two inmates", "actions": [{"subject": "the two inmates (causer of continuation)", "action": "continue (aspectual)", "object": "to fight each other (thing continuing)", "attributes": {}, "start_time": 2.0, "end_time": 4.0}]}]}
    ###

    # load data
    with open(data_json) as f:
        data = json.load(f)

    # get response
    response_strs = []
    for item in data["action_list"]:
        # print(item)
        for action in item["actions"]:
            response_strs.append({
                "subject": remove_parentheses(action["subject"]),
                "action": remove_parentheses(action["action"]),
                "object": remove_parentheses(action["object"])
            })
    response_str = json.dumps(response_strs, indent=4)
    ret = {
        "user": PROMPT_TEACHER,
        "assistant": response_str
    }
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(ret, f, indent=4)
    # print(ret['user'])
    # print(ret['assistant'])
    return ret

from glob import glob
from tqdm import tqdm
def prepare_sharegpt_style_data_for_llama_factory_single_process(vidsitu_dataset_root):
    all_data_video_version = []
    all_data_image_version = []
    # get all subdirs
    # subdirs = sorted(glob(f"{vidsitu_dataset_root}/*"))
    subdirs = glob(f"{vidsitu_dataset_root}/*")
    print(len(subdirs))

    for subdir in tqdm(subdirs):
        try:
            prompt_json = os.path.join(subdir, "prompt.json")
            if not os.path.exists(prompt_json):
                continue
            prompts = json.load(open(prompt_json))
            video_path = os.path.join(subdir, "segment.mp4")
            image_dir = os.path.join(subdir, "frames_v1")
            frames = sorted(glob(f"{image_dir}/*"))
            # print(len(frames))
            data_video = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<video>{prompts['user']}"
                    },
                    {
                        "role": "assistant",
                        "content": prompts["assistant"]
                    }
                ],
                "videos": [
                    video_path
                ]
            }

            data_image = {
                "messages": [
                    {
                        "role": "user",
                        "content": "<image>"*len(frames) + f"{prompts['user']}"
                    },
                    {
                        "role": "assistant",
                        "content": prompts["assistant"]
                    }
                ],
                "images": frames
            }
        except Exception as e:
            data_video, data_image = None, None
            print(f"Error processing subdir {subdir}: {e}")
            continue

        if data_video is not None and data_image is not None:
            # print(data_image)
            all_data_video_version.append(data_video)
            all_data_image_version.append(data_image)
    print(len(all_data_video_version))
    print(len(all_data_image_version))
    # shuffle
    import random
    random.seed(0)
    random.shuffle(all_data_video_version)
    random.shuffle(all_data_image_version)

    # with open("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/vidsitu_v3_325k_video.json", "w") as f:
    #     json.dump(all_data_video_version, f)
    # with open("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/vidsitu_v3_325k_image.json", "w") as f:
    #     json.dump(all_data_image_version, f)
    with open("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/tmp_img.json", "w") as f:
        json.dump(all_data_video_version, f)
    with open("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/tmp_video.json", "w") as f:
        json.dump(all_data_image_version, f)



import os
import json
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_subdir(subdir):
    try:
        prompt_json = os.path.join(subdir, "prompt.json")
        prompts = json.load(open(prompt_json))
        video_path = os.path.join(subdir, "segment.mp4")
        image_dir = os.path.join(subdir, "frames_v1")
        frames = sorted(glob(f"{image_dir}/*"))
        # print(len(frames))
        data_video = {
            "messages": [
                {
                    "role": "user",
                    "content": f"<video>{prompts['user']}"
                },
                {
                    "role": "assistant",
                    "content": prompts["assistant"]
                }
            ],
            "videos": [
                video_path
            ]
        }

        data_image = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>"*len(frames) + f"{prompts['user']}"
                },
                {
                    "role": "assistant",
                    "content": prompts["assistant"]
                }
            ],
            "images": frames
        }
    except Exception as e:
        print(f"Error processing subdir {subdir}: {e}")
        return None, None
    return data_video, data_image

def prepare_sharegpt_style_data_for_llama_factory_multi_process(vidsitu_dataset_root):
    subdirs = glob(os.path.join(vidsitu_dataset_root, "*"))
    print(f"Found {len(subdirs)} subdirectories.")

    all_data_video_version = []
    all_data_image_version = []

    # Use multiprocessing to process subdirectories in parallel
    max_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subdir, subdir): subdir for subdir in subdirs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subdirs"):
            data_video, data_image = future.result()
            if data_video and data_image:
                all_data_video_version.append(data_video)
                all_data_image_version.append(data_image)

    print(f"Processed {len(all_data_video_version)} video entries and {len(all_data_image_version)} image entries.")

    # Shuffle data
    import random
    random.seed(0)
    random.shuffle(all_data_video_version)
    random.shuffle(all_data_image_version)

    # # Write data to files
    # with open("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/vidsitu_v3_325k_video.json", "w") as f:
    #     json.dump(all_data_video_version, f)
    # with open("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/vidsitu_v3_325k_image.json", "w") as f:
    #     json.dump(all_data_image_version, f)
    # Write data to files
    with open("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/tmp_img.json", "w") as f:
        json.dump(all_data_video_version, f)
    with open("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/data/tmp_video.json", "w") as f:
        json.dump(all_data_image_version, f)

    print("Data has been written to the output files.")

if __name__ == "__main__":
    # data_json = "/shared/nas/data/m1/shared-resource/vision-language/data/raw/VidSitu/VidSitu/data/vidsitu_dataset/train/_-4Xy6CjAbw7_v__-4Xy6CjAbw_seg_25_35_0.0_4.0/data.json"
    # output_path = "./tmp.json"
    # get_teacher_prompt_and_response(data_json, output_path)

    vidsitu_dataset_root = "/shared/nas/data/m1/shared-resource/vision-language/data/raw/VidSitu/VidSitu/data/vidsitu_dataset_v3/train"
    prepare_sharegpt_style_data_for_llama_factory_single_process(vidsitu_dataset_root)
    # prepare_sharegpt_style_data_for_llama_factory_multi_process(vidsitu_dataset_root)