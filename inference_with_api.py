# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from openai import OpenAI
from transformers.utils.versions import require_version
import base64
import requests

import json
from glob import glob

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def get_messages(text, images):
    base64_images = [encode_image(image_path) for image_path in images]  # Encode all images
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": text},  # Single text input
        ],
    }
    
    # Add all images to the content
    for base64_image in base64_images:
        message["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        )
    return [message]  # Return the message as a list to match the API's expectations


PROMPT_INFERENCE='''You are given a sequence of frames from a video. 

First provide a detailed caption of the video. Then try to identify the main actions, subjects and objects in the video.

Think step by step and provide your answer in following format:

### Caption ###
Describe the video frames in detail. Focusing on the main events and actions.

### Actions ###
Provide a list of actions that are happening in the video. For each action, specify the subject, action and object if avaliable. Provide your answer in the following JSON format: 
[
    {"action": "", "subject": "", "object": ""}, 
    ...
]
'''
## UNC segmenter -> Qwen2-VL -> Caption; JSON

def get_vidsitu_instance(instance_dir):
    prompt_json = os.path.join(instance_dir, "prompt.json")
    if not os.path.exists(prompt_json):
        prompt = PROMPT_INFERENCE
    else:
        prompts = json.load(open(prompt_json))
        prompt = prompts['user']
    # video_path = os.path.join(instance_dir, "segment.mp4")
    image_dir = os.path.join(instance_dir, "frames_v1")
    frames = sorted(glob(f"{image_dir}/*"))
    return prompt, frames


def run_on_segmenter_results(client, segmenter_result_dirs):
    for segmented in segmenter_result_dirs:
        # glob all segment subdirs
        segs = sorted(glob(f"{segmented}/*"))
        for seg in segs:
            frames = glob(f"{seg}/images/*")
            frames = sorted(frames, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
            prompt = PROMPT_INFERENCE
            messages = get_messages(text = prompt, images = frames)
            result = client.chat.completions.create(messages=messages, model="test", temperature=0.2, max_tokens=1024, top_p=1.0)
            print("response:", result.choices[0].message.content)
            result_path = os.path.join(seg, "qwen_response.json")
            response = {
                "prompt": prompt,
                "frames": frames,
                "response": result.choices[0].message.content
            }
            with open(result_path, "w") as f:
                json.dump(response, f, indent=4)
            print(f"Saved response to {result_path}")

def main():
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
    )

    # example text prompt and video frames
    text_prompt = PROMPT_INFERENCE
    frames = sorted(glob("/shared/nas/data/m1/wangz3/CoPR_project/LLaMA-Factory/example_inputs/bridge2/bridge2_0/frames_v1/*"))
    print(text_prompt)
    print(frames, len(frames))

    # get messages
    messages = get_messages(text = text_prompt, images = frames)
    
    # send request (same args as openai api)
    result = client.chat.completions.create(messages=messages, model="test", temperature=0.2, max_tokens=1024, top_p=1.0)
    print("response:", result.choices[0].message.content)
    import pdb; pdb.set_trace()

def main_on_segmenter_results():
    client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
    )
    segmenter_result_dirs = [
        # "/shared/nas2/knguye71/ecole-new-system/data/test_data_2/bridge_1.mp4/bridge_1",
        # "/shared/nas2/knguye71/ecole-new-system/data/test_data_2/quiet_1.mp4/quiet_1",
        # "/shared/nas2/knguye71/ecole-new-system/data/test_data_2/v22.mp4/v22"
        "/shared/nas2/knguye71/ecole-new-system/data/test_data_3/test.mp4/test"
    ]
    run_on_segmenter_results(client, segmenter_result_dirs = segmenter_result_dirs)


if __name__ == "__main__":
    ## demo example
    # main()

    ## on segmenter results
    main_on_segmenter_results()

