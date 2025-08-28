import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import quaternion
import time
import transformers
import numpy as np

from typing import Any, Dict
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array

from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import dict_to_cuda
from utils.dist import *
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN


class VLNEvaluator:
    def __init__(
        self,
        sim_sensors_config: Dict,
        model: Any = None,
        tokenizer: Any = None,
        args: argparse.Namespace = None,
    ):
        self.args = args
        self.device = torch.device('cuda:0')
        self.sim_sensors_config = sim_sensors_config
        self.intrinsic_matrix = self.sim_sensors_config["camera_intrinsic"] #(4,4)
        self.image_processor = model.get_vision_tower().image_processor
        self.model : StreamVLNForCausalLM = model 
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.tokenizer.add_tokens(["<memory>"], special_tokens=True)
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })


        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]

        self.use_memory_tokens = True
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history
        
        print(f"num_frames: {self.num_frames}, num_future_steps: {self.num_future_steps}, num_history: {self.num_history}")
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.output_ids = None
        self.past_key_values = None
        self.step_id = 0
        self.last_image = None

    def reset_memory(self):
        print("!!!!!!!!!!!!!!!!!!!!!!!reset memory!!!!!!!!!!!!!!!!!!!!!!!")
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.intrinsic_list = []
        self.time_ids = []
        self.action_seq = []
        self.output_ids = None
        self.past_key_values = None
        self.step_id = 0
        self.last_image = None
        self.model.reset_for_env(0)
    
    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.",add_system: bool = False):
        roles = {"human": "user", "gpt": "assistant"}        
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start, im_end = tokenizer.additional_special_tokens_ids
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids
        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # Apply prompt templates
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            # prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN

            prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
            print(f"promt {prompt}")
            
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            input_id, target = [], []

            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)

                conv = [{"role" : role, "content" : content}]
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id

            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX

            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids,  conversations # tensor(bs x seq_len)

    def step(self, idx, rgb, instruction_text='', run_model=False):
        # Step 0. Fake some unused observation
        # Currently the model does not use voxel pooling. So we only use the rgb
        camera_pose = np.eye(4)
        depth = np.zeros((rgb.shape[0], rgb.shape[1], 1))
        intrinsic = torch.from_numpy(self.intrinsic_matrix).float()
        
        # Step 1. Preprocess images
        if run_model:
            image = Image.fromarray(rgb).convert('RGB')
            image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
            self.last_image = copy.deepcopy(image)

        if not run_model:
            image = self.last_image
        
        self.time_ids.append(self.step_id)    
        self.rgb_list.append(image)
        self.depth_list.append(torch.from_numpy(depth).float())
        self.pose_list.append(torch.from_numpy(camera_pose)) 
        self.intrinsic_list.append(intrinsic)
        
        # Step 2. Reset all conversation history when needed
        if not run_model:
            if self.use_memory_tokens and (self.step_id + 1) % self.num_frames == 0:
                print(f'Reset model at Step {self.step_id+1}')
                self.model.reset_for_env(idx)
                self.output_ids = None
                self.past_key_values = None
                self.time_ids = []
            return None, 0, None

        # Step 3. Prepare input for model
        if self.output_ids is None:
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
            if self.step_id != 0:
                sources[0]["value"] += f' You have visited these areas {DEFAULT_MEMORY_TOKEN}.'
            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction_text)
            add_system = True
            print(self.step_id, sources[0]["value"])
        else:
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            add_system = False
        
        input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
        if self.output_ids is not None:
            input_ids = torch.cat([self.output_ids,input_ids.to(self.output_ids.device)], dim=1)

        images = self.rgb_list[-1:]
        depths = self.depth_list[-1:]
        poses = self.pose_list[-1:]
        intrinsics = self.intrinsic_list[-1:]
        if self.use_memory_tokens:
            if self.step_id != 0 and self.step_id % self.num_frames == 0: 
                if self.num_history is None:
                    history_ids = slice(0, self.time_ids[0], self.num_future_steps)
                else:
                    history_ids = slice(0, self.time_ids[0], (self.time_ids[0] // self.num_history))
                images = self.rgb_list[history_ids] + images   
                depths = self.depth_list[history_ids] + depths
                poses = self.pose_list[history_ids] + poses
                intrinsics = self.intrinsic_list[history_ids] + intrinsics 

        input_dict_raw = {'images':torch.stack(images).unsqueeze(0), 'depths':torch.stack(depths).unsqueeze(0), \
                    'poses':torch.stack(poses).unsqueeze(0), 'intrinsics':torch.stack(intrinsics).unsqueeze(0), 'inputs':input_ids, 'env_id':idx, 'time_ids':[self.time_ids]}
        input_dict = dict_to_cuda(input_dict_raw.copy(), self.device)
        
        for key, value in input_dict.items():
            if key in ['images', 'depths', 'poses', 'intrinsics']:
                input_dict[key] = input_dict[key].to(torch.bfloat16)
        
        # Step 4. Generate
        tt = time.time()
        outputs = self.model.generate(**input_dict, task_ids=[0], do_sample=False, num_beams=1, max_new_tokens=10000, use_cache=True, return_dict_in_generate=True, past_key_values=self.past_key_values)
        generate_time = time.time() - tt
        print(f"generate: {generate_time}")
        self.output_ids = outputs.sequences
        
        # Step 5. Proprocess the output
        llm_outputs = self.tokenizer.batch_decode(self.output_ids, skip_special_tokens=False)[0].strip()
        print(llm_outputs, flush=True)
        self.past_key_values = outputs.past_key_values
        action_seq = self.parse_actions(llm_outputs)

        if len(action_seq) == 0: ## if generated llm without Specific values
            action_seq = [0]
        
        return action_seq, generate_time, llm_outputs
    



if __name__ == "__main__":
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/pjlab/yq_ws/StreamVLN/checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for testing')
    
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path,
                                                        model_max_length=args.model_max_length,
                                                        padding_side="right")
    
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
                args.model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=False,
                )
    model.model.num_history = args.num_history
    model.reset(1)
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()
    
    
    vln_sensor_config = {
        "rgb_height" : 1.25, 
        "camera_intrinsic" : np.array([[192.        ,   0.        , 191.42857143,   0.        ],
            [  0.        , 192.        , 191.42857143,   0.        ],
            [  0.        ,   0.        ,   1.        ,   0.        ],
            [  0.        ,   0.        ,   0.        ,   1.        ]]),
    }
    
    evaluator = VLNEvaluator(
        vln_sensor_config,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )
    
    # only for test
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=True)
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=False)
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=False)
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=False)
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=True)
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=False)
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=False)
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=False)
    evaluator.step(0, np.zeros((480, 640, 3), dtype=np.uint8), "move forward 25 cm", run_model=True)
    