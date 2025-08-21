

import habitat
import logging
import random
import json
import numpy as np
import argparse
import sys
import os
import torch

from omegaconf import OmegaConf
from PIL import Image

from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.config import read_write

from habitat_extensions import measures

DATASET = "rxr"
CONFIG_PATH = "./config/vln_r2r.yaml"
OUTPUT_PATH = f"./data/trajectory_data/{DATASET}"
DATA_PATH = None  # Set to None to use default dataset path

class StreamVLNHabitatRunner:
    def __init__(self, dataset: str, config_path: str, output_path: str, data_path: str = None):
        self.device = torch.device("cuda")
        self.dataset = dataset.lower()
        self.config_path = config_path
        self.output_path = output_path
        self.data_path = data_path

        self.config = get_habitat_config(self.config_path)

    def config_env(self, scene: str = None) -> habitat.Env:
        if self.data_path is not None:
            with read_write(self.config):
                self.config.habitat.dataset.update(
                    {
                        "data_path": self.data_path,
                    }
                )
        print(OmegaConf.to_yaml(self.config))
        return habitat.Env(config=self.config)

    def generate(self, rank: int = 0, world_size: int = 1) -> None:
        os.makedirs(os.path.join(self.output_path), exist_ok=True)
        env = self.config_env()

        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        annotations = []
        for scene_id in sorted(scene_episode_dict.keys()):
            scan = scene_id.split("/")[-2]
            episodes = scene_episode_dict[scene_id]
            print(f"scene_id: {scene_id}, scan: {scan}")

            for episode in episodes[rank::world_size]:
                env.current_episode = episode
                agent = ShortestPathFollower(
                    sim=env.sim, goal_radius=0.5, return_one_hot=False)

                instructions = episode.instruction.instruction_text
                trajectory_id = episode.trajectory_id
                scene_id = episode.scene_id.split('/')[-2]
                episode_id = int(episode.episode_id)
                ref_path = episode.reference_path

                observation = env.reset()

                # episode initialization
                rgb_list = []
                depth_list = []
                actions = [-1]
                next_waypoint_id = 1

                rgb_dir = os.path.join(
                    self.output_path, "images", f"{scene_id}_{self.dataset}_{episode_id:06d}", "rgb")
                os.makedirs(rgb_dir, exist_ok=True)
                while not env.episode_over:
                    rgb = observation["rgb"]
                    rgb_list.append(rgb)
                    Image.fromarray(rgb).convert("RGB").save(
                        os.path.join(rgb_dir, f"{len(rgb_list):03d}.jpg"))

                    next_action = agent.get_next_action(
                        ref_path[next_waypoint_id])

                    force_episode_over = False
                    while next_action == 0:
                        next_waypoint_id += 1
                        if next_waypoint_id == len(ref_path) - 1:
                            agent = ShortestPathFollower(
                                sim=env.sim, goal_radius=0.25, return_one_hot=False)
                        if next_waypoint_id >= len(ref_path):
                            force_episode_over = True
                            break
                        next_action = agent.get_next_action(
                            ref_path[next_waypoint_id])

                    if force_episode_over:
                        break

                    observation = env.step(next_action)
                    actions.append(next_action)

                if len(actions) > 498:
                    continue  # Skip episodes with too many actions

                assert len(actions) == len(
                    rgb_list), f"Actions length {len(actions)} does not match RGB frames length {len(rgb_list)}"
                annotations.append({
                    "id": episode_id,
                    "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                    "instructions": instructions if isinstance(instructions, list) else [instructions],
                    "actions": actions,
                })

                with open(os.path.join(self.output_path, "summary.json"), "a") as f:
                    result = {
                        "id": episode_id,
                        "video": os.path.join("images", f"{scene_id}_{self.dataset}_{episode_id:06d}"),
                        "instructions": instructions if isinstance(instructions, list) else [instructions],
                        "actions": actions,
                        "trajectory_id": trajectory_id,
                        "scene_id": scene_id,
                    }
                    f.write(json.dumps(result) + "\n")

            with open(os.path.join(self.output_path, f"annotations_{rank}.json"), "w") as f:
                json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--config_path", type=str, default=CONFIG_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    args = parser.parse_args()

    # world_size = os.environ['SLURM_NTASKS']
    # node_id = os.environ['SLURM_NODEID']
    # rank = os.environ['SLURM_PROCID']
    # local_rank = int(os.environ['SLURM_LOCALID'])
    # node_list = os.environ['SLURM_NODELIST']

    # print(
    #     f"rank: {rank}, world_size: {world_size}, node_id: {node_id}, local_rank: {local_rank}")

    # rank = int(rank)
    # world_size = int(world_size)
    # node_id = int(node_id)
    # local_rank = int(local_rank)

    runner = StreamVLNHabitatRunner(
        dataset=args.dataset,
        config_path=args.config_path,
        output_path=args.output_path,
        data_path=args.data_path
    )
    runner.run()
    # runner.generate(rank, world_size)
    # print(f"Trajectory generation completed. rank: {rank}, world_size: {world_size}")
