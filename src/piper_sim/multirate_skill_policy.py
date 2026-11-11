"""Separate visual memory and current joint feedback; cached vision at inference."""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from piper_sim.sensor_skill_learning import CausalSkillEncoder


class MultiRateSkillEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=192)
        vision_space = gym.spaces.Dict({"images": observation_space["visual_images"],
                                       "state": observation_space["visual_state"]})
        self.vision = CausalSkillEncoder(vision_space)
        self.robot = nn.Sequential(nn.Linear(37, 64), nn.ReLU())
        self._visual_key = self._visual_cache = None

    def forward(self, observations):
        cacheable = not self.training and observations["current_robot"].shape[0] == 1
        if self.training:
            self._visual_key = self._visual_cache = None
        key = tuple(observations["vision_token"][0].detach().cpu().tolist()) if cacheable else None
        if cacheable and key == self._visual_key and self._visual_cache is not None:
            vision = self._visual_cache
        else:
            vision = self.vision({"images": observations["visual_images"],
                                  "state": observations["visual_state"]})
            if cacheable:
                self._visual_key, self._visual_cache = key, vision.detach()
        return torch.cat([vision, self.robot(observations["current_robot"])], dim=-1)
