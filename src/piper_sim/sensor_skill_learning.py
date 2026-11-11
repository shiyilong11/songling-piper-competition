"""Causal geometry encoder and model-free public-sensor exploration.

The collector is deliberately not an IK solver: actions are physically tried,
scored with public sensors and changed after observing their consequences.
Positive transitions are self-imitation warm-start data, not expert successes.
"""

import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from piper_sim.sensor_skill_env import mask_tool_error


class CausalSkillEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=3), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
            nn.Linear(48*3*4, 96), nn.ReLU(),
        )
        self.state_net = nn.Sequential(nn.Linear(37, 64), nn.ReLU())
        self.gru = nn.GRU(160, features_dim, batch_first=True)

    def forward(self, observations):
        images, states = observations["images"], observations["state"]
        b, t, c, h, w = images.shape
        image_features = self.cnn(images.reshape(b*t, c, h, w)).reshape(b, t, -1)
        features = torch.cat([image_features, self.state_net(states)], dim=-1)
        # Padded history contributes no visual/state embedding, although the GRU
        # still observes the number of elapsed valid frames through its input.
        features = features*states[..., -1:]
        return self.gru(features)[0][:, -1]


class PublicJointExplorer:
    """Bounded joint-direction trials; no simulator model or inverse kinematics."""

    def __init__(self, skill, seed=0, *, posture_coupling=True):
        self.skill, self.rng = skill, np.random.default_rng(seed)
        self.directions = self.rng.choice([-1., 1.], 3)
        self.axis, self.block = 0, 0
        self.block_progress = 0.
        self.phase, self.phase_steps = "approach", 0
        self.last_height = None
        self.up_direction = -1.
        self.last_target, self.last_target_time = None, 0.
        self.posture_coupling = posture_coupling

    def action(self, estimate):
        a = np.zeros(7, np.float32)
        self.phase_steps += 1
        if self.phase == "approach":
            if self.skill == "pick":
                if estimate.target is not None:
                    self.last_target, self.last_target_time = estimate.target, estimate.timestamp
                target = self.last_target if (estimate.timestamp-self.last_target_time <= 6.
                                               and estimate.target_mask.sum() >= 30) else None
                distance = None if target is None else np.linalg.norm(estimate.tcp-target)
                a[6] = 1.
            else:
                distance = None if estimate.goal is None else np.linalg.norm(estimate.tcp-estimate.goal-[0, 0, .065])
                a[6] = -1.
            if distance is not None and distance < .028 and (self.skill == "place" or
                    (mask_tool_error(estimate) is not None and mask_tool_error(estimate) < .10)):
                self.phase = "close" if self.skill == "pick" else "release"
                self.phase_steps = 0
            else:
                if self.block % 5 == 0:
                    if self.block_progress < -.0003:
                        self.directions[self.axis] *= -1.
                    self.axis = (self.axis+1) % 3
                    self.block_progress = 0.
                a[self.axis] = .85*self.directions[self.axis]
                self.block += 1
        if self.phase in ("close", "release"):
            a[6] = -1. if self.phase == "close" else 1.
            if self.phase_steps >= 24:
                self.phase, self.phase_steps = "retreat", 0
        elif self.phase == "retreat":
            a[6] = -1. if self.skill == "pick" else 1.
            if self.last_height is not None and estimate.tcp[2] < self.last_height-.0001:
                self.up_direction *= -1.
            a[1] = .65*self.up_direction if self.phase_steps < 36 else 0.
            self.last_height = float(estimate.tcp[2])
        if self.posture_coupling:
            # A fixed joint-space posture prior, not a Cartesian pose solver:
            # counter-rotate joint5 when joint2/3 are physically probed. The .75
            # ratio converts the existing .3/.4 rad/s command speed limits.
            # No object pose, Jacobian, optimizer or IK routine is consulted.
            a[4] = -.75*(a[1]+a[2])
        return a

    def observe(self, info):
        self.block_progress += info["progress_m"] + .05*info["image_progress"]


def behavior_clone(model, observations, actions, selected, *, epochs=8, seed=0):
    """Fit actual issued actions on positive public-sensor transitions only."""
    indices = np.flatnonzero(selected)
    if len(indices) < 16:
        return {"trained": False, "reason": "fewer_than_16_positive_transitions",
                "selected_transitions": len(indices), "epochs": 0}
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=3e-4)
    losses = []
    model.policy.set_training_mode(True)
    for _ in range(epochs):
        epoch = []
        for batch in np.array_split(rng.permutation(indices), max(1, int(np.ceil(len(indices)/32)))):
            inputs = {k: torch.as_tensor(v[batch], device=model.device, dtype=torch.float32)
                      for k, v in observations.items()}
            targets = torch.as_tensor(actions[batch], device=model.device)
            means = model.policy.get_distribution(inputs).distribution.mean
            loss = torch.nn.functional.smooth_l1_loss(means, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.)
            optimizer.step()
            epoch.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch)))
    with torch.no_grad():
        model.policy.log_std.fill_(-1.)
    return {"trained": True, "selected_transitions": len(indices), "epochs": epochs,
            "training_action_loss": losses,
            "demonstration_kind": "positive_transition_self_imitation_not_expert_success"}
