import torch
from isaacgym import gymapi as g
from random import random


class RoughTerrainBlocks:
    def __init__(self, cfg, num_envs, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.y_margin = 2.0  # meters on either side of robots
        self.num_blocks = None

    # def init_terrain_pattern(self):
    #     """Every two meters the block size halves, spacing doubles, and
    #     height variation doubles.
    #     """

    #     width = self.cfg["robot_spacing"] * self.num_envs + 2 * self.y_margin
    #     num_levels = -int(self.cfg["distance"] // -2)  # ceil div
    #     first_level_block_size = self.cfg["min_block_size"] * 2 ** (num_levels - 1)
    #     first_level_spacing = self.cfg["max_block_spacing"] * 0.5 ** (num_levels - 1)
    #     first_level_num_blocks = -int(width // -(first_level_spacing + first_level_block_size))  # ceil div

    #     num_blocks = -first_level_num_blocks * (1 - 2**(num_levels))  # exponential sum formula, (denom is always 1)

    #     return num_blocks

    def create(self, gym, sim, env):
        """Loop through and create blocks in simuation. Heights and
        positions are not saved since they will not be used anyways.
        """
        exp = self.cfg["difficulty_growth_factor"]
        width = self.cfg["robot_spacing"] * self.num_envs + 2 * self.y_margin
        block_size = self.cfg["min_block_size"] * exp**(self.cfg["num_levels"] - 1)
        spacing = self.cfg["max_block_spacing"] * (1 / exp)**(self.cfg["num_levels"] - 1)
        height_variation = self.cfg["max_height_range"] * (1 / exp)**(self.cfg["num_levels"] - 1)

        asset_options = g.AssetOptions()
        asset_options.fix_base_link = True

        # create start block
        start_block_base_length = 2.0
        stone = gym.create_box(sim, start_block_base_length + self.cfg["level_length"], width, self.cfg["height"], asset_options)
        pose = g.Transform()
        pose.p = g.Vec3(start_block_base_length / 2.0, width / 2.0 - self.y_margin, self.cfg["height"] / 2.0)
        gym.create_actor(env, stone, pose, '', -1)

        self.num_blocks = 1
        pos = [start_block_base_length / 2.0 + self.cfg["level_length"],
               block_size / 2.0 - self.y_margin]

        for i in range(self.cfg["num_levels"]):
            length = 0.0
            stone = gym.create_box(sim, block_size, block_size, self.cfg["height"], asset_options)
            while length < self.cfg["level_length"]:
                dist = 0.0
                pos[1] = block_size / 2.0 - self.y_margin
                pos[0] += (block_size + spacing) / 2.0
                while dist < width:
                    rand_height = (random() - 0.5) * height_variation + self.cfg["height"]
                    pose.p = g.Vec3(pos[0], pos[1], rand_height / 2.0)
                    gym.create_actor(env, stone, pose, '', -1)
                    self.num_blocks += 1
                    dist += block_size + spacing
                    pos[1] += block_size + spacing
                pos[0] += (block_size + spacing) / 2.0
                length += block_size + spacing
            block_size *= 1 / exp
            spacing *= exp
            height_variation *= exp
        self.farthest_x_block = pos[0]



