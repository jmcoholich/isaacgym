import torch
from isaacgym import gymapi as g


class SteppingStones:
    def __init__(self, task, cfg, num_envs, device):
        """Creates a bed of stepping stones to be shared among all envs. """

        if isinstance(cfg['density'], float):
            if not 0.0 <= cfg["density"] <= 1.0:
                raise ValueError
        elif isinstance(cfg['density'], list):
            if not (0.0 <= cfg["density"][0] <= 1.0
                    and 0.0 <= cfg["density"][1] <= 1.0):
                raise ValueError()
        else:
            raise ValueError()
        self.task = task
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self.stepping_stones_height = self.cfg["height"]
        self.start_x = -0.5
        self.y_padding = 1.0
        # self.include_start_ss = self.cfg.get("include_starting_ss", True)
        (self.stone_pos, self.num_stones, self.eff_ss_size, self.ss_num_x,
        self.ss_num_y, self.is_stone, self.corner_ss_pos) = self.init_stepping_stone_positions()

        self.pi = 3.14159265358979

    def idcs2idcs(self, idcs, trans):
        """Takes a list of ss idcs, then applies the translation, then
        returns a list of new ss idcs.

        idcs is shape (...,)
        trans is shape (..., 2)

        output is shape (...,)
        NOTE: All indices correspond to 100% density stones
        """

        # do x translation
        output = idcs.clone()
        output += trans[..., 0] * self.ss_num_y

        # do this to prevent y-shifts from wrapping around
        output_test = output + trans[..., 1]
        bad_y_mask = \
            (torch.div(output_test, self.ss_num_y, rounding_mode='floor')
            != torch.div(output, self.ss_num_y, rounding_mode='floor'))
        trans[..., 1][bad_y_mask] = 0.0
        output += trans[..., 1]
        return output, bad_y_mask

    def pos2idcs(self, pos):
        """Input is a 2d tensor of 2d positions, output is a tensor
        of stepping stone idcs. This only works when stone density is 1.0
        because the stone positions are in order.

        shape of pos: (..., 2)

        shape of output: (...,)
        NOTE: All indices correspond to 100% density stones
        """

        # subtract position from first stone position
        trans = torch.round((pos - self.corner_ss_pos) / self.eff_ss_size).to(torch.long)
        return self.idcs2idcs(torch.zeros(pos.shape[:-1], device=self.device,
                                          dtype=torch.long), trans)

    def init_stepping_stone_positions(self):
        env_rows = -int(self.num_envs  # ceiling division
                       // -self.cfg['num_rows'])

        eff_ss_size = self.cfg['stone_dim'] + self.cfg["spacing"]
        ss_num_x = int(self.cfg["distance"] // eff_ss_size + 1)
        ss_num_y = int((self.cfg['robot_y_spacing'] * env_rows + 2
                        * self.y_padding) // eff_ss_size + 1)
        ss_grid_size = ss_num_x * ss_num_y

        # randomly select if there will or will not be a stepping stone at
        # each grid location
        if isinstance(self.cfg["density"], float):
            is_stone = (torch.rand(ss_num_x, ss_num_y, device=self.device)
                        < self.cfg["density"])
        elif isinstance(self.cfg["density"], list):
            rng = torch.rand(ss_num_x, ss_num_y, device=self.device)
            temp = torch.linspace(self.cfg["density"][0],
                self.cfg["density"][1], int(ss_num_y // 2), device=self.device)
            if ss_num_y % 2 == 0:
                mask = torch.cat([temp.flip(0), temp])
            else:
                mask = torch.cat([temp.flip(0), temp[:1], temp])

            is_stone = rng < mask
        else:
            raise ValueError()
        num_stones = is_stone.count_nonzero()

        if isinstance(self.cfg['height_range'], float):
            heights = ((torch.rand(is_stone.count_nonzero(), 1, device=self.device) - 0.5)
                    * self.cfg['height_range']) + self.cfg['height']
        elif isinstance(self.cfg['height_range'], list):
            # I want to linearly increase the height variance from front to back
            heights = torch.rand(ss_num_x, ss_num_y, device=self.device) - 0.5
            heights *= torch.linspace(self.cfg['height_range'][0],
                self.cfg['height_range'][1], ss_num_x, device=self.device).unsqueeze(-1)
            heights += self.cfg['height']
            heights = heights[is_stone].unsqueeze(-1)
        stone_pos = is_stone.nonzero()
        stone_pos = torch.cat([stone_pos, heights], dim=1)
        stone_pos[:, 0] = stone_pos[:, 0] * eff_ss_size + self.start_x
        stone_pos[:, 1] = stone_pos[:, 1] * eff_ss_size - self.y_padding

        corner_ss_pos = torch.tensor([self.start_x, -self.y_padding],
                                     device=self.device)

        # if self.include_start_ss:
        #     start_foot_pos = torch.tensor(
        #         [[0.2134, 0.1494, 0.0285],  # FL
        #         [0.2135, -0.1493, 0.0285],  # FR
        #         [-0.2694, 0.1495, 0.0285],  # RL
        #         [-0.2693, -0.1492, 0.0285]],  # RR
        #         device=self.device)

        # convert is_stone into indices into the stone_pos tensor
        is_stone = is_stone.to(torch.long).flatten()
        # where there is no stone, I can't use the is_stone value as an index
        is_stone[is_stone == 0] = -1
        is_stone[is_stone == 1] = \
            torch.arange((is_stone == 1).count_nonzero(), device=self.device)

        return stone_pos, num_stones, eff_ss_size, ss_num_x, ss_num_y, \
            is_stone, corner_ss_pos

    def create(self, gym, sim, env):
        asset_options_stone = g.AssetOptions()
        asset_options_stone.fix_base_link = True
        pose = g.Transform()
        pose.p = g.Vec3(0.0, 0.0, self.stepping_stones_height/2.0)
        stone_dim = self.cfg["stone_dim"]

        # start_block = gym.create_box(sim, 1, 2, self.stepping_stones_height, asset_options_stone)
        stone = gym.create_box(sim, stone_dim, stone_dim, self.cfg['height'], asset_options_stone)
        # gym.create_actor(self.envs[i], start_block, pose, "start_block", i)
        # pose.p = g.Vec3(self.cfg["distance"] + 1, 0.0, self.cfg['height']/2.0)
        # gym.create_actor(self.envs[i], start_block, pose, "end_block", i)

        for j in range(self.stone_pos.shape[0]):
            pose.p = g.Vec3(self.stone_pos[j, 0],
                            self.stone_pos[j, 1],
                            self.stone_pos[j, 2] / 2.0)
            actor = gym.create_actor(env, stone, pose, '', -1)

            # link_props = gym.get_actor_rigid_shape_properties(env, actor)
            # link_props[0].friction = 0.0  # 1.0 is the default.
            # # link_props[0].rolling_friction = 1.0  # 0.0 is the default.
            # gym.set_actor_rigid_shape_properties(env, actor, link_props)

    def get_large_state(self, foot_center_pos):
        return self.ss_state_calcs(
            foot_center_pos,
            10,
            torch.tensor([0.025, 0.075, 0.15, 0.225], device=self.device))

    def get_huge_state(self, foot_center_pos):
        return self.ss_state_calcs(
            foot_center_pos,
            20,
            torch.tensor([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3],
                         device=self.device))

    def get_slim_state(self, foot_center_pos):
        return self.ss_state_calcs(
            foot_center_pos,
            10,
            torch.tensor([0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2],
                         device=self.device))

    def get_state(self, foot_center_pos):
        return self.ss_state_calcs(
            foot_center_pos,
            9,
            torch.tensor([0.025, 0.05, 0.1], device=self.device))

    def ss_state_calcs(self, foot_center_pos, num_dirs, dists, clipping=1.0):
        # pos_per_foot = 1 + num_dirs * dists.shape[0]
        # output = torch.zeros(self.num_envs, pos_per_foot * 4,
        #                      device=self.device)
        # return output
        """Output a heightmap of several points around each foot.
        Assumes the only objects other than the robot are the stepping stones.
        """
        # xy direction tensor
        dirs = torch.zeros(num_dirs, 2, device=self.device)
        dirs[...] = torch.linspace(0, 2 * self.pi, num_dirs + 1,
                                   device=self.device)[:-1].unsqueeze(-1)
        dirs[:, 0] = dirs[:, 0].cos()
        dirs[:, 1] = dirs[:, 1].sin()

        # create a tensor of absolute xy positions to check
        pos_per_foot = 1 + num_dirs * dists.shape[0]
        positions = torch.zeros(self.num_envs, 4, pos_per_foot, 3,
                                device=self.device)
        # convert to position relative to footsteps
        # temp = torch.arange(self.num_envs, device=self.device,
        #                     dtype=torch.float) - 1
        # temp[0] = 0
        # foot_center_pos[:, :, 1] += temp.view(self.num_envs, 1) * self.cfg["robot_spacing"]
        foot_center_pos[:, :, :2] += self.task.env_offsets.view(self.num_envs, 1, 2)
        positions[:, :, 0] = foot_center_pos[:, :, :]
        positions[:, :, 1:, 2] = foot_center_pos[:, :, 2].view(self.num_envs, 4, 1)

        positions[:, :, 1:, :2] = (dists.view(dists.shape[0], 1, 1) * dirs.view(
            1, num_dirs, 2)).view(1, 1, num_dirs * dists.shape[0], 2) \
            + positions[:, :, 0, :2].view(self.num_envs, 4, 1, 2)

        # import time
        # st = time.time()
        positions = positions.view(self.num_envs, 4 * pos_per_foot, 3)

        ss_grid_idcs, bad_mask = self.pos2idcs(positions[:, :, :2])
        # santize by removing values off of the grid (values too high will
        # throw an error, negative values will index in a way not intended)
        # the issue here is that every position maps to a grid index, no matter what
        ss_grid_size = self.ss_num_y * self.ss_num_x
        ss_grid_idcs_clean = ss_grid_idcs.clone().clamp(0, ss_grid_size - 1)
        output = foot_center_pos[:, :, 2].repeat_interleave(pos_per_foot, dim=-1)

        # create boolean index for "output" and positions that only selects
        # scan points above a stone
        # shape (self.num_envs, pos_per_foot * 4)
        has_stone = (self.is_stone > -1)[ss_grid_idcs_clean]
        has_stone = has_stone & (ss_grid_idcs > -1)
        has_stone = has_stone & (ss_grid_idcs < ss_grid_size)
        has_stone = has_stone & ~bad_mask
        if has_stone.any():
            # nested fancy indexing eventually gets stone positions with
            # scan points above them
            assert (ss_grid_idcs[has_stone] == ss_grid_idcs_clean[has_stone]).all()
            relevant_stone_pos = self.stone_pos[self.is_stone[ss_grid_idcs[has_stone]]]
            diff = (relevant_stone_pos[..., :2] - positions[has_stone][..., :2]).abs().max(dim=-1)[0]
            assert diff.max().item() <= self.eff_ss_size / 2.0 + 1e-4

            # Falsify "has_stone" for scan positions that lie in the cracks
            # between stones
            still_has_stones = diff < self.cfg['stone_dim'] / 2.0
            has_stone[has_stone.clone()] = still_has_stones
            if has_stone.any():
                output[has_stone] = positions[has_stone][..., 2] - relevant_stone_pos[still_has_stones][..., 2]

        # et = time.time()
        # print(f"new method takes {et-st} seconds")

        # vertify that I get the same as prev approach
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        '''
        st = time.time()
        positions = positions.view(self.num_envs, 4 * pos_per_foot, 1, 3)

        # check for stepping stones
        # get diffence in xyz positions between every stepping stone and every position
        stone_pos = self.stone_pos.view(1, 1, self.num_stones, 3)
        diff = positions - stone_pos

        # there should only be one true value per position
        stone_match_idcs = (diff[..., :2].abs() < self.cfg['stone_dim'] / 2.0).all(-1)
        # idcs has shape (self.num_envs, 4 * pos_per_foot, self.num_stones)

        # default is distance from foot to the floor
        other_output = foot_center_pos[:, :, 2].repeat_interleave(pos_per_foot, dim=-1)

        # if there is any stone beneath a position, get the diff
        other_output[stone_match_idcs.any(-1)] = diff[stone_match_idcs][:, 2]

        et = time.time()
        print(f"OLD method takes {et - st} seconds")
        print()
        if not torch.allclose(output, other_output):
            breakpoint()
        '''
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        # visualize the computations for all robots
        if False:
            output_to_use = output
            self.task.gym.clear_lines(self.task.viewer)
            vertices = torch.zeros(self.num_envs, 4, pos_per_foot, 2, 3,
                                   device=self.device)
            # positions = positions.view(self.num_envs, 4, pos_per_foot, 3)
            # breakpoint()
            # positions[..., :2] += foot_center_pos[..., :2].view(self.num_envs, 4, 1, 2)


            # set start positions
            vertices[:, :, :, 0] = positions.view(self.num_envs, 4, pos_per_foot, 3)

            # set end positions
            vertices[:, :, :, 1] = positions.view(self.num_envs, 4, pos_per_foot, 3)
            # add 0.01 to show a gap between bottom of line and stop of stepping stone
            vertices[:, :, :, 1, 2] -= output_to_use.view(self.num_envs, 4, pos_per_foot) - 0.01
            vertices = vertices.cpu()

            # one color per foot's observation (color is rgb)
            colors = torch.rand(self.num_envs, 4, 3).repeat_interleave(pos_per_foot, dim=1)
            # colors = colors.view(self.num_envs * 4 * pos_per_foot, 3)
            # vertices = vertices.view(self.num_envs * 4 * pos_per_foot * 2, 3).cpu()

            # for i, env in enumerate(self.task.envs):
            vertices = vertices.flatten(end_dim=-2)
            colors = colors.flatten(end_dim=-2)
            self.task.gym.add_lines(self.task.viewer, self.task.envs[0],
                            4 * pos_per_foot * self.num_envs, vertices.numpy(),
                            colors.numpy())
        return output.clamp(-1.0, 1.0)
