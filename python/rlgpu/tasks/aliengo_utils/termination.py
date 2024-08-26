import torch


class Terminiations:
    def __init__(self, task, cfg):
        self.task = task
        self.device = self.task.device
        self.cfg = cfg

        self.conditions = {
            "base_z": self.base_z,
            "euler_angles": self.euler_angles,
            "timeout": self.timeout,
            "no_footstep_in": self.no_footstep_in,
            "out_of_footsteps": self.out_of_footsteps,
            "end_of_blocks_terrain": self.end_of_blocks_terrain,
            "out_of_stepping_stones": self.out_of_stepping_stones,
        }

        # assert all(part in self.conditions.keys() for part in self.cfg.keys())
        for part in self.cfg.keys():
            if part not in self.conditions.keys():
                raise ValueError(f"Termination condition {part} not found in available conditions")

        if self.task.is_stepping_stones:
            if isinstance(self.task.cfg['stepping_stones']['height_range'], float):
                hrange = self.task.cfg['stepping_stones']['height_range']
            elif isinstance(self.task.cfg['stepping_stones']['height_range'], list):
                hrange = self.task.cfg['stepping_stones']['height_range'][1]
            self.reset_height_lb = self.task.cfg['stepping_stones']['height'] \
                - hrange / 2.0
            self.reset_height_ub = self.task.cfg['stepping_stones']['height'] \
                + hrange / 2.0 + 10.0
        elif self.task.is_rough_terrain_blocks:
            self.reset_height_lb = self.task.cfg['rough_terrain_blocks']['height'] \
                - self.task.cfg['rough_terrain_blocks']['max_height_range'] / 2.0
            self.reset_height_ub = self.task.cfg['rough_terrain_blocks']['height'] \
                + self.task.cfg['rough_terrain_blocks']['max_height_range'] / 2.0 + 10.0
        else:
            self.reset_height_lb = self.cfg["base_z"][0]
            self.reset_height_ub = self.cfg["base_z"][1]

    def __call__(self):
        """Iterate through each termination condition. OR together all the
        termination bools and AND together all the bootstrap bools.

        Each method evaluates the termination condition, outputs a boolean
        tensor indicating if each env has terminated due to that condition,
        and a boolean that says whether or not the returns should be
        bootstrapped if that condition has occurred.
        """

        # assert not self.task.reset_buf.any()
        self.task.bootstrap_buf[:] = True
        for cond in self.cfg.keys():
            is_term, bootstrap, failure = self.conditions[cond](*self.cfg[cond])
            # if is_term.any():
            #     print(f"{is_term.count_nonzero()}, {cond}")
            self.task.reset_buf[:] = self.task.reset_buf | is_term
            self.task.bootstrap_buf[is_term] = self.task.bootstrap_buf[is_term] & bootstrap
            self.task.is_successful[is_term] = self.task.is_successful[is_term] & (not failure)
            # if is_term[0]:
            #     print()
            #     print(f"Termination due to condition: {cond}")
            #     print()

        # assert self.task.bootstrap_buf[~self.task.reset_buf].all()

    def base_z(self, lb, ub):
        height_term = torch.logical_or(
            self.task.base_pos[:, 2] > self.reset_height_ub,
            self.task.base_pos[:, 2] < self.reset_height_lb)
        return height_term, False, True

    def euler_angles(self, x, y, z):
        attitude_term = (self.task.base_euler.abs()
                         > (torch.tensor([x, y, z], device=self.device)
                            * 3.14159265359)).any(axis=1)
        return attitude_term, False, True

    def timeout(self, max_eps_len):
        """This means failure for the purposes of gathering stats"""
        timeout = self.task.progress_buf > max_eps_len
        return timeout, True, True

    def no_footstep_in(self, max_time_no_foostep):
        return self.task.footstep_generator.no_footstep_in(max_time_no_foostep), False, True

    def out_of_footsteps(self, placeholder_arg):
        return self.task.footstep_generator.out_of_footsteps(), True, False

    def end_of_blocks_terrain(self, placeholder_arg):
        """Check X position against length of terrain."""
        return self.task.base_pos[:, 0] > self.task.terrain.farthest_x_block - 0.4, True, False

    def out_of_stepping_stones(self, placeholder_arg):
        """Check X position against length of terrain. Since this is for evaluation,
        also terminate based on distance if there are no stepping stones"""
        if not self.task.is_stepping_stones:
            return self.task.base_pos[:, 0] > 10.0 - 1.5, True, False
        return self.task.base_pos[:, 0] > self.task.cfg["stepping_stones"]['distance'] - 1.5, True, False
