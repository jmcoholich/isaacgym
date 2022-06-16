from cv2 import accumulateSquare
import torch
import sys
import os
import pickle

class StatsGatherer:
    def __init__(self, task, max_episodes):
        """This will aggregate statistics from specified number of episodes.
        This will be faster the more environments are run in paralell.
        Call sys.exit() after max_episodes is reached.
        """
        self.task = task
        self.num_envs = task.num_envs
        self.max_episodes = max_episodes
        assert self.max_episodes % self.num_envs == 0
        assert self.max_episodes >= self.num_envs
        self.device = task.device
        self.episode_counter = 0
        self.current_rew_sum = torch.zeros(self.num_envs, device=self.device)
        self.current_eps_len = torch.zeros(self.num_envs, device=self.device)
        self.base_pos = torch.zeros(self.num_envs, device=self.device)
        self.success = torch.zeros(self.num_envs, device=self.device)
        self.still_running = torch.ones(self.num_envs, device=self.device,
                                        dtype=torch.bool)
        self.batches = 0
        self.data = None
        # self.rew_stats = RunningMeanStd("Reward")
        # self.eps_len_stats = RunningMeanStd("Episode_Length")
        # self.dist_traveled_stats = RunningMeanStd("Distance_Traveled")
        # self.success_stats = RunningMeanStd("Successful")
        self.parts = [
            'base_position',
            'base_6d_orientation',
            'base_velocity',
            'base_angular_velocity',
            'joint_torques',
            'joint_positions',
            'joint_velocities',
            'foot_positions',
            'foot_velocities',
            'trajectory_generator_phase',
            'foot_contact_binary',
            'footstep_target_distance',
            'foot_contact_forces',
            'previous_action',
            'base_roll',
            'base_pitch',
            'base_yaw',
            'base_roll_velocity',
            'base_pitch_velocity',
            'base_yaw_velocity',
            'footstep_generator_current_foot_one_hot',
        ]

    def init_data_storage(self, rew_dict):
        data = {"still_running": torch.zeros(self.max_episodes, self.task.cfg["termination"]["timeout"][0] + 2, 1, device=self.device),
                "reward": torch.zeros(self.max_episodes, self.task.cfg["termination"]["timeout"][0] + 2, 1, device=self.device),
                "succcessful": torch.zeros(self.max_episodes, device=self.device)}
        if self.task.is_footsteps:
            data.update({
                "footstep_targets": self.task.footstep_generator.footsteps.clone() ,
                "current_footstep": self.task.footstep_generator.current_footstep.clone()})
        if self.task.is_stepping_stones:
            data.update({"stepping_stone_positions": self.task.stepping_stones.stone_pos.clone()})

        for part in self.parts:
            part_len = self.task.observe.handles[part][1]
            data[part] = torch.zeros(self.max_episodes, self.task.cfg["termination"]["timeout"][0] + 2, part_len, device=self.device)
        for rew in rew_dict.keys():
            data[rew] = torch.zeros(self.max_episodes, self.task.cfg["termination"]["timeout"][0] + 2, 1, device=self.device)
        return data

    def update(self, rew_dict):
        # assert (self.task.progress_buf == self.task.progress_buf[0]).all()
        if self.data is None:
            self.data = self.init_data_storage(rew_dict)

        start = self.batches * self.num_envs
        end = (self.batches + 1) * self.num_envs
        for part in self.parts:
            self.data[part][start: end][self.still_running, self.task.progress_buf[0]] = self.task.observe(recalculating_obs=True, parts=[part], add_noise=False)[self.still_running]
            # self.data[part][start: end, self.task.progress_buf[0]] = self.task.observe(recalculating_obs=True, parts=[part], add_noise=False)

        for key, value in rew_dict.items():
            self.data[key][start: end][self.still_running, self.task.progress_buf[0], 0] = value[self.still_running]

        self.data["still_running"][start: end][self.still_running,  self.task.progress_buf[0], 0] = True
        self.data["reward"][start: end][self.still_running,  self.task.progress_buf[0], 0] = self.task.rew_buf[self.still_running]

    def save_data(self):
        # time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
        # fname = 'data' + time_stamp + '.pkl'
        # path = os.path.join(data_dir, fname)
        data_dir = self.task.args.data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        path = os.path.join(data_dir, self.task.args.save_fname.replace("/", "") + ".pkl")
        # if not os.path.exists(data_dir):
        #     os.makedirs(data_dir)
        with open(path, 'wb') as f:
            print(f"Writing gathered data to: {path}")
            pickle.dump(self.data, f)
            print(f"done")

    def log_final_stuff(self, just_terminated):
        """Logs the values for runs that just terminated"""
        idcs = torch.zeros(self.max_episodes, dtype=torch.bool, device=self.device)
        start = self.batches * self.num_envs
        end = (self.batches + 1) * self.num_envs
        idcs[start: end] = just_terminated
        self.data["succcessful"][idcs] = self.task.is_successful[idcs].clone().float()
        if self.task.is_footsteps:
            self.data["footstep_targets"][idcs] = self.task.footstep_generator.footsteps[just_terminated].clone()
            self.data["current_footstep"][idcs] = self.task.footstep_generator.current_footstep[just_terminated].clone()


    def __call__(self, rew_dict):
        """
        This method is called at the end of post_physics_step().

        Whenever a True value reset_buf is encountered, record stats
        and increment counter. If counter reaches max_episodes, terminate.

        The issue is that doing this naively biases the stats towards bad
        performance. Bad episodes terminate quickly and will get reset quickly
        and get logged first.

        To avoid this bias, I wait until all environments are finished.
        When I only need n < num_envs sample left, I wait for the first n
        environments to finish instead of just taking the n environments
        that finish first.
        """

        # if self.task.progress_buf[0] != 0:
        #     self.current_rew_sum += self.task.rew_buf[0]

        # idcs = (self.task.progress_buf != 0) & self.still_running
        self.update(rew_dict)

        terminated = (self.task.progress_buf != 0) & self.task.reset_buf
        just_terminated = self.still_running & terminated
        self.log_final_stuff(just_terminated)
        # self.base_pos[just_terminated] = self.task.base_pos[just_terminated][:, 0]
        # self.success[just_terminated] = self.task.is_successful[just_terminated].float()

        self.still_running = self.still_running & ~just_terminated
        # self.task.reset_buf[:] = False  # TODO I remember that there was some issue with doing this

        # this flag means the number of envs that have terminated is greater
        # than or equal the number of remaining samples needed
        if not self.still_running.any():
            self.batches += 1
            # collect all stats and reset environment
            # self.rew_stats.batch_update(self.current_rew_sum)
            # self.dist_traveled_stats.batch_update(self.base_pos)
            # self.eps_len_stats.batch_update(self.current_eps_len)
            # self.success_stats.batch_update(self.success)

            self.episode_counter += self.num_envs

            # reset all envs and totals
            self.task.reset_buf[:] = True
            # self.current_rew_sum[:] = 0
            # self.current_eps_len[:] = 0
            self.still_running[:] = True

            # check for completion of stats gathering
            if self.episode_counter == self.max_episodes:  # if completed
                # print()
                # for stat in [self.rew_stats, self.eps_len_stats,
                #         self.dist_traveled_stats, self.success_stats]:
                #     print(stat)
                self.save_data()
                sys.exit()
        assert self.episode_counter < self.max_episodes

# class RunningMeanStd(object):
#     # For a new value newValue, compute the new count, new mean, the new M2.
#     # mean accumulates the mean of the entire dataset
#     # M2 aggregates the squared distance from the mean
#     # count aggregates the number of samples seen so far

#     def __init__(self, name):
#         self.data = torch.tensor([])
#         self.name = name

#     # def __str__(self):
#     #     mean, _, sample_var = self.finalize()
#     #     info = f"mean: {mean}\nstd: {sample_var**0.5}"
#     #     return self.name + '\n' + info + '\n'

#     def __str__(self):
#         output = self.name
#         for i in range(len(self.data)):
#             output += '\n' + str(self.data[i].float().item())
#         output += '\n'
#         return output

#     def batch_update(self, values):
#         assert values.dim() == 1
#         self.data = self.data.to(values.device)
#         self.data = torch.cat((self.data, values))

#     # Retrieve the mean, variance and sample variance from an aggregate
#     def finalize(self):
#         if len(self.data) < 2:
#             raise ValueError("There are one or less samples, cannot compute std")
#         else:
#             return self.data.mean(), self.data.var(unbiased=True), self.data.var()

# class RunningMeanStd(object):
#     # For a new value newValue, compute the new count, new mean, the new M2.
#     # mean accumulates the mean of the entire dataset
#     # M2 aggregates the squared distance from the mean
#     # count aggregates the number of samples seen so far

#     def __init__(self, name):
#         self.existingAggregate = (0, 0.0, 0.0)
#         self.name = name

#     def __str__(self):
#         mean, _, sample_var = self.finalize()
#         info = f"mean: {mean}\nstd: {sample_var**0.5}"
#         return self.name + '\n' + info + '\n'

#     def batch_update(self, values):  # TODO make this not stupid lol
#         # or just avoid this alltogether and store all the values (maybe add this as a unit test.)
#         assert values.dim() == 1
#         for i in range(len(values)):
#             self.update(values[i])


#     def update(self, newValue):
#         (count, mean, M2) = self.existingAggregate
#         count += 1
#         delta = newValue - mean
#         mean += delta / count
#         delta2 = newValue - mean
#         M2 += delta * delta2
#         self.existingAggregate = (count, mean, M2)

#     # Retrieve the mean, variance and sample variance from an aggregate
#     def finalize(self):
#         (count, mean, M2) = self.existingAggregate
#         if count < 2:
#             return float("nan")
#         else:
#             (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
#             return (mean, variance, sampleVariance)


