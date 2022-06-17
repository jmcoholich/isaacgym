from datetime import datetime
import os
import wandb

from tasks.aliengo_utils.observation import Observation
from socket import gethostname


def save_load_config_file_names(args):
    if args.play or args.test:  # testing trained policy
        # overwrite config file names from  args with saved ones
        # set num_envs to 1 if I am testing and have not otherwise
        # specified a number of environments to try this on
        if args.num_envs == 0:
            args.num_envs = 1
        path = "./nn/" + str(args.checkpoint) + '/'
        if args.ws == -1:
            with open(os.path.join(path, 'run_args.txt'), 'r') as f:
                cfg_train = f.readline().split()[1]
                cfg_env = f.readline().split()[1]
        else:
            import paramiko
            ws = args.ws
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ws_ip = ['143.215.128.18',
                     '143.215.131.33',
                     '143.215.131.34',
                     '143.215.128.16',
                     '143.215.131.25',
                     '143.215.131.23',
                     '130.207.124.148',  # skynet head node
                     '143.215.128.197',  # my personal lab desktop
                     ]
            print('\n\nOpening Remote SSH Client...\n\n')
            if ws != 7:
                if args.username is None:
                    args.username = 'jcoholich'
                ssh_client.connect(ws_ip[ws - 1], 22, args.username)
            else:
                if args.username is None:
                    args.username = 'jcoholich3'
                ssh_client.connect(ws_ip[ws - 1], 22, args.username)
            print('Connected!\n\n')
            sftp_client = ssh_client.open_sftp()
            path = os.path.join('isaacgym/python/rlgpu/nn', args.checkpoint, 'run_args.txt')
            with sftp_client.open(path, 'r') as f:
                cfg_train = f.readline().split()[1]
                cfg_env = f.readline().split()[1]
        args.cfg_train = cfg_train
        if args.cfg_env_override is None:
            args.cfg_env = cfg_env
        else:
            args.cfg_env = args.cfg_env_override

        print('#' * 100)
        print("Using cfg_train: '{}' \nand cfg_env: '{}'".format(cfg_train, cfg_env))
        print('#' * 100)
    else:  # training
        # make ID and folder for run
        random_id = datetime.now().strftime("%y%m%d%H%M%S%f")
        print('#' * 100)
        print("Run ID: {}".format(random_id))
        print('#' * 100)

        path = "./nn/" + random_id + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            print("ID already exists, press enter to continue and overwrite")
            input()

        # save config args to the runs folder
        with open(os.path.join(path, 'run_args.txt'), 'w') as f:
            f.write('cfg_train: ' + args.cfg_train + '\n')
            f.write('cfg_env: ' + args.cfg_env)
        args.network_path = path
        args.run_id = random_id
    wandb.init(config=args, project=args.wandb_project,
               tags=[str(detect_workstation_id())],
               settings=wandb.Settings(_disable_stats=True))
    return args


def detect_workstation_id():
    ws_names = {
        "ripl-w1": 1,
        "ripl-w2": 2,
        "ripl-w3": 3,
        "ripl-w4": 4,
        "ripl-w5": 5,
        "ripl-w6": 6,
        "ripl-d1": 8,
    }
    host_name = gethostname()
    if host_name not in ws_names.keys():
        return 7  # I am running in a docker container on skynet
    else:
        return ws_names[host_name]


def aliengo_params_helper(cfg, cfg_train, args):
    """Populate fields in the cfg files to avoid having to change things in
    multiple places."""

    cfg['env']['max_epochs'] = cfg_train['params']['config']['max_epochs']
    # I want to change the behavior of the rl_games player object if I am
    # in stats-gather mode
    if args.add_perturb != -1:
        cfg["env"]["perturbations"] = args.add_perturb

    if args.finetune_value is not None:
        args.checkpoint = args.finetune_value
        cfg_train["params"]["config"]["finetune_value"] = True
        cfg_train["params"]["load_checkpoint"] = True
        cfg_train["params"]["config"]["steps_num"] = 200
        cfg["env"]["numEnvs"] = 1000
        cfg_train["params"]["config"]["minibatch_size"] = 200_000
        # cfg_train["params"]["network"]["mlp"]["regularizer"] = "l2_regularizer"  # TODO this doesn't do anything at all
        cfg_train["params"]["config"]["mini_epochs"] = 20
        # cfg_train["params"]["config"]["clip_value"] = False  # TODO try this
        cfg_train["params"]["config"]["learning_rate"] = 1e-4  # TODO try this
        cfg_train["params"]["config"]["lr_schedule"] = "fixed"  # TODO try this

    if args.no_ss:
        assert not args.add_ss
        if "stepping_stones" in cfg["env"]:
            cfg["env"].pop("stepping_stones")

    if args.footstep_targets_in_place:
        assert "footstep_target_parameters" in cfg["env"]
        cfg["env"]["footstep_target_parameters"]["step_length"] = 0.0
        cfg["env"]["footstep_target_parameters"]["step_length_rand"] = 0.0
        cfg["env"]["footstep_target_parameters"]["footstep_rand"] = 0.0
        cfg["env"]["footstep_target_parameters"]["n_cycles"] = 1_000  # NOTE this is more than enough for a timeout of 10_000

    if args.add_ss:
        cfg["env"]["stepping_stones"] = {
            "distance": 10.0,
            "stone_dim": 0.1,
            "spacing": 0.001,
            "height": 2.0,
            "include_starting_ss": False,
            "robot_y_spacing": 1.0,
            "robot_x_spacing": 0.5,
            "num_rows": 1
        }
        if args.ss_infill == -1.0 or args.ss_height_var == -1.0:
            raise ValueError("Must specify an infill and height variation of"
                             " stepping stones.")
        else:
            cfg["env"]["stepping_stones"]["density"] = args.ss_infill
            cfg["env"]["stepping_stones"]["height_range"] = args.ss_height_var

    if args.tau != -1:
        cfg_train["params"]['config']['tau'] = args.tau

    if args.play and args.num_envs > 1:
        if 'player' in cfg_train['params']["config"].keys():
            cfg_train['params']["config"]['player']['print_stats'] = False
        else:
            cfg_train['params']["config"]['player'] = {}
            cfg_train['params']["config"]['player']['print_stats'] = False
        # cfg_train['params']["config"]['player']['determenistic'] = args.determenistic
        # if args.gather_stats == -1:
        #     args.gather_stats = args.num_envs

    if args.gamma != -1:
        cfg_train['params']['config']['gamma'] = args.gamma

    if args.p_gain is not None:
        cfg["env"]["joint_stiffness"] = args.p_gain

    if args.gather_stats != -1:
        assert args.gather_stats > 0
        assert args.play
    cfg["env"]["wandb_log_interval"] = cfg_train['params']['config']['wandb_log_interval']
    cfg["env"]["steps_num"] = cfg_train['params']['config']['steps_num']
    if args.timeout != -1:
        assert args.play
        assert args.timeout > 0
        cfg["env"]["termination"]["timeout"] = [args.timeout]

    if args.plot_values:
        assert args.play
        cfg_train['params']['config']['plot_values'] = True
        cfg_train['params']['config']['des_dir'] = args.des_dir
        cfg_train['params']['config']['des_dir_coef'] = args.des_dir_coef
        cfg_train['params']['config']['start_after'] = args.start_after
        cfg_train['params']['config']['file_prefix'] = args.file_prefix
        cfg_train['params']['config']['box_len'] = args.box_len
        cfg_train['params']['config']['grid_points'] = args.grid_points
        cfg_train['params']['config']['random_footsteps'] = args.random_footsteps
    else:
        cfg_train['params']['config']['plot_values'] = False

    # process hyperparam overrides
    if args.num_neurons is not None:
        cfg_train["params"]["network"]["mlp"]["units"] = [args.num_neurons, args.num_neurons]
    if args.lr is not None:
        cfg_train["params"]["config"]["learning_rate"] = args.lr
    if args.entropy_coef is not None:
        cfg_train["params"]["config"]["entropy_coef"] = args.entropy_coef
    if args.mini_epochs is not None:
        cfg_train["params"]["config"]["mini_epochs"] = args.mini_epochs

    use_vision = "vision" in cfg["env"]

    # calculate observation sizes
    observation_object = Observation(cfg["env"]["observation"], None, env_cfg=cfg["env"])
    non_vision_size = observation_object.compute_obs_size_proprioception()
    if "stepping_stone_state" in cfg["env"]["observation"]:
        assert 'stepping_stones' in cfg["env"], "stepping stone in observation, but no stepping stone params passed"
        # stepping_stone_state_size = (cfg["env"]["stepping_stones"]["distance"]
        #                              * cfg["env"]["stepping_stones"]["density"]
        #                              * 3)
        # non_vision_size += stepping_stone_state_size
    if use_vision:
        vision_size = cfg["env"]["vision"]["size"]**2
    else:
        vision_size = 0
    total_obs_size = non_vision_size + vision_size
    cfg["env"]["pre_stack_numObservations"] = total_obs_size
    try:
        total_obs_size *= cfg["env"]["prev_obs_stacking"] + 1
    except KeyError:
        cfg["env"]["prev_obs_stacking"] = 0
    # fill in observation sizes
    cfg["env"]["numObservations"] = total_obs_size
    if use_vision and "cnn" in cfg_train["params"]["network"]:
        cfg_train["params"]["network"]["cnn"]["img_height"] = cfg["env"]["vision"]["size"]
        cfg_train["params"]["network"]["cnn"]["img_width"] = cfg["env"]["vision"]["size"]
        cfg_train["params"]["network"]["cnn"]["proprioception_size"] = non_vision_size

    # add vision to observation list if not present
    if use_vision:
        if "vision" not in cfg["env"]["observation"]:
            cfg["env"]["observation"] += ["vision"]

    # easy setting of device via argparse args
    if args.pipeline == 'cpu':
        cfg['device_type'] = 'cpu'
        args.rl_device = 'cpu'
        args.sim_device = 'cpu'
        cfg_train['params']['config']['device'] = 'cpu'
        # args.device_id = 'cpu'
    else:
        cfg['device_id'] = args.device_id
        args.sim_device = 'cuda:' + str(args.device_id)
        args.rl_device = args.device_id
        cfg_train['params']['config']['device'] = 'cuda:' + str(args.device_id)
    args.graphics_device_id = args.device_id
    try:
        cfg_train['params']['config']['network_path'] = args.network_path
    except AttributeError:
        pass
    return cfg, cfg_train, args
