from __future__ import print_function, division, absolute_import, unicode_literals
import os
import importlib.util
import time
import pickle
import shutil
import pdb
import torch
import gym
import logging
import numpy as np
import random
import sys
import argparse
import errno
import h5py
from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.utils.rnn as rnn
from tensorboardX import SummaryWriter

sys.path.insert(0, '.')
from pretrain import customargparse
from pretrain.BaseModel import BaseModel
from pretrain.SupervisedModel_option_new_vae import SupervisedModel
from pretrain.ppo_trainer_new_vae import PPOModel
from pretrain.sac_trainer_new_vae import SACModel
from pretrain.misc_utils import log_record_dict, create_directory
from pretrain.utils import convert_to_POMDP, convert_to_POMDP_np
from fetch_mapping import fetch_mapping
from rl.envs import make_vec_envs
from rl import utils

from karel_env.dsl import get_DSL_option_v2
from karel_env import karel_option as karel
from karel_env.generator_option import KarelStateGenerator


class ProgramDataset(Dataset):
    """Karel programs dataset."""

    def __init__(self, program_list, config, num_program_tokens, num_agent_actions, device):
        """ Init function for karel program dataset

        Parameters:
            :param program_list (list): list containing information about each program in dataset
            :param config (dict): all configs in dict format
            :param num_program_tokens (int): number of program tokens in karel DSL
            :param num_agent_actions (int): number of actions karel agent can take
            :param device(torch.device): dataset target device: torch.device('cpu') or torch.device('cuda:X')

        Returns: None
        """
        self.device = device
        self.config = config
        self.programs = program_list
        # need this +1 as DEF token is input to decoder, loss will be calculated only from run token
        self.max_program_len = config['dsl']['max_program_len'] + 1
        self.num_program_tokens = num_program_tokens
        self.num_agent_actions = num_agent_actions

    def _dsl_to_prl(self, program_seq):
        """ DSL tokens to PRL tokens mapping.
        PRL tokens refer to a shorter list of karel program tokens, which can be specified through mapping_karel2prl.txt

        Parameters:
            :param program_seq (list): program as a sequence of integers

        Returns: list
            :return: new program with PRL token mapping
        """
        def func(x):
            return self.config['prl_tokens'].index(self.config['dsl2prl_mapping'][self.config['dsl_tokens'][x]])
        return np.array(list(map(func, program_seq)), program_seq.dtype)

    def __len__(self):
        return len(self.programs)

    def __getitem__(self, idx):
        program_id, sample, exec_data = self.programs[idx]
        sample = self._dsl_to_prl(sample) if self.config['use_simplified_dsl'] else sample

        sample = torch.from_numpy(sample).to(self.device).to(torch.long)
        program_len = sample.shape[0]
        sample_filler = torch.tensor((self.max_program_len - program_len) * [self.num_program_tokens - 1],
                                     device=self.device, dtype=torch.long)
        sample = torch.cat((sample, sample_filler))

        mask = torch.zeros((self.max_program_len, 1), device=self.device, dtype=torch.bool)
        mask[:program_len] = 1

        # load exec data
        s_h_list = []
        s_h, partial_s_h, s_h_len, a_h, a_h_len = exec_data
        s_h = torch.tensor(s_h, device=self.device, dtype=torch.float32)
        partial_s_h = torch.tensor(partial_s_h, device=self.device, dtype=torch.float32)
        # print(f"s_h shape: {s_h.shape}")
        # print(f"partial_s_h shape: {partial_s_h.shape}")
        # from prog_policies.drl_train  import save_gif
        # np_sh = s_h.detach().cpu().numpy()
        # np_partial_sh = partial_s_h.detach().cpu().numpy()
        # for i in range(10):
        #     save_gif(os.path.join('/home/hubertchang/HPRL/pretrain/viz_data_state', f"prog{program_id}_roll{i}.gif"),  [frame for frame in np_sh[i]])
        #     save_gif(os.path.join('/home/hubertchang/HPRL/pretrain/viz_data_state', f"prog{program_id}_roll{i}_partial.gif"),  [frame for frame in np_partial_sh[i]])
        # print(f"prog_id:{program_id}")
        s_h_len = torch.tensor(s_h_len, device=self.device, dtype=torch.int16)
        a_h = torch.tensor(a_h, device=self.device, dtype=torch.int16)
        a_h_len = torch.tensor(a_h_len, device=self.device, dtype=torch.int16)

        packed_s_h = rnn.pack_padded_sequence(s_h, s_h_len.to("cpu"), batch_first=True, enforce_sorted=False)
        padded_s_h, s_h_len = rnn.pad_packed_sequence(packed_s_h, batch_first=True,
                                                      padding_value=0.0,
                                                      total_length=self.config['max_demo_length'])

        packed_partial_s_h = rnn.pack_padded_sequence(partial_s_h, s_h_len.to("cpu"), batch_first=True,
                                                      enforce_sorted=False)
        padded_partial_s_h, s_h_len = rnn.pad_packed_sequence(packed_partial_s_h, batch_first=True,
                                                              padding_value=0.0,
                                                              total_length=self.config['max_demo_length'])
        packed_a_h = rnn.pack_padded_sequence(a_h, a_h_len.to("cpu"), batch_first=True, enforce_sorted=False)
        padded_a_h, a_h_len = rnn.pad_packed_sequence(packed_a_h, batch_first=True,
                                                      padding_value=self.num_agent_actions-1,
                                                      total_length=self.config['max_demo_length'] - 1)

        s_h_list.append(padded_s_h)
        s_h_list.append(padded_partial_s_h)
        return sample, program_id, mask, s_h_list, s_h_len.to(self.device), padded_a_h, a_h_len.to(self.device)


def get_exec_data(hdf5_file, program_id, num_agent_actions):
    s_h = np.moveaxis(np.copy(hdf5_file[program_id]['s_h']), [-1, -2, -3], [-3, -1, -2])
    # print(f"s_h shape after get_exec_data: {s_h.shape}")
    patches, ok_mask = convert_to_POMDP_np(s_h)
    partial_s_h = patches

    # print("shape:", patches.shape)          # (R, T, 8, 3, 3)
    # print("frames with missing Karel:",
        # np.where(~ok_mask))               # indices you may want to drop
    a_h = np.copy(hdf5_file[program_id]['a_h'])
    s_h_len = np.copy(hdf5_file[program_id]['s_h_len'])
    a_h_len = np.copy(hdf5_file[program_id]['a_h_len'])

    # expand demo length if needed (only for 1-timestep demos)
    if s_h.shape[1] == 1:
        s_h = np.concatenate((s_h, s_h), axis=1)
        a_h = np.ones((s_h.shape[0], 1))

    # Add dummy actions for demos with a_h_len == 0
    for i in range(s_h_len.shape[0]):
        if a_h_len[i] == 0:
            assert s_h_len[i] == 1
            a_h_len[i] += 1
            s_h_len[i] += 1
            s_h[i][1] = s_h[i][0]
            a_h[i][0] = num_agent_actions - 1

    return s_h, partial_s_h, s_h_len, a_h, a_h_len


def make_datasets(datadir, config, num_program_tokens, num_agent_actions, device, logger, dsl):
    """ Given the path to main dataset, split the data into train, valid, test and create respective pytorch Datasets

    Parameters:
        :param datadir (str): patth to main dataset (should contain 'data.hdf5' and 'id.txt')
        :param config (dict):  all configs in dict format
        :param num_program_tokens (int): number of program tokens in karel DSL
        :param num_agent_actions (int): number of actions karel agent can take
        :param device(torch.device): dataset target device: torch.device('cpu') or torch.device('cuda:X')

    Returns:
        :return train_dataset(torch.utils.data.Dataset): training dataset
        :return valid_dataset(torch.utils.data.Dataset): validation dataset
        :return test_dataset(torch.utils.data.Dataset): test dataset

    """
    program_list = []
    r_eq_program_count = 0
    drop_program_count = 0
    seen_programs = set()

    # create karel_env for program run test
    s_gen = KarelStateGenerator(seed=config['seed'])
    _world = karel.Karel_world(make_error=False, env_task=config['env_task'], task_definition=config['rl']['envs']['executable']['task_definition'], reward_diff=config['reward_diff'], final_reward_scale=config['final_reward_scale']) 


    for file_name in os.listdir(datadir):
        if file_name.endswith("hdf5"):
            f_path = os.path.join(datadir, file_name)
            id_file_path = os.path.join(datadir, file_name.replace("data", "id").replace("hdf5", "txt"))
            
            logger.debug("Loading from files ...")
            logger.debug(f_path)
            logger.debug(id_file_path)

            hdf5_file = h5py.File(f_path, 'r')
            id_file = open(id_file_path, 'r')
            id_list = id_file.readlines()
            for program_id in tqdm(id_list):
                program_id = program_id.strip().split()[0]
                program = hdf5_file[program_id]['program'][()]
                valid_flag = True 
                
                random_code_str = dsl.intseq2str(program)
                
                if random_code_str in seen_programs:
                    continue

                if program.shape[0] < config['dsl']['max_program_len'] and valid_flag:
                    exec_data = get_exec_data(hdf5_file, program_id, num_agent_actions)
                    program_list.append((program_id, program, exec_data))
                    seen_programs.add(random_code_str)


            hdf5_file.close()
            id_file.close()
            logger.debug('Total programs with length <= {}: {}'.format(config['dsl']['max_program_len'], len(program_list)))
            logger.debug('filter out R= program number: {}'.format(r_eq_program_count))
            logger.debug('drop program number: {}'.format(drop_program_count))


    random.shuffle(program_list)

    train_r, val_r, test_r = 0.85, 0.15, 0.0
    split_idx1 = int(train_r*len(program_list))
    split_idx2 = int((train_r+val_r)*len(program_list))
    train_program_list = program_list[:split_idx1]
    valid_program_list = program_list[split_idx1:split_idx2]
    test_program_list = valid_program_list #program_list[split_idx2:]

    train_dataset = ProgramDataset(train_program_list, config, num_program_tokens, num_agent_actions, device)
    val_dataset = ProgramDataset(valid_program_list, config, num_program_tokens, num_agent_actions, device)
    test_dataset = ProgramDataset(test_program_list, config, num_program_tokens, num_agent_actions, device)
    return train_dataset, val_dataset, test_dataset


def run(config, logger):

    if config['logging']['wandb']:
        import wandb
        wandb.init(project="prl-nips", sync_tensorboard=True, name=config['outdir'].split('/')[-1])
    else:
        os.environ['WANDB_MODE'] = 'dryrun'

    # begin block: this block sets the device from the config
    if config['device'].startswith('cuda') and torch.cuda.is_available():
        device = torch.device(config['device'])
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        logger.warning('{} GPU not available, running on CPU'.format(__name__))

    # setup tensorboardX: create a summary writer
    writer = SummaryWriter(logdir=config['outdir'])

    # this line logs the device info
    logger.debug('{} Using device: {}'.format(__name__, device))

    # end block: this block looks good

    # begin block: this block sets random seed for the all the modules
    if config['seed'] is not None:
        logger.debug('{} Setting random seed'.format(__name__))
        seed = config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

    if config['device'].startswith('cuda') and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # end block: this block looks good. if we have specified a seed, then we set it

    global_logs = {'info': {}, 'result': {}}

    # make dummy env to get action and observation space based on the environment
    custom_kwargs = {"config": config['args']}
    custom = True if "karel" or "CartPoleDiscrete" in config['env_name'] else False
    logger.debug('Using environment: {}'.format(config['env_name']))
    envs = make_vec_envs(config['env_name'], config['seed'], 1,
                         config['gamma'], os.path.join(config['outdir'], 'openai'), device, False, custom_env=custom,
                         custom_kwargs=custom_kwargs)

    # call the training function using the dataloader and the model
    dsl = get_DSL_option_v2(seed=seed, environment=config['rl']['envs']['executable']['name'])
    config['dsl']['num_agent_actions'] = len(dsl.action_functions) + 1      # +1 for a no-op action, just for filling
    if config['algorithm'] == 'supervised':
        model = SupervisedModel(device, config, envs, dsl, logger, writer, global_logs, config['verbose'])
    elif config['algorithm'] == 'PPO_option':
        model = PPOModel(device, config, envs, dsl, logger, writer, global_logs, config['verbose'])
    elif config['algorithm'] == 'SAC_option':
        model = SACModel(device, config, envs, dsl, logger, writer, global_logs, config['verbose'])
    else:
        model = SupervisedModel(device, config, envs, dsl, logger, writer, global_logs, config['verbose'])

    # Add wandb logger to the model
    if config['logging']['wandb']:
        wandb.config.update(config)

    if config['algorithm'] in ['supervised']:
        # write the code to load the dataset and initiate the dataloader
        p_train_dataset, p_val_dataset, p_test_dataset = make_datasets(config['datadir'], config,
                                                                       model.num_program_tokens,
                                                                       config['dsl']['num_agent_actions'], device,
                                                                       logger, dsl)
        config_tr = config['train']
        config_val = config['valid']
        config_test = config['test']
        config_eval = config['eval']
        p_train_dataloader = DataLoader(p_train_dataset, batch_size=config_tr['batch_size'],
                                        shuffle=config_tr['shuffle'], **config['data_loader'])
        p_val_dataloader = DataLoader(p_val_dataset, batch_size=config_val['batch_size'],
                                      shuffle=config_val['shuffle'], **config['data_loader'])
        p_test_dataloader = DataLoader(p_test_dataset, batch_size=config_test['batch_size'],
                                       shuffle=config_test['shuffle'], **config['data_loader'])

        r_train_dataloader, r_val_dataloader = None, None
    
    # Save configs and models
    pickle.dump(config, file=open(os.path.join(config['outdir'], 'config.pkl'), 'wb'))
    shutil.copy(src=config['configfile'], dst=os.path.join(config['outdir'], 'configfile.py'))


    # start training
    if config['algorithm'] == 'supervised':
        if config['mode'] == 'train':
            tic = time.time()
            model.train(p_train_dataloader, p_val_dataloader, r_train_dataloader, r_val_dataloader,
                        max_epoch=config['train']['max_epoch'])
            toc = time.time()
            global_logs['tr_time'] = toc - tic

            # Save results
            logs_path = os.path.join(config['outdir'], config['record_file'])
            pickle.dump(global_logs, file=open(logs_path, 'wb'))

        elif config['mode'] == 'eval':
            assert config_eval['usage'] in ['train', 'valid', 'test'], 'usage should be one of [train, valid, test]'
            if config_eval['usage'] == 'train':
                data_loader = p_train_dataloader
            elif config_eval['usage'] == 'test':
                data_loader = p_test_dataloader
            elif config_eval['usage'] == 'valid':
                data_loader = p_val_dataloader

            # Evaluate on data
            tic = time.time()
            model.evaluate(data_loader)
            toc = time.time()
            global_logs['eval_time'] = toc - tic

            # Save results
            logs_path = os.path.join(config['outdir'], config['record_file'].replace('.pkl', '_eval.pkl'))
            pickle.dump(global_logs, file=open(logs_path, 'wb'))

        else:
            raise NotImplementedError('Not yet Implemented')
    elif config['algorithm'] == 'RL':
        model.train()
    elif config['algorithm'] == 'CEM':
        model.train()
    elif config['algorithm'] == 'PPO_option':
        model.train()
    elif config['algorithm'] == 'SAC_option':
        model.train()
    elif config['algorithm'] == 'debug':
        return model
    elif config['algorithm'] == 'output_dataset_split':
        dataset_list = [('train', p_train_dataset), ('valid', p_val_dataset), ('test', p_test_dataset)]
        for name, dataset in dataset_list:
            with open(os.path.join(config['outdir'],name+'_dataset_program_list.txt'),"w") as f:
                for program in dataset.programs:
                    f.write(dsl.intseq2str(program[1])+'\n')
    elif config['algorithm'] == 'output_dataset_eval':
        model.eval(p_train_dataloader, p_val_dataloader, p_test_dataloader)

    return


def _temp(config, args):

    args.task_file = config['rl']['envs']['executable']['task_file']
    args.grammar = config['dsl']['grammar']
    args.use_simplified_dsl = config['dsl']['use_simplified_dsl']
    args.task_definition = config['rl']['envs']['executable']['task_definition']
    args.execution_guided = config['rl']['policy']['execution_guided']

if __name__ == "__main__":
    # wandb_project_name = input("Please enter the wandb project name: ")
    from cfg_option_new_vae import config
    wandb_session_name = input("Please enter the wandb session name: ")
    wandb.init(project="DRLGS_latent", name=wandb_session_name, config=config)
    pwd = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(pwd, 'cfg_option_new_vae.py')
    wandb.save(config_path)

    #torch.set_num_threads(2)
    torch.set_num_threads(1)

    t_init = time.time()
    parser = customargparse.CustomArgumentParser(description='syntax learner')

    # Add arguments (including a --configfile)
    parser.add_argument('-o', '--outdir',
                        help='Output directory for results', default='pretrain/output_dir')
    parser.add_argument('-d', '--datadir',
                        help='dataset directory containing data.hdf5 and id.txt')
    parser.add_argument('-dc', '--datadirCheck',
                        help='dataset directory containing checked (valid) id.txt')
    parser.add_argument('-c', '--configfile',
                        help='Input file for parameters, constants and initial settings')
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')
    parser.add_argument('--linear_inter_units',
                        help='Intermediate unit numbers of linear layers', type=int, nargs='+')

    # Parse arguments
    args = parser.parse_args()

    # FIXME: This is only for backwards compatibility to old parser, should be removed once we change the original
    args.outdir = os.path.join(args.outdir, '%s-%s-%s-%s' % (args.prefix, args.grammar, args.seed, time.strftime("%Y%m%d-%H%M%S")))
    log_dir = os.path.expanduser(os.path.join(args.outdir, 'openai'))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    # fetch the mapping from prl tokens to dsl tokens
    if args.mapping_file is not None:
        args.dsl2prl_mapping, args.prl2dsl_mapping, args.dsl_tokens, args.prl_tokens = \
            fetch_mapping(args.mapping_file)
        args.use_simplified_dsl = True
        args.use_shorter_if = True if 'shorter_if' in args.mapping_file else False
    else:
        _, _, args.dsl_tokens, _ = fetch_mapping('mapping_karel2prl_new_vae_v2.txt')
        args.use_simplified_dsl = False

    config = customargparse.args_to_dict(args)
    config['args'] = args

    _temp(config, args)

    # TODO: shift this logic somewhere else
    # encode reward along with state and action if task defined by custom reward
    config['rl']['envs']['executable']['dense_execution_reward'] = config['rl']['envs']['executable'][
                                                                       'task_definition'] == 'custom_reward'

    # Create output directory if it does not already exist
    create_directory(config['outdir'])

    # Set up logger
    log_file = os.path.join(config['outdir'], config['logging']['log_file'])
    log_handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')]
    logging.basicConfig(handlers=log_handlers, format=config['logging']['fmt'], level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    print(config['logging'])
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    logger.disabled = (not config['verbose'])

    # Call the main method
    run_results = run(config, logger)

    # Final time
    t_final = time.time()
    logger.debug('{} Program finished in {} secs.'.format(__name__, t_final - t_init))
    print('{} Program finished in {} secs.'.format(__name__, t_final - t_init))
