#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 14:58
# @Author  : Yishi Li
# @Site    :
# @File    : train.py
# @Software: PyCharm
import os
import time
from collections import deque
from copy import deepcopy

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Params import env_configs, train_configs, dan_configs, test_configs
from utils.common_utils import *
from data_dev.excel_to_data import getdata
from env.fjsp_env_mult import FJSPEnvForSameOpNums
from env.get_case import casecoll
from utils.util import setup_seed

if train_configs.message == True:
    from model.PPO_ex3 import PPO_initialize
    from model.PPO_ex3 import Memory
else:
    from model.PPO import PPO_initialize
    from model.PPO import Memory

device = torch.device(train_configs.device)
str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


class Trainer:
    def __init__(self, env_configs, train_configs, dan_configs, counter):
        self.counter = counter
        self.maxlen = 1  # Save the best model
        self.best_models = deque()
        self.env_configs = env_configs
        self.train_configs = train_configs
        self.dan_configs = dan_configs
        self.max_updates = train_configs.max_iterations
        self.reset_env_timestep = train_configs.parallel_iter
        self.validate_timestep = train_configs.save_timestep
        self.num_envs = env_configs.batch_size

        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.model_name = '{}-{}-{}-{}'.format(env_configs.n_groups, env_configs.n_ops, env_configs.n_mas,
                                               env_configs.n_jobs)

        self.env_valid_configs = copy.deepcopy(env_configs)
        self.env_valid_configs.batch_size = env_configs.valid_batch_size

        self.env = FJSPEnvForSameOpNums(n_g=env_configs.n_groups,
                                        n_j=env_configs.n_jobs,
                                        n_m=env_configs.n_mas,
                                        n_ops=env_configs.n_ops,
                                        env_config=self.env_configs)

        # validation data set
        self.milp_data = pd.read_excel('data_dev/milpresult/{}.xlsx'.format(self.model_name), index_col=0)
        self.vali_data = getdata(self.model_name)
        self.vail_index = np.random.choice(range(len(self.vali_data)), size=self.env_valid_configs.batch_size,
                                           replace=False)

        self.vali_size_data = [self.vali_data[i] for i in self.vail_index]
        self.vail_milp_result = self.milp_data.iloc[self.vail_index]['cmax'].to_numpy()

        self.vali_env = FJSPEnvForSameOpNums(n_g=env_configs.n_groups,
                                             n_j=env_configs.n_jobs,
                                             n_m=env_configs.n_mas,
                                             n_ops=env_configs.n_ops,
                                             env_config=self.env_valid_configs)

        self.vali_env.set_initial_data(model='vali', data=self.vali_size_data)

        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=train_configs.gamma, gae_lambda=train_configs.gae_lambda)
        self.save_path = './save/train_{0}_{1}_{2}'.format(self.model_name, self.counter, str_time)
        self.data_path = './result/{0}/{1}/iter{2}'.format(self.model_name, str_time, self.counter)
        os.makedirs(self.save_path)
        os.makedirs(self.data_path)

        # Use visdom to visualize the training process
        self.is_viz = train_configs.viz
        if self.is_viz:
            self.board = SummaryWriter(log_dir=self.save_path, comment='_model_log', filename_suffix=str_time)

        self.valid_results = []
        self.valid_results_100 = []
        self.losscoll = []
        self.rewardcoll = []
        data_file = pd.DataFrame(np.arange(10, 10010, 10), columns=["iterations"])
        data_file.to_excel('{0}/training_ave_{1}.xlsx'.format(self.save_path, str_time), sheet_name='Sheet1',
                           index=False)
        data_file = pd.DataFrame(np.arange(10, 10010, 10), columns=["iterations"])
        data_file.to_excel('{0}/training_100_{1}.xlsx'.format(self.save_path, str_time), sheet_name='Sheet1',
                           index=False)

    def train(self):
        """
            train the model following the config
        """
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : random")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.model_name}")
        print("\n")

        self.train_st = time.time()

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                train_data = casecoll(self.env_configs.batch_size,
                                      n_group=self.env_configs.n_groups,
                                      n_job=self.env_configs.n_jobs,
                                      n_mas=self.env_configs.n_mas,
                                      max_proc=self.env_configs.max_proc,
                                      min_proc=self.env_configs.min_proc,
                                      group_shape=self.env_configs.group_shape)
                state = self.env.set_initial_data(model='train', data=train_data)
            else:
                state = self.env.reset()

            ep_rewards = - deepcopy(self.env.init_quality)
            actionlist = []
            while True:
                # state store
                self.memory.push(state)
                with torch.no_grad():

                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor,  # [sz_b, J]
                                                             fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                                             mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.new_dynamic_pair_mask_tensor,
                                                             # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)
                # state transition
                state, reward, done, act = self.env.step(actions=action_envs.cpu().numpy())
                actionlist.append(act)

                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            if self.is_viz:
                self.board.add_scalars(main_tag='data/makespan of envs',
                                       tag_scalar_dict={'makespan_train': np.mean(self.env.current_makespan)},
                                       global_step=i_update)
                self.board.flush()

            # if iter mod x = 0 then update the policy (x = 1 in paper)
            if i_update % self.train_configs.update_timestep == 0:
                loss, loss1, loss2, loss3 = self.ppo.update(self.memory)
                self.memory.clear_memory()
                mean_rewards_all_env = np.mean(ep_rewards)
                mean_makespan_all_env = np.mean(self.env.current_makespan)
                # save the mean rewards of all instances in current training data
                self.log.append([i_update, mean_rewards_all_env])
                self.losscoll.append(loss)
                self.rewardcoll.append(mean_rewards_all_env)
                if self.is_viz:
                    self.board.add_scalars(main_tag='data/loss of envs', tag_scalar_dict={'A_loss': np.array([loss1]),
                                                                                          'vf_loss': np.array([loss2]),
                                                                                          'entropy_loss': np.array(
                                                                                              [loss3]),
                                                                                          'total_loss': np.array(
                                                                                              [loss])},
                                           global_step=i_update)
                    self.board.add_scalars(main_tag='data/reward of envs',
                                           tag_scalar_dict={'reward': np.array([mean_rewards_all_env])},
                                           global_step=i_update)
                    self.board.flush()

            if self.is_viz:
                for name, param in self.ppo.policy.named_parameters():
                    self.board.add_histogram(name + '_grad', param.grad, i_update)
                    self.board.add_histogram(name + '_data', param, i_update)
                    self.board.flush()

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                vali_result = self.validate_envs_with_various_op_nums()
                self.valid_results.append(vali_result.mean())
                self.valid_results_100.append(vali_result)
                vail_gap = (vali_result - self.vail_milp_result) / self.vail_milp_result
                if self.is_viz:
                    self.board.add_scalars(main_tag='data/makespan of envs',
                                           tag_scalar_dict={'makespan_validate': np.array([vali_result.mean()]),
                                                            'gap_validate': np.array([vail_gap.mean()])},
                                           global_step=i_update)
                    self.board.flush()
                if vail_gap.mean() < self.record:
                    self.save_model(i_update)
                    self.record = vail_gap.mean()

                tqdm.write(f'The validation quality is: {vail_gap.mean()} (best : {self.record})')

            ep_et = time.time()

            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))

        self.train_et = time.time()

        # log results
        torch.save(self.ppo.policy.state_dict(),
                   '{0}/save_best_{1}_{2}.pt'.format(self.save_path, self.model_name, 'final'))
        self.save_training_log()

    def test(self, ckpt='final'):
        ep_st = time.time()
        self.ckpt = ckpt
        if self.ckpt == 'final':
            print(self.ckpt)
            self.ppo.policy.eval()
        elif self.ckpt == 'best':
            model_CKPT = torch.load(self.best_models[0], map_location=device)
            self.ppo.policy.load_state_dict(model_CKPT)
            self.ppo.policy_old.load_state_dict(model_CKPT)
            print(self.ckpt)
            self.ppo.policy.eval()

        self.env_test_configs = copy.deepcopy(self.env_configs)

        self.env_test_configs.batch_size = test_configs.num_ins

        # validation data set
        self.test_env = FJSPEnvForSameOpNums(n_g=self.env_test_configs.n_groups,
                                             n_j=self.env_test_configs.n_jobs,
                                             n_m=self.env_test_configs.n_mas,
                                             n_ops=self.env_test_configs.n_ops,
                                             env_config=self.env_test_configs)

        self.test_env.set_initial_data(model='vali', data=self.vali_data)

        vali_result = self.test_greedy_with_various_op_nums()
        print(vali_result.mean())
        resu = vali_result
        greedy_data = pd.DataFrame({'cmax': resu})

        greedy_data.to_excel('{0}/training_100_{1}_{2}.xlsx'.format(self.save_path, str_time, self.ckpt))

        if test_configs.sample == True:
            sample_data = pd.DataFrame()
            for ii in range(test_configs.num_sample):
                vali_result = self.test_sample_with_various_op_nums()
                sample_data['iter{}'.format(ii)] = vali_result

            cmax = sample_data.min(axis=1)

            sample_data = pd.concat([sample_data, cmax.rename('cmax')], axis=1)
            print(sample_data['cmax'].mean())
            sample_data.to_excel('{0}/training_100_100_min{1}_{2}.xlsx'.format(self.save_path, str_time, self.ckpt))

        gap1, gap2 = self.save_test_log(greedy_data, sample_data)
        ep_et = time.time()
        print(ep_et - ep_st)
        return gap1, gap2

    def save_test_log(self, greedy_data, sample_data):

        log = {}
        log['milpcmax'] = self.milp_data['cmax'].mean()
        log['milpcgap'] = self.milp_data['gap'].mean()
        log['milpcgapstd'] = self.milp_data['gap'].std()
        diff1 = (greedy_data['cmax'] - self.milp_data['cmax']) / self.milp_data['cmax']
        log['greedycmax'] = greedy_data['cmax'].mean()
        log['greedygap'] = diff1.mean()
        log['greedygapstd'] = diff1.std()
        diff2 = (sample_data['cmax'] - self.milp_data['cmax']) / self.milp_data['cmax']
        log['samplecmax'] = sample_data['cmax'].mean()
        log['samplegap'] = diff2.mean()
        log['samplegapstd'] = diff2.std()

        filename = '{0}/test_log{1}_{2}.txt'.format(self.data_path, train_configs.seed + self.counter, self.ckpt)
        with open(filename, 'w') as file:
            for name, value in log.items():
                file.write(f"{name}: {value}\n")
            file.write("\n")
        print('{}end'.format(self.counter))

        return log['greedygap'], log['samplegap']

    def test_greedy_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        state = self.test_env.reset()
        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.test_env.done_flag)
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.new_dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, J, M]

            action = greedy_select_action(pi)

            state, _, done, act = self.test_env.step(action.cpu().numpy())
            if done.all():
                break
        return self.test_env.current_makespan

    def test_sample_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        env1 = self.test_env
        ppo1 = self.ppo
        state = env1.reset()
        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(env1.done_flag)
                pi, _ = ppo1.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                    op_mask=state.op_mask_tensor[batch_idx],
                                    candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                    fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                    mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                    comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                    dynamic_pair_mask=state.new_dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                    fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, J, M]

            action, action_logprob_envs = sample_action(pi)
            state, _, done, act = env1.step(action.cpu().numpy())

            if done.all():
                break
        return env1.current_makespan

    def save_training_log(self):
        """
            save reward data & validation makespan data (during training) and the entire training time
        """
        # Save the data of training curve to files
        torch.save(self.ppo.policy.state_dict(),
                   '{0}/save_best_{1}_{2}.pt'.format(self.save_path, self.model_name, 'final'))
        lossc = pd.DataFrame({'loss': self.losscoll, 'reward': self.rewardcoll})
        lossc.to_excel('{0}/training_lossandreward_{1}.xlsx'.format(self.save_path, str_time), sheet_name='Sheet1')
        data = pd.DataFrame(np.array(self.valid_results).transpose(), columns=["res"])
        data.to_excel('{0}/training_ave_{1}.xlsx'.format(self.save_path, str_time), sheet_name='Sheet1', index=False,
                      startcol=1)
        column = [i_col for i_col in range(100)]
        data = pd.DataFrame(np.array(self.valid_results_100), columns=column)
        data.to_excel('{0}/training_100_{1}.xlsx'.format(self.save_path, str_time), sheet_name='Sheet1', index=False,
                      startcol=1)

        print("total_time: ", time.time() - self.train_st)

    def validate_envs_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.new_dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done, act = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def save_model(self, i):
        """
            save the model
        """
        if len(self.best_models) == self.maxlen:
            delete_file = self.best_models.popleft()
            os.remove(delete_file)
        save_file = '{0}/save_best_{1}.pt'.format(self.save_path, i)
        self.best_models.append(save_file)
        torch.save(self.ppo.policy.state_dict(), save_file)

def save_args_to_file(args_dict, filename):
    with open(filename, 'w') as file:
        for parser_name, args in args_dict.items():
            file.write(f"{parser_name}:\n")
            for arg, value in vars(args).items():
                file.write(f"{arg}: {value}\n")
            file.write("\n")


def save_settings(model_name):
    args_dict = {
        'train_settings': train_configs,
        'env_settings': env_configs,
        'dan_settings': dan_configs,
        'test_settings': test_configs
    }
    args_path = './result/{0}/{1}/args.txt'.format(model_name, str_time)
    save_args_to_file(args_dict, args_path)


def main(j, k):
    greedygap = []
    samplegap = []
    for counter in range(j, k):
        seed = train_configs.seed + counter
        setup_seed(seed)
        trainer = Trainer(env_configs, train_configs, dan_configs, counter)
        if counter == 0:
            save_settings(trainer.model_name)
        trainer.train()
        gap1, gap2 = trainer.test()
        greedygap.append(gap1)
        samplegap.append(gap2)

    gapdf = pd.DataFrame(
        {'greedygap': greedygap, 'samplegap': samplegap})
    gapdf.to_excel('./result/{0}/{1}/greedyandsamplegap.xlsx'.format(trainer.model_name, str_time))
    return


if __name__ == '__main__':
    env_configs.n_groups = 2
    main(0, 1)
