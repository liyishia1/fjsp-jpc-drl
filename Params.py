#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 15:39
# @Author  : Yishi Li
# @Site    :
# @File    :
# @Software: PyCharm

import argparse
import torch

train_parser = argparse.ArgumentParser(description='fjsp_ppo_gan')
# args for device
train_parser.add_argument('--message', type=bool, default=True, help='True or False')
train_parser.add_argument('--features', type=bool, default=True, help='True or False')
train_parser.add_argument('--simul_logic', type=bool, default=True, help='True or False')

train_parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='cuda or cpu')
train_parser.add_argument('--ckpt', type=str, default='result/20240419save_best_18_5_1750.pt', help='cuda or cpu')
train_parser.add_argument('--model', type=str, default='train', help='cuda or cpu')
train_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
train_parser.add_argument('--seed', type=int, default=20240705, help='Seed for training')
train_parser.add_argument('--gamma', type=float, default=1.0, help='instance')
train_parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE parameter')
train_parser.add_argument('--tau', type=float, default=0, help='Policy soft update coefficient')
train_parser.add_argument('--K_epochs', type=int, default=4, help='Seed for training')
train_parser.add_argument('--eps_clip', type=float, default=0.2, help='Seed for training')
train_parser.add_argument('--A_coeff', type=float, default=1.0, help='Seed')
train_parser.add_argument('--vf_coeff', type=float, default=0.5, help='epochs')
train_parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Seed')
train_parser.add_argument('--max_iterations', type=int, default=1000, help='No. of episodes of each env for training')
train_parser.add_argument('--save_timestep', type=int, default=10, help='Seed')
train_parser.add_argument('--update_timestep', type=int, default=1, help='epochs')
train_parser.add_argument('--viz', type=bool, default=False, help='Seed')
train_parser.add_argument('--minibatch_size', type=int, default=1024, help='Batch size for computing the gradient')
train_parser.add_argument('--parallel_iter', type=int, default=20, help='Batch size for training environments')
train_configs = train_parser.parse_args()

env_parser = argparse.ArgumentParser(description='fjsp_ppo_gan')
# args for device
env_parser.add_argument('--model', type=str, default='random', help='cuda or cpu')
# args for env
env_parser.add_argument('--n_jobs', type=int, default=6, help='instance')
env_parser.add_argument('--n_mas', type=int, default=5, help='instance')
env_parser.add_argument('--n_ops', type=int, default=12, help='instance')
env_parser.add_argument('--n_groups', type=int, default=2, help='instance')
env_parser.add_argument('--max_proc', type=int, default=200, help='instance')
env_parser.add_argument('--min_proc', type=int, default=100, help='instance')
env_parser.add_argument('--group_shape', type=list, default=[1, 1, 2, 2, 3, 3], help='instance')

env_parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training environments')
env_parser.add_argument('--ope_feat_dim', type=int, default=10, help='instance')
env_parser.add_argument('--ma_feat_dim', type=int, default=8, help='instance')
env_parser.add_argument('--edge_feat_dim', type=int, default=8, help='instance')
env_parser.add_argument('--valid_batch_size', type=int, default=100, help='instance')
env_configs = env_parser.parse_args()

dan_parser = argparse.ArgumentParser(description='fjsp_ppo_gan')
dan_parser.add_argument('--device', type=str, default=train_configs.device, help='cuda or cpu')
dan_parser.add_argument('--fea_j_input_dim', type=int, default=10, help='Dimension of operation raw feature vectors')
dan_parser.add_argument('--fea_m_input_dim', type=int, default=8, help='Dimension of machine raw feature vectors')
dan_parser.add_argument('--fea_edge_input_dim', type=int, default=8, help='Dimension of machine raw feature vectors')
dan_parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')

dan_parser.add_argument('--num_heads_OAB', nargs='+', type=int, default=[4, 4],
                    help='Number of attention head of operation message attention block')
dan_parser.add_argument('--num_heads_MAB', nargs='+', type=int, default=[4, 4],
                    help='Number of attention head of machine message attention block')
dan_parser.add_argument('--layer_fea_output_dim', nargs='+', type=int, default=[32, 8],
                    help='Output dimension of the DAN layers')

dan_parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Number of layers in Actor network')
dan_parser.add_argument('--hidden_dim_actor', type=int, default=64, help='Hidden dimension of Actor network')
dan_parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='Number of layers in Critic network')
dan_parser.add_argument('--hidden_dim_critic', type=int, default=64, help='Hidden dimension of Critic network')
dan_configs = dan_parser.parse_args()

test_parser = argparse.ArgumentParser(description='fjsp_ppo_gan')
# args for device
test_parser.add_argument('--num_ins', type=int, default=1000, help='instance')
test_parser.add_argument('--ttest_base', type=str, default='data_dev/ttest/20240318.xlsx', help='instance')
test_parser.add_argument('--name_ins', type=str, default='10002-12-5', help='instance')
test_parser.add_argument('--rules', type=list, default=["DRL"], help='Seed for training')
test_parser.add_argument('--sample', type=bool, default=True, help='instance')
test_parser.add_argument('--num_sample', type=int, default=20, help='Seed for training')
test_parser.add_argument('--num_average', type=int, default=10, help='Seed for training')
test_parser.add_argument('--public_ins', type=bool, default=True, help='Seed')
test_parser.add_argument('--data_path', type=str, default="inst1", help='epochs')
test_parser.add_argument('--ckpt', type=str, default='',
                         help='cuda or cpu')
test_configs = test_parser.parse_args()
