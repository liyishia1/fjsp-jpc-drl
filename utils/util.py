import copy
import random

import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def features1(n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec):
    num_op = len(proc_times)

    start_counter = []
    counter = 0
    for i in n_ops:
        for j in i:
            start_counter.append(counter)
            counter += j
    g_j_o = {}
    counter1 = 0
    for ii, i in enumerate(n_ops):
        for jj, j in enumerate(i):
            for k in range(j):
                g_j_o[(ii, jj, k)] = counter1
                counter1 += 1

    proc_times_batch = []
    ope_ma_adj_batch = []
    for op, proc in zip(avail_machs, proc_times):
        pro_time = np.zeros(n_machs)
        ope_ma_adj = np.zeros(n_machs)
        for i, j in zip(op, proc):
            pro_time[i] = j
            ope_ma_adj[i] = 1
        proc_times_batch.append(pro_time)
        ope_ma_adj_batch.append(ope_ma_adj)

    ope_pre_adj_batch = []
    kk = 0
    g_ = 0
    for m, n in zip(n_ops, jobs_prec):
        ope_ = {}
        for indexj, j in enumerate(m):
            for ij in range(j):
                ope_pre_adj = np.full(num_op, False)
                if ij == j - 1:
                    kk += 1
                else:
                    if kk < num_op:
                        ope_pre_adj[kk + 1] = True
                        kk += 1

                ope_[(indexj, ij)] = ope_pre_adj
        for indexi, i in enumerate(m):
            if n[indexi] is not None:
                for j in n[indexi]:
                    arra = ope_[(j, m[j] - 1)]
                    arra[g_j_o[(g_, indexi, 0)]] = True
        for value in ope_.values():
            ope_pre_adj_batch.append(value)
        g_ += 1

    ope_sub_adj_batch = []
    for i in range(num_op):
        ope_sub_adj_batch.append(np.full(num_op, False))
    for i in range(len(ope_pre_adj_batch)):
        for j in range(len(ope_pre_adj_batch[i])):
            if ope_pre_adj_batch[i][j] == True:
                ope_sub_adj_batch[j][i] = True
    nums_ope_batch = []
    for i in n_ops:
        for j in i:
            nums_ope_batch.append(j)

    nums_group_batch_end = []
    kkkkkk = 0
    for i in n_ops:
        kkkkkk += sum(i)
        nums_group_batch_end.append(kkkkkk - 1)

    state1 = np.array(proc_times_batch)
    # state2 = np.array(ope_ma_adj_batch)
    state2 = jobs_prec
    # state3 = torch.tensor(cal_cumul_adj_batch).unsqueeze(0).float()
    state4 = np.array(ope_pre_adj_batch)
    state5 = np.array(ope_sub_adj_batch)
    # state6 = torch.tensor(opes_appertain_batch).unsqueeze(0).int()
    # state7 = torch.tensor(num_ope_biases_batch).unsqueeze(0).int()
    state8 = np.array(nums_ope_batch)
    # state9 = torch.tensor(num_group_biases_batch).unsqueeze(0).int()
    state10 = np.array(nums_group_batch_end)

    return state1, state2, state4, state5, state8, state10


def features_ex4(n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec):
    # I, M, J, K, T, P = tote(n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec)
    num_op = len(proc_times)

    start_counter = []
    counter = 0
    for i in n_ops:
        for j in i:
            start_counter.append(counter)
            counter += j
    g_j_o = {}
    counter1 = 0
    for ii, i in enumerate(n_ops):
        for jj, j in enumerate(i):
            for k in range(j):
                g_j_o[(ii, jj, k)] = counter1
                counter1 += 1

    proc_times_batch = []
    ope_ma_adj_batch = []
    for op, proc in zip(avail_machs, proc_times):
        pro_time = np.zeros(n_machs)
        ope_ma_adj = np.zeros(n_machs)
        for i, j in zip(op, proc):
            pro_time[i] = j
            ope_ma_adj[i] = 1
        proc_times_batch.append(pro_time)
        ope_ma_adj_batch.append(ope_ma_adj)

    # cal_cumul_adj_batch = []
    # t = 0
    # for m in n_ops:
    #     for n in m:
    #         cal_cumul_adj = np.zeros(num_op)
    #         if n == 1:
    #             t += 1
    #         else:
    #             for j in range(n - 1):
    #                 if t < num_op:
    #                     cal_cumul_adj[t + 1] = 1
    #                     t += 1
    #         cal_cumul_adj_batch.append(cal_cumul_adj)

    ope_pre_adj_batch = []
    kk = 0
    g_ = 0
    for m, n in zip(n_ops, jobs_prec):
        ope_ = {}
        jj = 0
        for indexj, j in enumerate(m):
            tt = 0
            for ij in range(j):
                ope_pre_adj = np.full(num_op, False)
                if ij == j - 1:
                    kk += 1
                else:
                    if kk < num_op:
                        ope_pre_adj[kk + 1] = True
                        kk += 1

                ope_[(indexj, ij)] = ope_pre_adj
        for indexi, i in enumerate(m):
            if n[indexi] is not None:
                for j in n[indexi]:
                    arra = ope_[(j, m[j] - 1)]
                    arra[g_j_o[(g_, indexi, 0)]] = True
                    # ope_[(j, m[i] - 1)] = arra
        for value in ope_.values():
            ope_pre_adj_batch.append(value)
        g_ += 1

    ope_sub_adj_batch = []
    for i in range(num_op):
        ope_sub_adj_batch.append(np.full(num_op, False))
    for i in range(len(ope_pre_adj_batch)):
        for j in range(len(ope_pre_adj_batch[i])):
            if ope_pre_adj_batch[i][j] == True:
                ope_sub_adj_batch[j][i] = True

    # opes_appertain_batch = []
    # kkk = 0
    # for i in n_ops:
    #     for j in i:
    #         for m in range(j):
    #             opes_appertain_batch.append(kkk)
    #         kkk += 1

    # num_ope_biases_batch = []
    nums_ope_batch = []
    # kkkk = 0
    for i in n_ops:
        for j in i:
            # num_ope_biases_batch.append(kkkk)
            nums_ope_batch.append(j)
            # kkkk += j

    num_group_biases_batch = []
    nums_group_batch_end = []
    kkkkk = 0
    kkkkkk = 0
    for i in n_ops:
        for j in range(sum(i)):
            num_group_biases_batch.append(kkkkk)
        kkkkkk += sum(i)
        nums_group_batch_end.append(kkkkkk - 1)
        kkkkk += 1

    state1 = np.array(proc_times_batch)
    # state2 = np.array(ope_ma_adj_batch)
    state2 = jobs_prec
    # state3 = torch.tensor(cal_cumul_adj_batch).unsqueeze(0).float()
    state4 = np.array(ope_pre_adj_batch)
    state5 = np.array(ope_sub_adj_batch)
    # state6 = torch.tensor(opes_appertain_batch).unsqueeze(0).int()
    # state7 = torch.tensor(num_ope_biases_batch).unsqueeze(0).int()
    state8 = np.array(nums_ope_batch)
    state9 = np.array(num_group_biases_batch)
    state10 = np.array(nums_group_batch_end)

    return state1, state2, state4, state5, state8, state9, state10


def ini_start_and_end_ex4(nums_group_batch_end, num_group_biases_batch, ope_pre_adj_batch, proc_times_batch):
    st = []
    et = []
    npp = []
    nums_group_batch_start = []
    cal_cumul_adj_batch = []
    num_op = len(ope_pre_adj_batch[0])
    for i in range(len(nums_group_batch_end)):
        if i == 0:
            nums_group_batch_start.append(0)
        else:
            nums_group_batch_start.append(nums_group_batch_end[i - 1] + 1)
    for i in range(max(num_group_biases_batch) + 1):
        et_ = []
        np_ = []
        cal_cumul_adj_ = []
        group_index = [k for k, x in enumerate(num_group_biases_batch) if x == i]
        st_ = [0 for _ in range(len(group_index))]
        for j in group_index:
            opp = 0
            ett = 0
            cal_cumul_adj = np.zeros(num_op)
            cal_cumul_adj[j] = 1
            pre_index = [k for k, x in enumerate(ope_pre_adj_batch[j]) if x == True]
            st_[j - nums_group_batch_start[i]] = copy.deepcopy(max(0, st_[j - nums_group_batch_start[i]]))
            if pre_index:
                ij = j
                e = 0
                opp += 1
                ett += proc_times_batch[ij]
                while e != 1:
                    for m in pre_index:
                        ij = m
                        cal_cumul_adj[ij] = 1
                        opp += 1
                        st_[ij - nums_group_batch_start[i]] = copy.deepcopy(
                            max(ett, st_[ij - nums_group_batch_start[i]]))
                        ett += proc_times_batch[ij]
                        pre_index = [k for k, x in enumerate(ope_pre_adj_batch[ij]) if x == True]
                        if not pre_index:
                            e = 1
                np_.append(opp)
                et_.append(ett)
            else:
                np_.append(1)
                et_.append(proc_times_batch[j])
            cal_cumul_adj_.append(cal_cumul_adj)

        st += st_
        # et += [max(et_) for _ in range(len(group_index))]
        et += et_
        npp += np_
        cal_cumul_adj_batch += cal_cumul_adj_
    feat5 = np.array(st)
    # feat2 = np.array(et)
    feat2 = feat5 + proc_times_batch
    feat8 = np.array(npp)
    cal_cumul_adj_batchs = np.array(cal_cumul_adj_batch)
    # feat9 = np.dot(cal_cumul_adj_batchs, np.mean(proc_times_batch, axis=-1, keepdims=True))
    return feat2, feat8, 0, feat5, cal_cumul_adj_batchs


def start_and_end_ex4(scheduled, starttime, nums_group_batch_end, num_group_biases_batch, ope_pre_adj_batch,
                      proc_times_batch):
    st = []
    et = []
    nums_group_batch_start = []
    for i in range(len(nums_group_batch_end)):
        if i == 0:
            nums_group_batch_start.append(0)
        else:
            nums_group_batch_start.append(nums_group_batch_end[i - 1] + 1)
    for i in range(max(num_group_biases_batch) + 1):
        et_ = []
        group_index = [k for k, x in enumerate(num_group_biases_batch) if x == i]

        st_ = [0 for _ in range(len(group_index))]
        for j in group_index:
            pre_index = [k for k, x in enumerate(ope_pre_adj_batch[j]) if x == True]
            if scheduled[j] == 1:
                ett = starttime[j] + proc_times_batch[j]
                st_[j - nums_group_batch_start[i]] = starttime[j]
            else:
                ett = 0
                st_[j - nums_group_batch_start[i]] = copy.deepcopy(max(0, st_[j - nums_group_batch_start[i]]))
            if pre_index:
                ij = j
                e = 0
                if scheduled[ij] == 1:
                    ett = starttime[ij] + proc_times_batch[ij]
                else:
                    ett += proc_times_batch[ij]
                while e != 1:
                    for m in pre_index:
                        ij = m
                        pre_index = [k for k, x in enumerate(ope_pre_adj_batch[ij]) if x == True]
                        if scheduled[ij] == 1:
                            ett = starttime[ij] + proc_times_batch[ij]
                            st_[ij - nums_group_batch_start[i]] = starttime[ij]
                        else:
                            st_[ij - nums_group_batch_start[i]] = copy.deepcopy(
                                max(ett, st_[ij - nums_group_batch_start[i]]))
                            ett += proc_times_batch[ij]

                        if not pre_index:
                            e = 1
                et_.append(ett)
            else:
                et_.append(proc_times_batch[j])
        st += st_
        # et += [max(et_) for _ in range(len(group_index))]
        et += et_
    feat5 = np.array(st)
    # feat4 = np.array(et)
    feat4 = feat5 + proc_times_batch
    return feat5, feat4

def unavailable_jobs(prec, mask, mod):
    eligible_proc = np.full_like(mod, False)
    for i in range(len(prec)):
        env = tran_pre(prec[i])
        for j in range(len(env)):
            if env[j] is not None:
                prec_j = mask[i, env[j]]
                if np.alltrue(prec_j) == False:
                    eligible_proc[i, j] = True
    return eligible_proc


def diff_prec(next_unavailable_job, unavailable_job, prec):
    list1 = []
    list2 = []

    for i in range(len(unavailable_job)):

        env = tran_pre(prec[i])
        list2.append(env)

        list1.append([])
        for j in range(len(unavailable_job[i])):
            if unavailable_job[i, j, 0] == True:
                if next_unavailable_job[i, j, 0] == False:
                    list1[i].append(j)
    return list1, list2


def tran_pre(prec):
    counter = 0
    env = []
    for gr in prec:
        ng = len(gr)
        for grj in gr:
            if grj is None:
                env.append(grj)
            else:
                kk = []
                for kkk in grj:
                    kk.append(kkk + ng * counter)
                env.append(kk)
        counter += 1
    return env
