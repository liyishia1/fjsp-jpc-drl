#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 19:53
# @Author  : Yishi Li
# @Site    : 
# @File    : util.py
# @Software: PyCharm
import copy
import random

import numpy as np
from collections import Counter
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


def getjobs(n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec):
    machinesNb = n_machs
    jobs = []

    proc_times_batch = []
    ope_ma_adj_batch = []
    for op, proc in zip(avail_machs, proc_times):
        pro_time = np.zeros(n_machs).astype(int)
        ope_ma_adj = np.zeros(n_machs).astype(int)
        for i, j in zip(op, proc):
            pro_time[i] = j
            ope_ma_adj[i] = 1
        proc_times_batch.append(pro_time)
        ope_ma_adj_batch.append(ope_ma_adj)

    opcounter = 0
    for i in n_ops:
        for j in i:
            job = []
            for op in range(j):
                oplist = []
                pt = proc_times_batch[opcounter]
                ma = ope_ma_adj_batch[opcounter]
                opcounter += 1
                for k in range(len(ma)):
                    if ma[k] == 1:
                        oplist.append({'machine': k, 'processingTime': pt[k]})
                job.append(oplist)
            jobs.append(job)
    return machinesNb, jobs, jobs_prec


def jpc_check(os, env):
    for j in range(len(env)):
        if env[j] is not None:
            prec_j = env[j]
            for pj in prec_j:
                j_first = os.index(j)
                pj_last = len(os) - os[::-1].index(pj) - 1
                if j_first < pj_last:
                    return False
    return True


def encode(parameters):
    jobs = parameters['jobs']
    jpc = parameters['jpc']

    OS = []
    i = 0
    for job in jobs:
        for op in job:
            OS.append(i)
        i = i + 1
    osdic = Counter(OS)

    newos = []
    oss = copy.deepcopy(OS)
    while True:
        if len(oss) == 0:
            break
        elif len(oss) == 1:
            index = 0
        else:
            index = random.randint(0, len(oss)-1)
        selectjob = oss[index]
        ct = 0
        if jpc[selectjob] is not None:
            for pj in jpc[selectjob]:
                ct += osdic[pj]
        if ct == 0:
            newos.append(selectjob)
            osdic[selectjob] = osdic[selectjob] - 1
            del oss[index]

    return newos

