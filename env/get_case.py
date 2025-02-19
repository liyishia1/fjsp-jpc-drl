#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 14:28
# @Author  : Yishi Li
# @Site    : 
# @File    : get_case.py
# @Software: PyCharm

import numpy as np


def randomcase(n_group=2, n_job=6, n_mas=5, max_proc=200, min_proc=100, group_shape=[1, 1, 2, 2, 3, 3]):
    n_jobs = [n_job for _ in range(n_group)]
    # group_shape = np.random.randint(min_ope, max_ope + 1, n_job)
    group_shape = group_shape
    n_ops = []
    for _ in range(n_group):
        group_s = np.copy(group_shape)
        np.random.shuffle(group_s)
        n_ops.append(group_s)
    n_machs = n_mas
    avail_machs = []
    proc_times = []
    for i in n_ops:
        for j in i[:-1]:
            for k in range(j):
                avail_machs.append(np.random.choice(n_machs - 1, 2, replace=False))
                proc_times.append(np.random.randint(min_proc, max_proc, 2))
        for k in range(i[-1]):
            avail_machs.append([n_machs - 1])
            # proc_times.append(np.random.randint(min_proc * 2, max_proc * 2, 1))
            proc_times.append(np.random.randint(min_proc, max_proc, 1))

    jobs_prec = []
    for _ in range(n_group):
        job_p = [None for _ in range(n_job - 1)]
        class2 = np.random.choice(n_job - 1, 2, replace=False)
        class3 = []
        current_integer = 0
        while len(class3) < n_job - 1:  # 生成5个元素的列表
            class3.append(current_integer)
            current_integer += 1
        # 从原始列表中删除与随机选择的两个元素相同的元素
        for element in class2:
            if element in class3:
                class3.remove(element)
        job_p[np.random.choice(class3)] = class2
        job_p.append(class3)
        jobs_prec.append(job_p)
    return n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec


def casecoll(num, n_group, n_job, n_mas, max_proc, min_proc, group_shape):
    list1 = []
    for i in range(num):
        n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec = randomcase(n_group, n_job, n_mas, max_proc, min_proc, group_shape)
        list1.append((n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec))
    return list1


