#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/19 19:47
# @Author  : Yishi Li
# @Site    :
# @File    : test.py
# @Software: PyCharm

import pandas as pd
import re
import numpy as np
import ast


def convert_to_array(match):
    array_str = match.group(1)
    # 将array字符串转换为NumPy数组
    array = np.array(ast.literal_eval(array_str))
    # 将NumPy数组转换为字符串表示
    return str(array.tolist())


# 将字符串转换为列表
def convert(element):
    try:
        return ast.literal_eval(element)
    except:
        return element


def list_converted(elem):
    # 正则表达式匹配array对象
    pattern = r'array\((.*?)\)'
    # 将字符串中的array对象转换为字符串表示
    str_data_fixed = re.sub(pattern, lambda x: convert_to_array(x), elem)
    nested_list = ast.literal_eval(str_data_fixed)
    nested_list_converted = [[convert(element) for element in sublist] for sublist in nested_list]
    return nested_list_converted


def getdata(tag='2-12-5'):
    list1 = [tag]
    # result1 = []
    result2 = []
    for i in list1:
        df = pd.read_excel('data_dev/{}.xlsx'.format(i), index_col=0)
        for index, row in df.iterrows():
            n_jobs = ast.literal_eval(row['n_jobs'])
            n_machs = row['n_machs']
            n_ops = list_converted(row['n_ops'])
            avail_machs = list_converted(row['avail_machs'])
            proc_times = list_converted(row['proc_times'])
            jobs_prec = list_converted(row['jobs_prec'])
            result2.append((n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec))
            # I, M, J, K, T, P = utils.util.tote(n_jobs, n_machs, n_ops, avail_machs, proc_times, jobs_prec)
            # result1.append((I, M, J, K, T, P))

    return result2


def getresult(tag='2-12-5'):
    result1 = []
    for i in range(100):
        df = pd.read_excel('data_dev/imitate/{}-milp-{}.xlsx'.format(tag, i), index_col=0)
        result1.append(df)
    return result1