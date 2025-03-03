#!/usr/bin/env python

# This module creates a population of random OS and MS chromosomes

import random
from src import config
from src.utils.util import *

def generateOS(parameters):

    counter = 0
    while True:
        OS = encode(parameters)
        istrue = jpc_check(OS, parameters['jpc'])
        if istrue == True:
            return OS
        else:
            counter += 1
    # print('poc-5000 can not find jpc os')


def generateMS(parameters):
    jobs = parameters['jobs']

    MS = []
    for job in jobs:
        for op in job:
            randomMachine = random.randint(0, len(op)-1)
            MS.append(randomMachine)

    return MS


def initializePopulation(parameters):
    gen1 = []

    for i in range(config.popSize):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        gen1.append((OS, MS))

    return gen1
