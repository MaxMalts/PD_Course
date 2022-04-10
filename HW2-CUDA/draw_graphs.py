#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from subprocess import Popen, PIPE, STDOUT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", font_scale=2.5)

graphsDir = "graphs"
programNames = [
    #"01-add",
    #"02-mul",
    #"03-matrix-add",
    #"04-matrix-vector-mul",
    "05-scalar-mul",
    #"06-cosine-vector",
    #"07-matrix-mul"
]

blockSizes = [2 ** x for x in range(5, 11)] #[32] + range(128, 1025, 128)
nElements = [1] + range(10000, 1000001, 18000) #[10 ** x for x in range(1, 7)]

curTimes = [0] * len(nElements)
for programName in programNames:
    for blockSize in blockSizes:
        for i in range(len(nElements)):
            proc = Popen(["./" + programName], stdout=PIPE, stdin=PIPE, stderr=STDOUT, universal_newlines=True)
            proc_stdout = proc.communicate(input=str(nElements[i]) + ' ' + str(blockSize))[0]
            curTimes[i] = float(proc_stdout.decode())
        
        plt.figure(figsize=(16, 9))
        plt.title("Time on number of elements with block size " + str(blockSize))
        plt.xlabel("Number of elements")
        plt.ylabel("Time, ms")
        plt.plot(nElements, curTimes, linewidth=3)
        
        curDir = graphsDir + '/' + programName + '/'
        if not os.path.exists(curDir):
            os.makedirs(curDir)
        plt.savefig(curDir + "block_size_" + str(blockSize) + ".png")
        plt.close()
        
        print(curDir + "block_size_" + str(blockSize) + ".png" + " created.")