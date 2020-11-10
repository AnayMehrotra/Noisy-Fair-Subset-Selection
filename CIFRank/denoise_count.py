import os
import time
import numpy as np

import csv, datetime
import itertools

import numpy.random as random_numpy
from scipy.stats import truncnorm

rng = random_numpy.default_rng(1234) ## this is a newer and faster rand number generator
debug = lambda str : f"print(\"{str}\",\"=\",eval(\"{str}\"))"

folder_count_data = "./out/counterfactual_data/mv/m2/"

def denoise_count(ITER):
    for i in range(1,ITER+1):
        f = open(folder_count_data+'R'+str(i)+'_denoised.csv','w')
        f.write('G,R,X,Y,GR,UID,WrongR,qw,qb,X_count,Y_count,Y_count_resolve\n')
        #
        with open(folder_count_data+"R"+str(i)+"_count.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            lcnt=0
            for row in csv_reader:
                if lcnt==0: lcnt+=1; continue;
                f.write(row[0]+','+row[6]+','+row[2]+','+row[3]+','+\
                        row[4][0]+row[6]+','+row[5]+','+row[1]+','+row[7]+','+row[8]+','+\
                        row[9]+','+row[10]+','+row[11]+'\n')
                lcnt += 1
        f.close()

    time.sleep(1)
    for i in range(1,ITER+1):
        time.sleep(0.1)
        os.system('rm '+folder_count_data+'R'+str(i)+'_count.csv')
        time.sleep(0.1)
        os.system('mv '+folder_count_data+'R'+str(i)+'_denoised.csv '+folder_count_data+'R'+str(i)+'_count.csv')

if __name__ == '__main__':
    import sys
    denoise_count(ITER=int(sys.argv[1]))
