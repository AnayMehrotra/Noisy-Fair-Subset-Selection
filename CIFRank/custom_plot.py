import os
import time
import numpy as np

import csv, datetime
import itertools

import numpy.random as random_numpy
from scipy.stats import truncnorm

debug = lambda str : f"print(\"{str}\",\"=\",eval(\"{str}\"))"
rng = random_numpy.default_rng(1234) ## this is a newer and faster rand number generator

folder_eval_data = "./out/evaluation_res/mv/m2/"

def denoise_count(file, ranking, race):
    #
    with open(folder_eval_data+file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        res = []
        for row in csv_reader:
            if row[1] == ranking and row[2] == "200" and row[3] == race:
                res.append(float(row[4]))
    print(f"{file} ({ranking}) | R={race}: mean={np.mean(res)}, std={np.std(res)}")

file = "Eval_R20_select_rate_count.csv"
denoise_count(file, ranking="Y", race="M")
denoise_count(file, ranking="Y", race="F")

denoise_count(file, ranking="Y_count", race="M")
denoise_count(file, ranking="Y_count", race="F")

denoise_count(file, ranking="Y_count_resolve", race="M")
denoise_count(file, ranking="Y_count_resolve", race="F")


file = "Eval_R20_selection_lift_count.csv"
denoise_count(file, ranking="Y", race="XYZ")
denoise_count(file, ranking="Y_count", race="XYZ")
denoise_count(file, ranking="Y_count_resolve", race="XYZ")
