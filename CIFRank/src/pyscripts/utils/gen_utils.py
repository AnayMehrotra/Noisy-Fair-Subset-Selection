import numpy as np
import pandas as pd
from utils.basic_utils import writeToCSV
"""
    The below generation code comes from MirrorDataGenerator. Details in https://github.com/DataResponsibly/MirrorDataGenerator.

"""

def generateScores(x, att_formula, miu_dict, var_dict, value_dict=None, sensi_cols=None, inter_miu=0, inter_var=0):
    if value_dict:
        sensi_cols = list(value_dict.keys())
    else:
        if not sensi_cols:
            print("Need to specifiy sensitive attributes!")
            exit()

    cur_col = list(att_formula.keys())[0]
    score = np.zeros(x.shape[0])
    inter_flag = 0
    if sum(att_formula[cur_col].values()) > 0:
        for ai, ai_weight in att_formula[cur_col].items():
            if ai in sensi_cols:
                if isinstance(miu_dict[ai][x[ai].iloc[0]], dict):
                    cur_miu = miu_dict[ai][x[ai].iloc[0]][cur_col]
                else:
                    cur_miu = miu_dict[ai][x[ai].iloc[0]]
                cur_var = var_dict[ai][x[ai].iloc[0]]
                if ai == sensi_cols[0]: # record the value of G
                    inter_flag = x[ai].iloc[0]
                else:
                    ## the following if statement assumes there are at most 2 sensitive attributes
                    if value_dict[sensi_cols[0]].index(inter_flag) and value_dict[sensi_cols[1]].index(x[ai].iloc[0]): # for intersectional group, recall that 1 represent non-privileged group
                        cur_miu = cur_miu + inter_miu
                        cur_var = cur_var + inter_var
                score += ai_weight * np.random.normal(cur_miu, np.sqrt(cur_var), x.shape[0])
            else:
                score += ai_weight * x[ai].iloc[0] ## no noise for non-sensitive attributes
    else: # for independent column, e.g. U
        score += np.random.normal(miu_dict[cur_col], np.sqrt(var_dict[cur_col]), x.shape[0])
    x[cur_col] = score
    return x

def generate_data(file_name, para_dict, src_data=None):
    ## para_dict is the parameters of the causal model

    ## this if loop kust generates the sensitive attributes
    if "values" in para_dict:
        # generate the categorical columns for synthetic data
        dataset = pd.DataFrame()
        categoricalData = pd.DataFrame(columns=para_dict["values"])
        for ai, ai_values in para_dict["values"].items():
            ## ai is attribte, and attribte values, e.g., G, [M,F]
            ai_col = []
            for vi in ai_values:
                # vi_n fraction of elements with attribute vi
                vi_n = sum([para_dict["quotas_budget"][x] for x in para_dict["quotas_budget"] if len(x)==2 and vi in x])
                # convert frac to integer
                ai_size = int(np.ceil(para_dict["N"] * vi_n))
                # add these items.
                ai_col += [vi for _ in range(ai_size)] # or [vi] * ai_size
            np.random.shuffle(ai_col)
            categoricalData[ai] = ai_col ## for each  attribute we have a set of random candidates now.
        if categoricalData.shape[0] != para_dict["N"]:
            categoricalData.sample(n=para_dict["N"])
        # add categorical columns to dataset
        dataset = pd.concat([dataset, categoricalData], axis=1)
    else: # for semi real data
        dataset = pd.read_csv(src_data)

    # generate the continous columns
    for cur_formula in para_dict["edge_weights"]:
        key_col = list(cur_formula.keys())[0]
        if len(cur_formula[key_col]) > 1:  # multiple dependent columns
            if "values" in para_dict:
                dataset = dataset.groupby(list(cur_formula[key_col].keys())).apply(
                    lambda x: generateScores(x, cur_formula, para_dict["mius"], para_dict["vars"],\
                    value_dict=para_dict["values"], inter_miu=para_dict["intersectionality"]["miu"],\
                    inter_var=para_dict["intersectionality"]["var"]))
            else:
                dataset = dataset.groupby(list(cur_formula[key_col].keys())).apply(
                    lambda x: generateScores(x, cur_formula, para_dict["mius"], para_dict["vars"], sensi_cols=["G", "R"]))
        else:  # single dependent column
            depend_col, depend_weight = list(cur_formula[key_col].items())[0]
            dataset[key_col] = dataset[depend_col] * depend_weight

    if "values" in para_dict:
        dataset["".join(list(para_dict["values"].keys()))] = dataset.apply(lambda x: "".join([x[i] for i in para_dict["values"]]), axis=1)
        dataset["UID"] = list(range(1, para_dict["N"]+1))
        
        dataset = dataset[[x for x in dataset.columns if x not in ["X_i", "Y_d"]]]
    else:
        # rename columns
        for si in ["G", "R"]:
            dataset[si] = dataset[si].apply(lambda x: x[0].upper())
        dataset.drop(columns=['Y', 'X'], inplace=True)
        for new_coli in [list(x.keys())[0] for x in para_dict["edge_weights"]]:
            if "_" in new_coli:
                dataset.rename(columns={new_coli: new_coli[0]}, inplace=True)

    # save dataframe to a csv file
    writeToCSV(file_name, dataset)
