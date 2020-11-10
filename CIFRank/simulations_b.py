import os
import time
import numpy as np
import pandas as pd
import cvxpy as cp

import random
import itertools
import csv, datetime

from numpy import genfromtxt
import numpy.random as random_numpy
from scipy.stats import truncnorm

import string
import inspect
import copy, signal
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy.stats import johnsonsb, cauchy, invweibull, johnsonsu, uniform, dweibull, beta

#######################################################
# Constants
#######################################################
rng = random_numpy.default_rng(634) ## this is a newer and faster rand number generator

folder_cif_project = "./"
home_folder = folder_cif_project
folder_syn_data = folder_cif_project + "out/synthetic_data/mv/"
folder_eval_data = folder_cif_project + "out/evaluation_res/mv/m2/"
folder_count_data = folder_cif_project + "out/counterfactual_data/mv/m2/"


PYTHON = "/usr/local/Cellar/python@3.8/3.8.5/bin/python3.8 "

### Constants specific to experiments
ITER = 5
n = 100
num_bins = 20 # number of bins

ratio = np.array([0.63, 0.37]) # fixed in synthetic_data
ratio_intersectional = np.array([0.63*0.63, 0.63*0.37, 0.63*0.37, 0.37*0.37]) # fixed in synthetic_data
tau = 0.3
race = {'W': 0, 'B': 1}
inv_race = {0: 'W', 1: 'B'}


f_prob = np.array([np.array([1-tau,0+tau]), np.array([0+tau,1-tau])]) # white black // flip probability



#######################################################
# Helper functions
#######################################################


def update_f_prob():
    global f_prob
    f_prob = np.array([np.array([1-tau,0+tau]), np.array([0+tau,1-tau])]) # white black // flip probability

debug = lambda str : f"print(\"{str}\",\"=\",eval(\"{str}\"))"

rcParams.update({
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
        'figure.figsize': (10,6),
})

def file_str():
    """ Auto-generates file name."""
    now = datetime.datetime.now()
    return now.strftime("H%HM%MS%S_%m-%d-%y")

rand_string = lambda length: ''.join(random.choice(string.ascii_lowercase) for i in range(length))

def pdf_savefig():
    """ Saves figures as pdf """
    fname = file_str()+rand_string(20)
    plt.savefig(home_folder+f"/figs/{fname}.pdf")

def eps_savefig():
    """ Saves figure as encapsulated postscript file (vector format)
        so that it isn't pixelated when we put it into a pdf. """
    pdf_savefig()
#######################################################


def rand_round_solution(x,m,n):
    rx=np.array([0]*m);
    ind=rng.choice([i for i in range(len(x))], n, replace=False, p=x/np.sum(x))
    for i in ind: rx[i]=1
    rx=rx.reshape((len(rx),1))
    return np.array(rx)

#######################################################
# algorithms
#######################################################
def fair_select(w, q, u, n, lam=0, eps=0):
    # Outputs the ranking fair in expectation over the randomness in the candidates
    m = w.shape[0]
    p = q.shape[1]
    x = cp.Variable((m,1), boolean=True)
    o = np.ones((m,1))
    prob = cp.Problem( cp.Maximize(w.T*x),
                      [q.T@x<=u,o.T@x==n]);
    try:
        prob.solve(solver=cp.GUROBI, cplex_params={"timelimit": 30})
    except Exception as e:
        print(e)
        print("couldn't solve fair_select")
        return -1
    x=list(x)
    x=[y.value[0] for y in x]
    return np.array(x)


## LP based rounding algorithm
def fair_select_lp(w, q, u, n, lam=0, eps=0):
    # Outputs the ranking fair in expectation over the randomness in the candidates
    m=w.shape[0]
    p=q.shape[1]
    x = cp.Variable((m,1))
    o = np.ones((m,1))
    prob = cp.Problem( cp.Maximize(w.T*x),[q.T@x<=u,o.T@x==n,x<=1,x>=0]);
    try:
        #CPLEX automatically returns an extreme point solution
        prob.solve(solver=cp.GUROBI) #, cplex_params={"timelimit": 30})
    except Exception as e:
        print(e); print("couldn't solve fair_select_lp", flush=True)
        return -1
    x=list(x)
    x=np.array([np.clip(y.value[0],0,1) for y in x])
#     print("\nfair (frac):", np.sum(x), np.sum(x>=1e-2))
    return rand_round_solution(x,m,n)

def lagrangian_solver(w, tmpq, t, n, lam=1, eps=1e-3):
    # Outputs the ranking fair in expectation over the randomness in the candidates
    q = copy.deepcopy(tmpq)
    m=w.shape[0]
    p=q.shape[1]
    for i in range(m):
        k = np.argmax(q[i])
        q[i] =  np.zeros_like(q[i])
        q[i][k] = 1
    x = cp.Variable((m,1))
    o = np.ones((m,1)) #cp.log((q.T@x)*p), cp.norm(x, 2)
    t = np.array(t) ## this is the target now
    prob = cp.Problem( cp.Maximize(w.T*x - lam * cp.sum(cp.kl_div(q.T*x/n,t))),\
                      [o.T@x==n, x<=1, x>=0]);
    try: prob.solve(solver = cp.SCS, max_iters=10000, eps=1e-7, verbose=False) #, cplex_params={"timelimit": 30})
    except Exception as e: print(e); print("couldn't solve lagrangian_solver", flush=True); return -1
    fg=0
    if str(type(x)) == "<class 'NoneType'>" or str(type(x[0])) == "<class 'NoneType'>" or\
       str(type(x[0].value)) == "<class 'NoneType'>" or str(type(x[0].value[0])) == "<class 'NoneType'>": fg=1
    if fg: print("lagrangian didn't find a solution!"); return -1
    x=list(x)
    x=np.array([np.clip(y.value[0],0,1) for y in x])
    return rand_round_solution(x,m,n)

def thresh_select(w, q, u, n, lam=0, eps=0):
    # Outputs the fair ranking after thresholding the candidates
    m=w.shape[0]; p=q.shape[1]

    typ=[np.argmax(q[i]) for i in range(m)] # threshold

    li=[[-w[i], i] for i in range(m)]; li.sort() # sort

    y=[0]*m;sum=[0]*p;i=0;
    while(np.sum(sum)<n and i<m):
        l=typ[li[i][1]]
        if sum[l]<=u[l]: y[li[i][1]]=1; sum[l]+=1;
        i+=1

    if np.sum(sum) < n-1: raise Exception("Thresh found the problem infeasible!")

    y=np.array(y)
    y=y.reshape((len(y),1))

    return y

def blind_select(w, q, u, n, lam=0, eps=0):
    # Outputs the optimal ranking without fairness constraints
    m=w.shape[0]; p=q.shape[1]

    li=[[-w[i], i] for i in range(m)]; li.sort() # sort

    y=np.array([0]*m);
    for i in range(n): y[li[i][1]]=1
    y=y.reshape((len(y),1))
    return y
#######################################################


#######################################################
# Helper functions (Sampling candidates and utilities)
#######################################################
def gen_noise_causal(tau_int):
    global tau;
    tau = tau_int
    update_f_prob()
    #
    for i in range(1,ITER+1):
        f = open(folder_syn_data+'R'+str(i)+'_noisy.csv','w')
        f.write('G,R,X,Y,GR,UID,TrueR,qw,qb\n')
        #
        rand = [rng.choice([0,1], 2001, p=f_prob[0]), rng.choice([0,1], 2001, p=f_prob[1])]
        rng.shuffle(rand[0])
        rng.shuffle(rand[1])
        #
        with open(folder_syn_data+"R"+str(i)+".csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            lcnt=0
            cnt=0
            for row in csv_reader:
                if lcnt==0: lcnt+=1; continue;
                new_race = inv_race[rand[race[row[1]]] [lcnt]]
                if new_race != row[1]: cnt+=1
                f.write(row[0]+','+new_race+','+row[2]+','+row[3]+','+\
                        row[4][0]+new_race+','+row[5]+','+row[1]+','+str(-1)+','+str(-1)+'\n')
                lcnt += 1
            print(f"cnt({i}): ", cnt)
        f.close()
    #
    time.sleep(1)
    for i in range(1,ITER+1):
        time.sleep(0.01)
        os.system('mv '+folder_syn_data+'R'+str(i)+'.csv '+folder_syn_data+'R'+str(i)+'_copy.csv')
        time.sleep(0.01)
        os.system('mv '+folder_syn_data+'R'+str(i)+'_noisy.csv '+folder_syn_data+'R'+str(i)+'.csv')
    return

def gen_candidates_causal(ind,num_bins=num_bins):
    m = 2000
    q = np.zeros((m,2))
    qgrp = np.zeros((m,2))
    x = [["default", -1, -1] for i in range(m)]
    w = np.array([-1]*m)
    #
    df = pd.read_csv(folder_syn_data+"R"+str(ind)+"_copy.csv", sep=',')

    df['bin'] = pd.qcut(df['Y'], q=num_bins)

    ratio_bin = {}
    for i, r in df.iterrows():
        if r['bin'] not in ratio_bin: ratio_bin[r['bin']] = {'val': [-1,-1], 'W': 0, 'B': 0}
        ratio_bin[r['bin']][r['R']] += 1
        #
        for k in ratio_bin:
            ratio_bin[k]['val'][0] = ratio_bin[k]['W'] * 1.0 / (ratio_bin[k]['W'] + ratio_bin[k]['B'])
            ratio_bin[k]['val'][1] = 1 - ratio_bin[k]['val'][0]

    df_noisy = pd.read_csv(folder_syn_data+"R"+str(ind)+".csv", sep=',')
    w = np.array(df_noisy['Y']) # ORIGINAL

    for i in range(m):
        x[i][1] = df_noisy['TrueR'][i]
        x[i][2] = df_noisy['G'][i]

    df_noisy['bin'] = pd.qcut(df_noisy['Y'], q=num_bins)

    for i, c in df_noisy.iterrows():
        r = c['R']
        a = f_prob[0][int(r=='B')]*ratio_bin[c['bin']]['val'][0]
        b = f_prob[1][int(r=='B')]*ratio_bin[c['bin']]['val'][1]
        q[i][0] = a/(a+b)
        q[i][1] = 1.0 - q[i][0]

        a = f_prob[0][int(r=='B')]*ratio[0]
        b = f_prob[1][int(r=='B')]*ratio[1]

        qgrp[i][0]  = a/(a+b)
        qgrp[i][1] = 1.0 - qgrp[i][0]
        if np.isnan(q[i][0]) or np.isnan(q[i][1]):
            print(f"NAN! q[{i}]: {q[i]}")
            q[i][0] = 0.5; q[i][1] = 0.5
        if np.isnan(qgrp[i][0]) or np.isnan(qgrp[i][1]):
            print(f"NAN! qgrp[{i}]: {qgrp[i]}")
            qgrp[i][0] = 0.5; qgrp[i][1] = 0.5

        c=1 # to allow folding
    return x, w, q, qgrp

def get_fairness(num, p, fair_metric, return_raw=False, return_intersectional=False):
    n = np.sum(num);
    num = np.array(num).flatten()
    u = np.array(n * ratio)
    if return_intersectional:
        assert(return_raw)
        u = np.array(n * ratio_intersectional)

    if return_raw:
        assert(len(num)==len(u))
        return num/u
    selec_lft = False; r_diff = False; custom = False
    if fair_metric == 'selec_lft': selec_lft = True
    if fair_metric == 'r_diff': r_diff = True
    if fair_metric == 'custom': custom = True
    if selec_lft:
        mi = 2
        for i,j in itertools.product(range(p),range(p)):
            mi = min(mi, num[i]*u[j]/u[i]/num[j])
        return mi
    elif r_diff:
        mi = -1
        min_u = np.min(u) / n
        for i,j in itertools.product(range(p),range(p)):
            mi = max(mi, min_u * abs(num[i]/u[i]-num[j]/u[j]))
        return 1-mi
    elif custom:
        mi = 2
        for i in range(p):
            mi = min(mi, (n-num[i])/(n-u[i]))
        return mi
    else: print("Must choose a fairness definition!"); raise NotImplementedError

def get_causal_stats(p, ranking, fair_metric, return_raw=False, return_intersectional=False, blind_val=False):
    df = pd.read_csv(folder_eval_data+"Eval_R"+str(ITER)+"_raw_counts.csv", sep=',')
    dfval = pd.read_csv(folder_eval_data+"Eval_R"+str(ITER)+"_get_util.csv", sep=',')
    print(folder_eval_data+"Eval_R"+str(ITER)+"_raw_counts.csv")
    res=[]
    val=[]
    prop_list  = ['W', 'B'] if not return_intersectional else ['MW', 'MB', 'FW', 'FB']
    #
    for i in range(1,ITER+1):
        num = [ df[(df['rank']==ranking)&(df['k']==n)&(df['group']==r)&(df['run']==i)]['raw_counts']\
                for r in prop_list]
        num = np.array(num).T[0]
        #
        res.append(get_fairness(num, p=p, fair_metric=fair_metric, return_raw=return_raw, return_intersectional=return_intersectional))
        val.append(dfval[(dfval['rank']==ranking)&(dfval['k']==n)&(dfval['run']==i)&(dfval['group']=='VAL')]['get_util'])
    val = np.array(val) / np.array(blind_val)
    return (np.mean(res, axis=0), np.std(res, axis=0)), (np.mean(val), np.std(val))


#######################################################
# Simulation code
#######################################################
def subset_selection_causal(n=50, fair_metric='custom', lam=[1], return_raw=False, return_intersectional=False, verbose=0):
    # We compare the number of women selected by the thresholding and the fair in expectation scheme.
    # For each iteration, we sample the predictions
    # We use the following parameters: (m,n,p)

    m = 2000
    p = 2 # two genders

    algos = [fair_select_lp, thresh_select, fair_select_lp, lagrangian_solver, blind_select]
    algo_res = [[] for a in algos]
    algo_val = [[] for a in algos]
    algo_names = ["FairExpec", "Thresh", "FairExpecGrp", "MultObj", "Blind"]
    #
    prop_list = ['W', 'B']
    inter_list = ['MW', 'MB', 'FW', 'FB']
    #
    cnd=0; q=0 ;w=0;
    for i in range(1,ITER+1): # Generate candidates
        print("iter: ", i, end=",")

        st = time.time()
        cnd, w, q, qgrp = gen_candidates_causal(i)
        ave_w = np.max([1.0,np.mean(w)])


        if verbose:
            print("sample time: ", time.time()-st)
            print("Sum", [np.sum([(c[1]==l) for c in cnd]) for l in prop_list])
            print("Exp", np.sum([q[i] for i in range(m)], axis=0))

        u = np.array(n * ratio)+1 ## male, female
        t = copy.deepcopy(ratio); t = np.reshape(t, (p,1))
        w = np.reshape(w, (m,1)); q = np.reshape(q, (m,p)); u = np.reshape(u, (p,1))  ## male, female

        if verbose: eval(debug("alpha")); eval(debug("u"))

        def aggregate_stats(x,store_val=False,store_max=False,return_raw=False,return_intersectional=False):
            if return_intersectional:
                race_ind = [[(c[1]==l[1])*(c[2]==l[0]) for c in cnd] for l in inter_list]
                num_picked = [np.dot(race_ind[l],x)  for l in range(len(inter_list))]
                return get_fairness(num_picked, p=len(inter_list), fair_metric=fair_metric, return_raw=return_raw,return_intersectional=return_intersectional)
            #
            race_ind = [[(c[1]==l) for c in cnd] for l in prop_list]
            num_picked = [np.dot(race_ind[l],x)  for l in range(p)]

            val = np.sum([x[i]*w[i] for i in range(m)]) / np.sum(x) * n

            if store_val: return val
            elif store_max: return np.max(num_picked)
            else: return get_fairness(num_picked, p=p, fair_metric=fair_metric, return_raw=return_raw, return_intersectional=False)

        # Run algorithms
        fg=1
        st = time.time()
        cnt = [0 for i in range(len(algos))]
        slack = [0 for i in range(len(algos))]; slack[3] = 1.0
        while fg: # running algorithms
            fg=0
            for j, algo in enumerate(algos):
                if verbose: print('trying'+algo_names[j], end=" ")
                try:
                    if j == 3: x = algo(w,q,t,n,lam=lam[j]*ave_w/slack[j], eps=1e-2)
                    elif j == 2: x = algo(w,qgrp,u+slack[j],n,lam=lam[j]*ave_w, eps=1e-2)
                    else: x = algo(w,q,u+slack[j],n,lam=lam[j]*ave_w, eps=1e-2)
                    #
                    if type(x) != type(-1): x=list(x)
                    else: fg=1; print(algo_names[j]+" returned invalid solution\n")
                    if not fg and np.sum(x) > n*3/2: fg=1; print(algo_names[j]+" returned invalid solution2\n")
                except Exception as exc: fg=1; print(exc); print(algo_names[j]+'looping due to timeout')
                if fg and cnt[j] < 10:
                    for k in range(j): algo_res[k]=algo_res[k][:-1]; algo_val[k]=algo_val[k][:-1]; print(k)
                    print(f"breaking {j}, {algo_names[j]}!", slack[j])

                    slack[j] += 10
                    cnt[j] += 1
                    break
                elif fg: # some algorithm could not find the solution
                    print(f"{algo_names[j]} FAILED to solve! Putting NAN.")
                    algo_res[j].append(np.array([np.nan]))
                    algo_val[j].append(np.array([np.nan]))
                    fg=0
                else:
                    if verbose: print(algo_names[j],": ",np.sum(x), end= " | ")

                    algo_res[j].append(aggregate_stats(x,store_val=False,return_raw=return_raw,\
                                                    return_intersectional=return_intersectional))
                    algo_val[j].append(aggregate_stats(x,store_val=True))
        if verbose: print("solve time ", time.time()-st)
    #
    out = []
    print([len(res) for res in algo_res], [len(val) for val in algo_val])

    for v in algo_res: out.append([np.mean(v, axis=0), np.std(v, axis=0)]); print(v); print(out[-1])

    algo_val[4]=np.array(algo_val[4]) # for blind
    for i, v in enumerate(algo_val):
        if i == 2: out.append([1,0]); continue
        v=np.array(v);
        v=v/algo_val[4]; # for blind
        out.append([np.mean(v), np.std(v)])

    return out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], algo_val[4]

def experiment_causal(n=50, fair_metric='r_diff', verbose=0, save=1):
    os.system("rm "+folder_count_data+"*.csv")
    os.system("rm "+folder_cif_project+"*.mark")
    st = time.time()

    # Initialize
    m = 2000
    p = 2
    range_tau = np.linspace(0,0.5,3)

    algo_name = ['FairExpec', 'Thrsh', 'FairExpecGrp', 'MultObj', 'Blind', 'Causal-Resolving', 'Causal']
    algo_col = ['b', 'r', 'm', 'y', 'g', 'c', '#8c564b']
    algo_res = [{'mean':[], 'std':[]} for x in algo_name]
    algo_val = [{'mean':[], 'std':[]} for x in algo_name]
    results = algo_res + algo_val
    #
    st1 = time.time()
    os.system(folder_cif_project+f"custom_generate.sh {ITER}")
    while not os.path.isfile(folder_cif_project+"done_generation.mark"): time.sleep(1)
    os.system("rm "+folder_cif_project+"done_generation.mark")
    print("done generation")
    print("Time for datagen: ", time.time()-st1) #63.87 sec

    proc = min(5, ITER)

    for i in range(1, ITER+1, proc):
        st2 = time.time()
        for j in range(i, i+proc):
            os.system(folder_cif_project+f"custom_rscript.sh {ITER} {j} > tmp.txt 2>&1 &")
        for j in range(i, i+proc):
            while not os.path.isfile(folder_cif_project+f"done_causal_rsciprt{j}.mark"): time.sleep(1)
            os.system("rm "+folder_cif_project+f"done_causal_rsciprt{j}.mark")
        print(f"Time for {proc} mediations: ", time.time()-st2)

    # Run experiment
    for i, tau_int in enumerate(range_tau):

        st1 = time.time()
        os.system(folder_cif_project+f"custom_generate.sh {ITER}")
        while not os.path.isfile(folder_cif_project+"done_generation.mark"): time.sleep(1)
        os.system("rm "+folder_cif_project+"done_generation.mark")
        print("done generation")
        print("Time for datagen: ", time.time()-st1)

        gen_noise_causal(tau_int) # Generate noisy data
        print("done flipping") # ORIGINAL
        #
        os.system(folder_cif_project+f"custom_exp.sh {ITER} ")
        print("done causal exp")
        while not os.path.isfile(folder_cif_project+"done_causal.mark"): time.sleep(1)
        os.system("rm "+folder_cif_project+"done_causal.mark")

        st3 = time.time()
        temp = subset_selection_causal(n=n, fair_metric=fair_metric, lam=[0,0,0,500,0], verbose=verbose)

        # Store results (from subset_selection_causal, i.e., for FairExpec, Thresh, Blind, FairExpecGrp, MultObj)
        print("-"*25, f"tau={tau_int}", "-"*25)
        for j, xyz in enumerate(range(5)): results[xyz]['mean'].append(temp[j][0]); results[xyz]['std'].append(temp[j][1])
        for j, xyz in enumerate(range(7,12)): results[xyz]['mean'].append(temp[j+5][0]); results[xyz]['std'].append(temp[j+5][1])
        blind_val = temp[10]

        print("Time for one run of experiment: ", time.time()-st3)

        # Store results (of algorithms)
        tmp1, tmp2 = get_causal_stats(p, ranking='Y_count_resolve', fair_metric=fair_metric, blind_val=blind_val)
        algo_res[-2]['mean'].append(tmp1[0]); algo_res[-2]['std'].append(tmp1[1])
        algo_val[-2]['mean'].append(tmp2[0]); algo_val[-2]['std'].append(tmp2[1])

        tmp1, tmp2 = get_causal_stats(p, ranking='Y_count', fair_metric=fair_metric, blind_val=blind_val)
        algo_res[-1]['mean'].append(tmp1[0]); algo_res[-1]['std'].append(tmp1[1])
        algo_val[-1]['mean'].append(tmp2[0]); algo_val[-1]['std'].append(tmp2[1])

        for i, res in enumerate(algo_res): print(algo_name[i], ": ", res, algo_val[i])

    def plot(miny=0.5):
        x=np.array(range_tau)
        fig, ax = plt.subplots()
        ################################################
        # Make fairness vs tau plot
        ################################################
        algo_name = ['FairExpec', 'Thrsh', 'FairExpecGrp', 'MultObj', 'Blind', 'Causal-Resolving', 'Causal']
        for j in [0,2,3,1,5,6,4]:
            # for j, res in enumerate(algo_res):
            res = algo_res[j]
            val = algo_val[j]
            if miny == 0.8: print(algo_name[j], " (fairness): ", res['mean'])
            if miny == 0.8: print(algo_name[j], " (val): ", val['mean'])
            res['mean'] = np.array(res['mean'])
            res['std'] = np.array(res['std'])

            mask = np.isfinite(res['mean'])
            plt.errorbar(x[mask], res['mean'][mask], yerr=res['std'][mask]/np.sqrt(ITER), color=algo_col[j], label=algo_name[j], linewidth=4, alpha=0.7)

        ## Decorate
        plt.ylim(miny,1.0)
        plt.ylabel('Fairness achieved ($\\mathcal{F}$)', fontsize=25)

        legend = plt.legend(loc='best', shadow=False, fontsize=15)
        plt.xlabel('Noise ($\\tau$)',fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title(f'Causal experiment ($n=${n}) | iter={ITER} | metric={fair_metric}', fontsize=15)

        ## Show or save
        if True: pdf_savefig()
        else: plt.show()

        ################################################
        # Make value vs tau plot
        ################################################
        x = np.array(range_tau)
        fig, ax = plt.subplots()
        ## Plot: fairness vs tau
        for j in [0,2,3,1,5,6,4]:
            val = algo_val[j]

            val['mean'] = np.array(val['mean'])
            val['std'] = np.array(val['std'])
            mask = np.isfinite(val['mean'])
            plt.errorbar(x[mask], val['mean'][mask], yerr=val['std'][mask]/np.sqrt(ITER), color=algo_col[j], label=algo_name[j], linewidth=4, alpha=0.7)

        ## Decorate
        plt.ylim(miny,1.0)
        plt.ylabel('Utility Ratio ($\\mathcal{K}$)', fontsize=25)

        legend = plt.legend(loc='best', shadow=False, fontsize=15)
        plt.xlabel('Noise ($\\tau$)',fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title(f'Causal experiment ($n=${n}) | iter={ITER} | metric={fair_metric}', fontsize=15)

        ## Show or save
        if True: pdf_savefig()
        else: plt.show()
    #
    plot(0.5)
    plot(0.25)
    plot(0.0)
    #
    print("time taken: ", time.time()-st)

def experiment_causal_bar(n=50, fair_metric='r_diff', verbose=0, save=1):
    os.system("rm "+folder_count_data+"*.csv")
    os.system("rm "+folder_cif_project+"*.mark")
    st = time.time()

    # Initialize
    m = 2000
    p = 2
    range_tau = [0, 0.2, 0.5]

    algo_name = ['FairExpec', 'Thrsh', 'FairExpecGrp', 'MultObj', 'Blind', 'Causal-Resolving', 'Causal']
    algo_col = ['b', 'r', 'm', 'y', 'g', 'c', '#8c564b']

    # Run experiment
    for i, tau_int in enumerate(range_tau):
        #
        algo_res = [{'mean':[], 'std':[]} for x in algo_name]
        algo_val = [{'mean':[], 'std':[]} for x in algo_name]
        results = algo_res + algo_val

        st1 = time.time()
        os.system(folder_cif_project+f"custom_generate.sh {ITER}")
        while not os.path.isfile(folder_cif_project+"done_generation.mark"): time.sleep(1)
        os.system("rm "+folder_cif_project+"done_generation.mark")
        print("done generation")
        print("Time for datagen: ", time.time()-st1)

        proc = min(5, ITER)

        for i in range(1, ITER+1, proc):
            st2 = time.time()
            for j in range(i, i+proc):
                os.system(folder_cif_project+f"custom_rscript.sh {ITER} {j} > tmp.txt 2>&1 &")
            for j in range(i, i+proc):
                while not os.path.isfile(folder_cif_project+f"done_causal_rsciprt{j}.mark"): time.sleep(1)
                os.system("rm "+folder_cif_project+f"done_causal_rsciprt{j}.mark")
            print(f"Time for {proc} mediations: ", time.time()-st2)

        gen_noise_causal(tau_int) # Generate noisy data
        print("done flipping") # ORIGINAL

        os.system(folder_cif_project+f"custom_exp.sh {ITER} ")
        print("done causal exp")
        while not os.path.isfile(folder_cif_project+"done_causal.mark"): time.sleep(1)
        os.system("rm "+folder_cif_project+"done_causal.mark")

        def get_result(return_raw=False, return_intersectional=False):

            st3 = time.time()
            temp = subset_selection_causal(n=n, fair_metric=fair_metric, lam=[0,0,0,500,0], return_raw=return_raw, return_intersectional=return_intersectional, verbose=verbose)

            # Store results (from subset_selection_causal, i.e., for FairExpec, Thresh, Blind, FairExpecGrp, MultObj)
            print("-"*25, f"tau={tau_int}", "-"*25)
            for j, xyz in enumerate(range(5)): results[xyz]['mean'] = temp[j][0]; results[xyz]['std'] = temp[j][1]

            print("Time for one run of experiment: ", time.time()-st3)
            # Store results (of algorithms)
            tmp1, tmp2 = get_causal_stats(p, ranking='Y_count_resolve', fair_metric=fair_metric, return_raw=return_raw, return_intersectional=return_intersectional)
            algo_res[-2]['mean'] = tmp1[0]; algo_res[-2]['std'] = tmp1[1]

            tmp1, tmp2 = get_causal_stats(p, ranking='Y_count', fair_metric=fair_metric, return_raw=return_raw, return_intersectional=return_intersectional)
            algo_res[-1]['mean'] = tmp1[0]; algo_res[-1]['std'] = tmp1[1]

            for i, res in enumerate(algo_res): print(algo_name[i], ": ", res)

        def plot(miny=0.5, x_lab=['W', 'B']):
            x_pos = np.array(list(range(len(x_lab))))*(1+0.05*(len(x_lab)-2))
            xtick1 = []; xtick2 = []

            fig, ax = plt.subplots()

            ## Plot: fairness vs tau
            for j in [0,2,3,1,5,6,4]:
                res = algo_res[j]
                x_pos += 3+1.0*(len(x_lab)-2)
                if miny == 0.8: print(algo_name[j], " (fairness): ", res['mean'])
                res['mean'] = np.array(res['mean'])
                res['std'] = np.array(res['std'])
                plt.bar(x_pos, res['mean'], yerr=res['std']/np.sqrt(ITER), color=algo_col[j], label=f"{algo_name[j]}", linewidth=4, alpha=0.7)

                ## Decorate
                xtick1 += list(x_pos)
                xtick2 += list(x_lab)

            xx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 10)
            plt.plot(xx, np.ones_like(xx), color='black', linewidth=2)
            plt.legend(loc='best', shadow=False, fontsize=15)
            #
            plt.ylim(0.0,3.0)
            plt.ylabel(f'Selection rate', fontsize=25)
            #
            plt.xlabel('Race',fontsize=25)
            plt.xticks(xtick1, xtick2)
            plt.tick_params(axis='both', which='major', labelsize=16*2.0/len(x_lab))

            # Test C on N data\n Test FE on N data + {num_bins} eq-sz bins [[{f_prob[0][0]},{f_prob[0][1]}],[{f_prob[1][0]},{f_prob[1][1]}]]
            plt.title(f'Causal exp | $n=${n} | iter={ITER} | {num_bins} eq-sz bins | [[{f_prob[0][0]},{f_prob[0][1]}],[{f_prob[1][0]},{f_prob[1][1]}]].\nNoise: $\\tau=${tau_int}', fontsize=25)
            if True: pdf_savefig()
            else: plt.show()

        get_result(return_raw=True, return_intersectional=False)
        plot(0.0, x_lab=['W', 'B']) #Plot: fairness vs tau

    print("time taken: ", time.time()-st)


if __name__ == '__main__':
    import sys
    print(sys.argv[1])
    if sys.argv[1] == 'gen_noise':
        gen_noise_causal()
    elif sys.argv[1] == 'run_exp':
        verbose=int(eval(sys.argv[3]))
        save=int(eval(sys.argv[4]))
        n=int(sys.argv[2])

        # Generate Figure 2 and 13 (Risk Difference)
        experiment_causal(n=n, fair_metric='r_diff', verbose=verbose, save=save)
        # Generate Figure 12 (Selection lift)
        experiment_causal(n=n, fair_metric='selec_lft', verbose=verbose, save=save)
        # Generate Figure 14
        experiment_causal_bar(n=n, fair_metric='r_diff', verbose=verbose, save=save)
