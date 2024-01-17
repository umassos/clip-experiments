# experiment implementations for CLIP
# January 2024

import sys
import random
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lambertw
import seaborn as sns
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True,precision=3)

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

import matplotlib.style as style
style.use('tableau-colorblind10')
# style.use('seaborn-v0_8-paper')

import implementations as f
import clip as c


def experiment(U, d, beta):
    global f
    global c
    #################################### set up experiment parameters

    # get the parameters from the command line

    # lower and upper bounds
    L = 1

    # dimension of the decision space
    # d = int(sys.argv[2])

    # # beta, the maximum weight in the weighted L1 norm
    # beta = int(sys.argv[3])

    # generate the weight vector for the switching cost
    weights = np.random.uniform(0, beta, d)

    # specify std, the standard deviation (essentially the variance)
    # for each randomly generated cost function w.r.t the dimensions.
    std = (U)/5

    # specify adversarial factor xi
    # xi = float(sys.argv[5])

    # specify the number of instances to generate
    epochs = 1500

    opts = []
    roros = []
    lazys = []
    agnostics = []
    constThresholds = []
    minimizers = []
    clip2s = []
    baseline2s = []

    cost_opts = []
    cost_roros = []
    cost_lazys = []
    cost_agnostics = []
    cost_constThresholds = []
    cost_minimizers = []
    cost_clip2s = []
    cost_baseline2s = []

    alpha = 1 / (1 - (2*beta/U) + lambertw( ( ( (2*beta/U) + (L/U) - 1 ) * math.exp(2*beta/U) ) / math.e ) )

    for _ in range(epochs):
        #################################### generate cost functions (a sequence)

        # randomly generate $T$ for each instance (the integer deadline)
        T = np.random.randint(6, 24)

        # generate sequences of cost functions
        # each cost function is a random weight vector associated with it
        cost_functions = []
        for i in range(T):
            cost_functions.append(f.generate_cost_function(L, U, std, d))

        # compute L and U based on cost functions
        Lc = np.min([np.min(cost_functions[i]) for i in range(T)])
        Uc = np.max([np.max(cost_functions[i]) for i in range(T)])

        #################################### solve for the optimal solution

        # try to solve for the optimal solution using scipy minimization
        sol, solCost = f.optimalSolution(cost_functions, weights, d)
        # x_opt = sol.reshape((T, d))
        # print(sol)
        # print(x_opt)
        # print(solCost)
        # print(np.sum(x_opt))
        # obtain some advice based on the optimal solution

        # #################################### get the "bad" solution
        # bad, badCost = f.adversarialSolution(cost_functions, weights, d)
        adv = (sol).reshape((T, d))

        #################################### get the online RORO solution

        roro, roroCost = f.RORO(cost_functions, weights, d, Lc, Uc)

        #################################### get the online comparison solutions

        lazy, lazyCost = f.lazyAgnostic(cost_functions, weights, d)
        agn, agnCost = f.agnostic(cost_functions, weights, d)

        const, constCost = f.threshold(cost_functions, weights, d, L, U)

        mini, miniCost = f.moveMinimizer(cost_functions, weights, d)

        #################################### get the online baseline solution
        roro = np.array(roro)
        adv = np.array(adv)

        # epsilon = 0
        alpha = (1 / (1 - (2*beta/U) + lambertw( ( ( (2*beta/U) + (L/U) - 1 ) * math.exp(2*beta/U) ) / math.e ) )).real
        # lamda = (alpha - 1 - epsilon) / (alpha - 1)
        # baseline0 = (lamda*adv + (1-lamda)*roro).reshape((T, d))
        # baseline0Cost = f.objectiveFunction(baseline0, cost_functions, weights, d)

        epsilon = 2
        lamda = (alpha - 1 - epsilon) / (alpha - 1)
        baseline2 = (lamda*adv + (1-lamda)*roro).reshape((T, d))
        baseline2Cost = f.objectiveFunction(baseline2, cost_functions, weights, d)

        # epsilon = 5
        # lamda = (alpha - 1 - epsilon) / (alpha - 1)
        # baseline5 = (lamda*adv + (1-lamda)*roro).reshape((T, d))
        # baseline5Cost = f.objectiveFunction(baseline5, cost_functions, weights, d)

        # epsilon = 10
        # lamda = (alpha - 1 - epsilon) / (alpha - 1)
        # baseline10 = (lamda*adv + (1-lamda)*roro).reshape((T, d))
        # baseline10Cost = f.objectiveFunction(baseline10, cost_functions, weights, d)

        #################################### get the online CLIP solution
        # epsilon = 0
        # clip0, clip0Cost = adv, f.objectiveFunction(adv, cost_functions, weights, d)

        epsilon = 2
        clip2, clip2Cost = c.CLIP(cost_functions, weights, d, Lc, Uc, adv, epsilon)

        # epsilon = 5
        # clip5, clip5Cost = c.CLIP(cost_functions, weights, d, Lc, Uc, adv, epsilon)

        # epsilon = 10
        # clip10, clip10Cost = c.CLIP(cost_functions, weights, d, Lc, Uc, adv, epsilon)

        if roroCost > alpha*solCost:
            print("bug in RORO")
            print(sol)
            print(roro)
            print("alpha: ", alpha)
            print("cost: ", solCost)
            print("feasibility: ", np.sum(sol))
            print("roro cost: ", roroCost)
            print("roro feasibility: ", np.sum(roro))
            break

        if clip2Cost > 3*solCost:
            print("bug in CLIP")
            print("Advice: ")
            print(adv)
            print("CLIP: ")
            print(np.array(clip2))
            print("cost functions:")
            print(np.array(cost_functions))
            print("alpha: ", alpha)
            print("cost: ", f.objectiveFunction(adv, cost_functions, weights, d))
            print("feasibility: ", np.sum(adv))
            print("clip2 cost: ", f.objectiveFunction(np.array(clip2), cost_functions, weights, d))
            print("clip2 feasibility: ", np.sum(clip2))
            break

        opts.append(sol)
        roros.append(roro)
        lazys.append(lazy)
        agnostics.append(agn)
        constThresholds.append(const)
        minimizers.append(mini)
        # clip0s.append(clip0)
        clip2s.append(clip2)
        # clip5s.append(clip5)
        # clip10s.append(clip10)
        # baseline0s.append(baseline0)
        baseline2s.append(baseline2)
        # baseline5s.append(baseline5)
        # baseline10s.append(baseline10)

        cost_opts.append(solCost)
        cost_roros.append(roroCost)
        cost_lazys.append(lazyCost)
        cost_agnostics.append(agnCost)
        cost_constThresholds.append(constCost)
        cost_minimizers.append(miniCost)
        cost_clip2s.append(clip2Cost)
        cost_baseline2s.append(baseline2Cost)


    # compute competitive ratios
    cost_opts = np.array(cost_opts)
    cost_roros = np.array(cost_roros)
    cost_lazys = np.array(cost_lazys)
    cost_agnostics = np.array(cost_agnostics)
    cost_constThresholds = np.array(cost_constThresholds)
    cost_minimizers = np.array(cost_minimizers)
    cost_clip2s = np.array(cost_clip2s)
    cost_baseline2s = np.array(cost_baseline2s)

    crRORO = cost_roros/cost_opts
    crLazy = cost_lazys/cost_opts
    crAgnostic = cost_agnostics/cost_opts
    crConstThreshold = cost_constThresholds/cost_opts
    crMinimizer = cost_minimizers/cost_opts
    crClip2 = cost_clip2s/cost_opts
    crBaseline2 = cost_baseline2s/cost_opts

    # save the results (use a dictionary)
    results = {"opts": opts, "roros": roros, "lazys": lazys, "agnostics": agnostics, "constThresholds": constThresholds, "minimizers": minimizers, "clip2s": clip2s, "baseline2s": baseline2s,
                "cost_opts": cost_opts, "cost_roros": cost_roros, "cost_lazys": cost_lazys, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_minimizers": cost_minimizers, "cost_clip2s": cost_clip2s, "cost_baseline2s": cost_baseline2s}
    with open("CDF/cdf_results_d{}_s{}.pickle".format(d, beta), "wb") as file:
        pickle.dump(results, file)

    #legend = ["ALG1", "lazy agnostic", "agnostic", "simple threshold", "move to minimizer", "CLIP[$\\epsilon=0.1$]", "CLIP[$\\epsilon=2$]", "CLIP[$\\epsilon=5$]", "CLIP[$\\epsilon=10$]"]
    # legend = ["ALG1", "CLIP[$\\epsilon=0$]", "CLIP[$\\epsilon=2$]", "CLIP[$\\epsilon=5$]", "CLIP[$\\epsilon=10$]", "baseline[$\\epsilon=0$]", "baseline[$\\epsilon=2$]", "baseline[$\\epsilon=5$]", "baseline[$\\epsilon=10$]"]

    # print mean and 95th percentile of each competitive ratio
    print("Switching Cost: {}".format(beta))
    print("RORO: ", np.mean(crRORO), np.percentile(crRORO, 95))
    print("lazy agnostic: ", np.mean(crLazy), np.percentile(crLazy, 95))
    print("agnostic: ", np.mean(crAgnostic), np.percentile(crAgnostic, 95))
    print("simple threshold: ", np.mean(crConstThreshold), np.percentile(crConstThreshold, 95))
    print("move to minimizer: ", np.mean(crMinimizer), np.percentile(crMinimizer, 95))
    print("clip2: ", np.mean(crClip2), np.percentile(crClip2, 95))
    print("baseline2: ", np.mean(crBaseline2), np.percentile(crBaseline2, 95))
    print("alpha bound: ", alpha)



if __name__ == "__main__":
    # Us = np.arange(50, 1250, 100)
    ds = np.arange(5, 22, 4)
    betas = np.arange(0, 110, 10)
    U = 250
    for d in tqdm(ds):
        for beta in tqdm(betas):
            experiment(U, d, beta)