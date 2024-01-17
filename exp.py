# robust experiment implementations for robust algorithms
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
import sys
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True,precision=3)

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import implementations as f
import clip as c

import matplotlib.style as style
# style.use('tableau-colorblind10')
style.use('seaborn-v0_8-paper')

#################################### set up experiment parameters

# get the parameters from the command line

# lower and upper bounds
L = 1
U = float(sys.argv[1]) * L

# dimension of the decision space
d = int(sys.argv[2])

# beta, the maximum weight in the weighted L1 norm
beta = int(sys.argv[3])

# generate the weight vector for the switching cost
weights = np.random.uniform(0, beta, d)

# specify std, the standard deviation (essentially the variance)
# for each randomly generated cost function w.r.t the dimensions.
std = (U)/int(sys.argv[4])

# specify the number of instances to generate
epochs = 1000

opts = []
roros = []
lazys = []
agnostics = []
constThresholds = []
minimizers = []
cost_opts = []
cost_roros = []
cost_lazys = []
cost_agnostics = []
cost_constThresholds = []
cost_minimizers = []

alpha = 1 / (1 - (2*beta/U) + lambertw( ( ( (2*beta/U) + (L/U) - 1 ) * math.exp(2*beta/U) ) / math.e ) )

for _ in tqdm(range(epochs)):
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
    adv = sol.reshape((T, d))

    #################################### get the online RORO solution

    roro, roroCost = f.RORO(cost_functions, weights, d, Lc, Uc)

    #################################### get the online comparison solutions

    lazy, lazyCost = f.lazyAgnostic(cost_functions, weights, d)
    agn, agnCost = f.agnostic(cost_functions, weights, d)

    const, constCost = f.threshold(cost_functions, weights, d, L, U)

    mini, miniCost = f.moveMinimizer(cost_functions, weights, d)

    #################################### get the online CLIP solution
    # epsilon = 0.1
    # clip0, clip0Cost = c.CLIP(cost_functions, weights, d, Lc, Uc, adv, epsilon)

    # epsilon = 2
    # clip2, clip2Cost = c.CLIP(cost_functions, weights, d, Lc, Uc, adv, epsilon)


    opts.append(sol)
    roros.append(roro)
    lazys.append(lazy)
    agnostics.append(agn)
    constThresholds.append(const)
    minimizers.append(mini)
    # clip0s.append(clip0)
    # clip2s.append(clip2)

    cost_opts.append(solCost)
    cost_roros.append(roroCost)
    cost_lazys.append(lazyCost)
    cost_agnostics.append(agnCost)
    cost_constThresholds.append(constCost)
    cost_minimizers.append(miniCost)
    # cost_clip0s.append(clip0Cost)
    # cost_clip2s.append(clip2Cost)


# compute competitive ratios
cost_opts = np.array(cost_opts)
cost_roros = np.array(cost_roros)
cost_lazys = np.array(cost_lazys)
cost_agnostics = np.array(cost_agnostics)
cost_constThresholds = np.array(cost_constThresholds)
cost_minimizers = np.array(cost_minimizers)
# cost_clip0s = np.array(cost_clip0s)
# cost_clip2s = np.array(cost_clip2s)

crRORO = cost_roros/cost_opts
crLazy = cost_lazys/cost_opts
crAgnostic = cost_agnostics/cost_opts
crConstThreshold = cost_constThresholds/cost_opts
crMinimizer = cost_minimizers/cost_opts
# crClip0 = cost_clip0s/cost_opts
# crClip2 = cost_clip2s/cost_opts

# save the results (use a dictionary)
results = {"opts": opts, "roros": roros, "lazys": lazys, "agnostics": agnostics, "constThresholds": constThresholds, "minimizers": minimizers,
            "cost_opts": cost_opts, "cost_roros": cost_roros, "cost_lazys": cost_lazys, "cost_agnostics": cost_agnostics, "cost_constThresholds": cost_constThresholds, "cost_minimizers": cost_minimizers}
with open(str(sys.argv[1]) + "/robust_results_r{}_dim{}_s{}_d{}.pickle".format((U/L), d, beta, int(std)), "wb") as f:
    pickle.dump(results, f)

legend = ["ALG1", "lazy agnostic", "agnostic", "simple threshold", "move to minimizer"]

# CDF plot for competitive ratio (across all experiments)
plt.figure(figsize=(4,3), dpi=300)
linestyles = ["-.", ":", "--", (0, (3, 1, 1, 1, 1, 1)), "-", '-.', ":"]

for list in zip([crRORO, crLazy, crAgnostic, crConstThreshold, crMinimizer], linestyles):
    sns.ecdfplot(data = list[0], stat='proportion', linestyle = list[1])

plt.legend(legend)
plt.ylabel('cumulative probability')
plt.xlabel("empirical competitive ratio")
plt.tight_layout()
plt.xlim(left=1)
#plt.show()
plt.savefig(str(sys.argv[1]) + "/robust_cdf_r{}_dim{}_s{}_d{}.png".format((U/L), d, beta, std), facecolor='w', transparent=False, bbox_inches='tight')
plt.clf()

# print mean and 95th percentile of each competitive ratio
print("Switching Cost: {}".format(beta))
print("RORO: ", np.mean(crRORO), np.percentile(crRORO, 95))
print("lazy agnostic: ", np.mean(crLazy), np.percentile(crLazy, 95))
print("agnostic: ", np.mean(crAgnostic), np.percentile(crAgnostic, 95))
print("simple threshold: ", np.mean(crConstThreshold), np.percentile(crConstThreshold, 95))
print("move to minimizer: ", np.mean(crMinimizer), np.percentile(crMinimizer, 95))
# print("clip0: ", np.mean(crClip0), np.percentile(crClip0, 95))
# print("clip2: ", np.mean(crClip2), np.percentile(crClip2, 95))
print("alpha bound: ", alpha)
