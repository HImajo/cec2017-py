import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools, algorithms
from multiprocessing import get_context

import cec2017.functions as functions

# ==== GAの設定 ====
DIM = 10                 # 二次元
POP_SIZE = 100           # 個体数
GENS = 500               # 世代数
X_MIN, X_MAX = -100, 100  # 探索空間の範囲
TRIALS = 31

from cec2017.functions import f4 as f

# ==== DEAP セットアップ ====
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, X_MIN, X_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=DIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    x = np.asarray(individual, dtype=float).reshape(1, DIM)
    return (float(f(x)[0]),)

toolbox.register("evaluate", evaluate)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=[X_MIN] * DIM, up=[X_MAX] * DIM, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=[X_MIN] * DIM, up=[X_MAX] * DIM,
                 eta=20.0, indpb=1.0 / DIM)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_single_trial(seed):
    print(f"Trial {seed+1}")

    random.seed(seed)
    np.random.seed(seed)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.9, mutpb=1.0, ngen=GENS,
        stats=stats, halloffame=hof, verbose=False
    )

    # best_so_far を計算
    df = pd.DataFrame(log)
    df["best_so_far"] = df["min"].cummin()

    best_ind = hof[0][:]
    best_fit = hof[0].fitness.values[0]

    return {
        "seed": seed,
        "log": df["best_so_far"],
        "best_ind": best_ind,
        "best_fit": best_fit
    }

def main():
    ctx = get_context("spawn")   # MacOS 必須
    n_proc = max(1, ctx.cpu_count())

    print(f"Using {n_proc} processes")

    with ctx.Pool(processes=n_proc) as pool:
        results = pool.map(run_single_trial, range(TRIALS))

    # ======================
    # 結果の集約
    # ======================
    best_lists = []
    best_individuals = []

    for r in results:
        best_lists.append(r["log"])
        best_individuals.append(
            (r["best_ind"], r["best_fit"])
        )

    best_df = pd.concat(best_lists, axis=1)
    best_df["mean_best_so_far"] = best_df.mean(axis=1)
    best_df.reset_index(drop=True, inplace=True)
    best_df.to_csv("f4_best_so_far.csv", index=False)

    df_best = pd.DataFrame([
        {"trial": i+1, "x": ind[0], "y": ind[1], "fitness": val}
        for i, (ind, val) in enumerate(best_individuals)
    ])
    df_best.to_csv("f4_best_individuals.csv", index=False)

if __name__ == "__main__":
    main()