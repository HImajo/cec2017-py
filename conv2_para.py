import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools, algorithms
import os
from multiprocessing import get_context

from cec2017.functions import f4 as f
FUNCTION_NAME = "f4"

# ==== GAの設定 ====
DIM = 10                 # 次元数
POP_SIZE = 100           # 個体数
GENS = 500               # 世代数
X_MIN, X_MAX = -100, 100  # 探索空間の範囲
TRIALS = 31

# ==== 出力設定 ====
OUTPUT_DIR = f"pop{POP_SIZE}/data/{FUNCTION_NAME}/var{DIM}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE =  f"{FUNCTION_NAME}_best_so_far_helper2"
OUTPUT_FILE_BEST = f"{FUNCTION_NAME}_best_individuals_helper2"
OUTPUT_FILE_GENERATION_BEST = f"{FUNCTION_NAME}_generation_best_helper2"

# --- DEAP設定 ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, X_MIN, X_MAX)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=DIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=[X_MIN] * DIM, up=[X_MAX] * DIM, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=[X_MIN] * DIM, up=[X_MAX] * DIM,
                 eta=20.0, indpb=1.0 / DIM)
toolbox.register("select", tools.selNSGA2)

def select_parents(pop, n):
    # selTournamentDCDは4個単位で選択する
    n_main = (n // 4) * 4
    parents = []

    if n_main > 0:
        parents.extend(tools.selTournamentDCD(pop, n_main))

    # 4の倍数でない場合、残りを4個選択して必要数だけ使用
    n_remain = n - n_main
    if n_remain > 0:
        parents.extend(tools.selTournamentDCD(pop, 4)[:n_remain])

    return parents

# === メインループ ===
def run_single_trial_helper(args):
    a, trial = args

    print(f"Trial {trial + 1}")

    random.seed(trial)
    np.random.seed(trial)

    # =========================
    # 世代をまたいで保持する状態
    # =========================
    state = {
        "fmin": np.inf,
        "fmax": -np.inf,
        "epsilon": 0.0
    }

    # =========================
    # 評価関数（epsilon は state から参照）
    # =========================
    def with_helper(individual):
        x = np.asarray(individual, dtype=float).reshape(1, DIM)
        f1 = f(x)[0]
        noise = np.random.normal(0, 1)
        f2 = f(x)[0] + state["epsilon"] * noise
        return f1, f2

    toolbox.register("evaluate", with_helper)

    pop = toolbox.population(n=POP_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()

    # ---- 初期評価 + NSGA-II 初期化 ----
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # 初期集団にRankとCrowding Distanceを付与
    pop = toolbox.select(pop, len(pop))

    # ---- f1 の min / max を更新 ----
    f1_values = [ind.fitness.values[0] for ind in pop]

    state["fmin"] = min(state["fmin"], min(f1_values))
    state["fmax"] = max(state["fmax"], max(f1_values))

    # ---- epsilon 更新（その世代までの情報）----
    state["epsilon"] = (state["fmax"] - state["fmin"]) * a

    # ---- 統計 ----
    record = stats.compile(pop)
    logbook.record(gen=0, nvals=len(invalid_ind), **record)

    generation_best = []
    best_ind = min(
        pop,
        key=lambda ind: ind.fitness.values[0]
    )
    generation_best.append({
        "generation": 0,
        "best_fitness": best_ind.fitness.values[0],
        "helper_fitness": best_ind.fitness.values[1],
        "individual": list(best_ind)
    })

    for gen in range(1, GENS+1):
        parents = select_parents(pop, POP_SIZE)

        # ---- 進化操作 ----
        offspring = algorithms.varAnd(
            pop, toolbox,
            cxpb=0.9,
            mutpb=1.0
        )

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # ---- f1 の min / max を更新 ----
        f1_values = [ind.fitness.values[0] for ind in pop]

        state["fmin"] = min(state["fmin"], min(f1_values))
        state["fmax"] = max(state["fmax"], max(f1_values))

        # ---- epsilon 更新（その世代までの情報）----
        state["epsilon"] = (state["fmax"] - state["fmin"]) * a

        pop = toolbox.select(pop + offspring, POP_SIZE)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        best_ind = min(
            pop,
            key=lambda ind: ind.fitness.values[0]
        )

        generation_best.append({
            "generation": gen,
            "best_fitness": best_ind.fitness.values[0],
            "helper_fitness": best_ind.fitness.values[1],
            "individual": list(best_ind)
        })

    # =========================
    # best-so-far（f1）
    # =========================
    min_f1 = [rec["min"][0] for rec in logbook]
    best_so_far = pd.Series(min_f1).cummin()

    return {
        "trial": trial,
        "best_so_far": best_so_far,
        "best_ind": best_ind[:],
        "best_fit": best_ind.fitness.values[0],
        "generation_best": generation_best
    }

def main():
    ctx = get_context("spawn")   # MacOS 必須
    n_proc = max(1, ctx.cpu_count())

    print(f"Using {n_proc} processes")

    a_LIST = [0.1, 0.2]

    for a in a_LIST:
        print(f"\n=== Running for helper a = {a} ===")

        with ctx.Pool(processes=n_proc) as pool:
            results = pool.map(
                run_single_trial_helper,
                [(a, trial) for trial in range(TRIALS)]
            )

        # ===== 結果の集約 =====
        best_lists = []
        best_fitness = []
        generation_best_all = []

        for r in results:
            best_lists.append(r["best_so_far"])
            best_fitness.append((r["best_ind"], r["best_fit"]))

            for gen_result in r["generation_best"]:
                generation_best_all.append({
                    "trial": r["trial"] + 1,
                    "generation": gen_result["generation"],
                    "best_fitness": gen_result["best_fitness"],
                    "helper_fitness": gen_result["helper_fitness"],
                    "individual": gen_result["individual"]
                })

        best_df = pd.concat(best_lists, axis=1)
        best_df["mean_best_so_far"] = best_df.mean(axis=1)
        best_df.reset_index(drop=True)
        best_df.to_csv(f"{OUTPUT_DIR}/{OUTPUT_FILE}_{a}.csv", index=False)

        df_best = pd.DataFrame([
            {
                "trial": i + 1,
                "best_fitness": best_fitness[i][1],
                "individual": best_fitness[i][0]
            }
            for i in range(TRIALS)
        ])
        df_best.to_csv(f"{OUTPUT_DIR}/{OUTPUT_FILE_BEST}_{a}.csv", index=False)

        df_generation_best = pd.DataFrame(generation_best_all)
        df_generation_best.to_csv(f"{OUTPUT_DIR}/{OUTPUT_FILE_GENERATION_BEST}_{a}.csv", index=False)

if __name__ == "__main__":
    main()