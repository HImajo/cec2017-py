import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools, algorithms
import os
from multiprocessing import get_context
import time
import torch
from kan import KAN

from cec2017.functions import f4 as f
FUNCTION_NAME = "f4"
MULTI_NUM = 7

# ==== GAの設定 ====
DIM = 50                 # 次元数
POP_SIZE = 50           # 個体数
GENS = 500               # 世代数
X_MIN, X_MAX = -100, 100  # 探索空間の範囲
TRIALS = 31

# ==== KANの設定 ====
KAN_GRID = 3
KAN_WIDTH = [DIM, 1]
KAN_STEPS = 50
KAN_N_TRAIN = 50

# ==== 出力設定 ====
OUTPUT_DIR = f"pop{POP_SIZE}/data/{FUNCTION_NAME}/var{DIM}"
OUTPUT_MODEL = f"pop{POP_SIZE}/model/{FUNCTION_NAME}/var{DIM}/{KAN_WIDTH}_g{KAN_GRID}_static"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_best_so_far_KAN_static_{KAN_WIDTH}_g{KAN_GRID}.csv")
OUTPUT_FILE_BEST = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_best_individuals_KAN_static_{KAN_WIDTH}_g{KAN_GRID}.csv")
OUTPUT_FILE_GENERATION_BEST = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_generation_best_KAN_static_{KAN_WIDTH}_g{KAN_GRID}.csv")
OUTPUT_FILE_TIME = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_trial_time_KAN_static_{KAN_WIDTH}_g{KAN_GRID}.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_target(individual):
    """
    個体の真の目的関数値を取得する。
    fには必ず (1, DIM) の形状で渡す。
    """
    x = np.asarray(individual, dtype=float).reshape(1, DIM)
    return float(f(x)[0])


def train_kan(archive, seed):
    """
    アーカイブを学習データとしてKANを学習する。

    Parameters
    ----------
    archive : list
        [[individual, f_value], ...]
    seed : int
        乱数シード

    Returns
    -------
    model : KAN
        学習済みモデル
    train_X : ndarray
        学習に使用した個体
    train_y : ndarray
        対応する真の目的関数値
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # アーカイブから個体と評価値を取り出す
    X_train = np.asarray([entry[0] for entry in archive], dtype=np.float32).reshape(-1, DIM)
    y_train = np.asarray([entry[1] for entry in archive], dtype=np.float32).reshape(-1, 1)

    train_input = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    train_label = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    dataset = {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": train_input,
        "test_label": train_label,
    }

    model_folder = os.path.join(
        OUTPUT_MODEL,
        f"trial{seed}"
    )

    os.makedirs(model_folder, exist_ok=True)

    model = KAN(
        width=KAN_WIDTH,
        grid=KAN_GRID,
        seed=seed,
        device=DEVICE,
        ckpt_path=model_folder
    )

    model.fit(
        dataset,
        opt="LBFGS",
        steps=KAN_STEPS
    )

    model.eval()

    return model, X_train, y_train


# --- DEAP設定 ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# ============================================================
# 5. 初期個体群生成用のtoolbox
# ============================================================
def make_population_toolbox():
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, X_MIN, X_MAX)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=DIM)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


def make_toolbox(model):
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

    # KANの予測関数
    def helper_function(individual):
        x = np.asarray(individual, dtype=np.float32).reshape(1, DIM)
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            prediction = model(x_tensor)

        return float(prediction.detach().cpu().numpy().reshape(-1)[0])

    # 2目的関数
    def evaluate(individual):
        x = np.asarray(individual, dtype=float).reshape(1, DIM)
        f1 = float(f(x)[0])
        f2 = helper_function(individual)

        return f1, f2

    toolbox.register("evaluate", evaluate)

    return toolbox

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


def run_single_trial_helper(trial):

    seed = trial
    print(f"Trial {trial + 1}: start")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------------
    # 1. GAの初期個体群を生成
    # --------------------------------------------------------
    pop_toolbox = make_population_toolbox()
    pop = pop_toolbox.population(n=POP_SIZE)

    # --------------------------------------------------------
    # 2. 初期個体群の評価とアーカイブの作成
    # --------------------------------------------------------
    archive = []

    for ind in pop:
        x = np.asarray(ind, dtype=float).copy()
        y = evaluate_target(ind)

        archive.append([x, y])

    # --------------------------------------------------------
    # 3. 初期アーカイブを使ってKANを学習
    # --------------------------------------------------------
    print(f"Trial {trial + 1}: initial KAN training")

    model, train_X, train_y = train_kan(
        archive=archive,
        seed=seed
    )

    # --------------------------------------------------------
    # 4. 学習済みKANを使うtoolboxを作成
    # --------------------------------------------------------
    toolbox = make_toolbox(model)

    # --------------------------------------------------------
    # 5. 初期個体群を2目的で評価
    # --------------------------------------------------------
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
        

    best_idx = int(np.argmin([
        entry[1] for entry in archive
    ]))

    best_so_far = float(archive[best_idx][1])
    best_ind = archive[best_idx][0].copy().tolist()


    # 各世代のbest-so-far
    best_hist = [best_so_far]

    # --------------------------------------------------------
    # 6. 初期個体群を多目的選択
    # --------------------------------------------------------
    pop = toolbox.select(pop, POP_SIZE)

    # ========================================================
    # 7. 進化計算のメインループ
    # ========================================================
    for gen in range(1, GENS + 1):

        parents = select_parents(pop, POP_SIZE)

        # ----------------------------------------------------
        # 7.1 子個体群を生成
        # ----------------------------------------------------
        offspring = algorithms.varAnd(
            parents,
            toolbox,
            cxpb=0.9,
            mutpb=1.0
        )

        # ----------------------------------------------------
        # 7.2 未評価の子個体を評価
        # ----------------------------------------------------
        invalid_ind = [
            ind for ind in offspring
            if not ind.fitness.valid
        ]

        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # ----------------------------------------------------
        # 7.3 評価済みの子個体をアーカイブに追加
        # ----------------------------------------------------

        for ind in offspring:
            x = np.asarray(ind, dtype=float).copy()
            y = float(ind.fitness.values[0])

            archive.append([x, y])


        # ----------------------------------------------------
        # 7.5 ターゲット目的関数の最良値を更新
        # ----------------------------------------------------
        offspring_best_idx = int(np.argmin([
            ind.fitness.values[0]
            for ind in offspring
        ]))
        offspring_best = offspring[offspring_best_idx]
        current_best = float(offspring_best.fitness.values[0])

        if current_best < best_so_far:
            best_so_far = current_best
            best_ind = list(offspring_best)

        # best-so-farを記録
        best_hist.append(best_so_far)

        pop = toolbox.select(pop + offspring, POP_SIZE)

    # ========================================================
    # 8. 結果を返す
    # ========================================================
    result = {
        "trial": trial,
        "best_so_far": pd.Series(best_hist),
        "best_ind": best_ind,
        "best_fit": float(best_so_far),
    }

    print(f"Trial {trial + 1}: finished")

    return result

def run_single_trial_with_time(trial):
    start_time = time.perf_counter()

    result = run_single_trial_helper(trial)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return trial, result, elapsed_time

# ============================================================
# 7. 並列実行と結果の集約
# ============================================================
def main():

    ctx = get_context("spawn")
    # n_proc = min(TRIALS, ctx.cpu_count())
    n_proc = min(TRIALS, MULTI_NUM)

    print(f"Using {n_proc} processes")
    print(f"Total trials: {TRIALS}")

    with ctx.Pool(processes=n_proc) as pool:
        trial_results = pool.map(
            run_single_trial_with_time,
            range(TRIALS)
        )

    results = [r[1] for r in trial_results]
    execution_times = [r[2] for r in trial_results]

    # ========================================================
    # 結果の集約
    # ========================================================
    best_lists = []
    best_fitness = []

    for r in results:
        best_lists.append(r["best_so_far"])
        best_fitness.append(
            (r["best_ind"], r["best_fit"])
        )

    # --------------------------------------------------------
    # 各試行のbest-so-farを保存
    # --------------------------------------------------------
    best_df = pd.concat(
        best_lists,
        axis=1
    )

    best_df.columns = [
        str(i)
        for i in range(TRIALS)
    ]

    best_df["mean_best_so_far"] = best_df.mean(axis=1)
    best_df.reset_index(drop=True, inplace=True)
    best_df.to_csv(OUTPUT_FILE, index=False)


    # ========================================================
    # 各試行の実行時間を保存
    # ========================================================
    time_df = pd.DataFrame({
        "trial": range(TRIALS),
        "execution_time_sec": execution_times,
        "execution_time_min": [
            t / 60 for t in execution_times
        ]
    })

    time_df.to_csv(OUTPUT_FILE_TIME, index=False)


    # --------------------------------------------------------
    # 各試行の最良個体を保存
    # --------------------------------------------------------
    df_best = pd.DataFrame([
        {
            "trial": i + 1,
            "best_fitness": best_fitness[i][1],
            "individual": best_fitness[i][0]
        }
        for i in range(TRIALS)
    ])

    df_best.to_csv(OUTPUT_FILE_BEST, index=False)

    print("All trials finished.")
    print("Results saved.")


if __name__ == "__main__":
    main()