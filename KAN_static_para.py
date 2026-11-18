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

# ==== GAの設定 ====
DIM = 30                 # 次元数
POP_SIZE = 50           # 個体数
GENS = 500               # 世代数
X_MIN, X_MAX = -100, 100  # 探索空間の範囲
TRIALS = 31

# ==== KANの設定 ====
KAN_GRID = 3
KAN_WIDTH = [DIM, 1]
KAN_STEPS = 50
KAN_N_TRAIN = 50

# # ==== 停滞判定の設定 ====
# PATIENCE = 10
# IMPROVEMENT_THRESHOLD = 1e-6
# MAX_MIN_N_SELECT = 50
# DUPLICATE_TOL = 1e-8

# ==== 出力設定 ====
OUTPUT_DIR = f"pop{POP_SIZE}/data/{FUNCTION_NAME}/var{DIM}"
OUTPUT_MODEL = f"pop{POP_SIZE}/model/{FUNCTION_NAME}/var{DIM}/{KAN_WIDTH}_p{PATIENCE}_g{KAN_GRID}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_MODEL, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_best_so_far_KAN_static_{KAN_WIDTH}_p{PATIENCE}_g{KAN_GRID}.csv")
OUTPUT_FILE_BEST = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_best_individuals_KAN_static_{KAN_WIDTH}_p{PATIENCE}_g{KAN_GRID}.csv")
OUTPUT_FILE_GENERATION_BEST = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_generation_best_KAN_static_{KAN_WIDTH}_p{PATIENCE}_g{KAN_GRID}.csv")
# OUTPUT_FILE_UPDATE = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_update_history_KAN_static_{KAN_WIDTH}_p{PATIENCE}_g{KAN_GRID}.csv")
OUTPUT_FILE_TIME = os.path.join(OUTPUT_DIR, f"{FUNCTION_NAME}_trial_time_KAN_static_{KAN_WIDTH}_p{PATIENCE}_g{KAN_GRID}.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MULTI_NUM = 6

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


# def update_kan(
#     model,
#     train_X,
#     train_y,
#     new_archive,
#     seed
# ):
#     """
#     新しいアーカイブのデータを追加し、
#     既存のKANを追加学習する。
#     """

#     if len(new_archive) == 0:
#         return model, train_X, train_y

#     # 新しい個体と目的関数値を取得
#     new_X = np.asarray([entry[0] for entry in new_archive], dtype=np.float32).reshape(-1, DIM)
#     new_y = np.asarray([entry[1] for entry in new_archive], dtype=np.float32).reshape(-1, 1)

#     # 学習データを追加
#     train_X = np.vstack([train_X, new_X])
#     train_y = np.vstack([train_y, new_y])

#     train_input = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
#     train_label = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)

#     dataset = {
#         "train_input": train_input,
#         "train_label": train_label,
#         "test_input": train_input,
#         "test_label": train_label,
#     }

#     # 既存モデルのパラメータを引き継いで追加学習
#     model.fit(dataset, opt="LBFGS", steps=KAN_STEPS)

#     model.eval()

#     return model, train_X, train_y


# def max_min_selection(
#     archive,
#     train_X,
#     n_select=50,
#     duplicate_tol=1e-8
# ):
#     """
#     アーカイブからMax-min selectionで学習候補を選択する。

#     Parameters
#     ----------
#     archive : list
#         [[individual, f_value], ...]
#     train_X : ndarray
#         これまでKANの学習に使用した個体
#     n_select : int
#         選択する個体数
#     duplicate_tol : float
#         既存学習データとの重複判定閾値

#     Returns
#     -------
#     selected_archive : list
#         [[individual, f_value], ...]
#     """

#     if len(archive) == 0:
#         return []

#     # 個体と評価値を取り出す
#     candidates = np.asarray([entry[0] for entry in archive], dtype=float).reshape(-1, DIM)
#     values = np.asarray([entry[1] for entry in archive], dtype=float)
#     train_X = np.asarray(train_X, dtype=float).reshape(-1, DIM)

#     # アーカイブ内の重複個体を除去
#     _, unique_indices = np.unique(
#         candidates,
#         axis=0,
#         return_index=True
#     )

#     unique_indices = np.sort(unique_indices)

#     candidates = candidates[unique_indices]
#     values = values[unique_indices]

#     # 既存のKAN学習点との重複を除外
#     if len(train_X) > 0:

#         distances = np.linalg.norm(
#             candidates[:, None, :] - train_X[None, :, :],
#             axis=2
#         )

#         min_dist_to_train = np.min(distances, axis=1)

#         valid = min_dist_to_train > duplicate_tol

#         candidates = candidates[valid]
#         values = values[valid]

#     if len(candidates) == 0:
#         return []

#     n_select = min(n_select, len(candidates))

#     # 既存学習点との最小距離
#     if len(train_X) > 0:

#         distances = np.linalg.norm(
#             candidates[:, None, :] - train_X[None, :, :],
#             axis=2
#         )

#         min_dist = np.min(distances, axis=1)

#     else:
#         min_dist = np.full(len(candidates), np.inf)

#     selected_indices = []
#     available = np.ones(len(candidates), dtype=bool)

#     for _ in range(n_select):

#         scores = np.where(
#             available,
#             min_dist,
#             -np.inf
#         )

#         idx = int(np.argmax(scores))

#         if not available[idx]:
#             break

#         selected_indices.append(idx)
#         available[idx] = False

#         x_new = candidates[idx]

#         new_distances = np.linalg.norm(
#             candidates - x_new,
#             axis=1
#         )

#         min_dist = np.minimum(
#             min_dist,
#             new_distances
#         )

#     # 個体と評価値をまとめて返す
#     selected_archive = [
#         [
#             candidates[i].copy(),
#             float(values[i])
#         ]
#         for i in selected_indices
#     ]

#     return selected_archive


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

    # # 停滞世代数
    # no_improve = 0

    # # KAN更新を実施した世代
    # update_gens = []

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


        # offspring_best = offspring[offspring_best_idx]
        # current_best = float(offspring_best.fitness.values[0])

        # if current_best < best_so_far - IMPROVEMENT_THRESHOLD:

        #     best_so_far = current_best
        #     best_ind = list(offspring_best)

        #     no_improve = 0

        # else:
        #     no_improve += 1

        # best-so-farを記録
        best_hist.append(best_so_far)

        pop = toolbox.select(pop + offspring, POP_SIZE)

        # ----------------------------------------------------
        # 7.6 停滞時にKANを追加学習
        # ----------------------------------------------------
        # if no_improve >= PATIENCE:

        #     print(
        #         f"Trial {trial + 1}, Gen {gen}: "
        #         f"stagnation detected. Updating KAN."
        #     )

        #     # アーカイブからMax-min selectionで新しい学習点を選択
        #     new_archive = max_min_selection(
        #         archive=archive,
        #         train_X=train_X,
        #         n_select=MAX_MIN_N_SELECT,
        #         duplicate_tol=DUPLICATE_TOL
        #     )

        #     # 新しい学習点が存在する場合のみKANを更新
        #     if len(new_archive) > 0:

        #         model, train_X, train_y = update_kan(
        #             model=model,
        #             train_X=train_X,
        #             train_y=train_y,
        #             new_archive=new_archive,
        #             seed=seed + gen
        #         )

        #         # 更新したKANを使うtoolboxに置き換える
        #         toolbox = make_toolbox(model)

        #         # 補助関数が変化したため、現在の個体群を再評価
        #         for ind in pop:
        #             ind.fitness.values = toolbox.evaluate(ind)

        #         pop = toolbox.select(pop, POP_SIZE)

        #         update_gens.append(gen)

        #         print(
        #             f"Trial {trial + 1}, Gen {gen}: "
        #             f"KAN updated. "
        #             f"Training samples = {len(train_X)}"
        #         )

        #         no_improve = 0

        #     else:
        #         print(
        #             f"Trial {trial + 1}, Gen {gen}: "
        #             "No new training points available."
        #         )

        #         # 新しい学習点がない場合は、再度同じ更新を
        #         # 繰り返さないよう停滞カウンタをリセット
        #         no_improve = 0

    # ========================================================
    # 8. 結果を返す
    # ========================================================
    result = {
        "trial": trial,
        "best_so_far": pd.Series(best_hist),
        "best_ind": best_ind,
        "best_fit": float(best_so_far),
        # "update_gens": update_gens,
        # "n_train": len(train_X)
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

    # --------------------------------------------------------
    # KANの更新履歴を保存
    # --------------------------------------------------------
    # df_updates = pd.DataFrame([
    #     {
    #         "trial": r["trial"],
    #         "update_gens": r["update_gens"],
    #         "n_train": r["n_train"]
    #     }
    #     for r in results
    # ])

    # df_updates.to_csv(OUTPUT_FILE_UPDATE, index=False)

    print("All trials finished.")
    print("Results saved.")


if __name__ == "__main__":
    main()