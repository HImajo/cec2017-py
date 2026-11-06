import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools, algorithms
from multiprocessing import get_context

import torch
from kan import KAN

# ==== GAの設定 ====
DIM = 10                 # 次元数
POP_SIZE = 50           # 個体数
GENS = 500               # 世代数
X_MIN, X_MAX = -100, 100  # 探索空間の範囲
TRIALS = 31

# ==== KANの設定 ====
KAN_GRID = 3
KAN_WIDTH = [DIM, 1]
KAN_STEPS = 50
KAN_N_TRAIN = 50

# ==== 停滞判定の設定 ====
PATIENCE = 20
IMPROVEMENT_THRESHOLD = 1e-6
MAX_MIN_N_SELECT = 50
DUPLICATE_TOL = 1e-8


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==== Rosenbrock 関数（シフト・回転版）定義 ====
o_4 = np.loadtxt("def/shift_data_4.txt")[:DIM]  # shape: (DIM,)
M = np.loadtxt("def/M_4_D10.txt")                  # shape: (DIM, DIM)

assert o_4.shape[0] == DIM
assert M.shape == (DIM, DIM)

F4_star = 400.0


def rosenbrock_base(z):
    return np.sum(
        100.0 * (z[:-1]**2 - z[1:])**2 +
        (z[:-1] - 1.0)**2
    )

def function_shifted_rotated(individual):
    x = np.array(individual)
    # (x - o_4)
    z = x - o_4
    # 2.048 * (x - o_4) / 100
    z = 2.048 * z / 100.0
    # 回転
    z = M @ z
    # + 1 ベクトル
    z = z + 1.0
    # Rosenbrock
    return rosenbrock_base(z) + F4_star,



# ============================================================
# 4. 各試行のKAN学習
# ============================================================
def train_kan(pop, seed):
    """
    GAの初期個体群を学習データとしてKANを学習する。

    Parameters
    ----------
    pop : list
        GAの初期個体群
    seed : int
        乱数シード

    Returns
    -------
    model : KAN
        学習済みモデル
    X_train : ndarray
        学習に使用した個体
    y_train : ndarray
        対応する真の目的関数値
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GAの初期個体群をそのまま学習データに使用
    X_train = np.asarray(pop, dtype=np.float32)

    # 各個体の真の目的関数値
    y_train = np.asarray([
        float(function_shifted_rotated(x)[0])
        for x in X_train
    ], dtype=np.float32).reshape(-1, 1)

    train_input = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=DEVICE
    )

    train_label = torch.tensor(
        y_train,
        dtype=torch.float32,
        device=DEVICE
    )

    dataset = {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": train_input,
        "test_label": train_label,
    }

    model = KAN(
        width=KAN_WIDTH,
        grid=KAN_GRID,
        seed=seed,
        device=DEVICE
    )

    model.fit(
        dataset,
        opt="LBFGS",
        steps=KAN_STEPS
    )

    model.eval()

    return model, X_train, y_train


def update_kan(
    model,
    train_X,
    train_y,
    new_X,
    seed
):
    """
    新しい学習データを追加し、既存のKANを追加学習する。
    """

    if len(new_X) == 0:
        return model, train_X, train_y

    # 新しい個体の真の目的関数値を取得
    new_y = np.asarray([
        float(function_shifted_rotated(x)[0])
        for x in new_X
    ], dtype=np.float32).reshape(-1, 1)

    # 学習データを蓄積
    train_X = np.vstack([
        train_X,
        new_X
    ])

    train_y = np.vstack([
        train_y,
        new_y
    ])

    train_input = torch.tensor(
        train_X,
        dtype=torch.float32,
        device=DEVICE
    )

    train_label = torch.tensor(
        train_y,
        dtype=torch.float32,
        device=DEVICE
    )

    dataset = {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": train_input,
        "test_label": train_label,
    }

    # 既存モデルのパラメータを引き継いで追加学習
    model.fit(
        dataset,
        opt="LBFGS",
        steps=KAN_STEPS
    )

    model.eval()

    return model, train_X, train_y


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
#     archive : ndarray, shape (n_archive, DIM)
#         過去に評価した全個体
#     train_X : ndarray, shape (n_train, DIM)
#         これまでKANの学習に使用した全個体
#     n_select : int
#         選択する個体数
#     duplicate_tol : float
#         既存の学習データとの重複判定閾値

#     Returns
#     -------
#     selected_X : ndarray, shape (n_select, DIM) 以下
#         選択された個体
#     """

#     archive = np.asarray(archive, dtype=float)
#     train_X = np.asarray(train_X, dtype=float)

#     if len(archive) == 0:
#         return np.empty((0, DIM))

#     # アーカイブ内の重複個体を除去
#     candidates = np.unique(archive, axis=0)

#     # すでに学習に使用した個体は候補から除外
#     remaining = []

#     for x in candidates:
#         distances = np.linalg.norm(train_X - x, axis=1)

#         if np.all(distances > duplicate_tol):
#             remaining.append(x)

#     if len(remaining) == 0:
#         return np.empty((0, DIM))

#     remaining = np.asarray(remaining)

#     # 既存の学習データに対する各候補の最小距離
#     min_dist = np.empty(len(remaining))

#     for i, x in enumerate(remaining):
#         min_dist[i] = np.min(
#             np.linalg.norm(train_X - x, axis=1)
#         )

#     selected = []

#     n_select = min(n_select, len(remaining))

#     for _ in range(n_select):

#         # 最小距離が最大の候補を選択
#         idx = int(np.argmax(min_dist))

#         x_new = remaining[idx]
#         selected.append(x_new)

#         # 選択済み候補を除外
#         remaining = np.delete(remaining, idx)
#         min_dist = np.delete(min_dist, idx)

#         if len(remaining) == 0:
#             break

#         # 新しく選んだ個体からの距離を計算
#         new_distances = np.linalg.norm(
#             remaining - x_new,
#             axis=1
#         )

#         # 既存の学習データと選択済み個体の両方からの
#         # 最小距離を更新
#         min_dist = np.minimum(
#             min_dist,
#             new_distances
#         )

#     if len(selected) == 0:
#         return np.empty((0, DIM))

#     return np.asarray(selected)


def max_min_selection(
    archive,
    train_X,
    n_select=50,
    duplicate_tol=1e-8
):
    """
    アーカイブからMax-min selectionで学習候補を選択する。

    Parameters
    ----------
    archive : ndarray, shape (n_archive, DIM)
        過去に評価した全個体
    train_X : ndarray, shape (n_train, DIM)
        これまでKANの学習に使用した全個体
    n_select : int
        選択する個体数
    duplicate_tol : float
        既存の学習データとの重複判定閾値

    Returns
    -------
    selected_X : ndarray, shape (n_selected, DIM)
        選択された個体
    """

    archive = np.asarray(archive, dtype=float).reshape(-1, DIM)
    train_X = np.asarray(train_X, dtype=float).reshape(-1, DIM)

    # アーカイブが空の場合
    if len(archive) == 0:
        return np.empty((0, DIM), dtype=float)

    # アーカイブ内の重複個体を除去
    candidates = np.unique(archive, axis=0)

    # 既存のKAN学習点と重複する候補を除外
    if len(train_X) > 0:
        distances = np.linalg.norm(
            candidates[:, None, :] - train_X[None, :, :],
            axis=2
        )

        min_dist_to_train = np.min(distances, axis=1)

        candidates = candidates[
            min_dist_to_train > duplicate_tol
        ]

    # 選択可能な候補がない場合
    if len(candidates) == 0:
        return np.empty((0, DIM), dtype=float)

    # 選択数を候補数以下に制限
    n_select = min(n_select, len(candidates))

    # 各候補と既存学習点との最小距離を計算
    if len(train_X) > 0:
        distances = np.linalg.norm(
            candidates[:, None, :] - train_X[None, :, :],
            axis=2
        )
        min_dist = np.min(distances, axis=1)
    else:
        # 既存学習点がない場合は、最初の候補を選べるようにする
        min_dist = np.full(len(candidates), np.inf)

    # 選択済み候補を管理
    selected_indices = []
    available = np.ones(len(candidates), dtype=bool)

    for _ in range(n_select):

        # 選択可能な候補のうち、最小距離が最大の候補を選ぶ
        scores = np.where(available, min_dist, -np.inf)
        idx = int(np.argmax(scores))

        # 選択可能な候補が残っていない場合
        if not available[idx]:
            break

        selected_indices.append(idx)
        available[idx] = False

        # 今回選択した個体
        x_new = candidates[idx]

        # すべての候補と選択個体との距離を計算
        new_distances = np.linalg.norm(
            candidates - x_new,
            axis=1
        )

        # 各候補について、既存学習点および選択済み個体との
        # 最小距離を更新
        min_dist = np.minimum(min_dist, new_distances)

    selected_X = candidates[selected_indices]

    return selected_X


# --- DEAP設定 ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

# ============================================================
# 5. 初期個体群生成用のtoolbox
# ============================================================
def make_population_toolbox():
    toolbox = base.Toolbox()

    toolbox.register(
        "attr_float",
        random.uniform,
        X_MIN,
        X_MAX
    )

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=DIM
    )

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

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
                 eta=20.0, indpb=0.2)

    toolbox.register("select", tools.selNSGA2)

    # KANの予測関数
    def helper_function(individual):
        x = np.asarray(
            individual,
            dtype=np.float32
        ).reshape(1, DIM)

        x_tensor = torch.as_tensor(
            x,
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():
            prediction = model(x_tensor)

        return float(
            prediction.detach().cpu().numpy().reshape(-1)[0]
        )

    # 2目的関数
    def evaluate(individual):
        f1 = function_shifted_rotated(individual)[0]
        f2 = helper_function(individual)

        return f1, f2

    toolbox.register("evaluate", evaluate)

    return toolbox


# ============================================================
# 6. 試行ごとの評価関数
# ============================================================
def run_single_trial_helper(trial):

    seed = trial

    print(f"Trial {trial + 1}: start")

    # --------------------------------------------------------
    # 乱数シードの設定
    # --------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------------
    # 1. GAの初期個体群を生成
    # --------------------------------------------------------
    pop_toolbox = make_population_toolbox()
    pop = pop_toolbox.population(n=POP_SIZE)

    # --------------------------------------------------------
    # 2. 初期個体群を使ってKANを学習
    # --------------------------------------------------------
    print(f"Trial {trial + 1}: initial KAN training")

    # 初期個体群をそのままKANの学習データとして使用する
    model, train_X, train_y = train_kan(
        pop=pop,
        seed=seed
    )

    # --------------------------------------------------------
    # 3. 学習済みKANを使うtoolboxを作成
    # --------------------------------------------------------
    toolbox = make_toolbox(model)

    # --------------------------------------------------------
    # 4. アーカイブを初期化
    # --------------------------------------------------------
    # 初期個体群を、これまでに評価した個体のアーカイブとする
    archive = [
        np.asarray(ind, dtype=float).copy()
        for ind in pop
    ]

    # --------------------------------------------------------
    # 5. 初期個体群を2目的で評価
    # --------------------------------------------------------
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # 初期個体群のターゲット目的関数の最良値
    best_so_far = min(
        float(function_shifted_rotated(ind)[0])
        for ind in pop
    )

    # 各世代のbest-so-farを保存
    best_hist = [best_so_far]

    # 最良個体を記録
    best_ind = min(
        pop,
        key=lambda ind: float(function_shifted_rotated(ind)[0])
    )
    best_ind = list(best_ind)

    # 停滞世代数
    no_improve = 0

    # KAN更新を実施した世代
    update_gens = []

    # --------------------------------------------------------
    # 6. 初期個体群を多目的選択
    # --------------------------------------------------------
    pop = toolbox.select(pop, POP_SIZE)

    # ========================================================
    # 7. 進化計算のメインループ
    # ========================================================
    for gen in range(1, GENS + 1):

        # ----------------------------------------------------
        # 7.1 子個体群を生成
        # ----------------------------------------------------
        offspring = algorithms.varOr(
            pop,
            toolbox,
            lambda_=POP_SIZE,
            cxpb=0.7,
            mutpb=0.2
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
        archive.extend(
            np.asarray(ind, dtype=float).copy()
            for ind in offspring
        )

        # ----------------------------------------------------
        # 7.4 親個体群と子個体群から次世代を選択
        # ----------------------------------------------------
        pop = toolbox.select(
            pop + offspring,
            POP_SIZE
        )

        # ----------------------------------------------------
        # 7.5 ターゲット目的関数の最良値を更新
        # ----------------------------------------------------
        # アーカイブに含まれる全評価済み個体から求める
        archive_best_ind = min(
            archive,
            key=lambda x: float(function_shifted_rotated(x)[0])
        )

        current_best = float(
            function_shifted_rotated(archive_best_ind)[0]
        )

        # best-so-farが十分改善したか判定
        if current_best < best_so_far - IMPROVEMENT_THRESHOLD:

            best_so_far = current_best
            best_ind = np.asarray(
                archive_best_ind,
                dtype=float
            ).tolist()

            no_improve = 0

        else:
            no_improve += 1

        # この世代のbest-so-farを記録
        best_hist.append(best_so_far)

        # ----------------------------------------------------
        # 7.6 停滞時にKANを追加学習
        # ----------------------------------------------------
        if no_improve >= PATIENCE:

            print(
                f"Trial {trial + 1}, Gen {gen}: "
                f"stagnation detected. Updating KAN."
            )

            # アーカイブ内の個体から、過去のKAN学習点との
            # 最小距離が最大になる個体を選択
            archive_X = np.asarray(
                archive,
                dtype=float
            )

            new_X = max_min_selection(
                archive=archive_X,
                train_X=train_X,
                n_select=MAX_MIN_N_SELECT,
                duplicate_tol=DUPLICATE_TOL
            )

            # 新しい学習点が存在する場合のみKANを更新
            if len(new_X) > 0:

                model, train_X, train_y = update_kan(
                    model=model,
                    train_X=train_X,
                    train_y=train_y,
                    new_X=new_X,
                    seed=seed + gen
                )

                # 更新したKANを使うtoolboxに置き換える
                toolbox = make_toolbox(model)

                # 補助関数が変化したため、現在の個体群を再評価
                for ind in pop:
                    ind.fitness.values = toolbox.evaluate(ind)

                update_gens.append(gen)

                print(
                    f"Trial {trial + 1}, Gen {gen}: "
                    f"KAN updated. "
                    f"Training samples = {len(train_X)}"
                )

                # 次の停滞判定を開始
                no_improve = 0

            else:
                print(
                    f"Trial {trial + 1}, Gen {gen}: "
                    "No new training points available."
                )

                # 新しい学習点がない場合は、再度同じ更新を
                # 繰り返さないよう停滞カウンタをリセット
                no_improve = 0

    # ========================================================
    # 8. 結果を返す
    # ========================================================
    result = {
        "trial": trial,
        "best_so_far": pd.Series(best_hist),
        "best_ind": best_ind,
        "best_fit": float(
            function_shifted_rotated(best_ind)[0]
        ),
        "update_gens": update_gens,
        "n_train": len(train_X)
    }

    print(f"Trial {trial + 1}: finished")

    return result


# ============================================================
# 7. 並列実行と結果の集約
# ============================================================
def main():

    os.makedirs("data", exist_ok=True)

    ctx = get_context("spawn")

    n_proc = min(TRIALS, ctx.cpu_count())

    print(f"Using {n_proc} processes")
    print(f"Total trials: {TRIALS}")

    with ctx.Pool(processes=n_proc) as pool:
        results = pool.map(
            run_single_trial_helper,
            range(TRIALS)
        )

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

    best_df.to_csv(
        "data/f4_best_so_far_KAN_dynamic.csv",
        index=False
    )

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

    df_best.to_csv(
        "data/f4_best_individuals_KAN_dynamic.csv",
        index=False
    )

    # --------------------------------------------------------
    # KANの更新履歴を保存
    # --------------------------------------------------------
    df_updates = pd.DataFrame([
        {
            "trial": r["trial"] + 1,
            "update_gens": r["update_gens"],
            "n_train": r["n_train"]
        }
        for r in results
    ])

    df_updates.to_csv(
        "data/f4_KAN_update_history.csv",
        index=False
    )

    print("All trials finished.")
    print("Results saved.")


if __name__ == "__main__":
    main()