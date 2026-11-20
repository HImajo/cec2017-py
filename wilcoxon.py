import pandas as pd
import numpy as np
import os
from scipy.stats import wilcoxon

FUNCTION_NAME = "f10"
DIM = 10
POP_SIZE = 50
KAN_WIDTH = [DIM, 1]
PATIENCE = 10
GRID = 3

INPUT_DIR = f"pop{POP_SIZE}/data/{FUNCTION_NAME}/var{DIM}"

conditions = [
    # ("helper1", "1/f(x)"),
    # ("helper2_0.1", "f(x)+ε"),
    (f"KAN_dynamic_{KAN_WIDTH}_p{PATIENCE}_g{GRID}", "Dynamic KAN"),
    (f"KAN_dynamic_{KAN_WIDTH}_p20_g{GRID}", "p=20"),
    (f"KAN_dynamic_{KAN_WIDTH}_p{PATIENCE}_g5", "g=5"),
    # (f"KAN_dynamic_[10, 21, 1]_p{PATIENCE}_g{GRID}", "width=[10,21,1]"),
    # (f"KAN_static_{KAN_WIDTH}_g{GRID}", "Static KAN"),
]

baseline = "Dynamic KAN"


def load_final_values(file):
    """
    CSVファイルから最終世代の各試行のbest-so-farを取得する。

    Parameters
    ----------
    file : str
        FUNCTION_NAME_best_so_far_ に続くファイル名

    Returns
    -------
    values : np.ndarray
        各試行の最終世代の評価値
    """
    input_file = os.path.join(INPUT_DIR, f"{FUNCTION_NAME}_best_so_far_{file}.csv")

    df = pd.read_csv(input_file)

    # mean_best_so_far列を除外
    seed_cols = [
        c for c in df.columns
        if c != "mean_best_so_far"
    ]

    # 最終世代のデータを取得
    values = df.iloc[-1][seed_cols].astype(float).values

    return values

def calc_statistics(x):
    """
    平均、標準偏差、中央値、四分位数を計算する。
    """
    mean = np.mean(x)
    std = np.std(x)

    median = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "q1": q1,
        "q3": q3
    }

results = {}

# ==========================================
# 統計量の計算
# ==========================================
for file, label in conditions:
    values = load_final_values(file)
    stats = calc_statistics(values)

    results[label] = {
        "values": values,
        **stats
    }

    print(
        f"{label}:\n"
        f"Mean = {stats['mean']:.4f}, "
        f"Std = {stats['std']:.4f}"
    )


# ==========================================
# Wilcoxon検定
# ==========================================

print(f"\nWilcoxon signed-rank test ({baseline} vs others)")

for file, label in conditions[0:]:
    if (label != baseline):
        stat, p_value = wilcoxon(
            results[baseline]["values"],
            results[label]["values"],
            alternative="less"
        )

        print(
            f"{baseline} < {label}:\n"
            # f"Statistic = {stat:.4f}, "
            f"p-value = {p_value:.6f}"
        )



# CSVの読み込み
# df_conv1 = pd.read_csv(f"{INPUT_DIR}/f4_best_so_far_helper.csv")
# df_conv2 = pd.read_csv(f"{INPUT_DIR}/f4_best_so_far_helper2_0.1.csv")
# seed_cols = [c for c in df_ga.columns if c != "mean_best_so_far"]

# conv1_final = df_conv1.iloc[-1][seed_cols].astype(float).values
# conv2_final = df_conv2.iloc[-1][seed_cols].astype(float).values


# stat, p_value = wilcoxon(dynamic_final, ga_final, alternative="less")
# print("Wilcoxon signed-rank test片側検定 (KAN < GA)")
# # print(f"Statistic = {stat}")
# print(f"p-value   = {p_value}")

# def median_iqr(x):
#     median = np.median(x)
#     q1 = np.percentile(x, 25)
#     q3 = np.percentile(x, 75)
#     return median, q1, q3

# def mean_std(x):
#     mean = np.mean(x)
#     std = np.std(x)
#     return mean, std

# conv1_mean, conv1_std = mean_std(conv1_final)
# conv2_mean, conv2_std = mean_std(conv2_final)

# print("\nFinal generation statistics:")
# print(f"{conv1_mean:.4f}|{conv1_std:.4f}")
# print(f"{conv2_mean:.4f}|{conv2_std:.4f}")