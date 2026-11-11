import pandas as pd
import matplotlib.pyplot as plt
import os


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    "font.size": 18,       # 全体の基本フォントサイズ
    "axes.labelsize": 34,  # 軸ラベル
    "xtick.labelsize": 26, # x軸目盛り
    "ytick.labelsize": 26, # y軸目盛り
    "legend.fontsize": 35,  # 凡例
    "font.family": "Times New Roman"  # フォントをTimes New Romanに設定
})

FUNCTION_NAME = "f10"
DIM = 30
POP_SIZE = 50
KAN_WIDTH = [DIM, 1]
PATIENCE = 20

BOTTOM = 1000
TOP = 10000

INPUT_DIR = f"pop{POP_SIZE}/data/{FUNCTION_NAME}/var{DIM}"
OUTPUT_DIR = f"pop{POP_SIZE}/fig/{FUNCTION_NAME}/var{DIM}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{FUNCTION_NAME}_var{DIM}_diffh.pdf"

files = [
    ("helper1", "1/f(x)"),
    ("helper2_0.1", "f(x)+ε"),
    (f"KAN_dynamic_{KAN_WIDTH}_p{PATIENCE}", "Dynamic KAN"),
]

def calc_mean_std(file):
    input_file = os.path.join(INPUT_DIR, f"{FUNCTION_NAME}_best_so_far_{file}.csv")
    df = pd.read_csv(input_file)
    best_df = df.iloc[:, :-1]
    mean = best_df.mean(axis=1)
    std = best_df.std(axis=1)

    return mean, std

def plot_convergence(results, filename):
    """
    平均と標準偏差を用いた収束曲線を描画する。

    Parameters
    ----------
    results : list
        [(mean, std, label), ...]
    filename : str
        保存するファイル名
    """

    plt.figure(figsize=(8, 6))

    for mean, std, label in results:
        generations = range(len(mean))

        plt.plot(generations, mean, label=label, linewidth=2)

        plt.fill_between(generations, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Generation")
    plt.ylabel("Mean of best fitness")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 500)
    plt.ylim(bottom=BOTTOM, top=TOP)
    plt.tight_layout()

    output_file = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_file, format="pdf")
    plt.show()
    plt.close()


files = [
    ("helper1", "1/f(x)"),
    ("helper2_0.1", "f(x)+ε"),
    (f"KAN_dynamic_{KAN_WIDTH}_p{PATIENCE}", "Dynamic KAN"),
]

results = []

for file, label in files:
    mean, std = calc_mean_std(file)
    results.append((mean, std, label))

plot_convergence(
    results,
    filename=OUTPUT_FILE
)

# CSVの読み込み
# df = pd.read_csv("data/f10_best_so_far.csv")
# df_p = pd.read_csv("data/f10_best_so_far_helper.csv")

# # 試行ごとのデータ（最後の列を除く）
# best_df = df.iloc[:, :-1]
# best_df_p = df_p.iloc[:, :-1]

# # 各世代での平均と標準偏差
# mean = best_df.mean(axis=1)
# std = best_df.std(axis=1)
# mean_p = best_df_p.mean(axis=1)
# std_p = best_df_p.std(axis=1)

# # 世代の番号（0から）
# generations = range(len(mean_h0))

# plt.figure(figsize=(8, 6))
# plt.plot(generations, mean_h2_1, label="0.1")
# plt.fill_between(generations, mean_h2_1 - std_h2_1, mean_h2_1 + std_h2_1, alpha=0.2)
# plt.plot(generations, mean_h2_2, label="0.2")
# plt.fill_between(generations, mean_h2_2 - std_h2_2, mean_h2_2 + std_h2_2, alpha=0.2)
# plt.xlabel("Generation")
# plt.ylabel("Mean of best fitness")
# plt.legend()
# plt.grid(True)
# plt.xlim(0,500)
# # plt.ylim(bottom=900, top=1000)
# plt.tight_layout()
# plt.savefig("fig/f10_var30_conv2.pdf", format='pdf')
# plt.show()

