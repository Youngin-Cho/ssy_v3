import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


def draw_boxplot(file_path):
    df = pd.read_excel(file_path)
    df = df.iloc[:-1]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot([df['RL'], df['SD'], df['MA'], df['RAND']])
    ax.set_xticks([1, 2, 3, 4], ['RL', 'SD', 'MA', 'RAND'])
    ax.set_ylabel('Makespan')

    fig.savefig("./boxplot.png")


def perform_paired_t_test(file_path):
    df = pd.read_excel(file_path)
    df = df.iloc[:-1]

    res = stats.ttest_rel(df["RL"], df["SD"])
    print("(RL vs SD) p-value: %f" % res.pvalue)

    res = stats.ttest_rel(df["RL"], df["MA"])
    print("(RL vs MA) p-value: %f" % res.pvalue)

    res = stats.ttest_rel(df["RL"], df["RAND"])
    print("(RL vs RAND) p-value: %f" % res.pvalue)


if __name__ == "__main__":
    file_path = "./output/test/case0/test_results.xlsx"
    perform_paired_t_test(file_path)


