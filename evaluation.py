import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    file_path = "./output/test/case0/test_results.xlsx"
    df = pd.read_excel(file_path)
    df = df.iloc[:-1]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot([df['RL'], df['SD'], df['MA'], df['RAND']])
    ax.set_xticks([1, 2, 3, 4], ['RL', 'SD', 'MA', 'RAND'])
    ax.set_ylabel('Makespan')

    # plt.subplots_adjust(left=0.2)
    plt.show()

    fig.savefig("./boxplot.png")




