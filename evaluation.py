import os
import numpy as np
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


def perform_paired_t_test(log_dir):
    p_values = []
    for log_dir_temp in log_dir:
        df_RL_GNN = pd.read_excel(log_dir_temp + '(RL-GNN) test_results.xlsx', index_col=0)
        df_RL_GNN = df_RL_GNN.rename(columns={"RL": "RL-GNN"})

        df_RL_MLP = pd.read_excel(log_dir_temp + '(RL-MLP) test_results.xlsx', index_col=0)
        df_RL_MLP = df_RL_MLP.rename(columns={"RL": "RL-MLP"})

        df_GP = pd.read_excel(log_dir_temp + '(GP) test_results.xlsx', index_col=0)
        df_Heuristics = pd.read_excel(log_dir_temp + '(Heuristics) test_results.xlsx', index_col=0)

        df_all = pd.concat([df_RL_GNN, df_RL_MLP, df_GP, df_Heuristics], axis=1)
        df_all = df_all.iloc[:-1]

        temp = []
        for name in df_all.columns:
            if name != "RL-GNN":
                res = stats.ttest_rel(df_all["RL-GNN"], df_all[name])
                pvalue = res.pvalue
                temp.append(pvalue)
        p_values.append(temp)

    df_results = pd.DataFrame(p_values, columns=df_all.columns[1:])
    writer = pd.ExcelWriter('./output/test/ttest.xlsx')
    df_results.to_excel(writer, sheet_name="results")
    writer.close()

def get_lowerbound(row_range=("A", "B"), bay_range=(1, 40), output_points=(23, 27, 44)):
    x_velocity = 0.5
    y_velocity = 1.0

    row_list = [chr(i) for i in range(ord(row_range[0]), ord(row_range[1]) + 1)]
    bay_list = [i for i in range(bay_range[0] - 1, bay_range[1] + 1)]
    coord_mapping = {}
    for i, row_id in enumerate(row_list):
        for j, bay_id in enumerate(bay_list):
            name = row_id + str(bay_id).rjust(2, '0')
            x_coord = j + 1
            y_coord = i + 1
            if output_points[0] <= x_coord:
                x_coord += 1
            if output_points[1] <= x_coord:
                x_coord += 1
            coord_mapping[name] = (x_coord, y_coord)

    # data_dir = ["./input/test/basic_test/5-10-10/",
    #             "./input/test/scalability_test/5-10-15/",
    #             "./input/test/scalability_test/5-10-20/",
    #             "./input/test/scalability_test/5-15-10/",
    #             "./input/test/scalability_test/5-15-15/",
    #             "./input/test/scalability_test/5-15-20/",
    #             "./input/test/scalability_test/5-20-10/",
    #             "./input/test/scalability_test/5-20-15/",
    #             "./input/test/scalability_test/5-20-20/"]
    # log_dir = ["./output/test/basic_test/5-10-10/",
    #            "./output/test/scalability_test/5-10-15/",
    #            "./output/test/scalability_test/5-10-20/",
    #            "./output/test/scalability_test/5-15-10/",
    #            "./output/test/scalability_test/5-15-15/",
    #            "./output/test/scalability_test/5-15-20/",
    #            "./output/test/scalability_test/5-20-10/",
    #            "./output/test/scalability_test/5-20-15/",
    #            "./output/test/scalability_test/5-20-20/"]

    data_dir = ["./input/test/computational_complexity/1600/",
                "./input/test/computational_complexity/1800/",
                "./input/test/computational_complexity/2200/",
                "./input/test/computational_complexity/2400/"]
    log_dir = ["./output/test/computational_complexity/1600/",
               "./output/test/computational_complexity/1800/",
               "./output/test/computational_complexity/2200/",
               "./output/test/computational_complexity/2400/"]


    for log_dir_temp in log_dir:
        if not os.path.exists(log_dir_temp):
            os.makedirs(log_dir_temp)

    for data_dir_temp, log_dir_temp in zip(data_dir, log_dir):
        file_paths = os.listdir(data_dir_temp)
        index = ["P%d" % i for i in range(1, len(file_paths) + 1)] + ["avg"]
        columns = ["LB"]
        lower_bounds = []

        df_lowerbound = pd.DataFrame(index=index, columns=columns)
        for prob, path in zip(index, file_paths):
            df_storage = pd.read_excel(data_dir_temp + path, sheet_name="storage", engine='openpyxl')
            df_reshuffle = pd.read_excel(data_dir_temp + path, sheet_name="reshuffle", engine='openpyxl')

            df_storage["from_coord"] = df_storage["pileno"].apply(lambda x: coord_mapping[x])
            df_storage["to_coord"] = df_storage["topile"].apply(lambda x: coord_mapping[x])
            df_storage["moving_time"] = df_storage.apply(lambda row: max(abs(row["to_coord"][0] - row["from_coord"][0]) / x_velocity,
                                                                         abs(row["to_coord"][1] - row["from_coord"][1]) / y_velocity), axis=1)

            df_reshuffle["from_coord"] = df_reshuffle["pileno"].apply(lambda x: coord_mapping[x])
            df_reshuffle["to_coord"] = df_reshuffle["topile"].apply(lambda x: coord_mapping[x])
            df_reshuffle["moving_time"] = df_reshuffle.apply(lambda row: max(abs(row["to_coord"][0] - row["from_coord"][0]) / x_velocity,
                                                                             abs(row["to_coord"][1] - row["from_coord"][1]) / y_velocity), axis=1)

            travel_time_sum = (df_storage["moving_time"].sum() + df_reshuffle["moving_time"].sum()) / 2
            lower_bounds.append(travel_time_sum)

        df_lowerbound["LB"] = lower_bounds + [sum(lower_bounds) / len(lower_bounds)]

        writer = pd.ExcelWriter(log_dir_temp + 'LB.xlsx')
        df_lowerbound.to_excel(writer, sheet_name="makespan")
        writer.close()

def summarize(log_dir):
    for log_dir_temp in log_dir:
        df_RL_GNN = pd.read_excel(log_dir_temp + '(RL-GNN) test_results.xlsx', index_col=0)
        df_RL_GNN = df_RL_GNN.rename(columns={"RL": "RL-GNN"})

        df_RL_MLP = pd.read_excel(log_dir_temp + '(RL-MLP) test_results.xlsx', index_col=0)
        df_RL_MLP = df_RL_MLP.rename(columns={"RL": "RL-MLP"})

        df_GP = pd.read_excel(log_dir_temp + '(GP) test_results.xlsx', index_col=0)
        df_Heuristics = pd.read_excel(log_dir_temp + '(Heuristics) test_results.xlsx', index_col=0)

        df_LB = pd.read_excel(log_dir_temp + 'LB.xlsx', index_col=0)
        df_LB = df_LB.iloc[:-1]

        df_all = pd.concat([df_RL_GNN, df_RL_MLP, df_GP, df_Heuristics], axis=1)
        df_all = df_all.iloc[:-1]

        df_results = (df_all.values - df_LB.values) / df_LB.values * 100
        df_results = pd.DataFrame(df_results, columns=df_all.columns)
        df_results.loc["avg"] = df_results.apply(np.mean, axis=0)

        writer = pd.ExcelWriter(log_dir_temp + 'summary.xlsx')
        df_results.to_excel(writer, sheet_name="results")
        writer.close()


if __name__ == "__main__":
    log_dir = ["./output/test/basic_test/5-10-10/",
               "./output/test/scalability_test/5-10-15/",
               "./output/test/scalability_test/5-10-20/",
               "./output/test/scalability_test/5-15-10/",
               "./output/test/scalability_test/5-15-15/",
               "./output/test/scalability_test/5-15-20/",
               "./output/test/scalability_test/5-20-10/",
               "./output/test/scalability_test/5-20-15/",
               "./output/test/scalability_test/5-20-20/"]

    perform_paired_t_test(log_dir)


