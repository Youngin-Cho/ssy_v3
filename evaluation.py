import os
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


def get_lowerbound(row_range=("A", "B"), bay_range=(1, 40), output_points=(22, 26, 44)):
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

    data_dir = ["./input/data/test/basic_test/5-10-10/",
                "./input/data/test/scalability_test/storage_plan/3-10-10/",
                "./input/data/test/scalability_test/storage_plan/4-10-10/",
                "./input/data/test/scalability_test/storage_plan/6-10-10/",
                "./input/data/test/scalability_test/storage_plan/7-10-10/",
                "./input/data/test/scalability_test/reshuffling_plan/5-8-10/",
                "./input/data/test/scalability_test/reshuffling_plan/5-9-10/",
                "./input/data/test/scalability_test/reshuffling_plan/5-11-10/",
                "./input/data/test/scalability_test/reshuffling_plan/5-12-10/",
                "./input/data/test/scalability_test/reshuffling_plan/5-10-8/",
                "./input/data/test/scalability_test/reshuffling_plan/5-10-9/",
                "./input/data/test/scalability_test/reshuffling_plan/5-10-11/",
                "./input/data/test/scalability_test/reshuffling_plan/5-10-12/"]

    res_dir = ["./output/test/basic_test/5-10-10/",
                "./output/test/scalability_test/storage_plan/3-10-10/",
                "./output/test/scalability_test/storage_plan/4-10-10/",
                "./output/test/scalability_test/storage_plan/6-10-10/",
                "./output/test/scalability_test/storage_plan/7-10-10/",
                "./output/test/scalability_test/reshuffling_plan/5-8-10/",
                "./output/test/scalability_test/reshuffling_plan/5-9-10/",
                "./output/test/scalability_test/reshuffling_plan/5-11-10/",
                "./output/test/scalability_test/reshuffling_plan/5-12-10/",
                "./output/test/scalability_test/reshuffling_plan/5-10-8/",
                "./output/test/scalability_test/reshuffling_plan/5-10-9/",
                "./output/test/scalability_test/reshuffling_plan/5-10-11/",
                "./output/test/scalability_test/reshuffling_plan/5-10-12/"]

    for res_dir_temp in res_dir:
        if not os.path.exists(res_dir_temp):
            os.makedirs(res_dir_temp)

    for data_dir_temp, res_dir_temp in zip(data_dir, res_dir):
        file_paths = os.listdir(data_dir_temp)
        index = ["P%d" % i for i in range(1, len(file_paths) + 1)] + ["avg"]
        columns = ["Ideal"]
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

        df_lowerbound["lb"] = lower_bounds + [sum(lower_bounds) / len(lower_bounds)]

        writer = pd.ExcelWriter(res_dir_temp + 'baseline.xlsx')
        df_lowerbound.to_excel(writer, sheet_name="lc")
        writer.close()

if __name__ == "__main__":
    get_lowerbound()


