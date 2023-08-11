import os
import random
import itertools
import numpy as np
import pandas as pd

from utilities import get_coord


def read_data(file_name, bay=1, num_crane=1):
    df = pd.read_csv(file_name, encoding="CP949")

    if bay == 1:
        bay_list = ["A", "B"]
    elif bay == 2:
        bay_list = ["C", "D"]
    elif bay == 3:
        bay_list = ["E", "F"]
    elif bay == 4:
        bay_list = ["S", "T"]
    else:
        raise KeyError("invalid input value: {0} BAY".format(bay))

    data = {"Crane-%d" % i : {} for i in range(1, num_crane + 1)}
    for i in range(1, num_crane + 1):
        pile_range = range(int(40 * (i - 1) / num_crane) + 1, int(40 * i / num_crane) + 1)
        pile_list = [row_id + str(col_id).rjust(2, '0') for row_id in bay_list for col_id in pile_range]
        df_sub = df[df['pileno'].isin(pile_list)]

        df_sorting = df_sub[df_sub["topile"].isin(pile_list)]
        from_piles = list(df_sorting["pileno"].unique())
        to_piles = {}
        for from_pile in from_piles:
            temp = df_sorting[df_sorting["pileno"] == from_pile]
            temp = temp[temp["topile"].isin(from_piles)]
            to_piles[from_pile] = list(temp["topile"].unique())

        remove_list = {i:[] for i in from_piles}
        for from_pile in from_piles:
            for to_pile in to_piles[from_pile]:
                if from_pile in to_piles[to_pile]:
                    remove_list[from_pile].append(to_pile)

        for key, value in remove_list.items():
            df_sorting = df_sorting[~((df_sorting["pileno"] == key) & (df_sorting["topile"].isin(value)))]
        df_sorting = df_sorting.sort_values(by=["pileno", "pileseq"])

        df_release = df_sub[df_sub["topile"].isin(["CN1", "CN2"])]
        df_release = df_release[~df_release["pileno"].isin(df_sorting["pileno"])]
        df_release = df_release.sort_values(by=["pileno", "pileseq"])

        data["Crane-%d" % i]["num_from_pile"] = len(df_sorting["pileno"].unique())
        data["Crane-%d" % i]["num_to_pile"] = len(df_sorting["topile"].unique())
        data["Crane-%d" % i]["num_release_pile"] = 0  # len(df_release["pileno"].unique())
        data["Crane-%d" % i]["sorting_plan"] = df_sorting
        data["Crane-%d" % i]["release_plan"] = df_release

    return data


class DataGenerator:
    def __init__(self, rows=("A", "B"),  # row 이름
                       storage=True,  # 적치 계획 데이터를 생성할 지 여부
                       reshuffle=True,  # 선별 계획 데이터를 생성할 지 여부
                       retrieval=True,  # 출고 계획 데이터를 생성할 지 여부
                       n_bays_in_area1=15,  # 1번 영역 내 bay의 수
                       n_bays_in_area2=6,  # 2번 영역 내 bay의 수
                       n_bays_in_area3=3,  # 3번 영역 내 bay의 수
                       n_bays_in_area4=6,  # 4번 영역 내 bay의 수
                       n_bays_in_area5=10,  # 5번 영역 내 bay의 수
                       n_bays_in_area6=1,  # 6번 영역 내 bay의 수
                       n_from_piles_storage=2,  # 적치 작업의 대상 강재가 적치된 가상 파일의 수 (from pile)
                       n_to_piles_storage=5,  # 적치 작업의 대상 강재가 이동할 파일의 수 (to pile)
                       n_from_piles_reshuffle=10,  # 선별 작업의 대상 강재가 적치된 파일의 수 (from pile)
                       n_to_piles_reshuffle=10,   # 선별 작업의 대상 강재가 이동할 파일의 수 (to pile)
                       n_from_piles_retrieval_cn1=5,  # cn1 출고 작업의 대상 강재가 적치된 파일의 수 (from pile)
                       n_from_piles_retrieval_cn2=5,  # cn2 출고 작업의 대상 강재가 적치된 파일의 수 (from pile)
                       n_from_piles_retrieval_cn3=2,  # cn3 출고 작업의 대상 강재가 적치된 파일의 수 (from pile)
                       n_plates_storage=150,  # 입고 지점에 위치한 적치 대상 강재의 평균 개수
                       n_plates_reshuffle=150,  # 각 파일에 위치한 선별 대상 강재의 평균 개수
                       n_plates_retrieval=150,  # 각 파일에 위치한 출고 대상 강재의 평균 개수
                       working_crane_ids=("Crane-1", "Crane-2"),  # 작업을 수행할 크레인
                       safety_margin=5,  # 크레인 간 안전 거리
                 ):

        self.rows = rows
        self.storage = storage
        self.reshuffle = reshuffle
        self.retrieval = retrieval
        self.n_bays_in_area1 = n_bays_in_area1
        self.n_bays_in_area2 = n_bays_in_area2
        self.n_bays_in_area3 = n_bays_in_area3
        self.n_bays_in_area4 = n_bays_in_area4
        self.n_bays_in_area5 = n_bays_in_area5
        self.n_bays_in_area6 = n_bays_in_area6
        self.n_from_piles_storage = n_from_piles_storage
        self.n_to_piles_storage = n_to_piles_storage
        self.n_from_piles_reshuffle = n_from_piles_reshuffle
        self.n_to_piles_reshuffle = n_to_piles_reshuffle
        self.n_from_piles_retrieval_cn1 = n_from_piles_retrieval_cn1
        self.n_from_piles_retrieval_cn2 = n_from_piles_retrieval_cn2
        self.n_from_piles_retrieval_cn3 = n_from_piles_retrieval_cn3
        self.n_plates_storage = n_plates_storage
        self.n_plates_reshuffle = n_plates_reshuffle
        self.n_plates_retrieval = n_plates_retrieval
        self.working_crane_ids = working_crane_ids
        self.safety_margin = safety_margin

    def generate(self, file_path=None):
        # 입고, 선별, 출고 데이터를 저장하기 위한 데이터프레임 생성
        df_storage = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])
        df_reshuffle = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])
        df_retrieval = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])

        # 강재 적치장 내 모든 파일이 포함된 리스트 생성
        mapping_from_pile_to_x = {}

        piles_in_area0 = []
        for row_id in self.rows:
            pile = row_id + "00"
            piles_in_area0.append(pile)
            mapping_from_pile_to_x[pile] = 1

        piles_in_area1, piles_in_area2, piles_in_area3, piles_in_area4, piles_in_area5, piles_in_area6 = [], [], [], [], [], []
        n_bays_cum = np.cumsum([self.n_bays_in_area1, self.n_bays_in_area2, self.n_bays_in_area3,
                                self.n_bays_in_area4, self.n_bays_in_area5, self.n_bays_in_area6])

        for row_id in self.rows:
            for col_id in range(1, n_bays_cum[-1] + 1):
                pile = row_id + str(col_id).rjust(2, '0')
                if col_id <= n_bays_cum[0]:
                    piles_in_area1.append(pile)
                    mapping_from_pile_to_x[pile] = col_id + 1
                elif n_bays_cum[0] < col_id <= n_bays_cum[1]:
                    piles_in_area2.append(pile)
                    mapping_from_pile_to_x[pile] = col_id + 1
                elif n_bays_cum[1] < col_id <= n_bays_cum[2]:
                    piles_in_area3.append(pile)
                    mapping_from_pile_to_x[pile] = col_id + 2
                elif n_bays_cum[2] < col_id <= n_bays_cum[3]:
                    piles_in_area4.append(pile)
                    mapping_from_pile_to_x[pile] = col_id + 3
                elif n_bays_cum[3] < col_id <= n_bays_cum[4]:
                    piles_in_area5.append(pile)
                    mapping_from_pile_to_x[pile] = col_id + 3
                else:
                    piles_in_area6.append(pile)
                    mapping_from_pile_to_x[pile] = col_id + 3

        piles_all = piles_in_area0 + piles_in_area1 + piles_in_area2 \
                    + piles_in_area3 + piles_in_area4 + piles_in_area5 + piles_in_area6
        x_max = max(mapping_from_pile_to_x.values()) + 1

        # 출고 계획 생성
        if self.retrieval:
            candidates = piles_in_area2 + piles_in_area3 + piles_in_area4
            # candidates = [i for i in candidates if not i in ["A22", "A23", "A24", "B22", "B23", "B24"]]
            from_piles_retrieval_cn1 = random.sample(candidates, self.n_from_piles_retrieval_cn1)
            candidates = [i for i in candidates if not i in from_piles_retrieval_cn1]
            from_piles_retrieval_cn2 = random.sample(candidates, self.n_from_piles_retrieval_cn2)
            if "Crane-2" in self.working_crane_ids:
                candidates = piles_in_area6
                from_piles_retrieval_cn3 = random.sample(candidates, self.n_from_piles_retrieval_cn3)
            else:
                from_piles_retrieval_cn3 = []
            from_piles_retrieval = from_piles_retrieval_cn1 + from_piles_retrieval_cn2 + from_piles_retrieval_cn3
            for pile in from_piles_retrieval:
                num_of_plates = random.randint(int(0.9 * self.n_plates_retrieval), int(1.1 * self.n_plates_retrieval))
                pileno = [pile] * num_of_plates
                pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
                markno = ["SP-RT-%s-%s" % (pile, i) for i in pileseq]
                unitw = np.random.uniform(0.141, 19.294, num_of_plates)
                if pile in from_piles_retrieval_cn1:
                    topile = ["cn1"] * num_of_plates
                elif pile in from_piles_retrieval_cn2:
                    topile = ["cn2"] * num_of_plates
                else:
                    topile = ["cn3"] * num_of_plates
                df_temp = pd.DataFrame(
                    {"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
                df_retrieval = pd.concat([df_retrieval, df_temp], ignore_index=True)
        else:
            from_piles_retrieval = []

        # 선별 계획 생성
        if self.reshuffle:
            candidates = piles_in_area1 + piles_in_area5
            if not "Crane-1" in self.working_crane_ids:
                candidates = [i for i in candidates if mapping_from_pile_to_x[i] >= 1 + self.safety_margin]
            if not "Crane-2" in self.working_crane_ids:
                candidates = [i for i in candidates if mapping_from_pile_to_x[i] <= x_max - self.safety_margin]
            from_piles_reshuffle = random.sample(candidates, self.n_from_piles_reshuffle)
            candidates = [i for i in piles_all if (i not in from_piles_retrieval) and (i not in piles_in_area0)]
            candidates = [i for i in candidates if i not in from_piles_reshuffle]
            if not "Crane-1" in self.working_crane_ids:
                candidates = [i for i in candidates if mapping_from_pile_to_x[i] >= 1 + self.safety_margin]
            if not "Crane-2" in self.working_crane_ids:
                candidates = [i for i in candidates if mapping_from_pile_to_x[i] <= x_max - self.safety_margin]
            to_piles_reshuffle = random.sample(candidates, self.n_to_piles_reshuffle)
            # to_piles_reshuffle = ["A22", "A23", "A24", "B22", "B23", "B24"]

            for pile in from_piles_reshuffle:
                x = mapping_from_pile_to_x[pile]
                if x < 1 + self.safety_margin:
                    to_piles_reshuffle_rev = [i for i in to_piles_reshuffle
                                              if mapping_from_pile_to_x[i] <= x_max - self.safety_margin]
                elif x > x_max - self.safety_margin:
                    to_piles_reshuffle_rev = [i for i in to_piles_reshuffle
                                              if mapping_from_pile_to_x[i] >= 1 + self.safety_margin]
                else:
                    to_piles_reshuffle_rev = to_piles_reshuffle
                num_of_plates = random.randint(int(0.9 * self.n_plates_reshuffle), int(1.1 * self.n_plates_reshuffle))
                pileno = [pile] * num_of_plates
                pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
                markno = ["SP-RS-%s-%s" % (pile, i) for i in pileseq]
                unitw = np.random.uniform(0.141, 19.294, num_of_plates)
                topile = random.choices(to_piles_reshuffle_rev, k=num_of_plates)
                df_temp = pd.DataFrame(
                    {"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
                df_reshuffle = pd.concat([df_reshuffle, df_temp], ignore_index=True)
        else:
            from_piles_reshuffle = []

        # 적치 계획 생성
        if self.storage:
            if "Crane-1" in self.working_crane_ids:
                from_piles_storage = random.sample(piles_in_area0, self.n_from_piles_storage)
            else:
                from_piles_storage = []
            candidates = piles_in_area1 + piles_in_area5
            candidates = [i for i in candidates if i not in from_piles_reshuffle]
            candidates = [i for i in candidates if mapping_from_pile_to_x[i] <= x_max - self.safety_margin]
            to_piles_storage = random.sample(candidates, self.n_to_piles_storage)
            # to_piles_storage = ["A22", "A23", "A24", "B22", "B23", "B24"]

            for pile in from_piles_storage:
                num_of_plates = random.randint(int(0.9 * self.n_plates_storage), int(1.1 * self.n_plates_storage))
                pileno = [pile] * num_of_plates
                pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
                markno = ["SP-ST-%s-%s" % (pile, i) for i in pileseq]
                unitw = np.random.uniform(0.141, 19.294, num_of_plates)
                topile = random.choices(to_piles_storage, k=num_of_plates)
                df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
                df_storage = pd.concat([df_storage, df_temp], ignore_index=True)

        if file_path is not None:
            writer = pd.ExcelWriter(file_path)
            df_storage.to_excel(writer, sheet_name="storage", index=False)
            df_reshuffle.to_excel(writer, sheet_name="reshuffle", index=False)
            df_retrieval.to_excel(writer, sheet_name="retrieval", index=False)
            writer.save()

        return df_storage, df_reshuffle, df_retrieval


if __name__ == '__main__':
    rows = ("A", "B")

    storage = True
    reshuffle = True
    retrieval = True

    n_bays_in_area1 = 15
    n_bays_in_area2 = 6
    n_bays_in_area3 = 3
    n_bays_in_area4 = 6
    n_bays_in_area5 = 9
    n_bays_in_area6 = 1

    n_from_piles_storage = 1
    n_to_piles_storage = 5
    n_from_piles_reshuffle = 10
    n_to_piles_reshuffle = 10
    n_from_piles_retrieval_cn1 = 5
    n_from_piles_retrieval_cn2 = 5
    n_from_piles_retrieval_cn3 = 2

    n_plates_storage = 500
    n_plates_reshuffle = 150
    n_plates_retrieval = 150

    working_crane_ids = ("Crane-1", "Crane-2")
    safety_margin = 5
    file_dir = "../input/data/validation/"

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    data_src = DataGenerator(rows=rows,
                             storage=storage,
                             reshuffle=reshuffle,
                             retrieval=retrieval,
                             n_bays_in_area1=n_bays_in_area1,
                             n_bays_in_area2=n_bays_in_area2,
                             n_bays_in_area3=n_bays_in_area3,
                             n_bays_in_area4=n_bays_in_area4,
                             n_bays_in_area5=n_bays_in_area5,
                             n_bays_in_area6=n_bays_in_area6,
                             n_from_piles_storage=n_from_piles_storage,
                             n_to_piles_storage=n_to_piles_storage,
                             n_from_piles_reshuffle=n_from_piles_reshuffle,
                             n_to_piles_reshuffle=n_to_piles_reshuffle,
                             n_from_piles_retrieval_cn1=n_from_piles_retrieval_cn1,
                             n_from_piles_retrieval_cn2=n_from_piles_retrieval_cn2,
                             n_from_piles_retrieval_cn3=n_from_piles_retrieval_cn3,
                             n_plates_storage=n_plates_storage,
                             n_plates_reshuffle=n_plates_reshuffle,
                             n_plates_retrieval=n_plates_retrieval,
                             working_crane_ids=working_crane_ids,
                             safety_margin=safety_margin)

    iteration = 5
    for i in range(1, iteration + 1):
        file_path = file_dir + "instance-{0}.xlsx".format(i)
        df_storage, df_reshuffle, df_retrieval = data_src.generate(file_path=file_path)