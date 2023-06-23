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


def generate_data_pre(num_of_storage_to_piles=10,  # 적치 작업 시 강재를 적치할 파일의 수
                  num_of_reshuffle_from_piles=10,  # 선별 작업 시 이동할 강재가 적치된 파일의 수
                  num_of_reshuffle_to_piles=20,   # 선별 작업 시 강재가 이동할 파일의 수
                  num_of_retrieval_from_piles=4,   # 출고 작업 시 이동할 강재가 적치된 파일의 수
                  bays=("A", "B"),  # 강재적치장의 베이 이름
                  safety_margin=5,
                  file_path=None):

    # 입고, 선별, 출고 데이터를 저장하기 위한 데이터프레임 생성
    df_storage = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])
    df_reshuffle = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])
    df_retrieval = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])

    # 강재 적치장 내 모든 파일이 포함된 리스트 생성
    piles_all = [row_id + str(col_id).rjust(2, '0') for row_id in bays for col_id in range(1, 41)]

    # 출고일이 2주 이하로 남은 강재가 적치되는 파일 리스트
    piles_conveyor = [row_id + str(col_id).rjust(2, '0') for row_id in bays for col_id in range(16, 31)]
    piles_truck = [bays[0] + "40", bays[1] + "40"]
    # 출고일이 2주 이상 남은 강재가 적치되는 파일 리스트
    piles_misc = [i for i in piles_all if (i not in piles_conveyor) and (i not in piles_truck)]

    # 출고 작업 대상 강재 데이터 생성
    if num_of_retrieval_from_piles != 0:
        # piles_conveyor = [i for i in piles_conveyor if not i in ["A22", "A23", "A24", "B22", "B23", "B24"]]
        retrieval_from_piles = random.sample(piles_conveyor, num_of_retrieval_from_piles) + ["A40"]
        retrieval_from_piles_to_cn1 = random.sample(retrieval_from_piles[:-1], int(num_of_retrieval_from_piles / 2))
        retrieval_from_piles_to_cn2 = [i for i in retrieval_from_piles[:-1] if i not in retrieval_from_piles_to_cn1]
        # retrieval_from_piles_to_cn1 = retrieval_from_piles[:-1]
        # retrieval_from_piles_to_cn2 = []
        for pile in retrieval_from_piles:
            # num_of_plates = random.randint(150, 201)
            num_of_plates = 150
            pileno = [pile] * num_of_plates
            pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
            markno = ["SP-RT-%s-%s" % (pile, i) for i in pileseq]
            unitw = np.random.uniform(0.141, 19.294, num_of_plates)
            if pile in retrieval_from_piles_to_cn1:
                topile = ["cn1"] * num_of_plates
            elif pile in retrieval_from_piles_to_cn2:
                topile = ["cn2"] * num_of_plates
            else:
                topile = ["cn3"] * num_of_plates
            df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
            df_retrieval = pd.concat([df_retrieval, df_temp], ignore_index=True)
    else:
        retrieval_from_piles = []

    piles_all = [i for i in piles_all if i not in retrieval_from_piles]

    if num_of_reshuffle_from_piles != 0:
        # 선별 작업 대상 강재 데이터 생성
        reshuffle_from_piles = random.sample(piles_misc, num_of_reshuffle_from_piles)
        piles_all = [i for i in piles_all if i not in reshuffle_from_piles]
        reshuffle_to_piles = random.sample(piles_all, num_of_reshuffle_to_piles)
        # reshuffle_to_piles = ["A22", "A23", "A24", "B22", "B23", "B24"]

        for pile in reshuffle_from_piles:
            x_coord = get_coord(pile)[0]
            if x_coord < 1 + safety_margin:
                reshuffle_to_piles_rev = [i for i in reshuffle_to_piles if get_coord(i)[0] <= 43 - safety_margin]
            elif x_coord > 43 - safety_margin:
                reshuffle_to_piles_rev = [i for i in reshuffle_to_piles if get_coord(i)[0] >= 1 + safety_margin]
            else:
                reshuffle_to_piles_rev = reshuffle_to_piles
            # num_of_plates = random.randint(5, 15)
            num_of_plates = 150
            pileno = [pile] * num_of_plates
            pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
            markno = ["SP-RS-%s-%s" % (pile, i) for i in pileseq]
            unitw = np.random.uniform(0.141, 19.294, num_of_plates)
            topile = random.choices(reshuffle_to_piles_rev, k=num_of_plates)
            df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
            df_reshuffle = pd.concat([df_reshuffle, df_temp], ignore_index=True)

        piles_misc = [i for i in piles_all if i not in reshuffle_from_piles]

    piles_misc = [i for i in piles_misc if get_coord(i)[0] <= 43 - safety_margin]

    # 입고 작업 대상 강재 데이터 생성
    if num_of_storage_to_piles != 0:
        storage_from_piles = [bays[0] + "00"]  # [bays[0] + "00", bays[1] + "00"]
        storage_to_piles = random.sample(piles_misc, num_of_storage_to_piles)
        # storage_to_piles = ["A22", "A23", "A24", "B22", "B23", "B24"]
        for pile in storage_from_piles:
            # num_of_plates = random.randint(35, 45)
            num_of_plates = 500
            pileno = [pile] * num_of_plates
            pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
            markno = ["SP-ST-%s-%s" % (pile, i) for i in pileseq]
            unitw = np.random.uniform(0.141, 19.294, num_of_plates)
            topile = random.choices(storage_to_piles, k=num_of_plates)
            df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
            df_storage = pd.concat([df_storage, df_temp], ignore_index=True)

    if file_path is not None:
        writer = pd.ExcelWriter(file_path)
        df_storage.to_excel(writer, sheet_name="storage", index=False)
        df_reshuffle.to_excel(writer, sheet_name="reshuffle", index=False)
        df_retrieval.to_excel(writer, sheet_name="retrieval", index=False)
        writer.save()

    return df_storage, df_reshuffle, df_retrieval


def generate_data_temp(num_of_storage_to_piles=10,  # 적치 작업 시 강재를 적치할 파일의 수
                       num_of_reshuffle_from_piles=10,  # 선별 작업 시 이동할 강재가 적치된 파일의 수
                       num_of_reshuffle_to_piles=20,   # 선별 작업 시 강재가 이동할 파일의 수
                       num_of_retrieval_from_piles=4,   # 출고 작업 시 이동할 강재가 적치된 파일의 수
                       bays=("A", "B"),  # 강재적치장의 베이 이름
                       safety_margin=5,
                       file_path=None):

    # 입고, 선별, 출고 데이터를 저장하기 위한 데이터프레임 생성
    df_storage = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])
    df_reshuffle = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])
    df_retrieval = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])

    # 강재 적치장 내 모든 파일이 포함된 리스트 생성
    piles_all = [row_id + str(col_id).rjust(2, '0') for row_id in bays for col_id in range(1, 52)]

    # 출고일이 2주 이하로 남은 강재가 적치되는 파일 리스트
    piles_conveyor = [row_id + str(col_id).rjust(2, '0') for row_id in bays for col_id in range(19, 39)]
    piles_truck = [bays[0] + "51", bays[1] + "51"]
    # 출고일이 2주 이상 남은 강재가 적치되는 파일 리스트
    piles_misc = [i for i in piles_all if (i not in piles_conveyor) and (i not in piles_truck)]

    # 출고 작업 대상 강재 데이터 생성
    if num_of_retrieval_from_piles != 0:
        # piles_conveyor = [i for i in piles_conveyor if not i in ["A22", "A23", "A24", "B22", "B23", "B24"]]
        retrieval_from_piles = random.sample(piles_conveyor, num_of_retrieval_from_piles) + ["A51"]
        retrieval_from_piles_to_cn1 = random.sample(retrieval_from_piles[:-1], int(num_of_retrieval_from_piles / 2))
        retrieval_from_piles_to_cn2 = [i for i in retrieval_from_piles[:-1] if i not in retrieval_from_piles_to_cn1]
        # retrieval_from_piles_to_cn1 = retrieval_from_piles[:-1]
        # retrieval_from_piles_to_cn2 = []
        for pile in retrieval_from_piles:
            # num_of_plates = random.randint(150, 201)
            num_of_plates = 150
            pileno = [pile] * num_of_plates
            pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
            markno = ["SP-RT-%s-%s" % (pile, i) for i in pileseq]
            unitw = np.random.uniform(0.141, 19.294, num_of_plates)
            if pile in retrieval_from_piles_to_cn1:
                topile = ["cn1"] * num_of_plates
            elif pile in retrieval_from_piles_to_cn2:
                topile = ["cn2"] * num_of_plates
            else:
                topile = ["cn3"] * num_of_plates
            df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
            df_retrieval = pd.concat([df_retrieval, df_temp], ignore_index=True)
    else:
        retrieval_from_piles = []

    piles_all = [i for i in piles_all if i not in retrieval_from_piles]

    if num_of_reshuffle_from_piles != 0:
        # 선별 작업 대상 강재 데이터 생성
        reshuffle_from_piles = random.sample(piles_misc, num_of_reshuffle_from_piles)
        piles_all = [i for i in piles_all if i not in reshuffle_from_piles]
        reshuffle_to_piles = random.sample(piles_all, num_of_reshuffle_to_piles)
        # reshuffle_to_piles = ["A22", "A23", "A24", "B22", "B23", "B24"]

        for pile in reshuffle_from_piles:
            x_coord = get_coord(pile)[0]
            if x_coord < 1 + safety_margin:
                reshuffle_to_piles_rev = [i for i in reshuffle_to_piles if get_coord(i)[0] <= 54 - safety_margin]
            elif x_coord > 54 - safety_margin:
                reshuffle_to_piles_rev = [i for i in reshuffle_to_piles if get_coord(i)[0] >= 1 + safety_margin]
            else:
                reshuffle_to_piles_rev = reshuffle_to_piles
            # num_of_plates = random.randint(5, 15)
            num_of_plates = 150
            pileno = [pile] * num_of_plates
            pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
            markno = ["SP-RS-%s-%s" % (pile, i) for i in pileseq]
            unitw = np.random.uniform(0.141, 19.294, num_of_plates)
            topile = random.choices(reshuffle_to_piles_rev, k=num_of_plates)
            df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
            df_reshuffle = pd.concat([df_reshuffle, df_temp], ignore_index=True)

        piles_misc = [i for i in piles_all if i not in reshuffle_from_piles]

    piles_misc = [i for i in piles_misc if get_coord(i)[0] <= 54 - safety_margin]

    # 입고 작업 대상 강재 데이터 생성
    if num_of_storage_to_piles != 0:
        storage_from_piles = [bays[0] + "00"]  # [bays[0] + "00", bays[1] + "00"]
        storage_to_piles = random.sample(piles_misc, num_of_storage_to_piles)
        # storage_to_piles = ["A22", "A23", "A24", "B22", "B23", "B24"]
        for pile in storage_from_piles:
            # num_of_plates = random.randint(35, 45)
            num_of_plates = 500
            pileno = [pile] * num_of_plates
            pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
            markno = ["SP-ST-%s-%s" % (pile, i) for i in pileseq]
            unitw = np.random.uniform(0.141, 19.294, num_of_plates)
            topile = random.choices(storage_to_piles, k=num_of_plates)
            df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
            df_storage = pd.concat([df_storage, df_temp], ignore_index=True)

    if file_path is not None:
        writer = pd.ExcelWriter(file_path)
        df_storage.to_excel(writer, sheet_name="storage", index=False)
        df_reshuffle.to_excel(writer, sheet_name="reshuffle", index=False)
        df_retrieval.to_excel(writer, sheet_name="retrieval", index=False)
        writer.save()

    return df_storage, df_reshuffle, df_retrieval

def generate_data(rows=("A", "B"),  # row 이름
                  storage=True,  # 적치 계획 데이터를 생성할 지 여부
                  reshuffle=True,  # 선별 계획 데이터를 생성할 지 여부
                  retrieval=True,  # 출고 계획 데이터를 생성할 지 여부
                  n_bays_in_area0=1, # 0번 영역 내 bay의 수
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
                  n_plates_for_storage=150,  # 입고 지점에 위치한 적치 대상 강재의 평균 개수
                  n_plates_for_reshuffle=150,  # 각 파일에 위치한 선별 대상 강재의 평균 개수
                  n_plates_for_retrieval=150,  # 각 파일에 위치한 출고 대상 강재의 평균 개수
                  safety_margin=5,  # 크레인 간 안전 거리
                  file_path=None):

    # 입고, 선별, 출고 데이터를 저장하기 위한 데이터프레임 생성
    df_storage = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])
    df_reshuffle = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])
    df_retrieval = pd.DataFrame(columns=["pileno", "pileseq", "markno", "unitw", "topile"])

    # 강재 적치장 내 모든 파일이 포함된 리스트 생성
    mapping_from_pile_to_x = {}

    piles_in_area0 = []
    for row_id in rows:
        for col_id in range(0, n_bays_in_area0):
            # pile = "I" + row_id + str(col_id).rjust(2, '0')
            pile = row_id + str(col_id).rjust(2, '0')
            piles_in_area0.append(pile)
            mapping_from_pile_to_x[pile] = col_id

    piles_in_area1, piles_in_area2, piles_in_area3, piles_in_area4, piles_in_area5, piles_in_area6 = [], [], [], [], [], []
    n_bays_cum = np.cumsum([n_bays_in_area1, n_bays_in_area2, n_bays_in_area3, n_bays_in_area4, n_bays_in_area5, n_bays_in_area6])
    for row_id in rows:
        for col_id in range(1, n_bays_cum[-1] + 1):
            pile = row_id + str(col_id).rjust(2, '0')
            if col_id <= n_bays_cum[0]:
                piles_in_area1.append(pile)
                mapping_from_pile_to_x[pile] = n_bays_in_area0 + col_id
            elif n_bays_cum[0] < col_id <= n_bays_cum[1]:
                piles_in_area2.append(pile)
                mapping_from_pile_to_x[pile] = n_bays_in_area0 + col_id
            elif n_bays_cum[1] < col_id <= n_bays_cum[2]:
                piles_in_area3.append(pile)
                mapping_from_pile_to_x[pile] = n_bays_in_area0 + col_id + 1
            elif n_bays_cum[2] < col_id <= n_bays_cum[3]:
                piles_in_area4.append(pile)
                mapping_from_pile_to_x[pile] = n_bays_in_area0 + col_id + 2
            elif n_bays_cum[3] < col_id <= n_bays_cum[4]:
                piles_in_area5.append(pile)
                mapping_from_pile_to_x[pile] = n_bays_in_area0 + col_id + 2
            else:
                piles_in_area6.append(pile)
                mapping_from_pile_to_x[pile] = n_bays_in_area0 + col_id + 2

    piles_all = piles_in_area0 + piles_in_area1 + piles_in_area2 \
                + piles_in_area3 + piles_in_area4 + piles_in_area5 + piles_in_area6
    x_max = max(mapping_from_pile_to_x.values()) + 1


    # 출고 계획 생성
    if retrieval:
        candidates = piles_in_area2 + piles_in_area3 + piles_in_area4
        # candidates = [i for i in candidates if not i in ["A22", "A23", "A24", "B22", "B23", "B24"]]
        from_piles_retrieval_cn1 = random.sample(candidates, n_from_piles_retrieval_cn1)
        candidates = [i for i in candidates if not i in from_piles_retrieval_cn1]
        from_piles_retrieval_cn2 = random.sample(candidates, n_from_piles_retrieval_cn2)
        candidates = piles_in_area6
        from_piles_retrieval_cn3 = random.sample(candidates, n_from_piles_retrieval_cn3)
        from_piles_retrieval = from_piles_retrieval_cn1 + from_piles_retrieval_cn2 + from_piles_retrieval_cn3
        for pile in from_piles_retrieval:
            num_of_plates = random.randint(int(0.9 * n_plates_for_retrieval), int(1.1 * n_plates_for_retrieval))
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
            df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
            df_retrieval = pd.concat([df_retrieval, df_temp], ignore_index=True)
    else:
        from_piles_retrieval = []

    # 선별 계획 생성
    if reshuffle:
        candidates = piles_in_area1 + piles_in_area5
        from_piles_reshuffle = random.sample(candidates, n_from_piles_reshuffle)
        candidates = [i for i in piles_all if (i not in from_piles_retrieval) and (i not in piles_in_area0)]
        candidates = [i for i in candidates if i not in from_piles_reshuffle]
        to_piles_reshuffle = random.sample(candidates, n_to_piles_reshuffle)
        # to_piles_reshuffle = ["A22", "A23", "A24", "B22", "B23", "B24"]

        for pile in from_piles_reshuffle:
            x = mapping_from_pile_to_x[pile]
            if x < 1 + safety_margin:
                to_piles_reshuffle_rev = [i for i in to_piles_reshuffle
                                          if mapping_from_pile_to_x[i] <= x_max + 1 - safety_margin]
            elif x > x_max + 1 - safety_margin:
                to_piles_reshuffle_rev = [i for i in to_piles_reshuffle
                                          if mapping_from_pile_to_x[i] >= 1 + safety_margin]
            else:
                to_piles_reshuffle_rev = to_piles_reshuffle
            num_of_plates = random.randint(int(0.9 * n_plates_for_reshuffle), int(1.1 * n_plates_for_reshuffle))
            pileno = [pile] * num_of_plates
            pileseq = [str(i).rjust(3, '0') for i in range(1, num_of_plates + 1)]
            markno = ["SP-RS-%s-%s" % (pile, i) for i in pileseq]
            unitw = np.random.uniform(0.141, 19.294, num_of_plates)
            topile = random.choices(to_piles_reshuffle_rev, k=num_of_plates)
            df_temp = pd.DataFrame({"pileno": pileno, "pileseq": pileseq, "markno": markno, "unitw": unitw, "topile": topile})
            df_reshuffle = pd.concat([df_reshuffle, df_temp], ignore_index=True)
    else:
        from_piles_reshuffle = []

    # 적치 계획 생성
    if storage:
        from_piles_storage = random.sample(piles_in_area0, n_from_piles_storage)
        candidates = piles_in_area1 + piles_in_area5
        candidates = [i for i in candidates if i not in from_piles_reshuffle]
        candidates = [i for i in candidates if mapping_from_pile_to_x[i] <= x_max + 1 - safety_margin]
        to_piles_storage = random.sample(candidates, n_to_piles_storage)
        # to_piles_storage = ["A22", "A23", "A24", "B22", "B23", "B24"]

        for pile in from_piles_storage:
            num_of_plates = random.randint(int(0.9 * n_plates_for_storage), int(1.1 * n_plates_for_storage))
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

    n_bays_in_area0 = 1
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

    n_plates_for_storage = 500
    n_plates_for_reshuffle = 150
    n_plates_for_retrieval = 150

    safety_margin = 5
    file_dir = "../input/data/case_study/case2/case2-4/case2-4-1/"

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    iteration = 50
    for i in range(1, iteration + 1):
        file_path = file_dir + "instance-{0}.xlsx".format(i)
        data_storage, data_reshuffle, data_retrieval \
            = generate_data(rows=rows,
                            storage=storage,
                            reshuffle=reshuffle,
                            retrieval=retrieval,
                            n_bays_in_area0=n_bays_in_area0,
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
                            n_plates_for_storage=n_plates_for_storage,
                            n_plates_for_reshuffle=n_plates_for_reshuffle,
                            n_plates_for_retrieval=n_plates_for_retrieval,
                            safety_margin=safety_margin,
                            file_path=file_path)