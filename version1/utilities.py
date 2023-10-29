def get_coord(pile_name):
    if "I" in pile_name:
        x = 1
    else:
        x = float(pile_name[1:]) + 1
    y = 1

    if 1 <= x <= 22:
        x = x
    elif 23 <= x <= 25:
        x = x + 1
    elif x >= 26:
        x = x + 2

    if "I" in pile_name:
        row_id = pile_name[1]
    else:
        row_id = pile_name[0]

    if row_id in ["A", "C", "E", "S"]:
        y = 1
    elif row_id in ["B", "D", "F", "T"]:
        y = 2

    return [x, y]


def get_location_id(pile_name):
    x, y = get_coord(pile_name)
    id = int((x - 1) + 44 * (y - 1))
    return id


def get_moving_time(from_xcoord=None, to_xcoord=None):
    x_velocity = 0.5
    x_moving_time = abs(to_xcoord - from_xcoord) / x_velocity
    return x_moving_time

def get_layout(n_rows,  # 강재적치장 내 row의 개수
               n_bays,  # 강재적치장 내 bay의 개수
               n_input,  # 입고 지점의 개수
               n_output,  # 출고 지점의 개수
               loc_input,  # 입고 지점의 위치
               loc_output,  # 출고 지점의 위치
               range_piles_for_retrieval,  # 공정 파일의 범위
               file_path=None  # 데이터 저장 경로
               ):
    pass

