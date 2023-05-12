def get_coord(pile_name):
    x = float(pile_name[1:]) + 1
    y = 1

    if 0 <= x <= 22:
        x = x
    elif 23 <= x <= 25:
        x = x + 1
    elif x >= 26:
        x = x + 2

    if pile_name[0] in ["A", "C", "E", "S"]:
        y = 1
    elif pile_name[0] in ["B", "D", "F", "T"]:
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