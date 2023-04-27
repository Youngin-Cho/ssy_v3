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


def get_moving_time(from_location, to_location):
    if type(from_location).__name__ == "Conveyor" or type(to_location).__name__ == "Conveyor":
        x_time = 2 * abs(to_location.coord[0] - from_location[0])
        y_time = 0.0
    else:
        x_time = 2 * abs(to_location.coord[0] - from_location[0])
        y_time = abs(to_location.coord[1] - from_location.coord[1])

    moving_time = max(x_time, y_time)

    return moving_time