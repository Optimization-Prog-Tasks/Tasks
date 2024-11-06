import copy
import numpy as np

def north_west_algo(c):
    i = 0
    j = 0
    x_vector = []
    while i < 3 and j < 5:
        supply = c[i][-1]
        demand = c[-1][j]

        allocation = min(supply, demand)
        x_vector.append([i, j, allocation])

        c[i][-1] -= allocation
        c[-1][j] -= allocation

        if c[i][-1] == 0:
            i += 1
        if c[-1][j] == 0:
            j += 1

    return x_vector


def russells_approximation(c):
    x_vector = []
    while len(c[0]) > 2 or len(c) > 2:
        addition_matrix = copy.deepcopy(c)
        m = len(c) - 1
        n = len(c[0]) - 1

        u = [0] * m
        v = [0] * n

        for i in range(m):
            u[i] = max(c[i][j] for j in range(n))

        for j in range(n):
            v[j] = max(c[i][j] for i in range(m))

        for i in range(m):
            for j in range(n):
                addition_matrix[i][j] = c[i][j] - u[i] - v[j]

        min_value = float('inf')
        min_position = (0, 0)

        for i, row in enumerate(addition_matrix):
            for j, elem in enumerate(row):
                if elem < min_value:
                    min_value = elem
                    min_position = (i, j)

        if c[-1][min_position[1]] > c[min_position[0]][-1]:
            x_vector.append([min_position[0], min_position[1],  c[min_position[0]][-1]])
            c[-1][min_position[1]] -= c[min_position[0]][-1]
            c.pop(min_position[0])
        else:
            x_vector.append(
                [min_position[0], min_position[1],  c[-1][min_position[1]]])
            c[min_position[0]][-1] -= c[-1][min_position[1]]

            for row in c:
                del row[min_position[1]]

    x_vector.append([0, 0, c[0][-1]])
    return x_vector


def vogel_approximation(supply, demand, costs):
    INF = 10 ** 10
    n, m = len(costs), len(costs[0])
    result = []

    def get_differences(costs):
        row_diff, col_diff = [], []

        for row in costs:
            sorted_row = sorted(row)
            row_diff.append(sorted_row[1] - sorted_row[0])

        for col_idx in range(m):
            sorted_col = sorted([costs[row_idx][col_idx] for row_idx in range(n)])
            col_diff.append(sorted_col[1] - sorted_col[0])

        return row_diff, col_diff

    while max(supply) > 0 or max(demand) > 0:
        row_diff, col_diff = get_differences(costs)

        max_row_diff = max(row_diff)
        max_col_diff = max(col_diff)

        if max_row_diff >= max_col_diff:
            row_idx = row_diff.index(max_row_diff)
            min_cost = min(costs[row_idx])
            col_idx = costs[row_idx].index(min_cost)

            transport_quantity = min(supply[row_idx], demand[col_idx])

            result.append([row_idx, col_idx, transport_quantity])

            supply[row_idx] -= transport_quantity
            demand[col_idx] -= transport_quantity

            if demand[col_idx] == 0:
                for i in range(n):
                    costs[i][col_idx] = INF
            else:
                costs[row_idx] = [INF] * m

        else:
            col_idx = col_diff.index(max_col_diff)
            min_cost = min(costs[row_idx][col_idx] for row_idx in range(n))
            row_idx = next(i for i in range(n) if costs[i][col_idx] == min_cost)

            transport_quantity = min(supply[row_idx], demand[col_idx])

            result.append([row_idx, col_idx, transport_quantity])

            supply[row_idx] -= transport_quantity
            demand[col_idx] -= transport_quantity
            if demand[col_idx] == 0:
                for i in range(n):
                    costs[i][col_idx] = INF
            else:
                costs[row_idx] = [INF] * m

    return result


def is_balanced(supply, demand):
    return sum(supply) == sum(demand)

def print_matrix(costs, supply, demand):
    extended_matrix = np.zeros((4, 5), dtype=int)
    extended_matrix[:3, :4] = costs
    extended_matrix[:3, 4] = supply
    extended_matrix[3, :4] = demand
    extended_matrix[3, 4] = sum(supply)
    print(extended_matrix)


def convert_to_matrix(costs, supply, demand):
    matrix = [[0 for _ in range(5)] for _ in range(4)]

    for i in range(3):
        for j in range(4):
            matrix[i][j] = costs[i][j]

    for i in range(3):
        matrix[i][4] = supply[i]

    for j in range(4):
        matrix[3][j] = demand[j]

    matrix[3][4] = sum(supply)

    return matrix


def perform_algorithms(supply, demand, costs):
    if not is_balanced(supply, demand):
        print("The problem is not balanced!")
        return
    table1 = convert_to_matrix(costs, supply, demand)
    table3 = convert_to_matrix(costs, supply, demand)
    print_matrix(costs, supply, demand)

    northwest = north_west_algo(table1)
    vogel = vogel_approximation(supply, demand, costs)
    russell = russells_approximation(table3)

    print("\nNorth-West Corner Method Solution:")
    print(northwest)

    print("\nVogel’s Approximation Method Solution:")
    print(vogel)

    print("\nRussell’s Approximation Method Solution:")
    print(russell)

    print("-" * 50)


supply = [[20, 30, 25], [25, 20, 25], [15, 35, 30]]
demand = [[10, 15, 30, 20], [20, 10, 20, 20], [25, 20, 15, 20]]
costs = [[
    [8, 6, 10, 9],
    [9, 12, 7, 5],
    [4, 14, 10, 6]
],
    [
        [7, 5, 9, 6],
        [6, 8, 7, 3],
        [5, 9, 6, 4]
    ],
    [
        [6, 7, 8, 5],
        [5, 8, 6, 9],
        [7, 5, 10, 6]
    ],
]
for i in range(len(supply)):
    perform_algorithms(supply[i], demand[i], costs[i])
