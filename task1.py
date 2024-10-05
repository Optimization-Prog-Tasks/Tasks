def negative_checker(arr: list[float], eps: float) -> bool:
    return any(el < -eps for el in arr)

def simplex_func(a: list[list[int]], c: list[int], b: list[int], t: str, eps: float) -> dict:

    for i in range(len(a)):
        for j in range(len(a)):
            if i == j:
                a[i].append(1)
            else:
                a[i].append(0)
        a[i].append(b[i])

    c += [0] * (len(a) + 1)
    a.insert(0, c)

    arr: list[list[float]] = [[float(element) for element in row] for row in a]

    if t == "max":
        arr[0] = [-x for x in arr[0]]

    num_var = len(arr) - 1
    x_sol: list[int] = [0] * num_var
    x_vector: list[float] = [0] * num_var

    while negative_checker(arr[0], eps):
        indexRow: int = arr[0].index(min(arr[0]))
        indexCol: int = -1
        min_rat: float = float('inf')

        for i in range(1, len(arr)):
            if arr[i][indexRow] > eps:
                ratio = arr[i][-1] / arr[i][indexRow]
                if ratio < min_rat:
                    min_rat = ratio
                    indexCol = i

        if indexCol == -1:
            return {"solver_state": "unbounded"}

        x_sol[indexCol - 1] = indexRow + 1
        pivot: float = arr[indexCol][indexRow]

        for i in range(len(arr[indexCol])):
            arr[indexCol][i] /= pivot

        for i in range(len(arr)):
            if i != indexCol:
                k = arr[i][indexRow] / arr[indexCol][indexRow]
                for j in range(len(arr[0])):
                    arr[i][j] = arr[i][j] - (arr[indexCol][j] * k)

    for i in range(len(x_sol)):
        for j in range(len(x_sol)):
            if i + 1 == x_sol[j]:
                x_vector[i] = arr[j + 1][-1]

    return {
        "solver_state": "solved",
        "vector": x_vector,
        "solution": arr[0][-1]
    }

def main():
    eps =  1e-7
    tests = [
        [
            "max",
            [3, 5],
            [[1, 2], [3, 2]],
            [6, 12],
        ],

        [
            "max",
            [4, 3],
            [[2, 1], [1, 3]],
            [8, 9],
        ],

        [
            "max",
            [9, 10, 16],
            [[18, 15, 12], [6, 4, 8], [5, 3, 3]],
            [360, 192, 180],
        ],

        [
            "max",
            [3, 9],
            [[1, 4], [1, 2]],
            [8, 4],
        ],

        [
            "max",
            [2, 3],
            [[1, -1]],
            [4],

        ]
    ]

    for el in tests:
        solution = simplex_func(el[2], el[1], el[3], el[0], eps)

        print(solution["solver_state"])
        if solution["solver_state"] == "solved":
            print("A vector of decision variables:", solution["vector"])
            print(f"{el[0]} value of the objective function:", solution["solution"])
        print("-" * 50)

if __name__ == '__main__':
    main()
