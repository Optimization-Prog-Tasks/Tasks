import numpy as np
from numpy.linalg import norm

def negative_checker(arr: list[float], eps: float) -> bool:
    return any(el < -eps for el in arr)

def generate_initial_point(A, b):
    n = A.shape[1]
    x = np.random.randint(0, 10, n)

    while np.any(A @ x > b):
        x = np.random.rand(n)

    return x

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
    if t == "max":
        return {
            "solver_state": "solved",
            "vector": x_vector,
            "solution": arr[0][-1]
        }
    else:
        return {
            "solver_state": "solved",
            "vector": x_vector,
            "solution": arr[0][-1] * -1
        }

def interior_func(t: str, x: np.array, a: np.array, c: np.array, alp: float, eps: float):
    if t == "min":
        c = -c

    while True:
        v = x
        D = np.diag(x)
        AA = np.dot(a, D)

        if(len(a) != len(c) - len(a)):
            return {"solver_state": "unbounded"}

        cc = np.dot(D, c)
        I = np.eye(len(c))
        F = np.dot(AA, np.transpose(AA))  # Regularized F
        FI = np.linalg.inv(F)
        H = np.dot(np.transpose(AA), FI)
        P = np.subtract(I, np.dot(H, AA))
        cp = np.dot(P, cc)
        nu = np.absolute(np.min(cp))

        if nu <= 1e-10:
            return {"solver_state": "unbounded"}

        y = np.add(np.ones(len(x), float), (alp / nu) * cp)
        yy = np.dot(D, y)
        x = yy
        if norm(np.subtract(yy, v), ord=2) < 0.0001:
            break

    if t == "max":
        return {
            "solver_state": "solved",
            "vector": x[0:3],
            "solution": np.dot(c, x)
        }
    else:
        return {
            "solver_state": "solved",
            "vector": x[0:3],
            "solution": np.dot(c, x) * -1
        }

def main():
    eps = 1e-7
    tests = [
        ["max", [9, 10, 16], [[18, 15, 12], [6, 4, 8], [5, 3, 3]], [360, 192, 180], [1, 1, 1, 315, 174, 169]],
        ["max", [3, 5], [[2, 1], [1, 3]], [6, 12], [1, 1, 3, 8]],
        ["max", [4, 3], [[2, 1], [1, 3]], [8, 9], [2, 2, 2, 1]],
        ["max", [9, 10, 16], [[18, 15, 12], [6, 4, 8], [5, 3, 3]], [360, 192, 180], [1, 1, 10, 207, 102, 142]],
        ["min", [-2, 2, -6], [[2, 1, -2], [1, 2, 4], [1, -1, 2]], [24, 23, 10], [1, 1, 1, 23, 16, 8]],
        ["max", [2, 3], [[1, -1]], [4], [4, 0, 0]]
    ]
    for el in tests:
        c = np.array(el[1], dtype=float)
        a = np.array(el[2], dtype=float)
        identity_matrix = np.eye(a.shape[0])
        extended_matrix = np.hstack((a, identity_matrix))
        zeros_matrix = np.zeros((a.shape[0], len(c)))
        A_extended = np.hstack((a, zeros_matrix))
        c_extended = np.concatenate((c, np.zeros(A_extended.shape[1] - len(c))))

        solutions = [[], [], []]
        names = ["Interior point method with a = 0.5:", "Interior point method with a = 0.9:", "Simplex method:"]

        solutions[0] = interior_func(el[0], el[4], extended_matrix, c_extended, 0.5, eps)
        solutions[1] = interior_func(el[0], el[4], extended_matrix, c_extended, 0.9, eps)
        solutions[2] = simplex_func(el[2], el[1], el[3], el[0], eps)

        for i in range(len(solutions)):
            print(names[i])
            if solutions[i]["solver_state"] == "solved":
                print("A vector of decision variables:", solutions[i]["vector"])
                print(f"{el[0]} value of the objective function:", solutions[i]["solution"])
            elif solutions[i]["solver_state"] == "unbounded":
                print("The problem is unbounded")
            print("-" * 50)
        print("\n\n")


if __name__ == '__main__':
    main()