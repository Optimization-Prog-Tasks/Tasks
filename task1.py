def negative_checker(arr: list[float]) -> bool:
    return any(el < 0 for el in arr)

def simplex_func(arr: list[list[float]], t: str) -> dict:
    if t == "max":
        arr[0] = [-x for x in arr[0]]

    num_var = len(arr) - 1
    x_sol: list[int] = [0] * num_var
    x_vector: list[float] = [0] * num_var

    while negative_checker(arr[0]):
        indexRow: int = arr[0].index(min(arr[0]))
        indexCol: int = -1
        min_rat: float = float('inf')

        for i in range(1, len(arr)):
            if arr[i][indexRow] > 0:
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
    t: str = input().strip()
    c = eval(input())
    a  = eval(input())
    b = eval(input())

    for i in range(len(a)):
        for j in range(len(a)):
            if i == j:
                a[i].append(1)
            else:
                a[i].append(0)
        a[i].append(b[i])

    c += [0] * (len(a) + 1)
    a.insert(0, c)

    solution = simplex_func([[float(element) for element in row] for row in a], t)

    print(solution["solver_state"])
    if solution["solver_state"] == "solved":
        print("A vector of decision variables:", solution["vector"])
        print(f"{t} value of the objective function:", solution["solution"])

if __name__ == '__main__':
    main()
