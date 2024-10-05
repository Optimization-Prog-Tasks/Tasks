def negative_checker(arr) -> bool:
    for el in arr:
        if el < 0:
            return True
    return False

def simplex_func(arr: list[list], t: str) -> dict:
    global indexCol, indexRow
    if t.strip() == "max":
        for i in range(len(arr[0])):
            arr[0][i] *= -1

    x_sol = [0] * (len(arr) - 1)
    x_vector = [0.0] * (len(arr) - 1)
    while negative_checker(arr[0]):
        min_num: float = 1000
        indexRow: int
        indexCol: int
        min_sol: float = 1000
        div_num: float

        for i in range(len(arr[0])):
            if arr[0][i] < min_num:
                min_num = arr[0][i]
                indexRow = i

        unbounded = True
        for i in range(1, len(arr)):
            if arr[i][indexRow] > 0:
                ratio = arr[i][-1] / arr[i][indexRow]
                if 0 <= ratio < min_sol:
                    min_sol = ratio
                    indexCol = i
                    unbounded = False

        if unbounded:
            return {"solver_state": "unbounded"}

        x_sol[indexCol - 1] = indexRow + 1

        div_num = arr[indexCol][indexRow]

        for i in range(len(arr[indexCol])):
            arr[indexCol][i] /= div_num

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
    t: str = input()
    c = eval(input())
    a  = eval(input())
    b = eval(input())

    for i in range(len(c) + 1):
        c.append(0)

    for i in range(len(a)):
        for j in range(len(a)):
            if i == j:
                a[i].append(1)
            else:
                a[i].append(0)
        a[i].append(b[i])

    a.insert(0, c)

    solution = simplex_func([[float(element) for element in row] for row in a], t)

    print(solution["solver_state"])
    if solution["solver_state"] == "solved":
        print("A vector of decision variables:", solution["vector"])
        print(f"{t} value of the objective function:", solution["solution"])

if __name__ == '__main__':
    main()
