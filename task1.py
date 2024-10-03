def negative_checker(arr) -> bool:
    for el in arr:
        if el < 0:
            return True
    return False

def simplex_func(arr: list[list], t: str):
    if t.strip() == "max":
        for i in range(len(arr[0])):
            arr[0][i] *= -1

    x_sol = [0] * (len(arr) - 1)
    x_vector = [0.0] * (len(arr) - 1)
    while negative_checker(arr[0]):
        min_num: float = 1000
        indexRow: float
        indexCol: float
        min_sol: float = 1000
        div_num: float
        count = 0

        for i in range(len(arr[0])):
            if arr[0][i] < min_num:
                min_num = arr[0][i]
                indexRow = i

        for i in range(len(arr)):
            if min_sol > arr[i][-1] / arr[i][indexRow] > 0:
                min_sol = arr[i][-1] / arr[i][indexRow]
                indexCol = i
                count = 1
            elif min_sol == arr[i][-1] / arr[i][indexRow]:
                count += 1

        if count == 2 or min_sol == 1000:
            print("The method is not applicable!")
            exit(1)

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

    print("A vector of decision variables:", x_vector)
    print(f"{t} value of the objective function:", arr[0][-1])


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

    simplex_func([[float(element) for element in row] for row in a], t)

if __name__ == '__main__':
    main()
