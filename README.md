# Parallel algorithms course

Лектор: Виталий Аксёнов

Конспект подготовил: студент группы М3437, Хлытин Григорий.

---

# Lecture 1

---

## Раздел 1: Concurrency is not Parallelism!

---

### Параграф 1: Многопоточные _(concurrency)_ структуры данных и алгоритмы.

Допустим, есть
![N](https://latex.codecogs.com/svg.latex?N) потоков.

Мы хотим &mdash; реализовать структуру данных `vector`, чтобы можно было взаимодействовать с `vector` параллельно из
![N](https://latex.codecogs.com/svg.latex?N) потоков (например добавлять в `vector` элементы с помощью
операции `push_back`).

Какие проблемы при этом у нас могут возникнуть?

Нам нужна _синхронизация_!

***Пояснение:***
> Операция `push_back` сама по себе не является _атомарной_!
> Если два различных потока одновременно начнут делать эту операцию, то состояние `vector` может стать некорректным.

***Вывод:***
> В многопоточном программировании мы занимаемся _синхронизацией_ нашего кода между потоками
> (например при помощи взятия _блокировок_ на объекты, или используя _lock-free_ стратегии).

---

### Параграф 2: Параллельные _(parallel)_ алгоритмы.

Допустим, у нас в распоряжении есть
![N](https://latex.codecogs.com/svg.latex?N) потоков.

Мы хотим &mdash; использовать
![N](https://latex.codecogs.com/svg.latex?N) потоков так, чтобы ускорить решение какой-то "задачи".

Что это значит?

В отличие от предыдущего параграфа, наш код не будет использоваться в нескольких потоках, а наоборот, мы будем
использовать
![N](https://latex.codecogs.com/svg.latex?N) потоков внутри нашего кода. То есть, мы разбиваем всю нашу
"задачу" на несколько непересекающихся подзадач, и выполняем их параллельно. При этом, так как подзадачи (почти) не
пересекаются, нам (почти) не нужно заниматься _синхронизацией_
потоков между собой.

***Вывод:***
> В параллельных алгоритмах мы почти не занимаемся вопросами _синхронизации_,
> так как мы разбиваем общую "задачу" на несколько непересекающихся подзадач и выполняем их параллельно в
> ![N](https://latex.codecogs.com/svg.latex?N)
> потоков, для ускорения решения общей "задачи".

---

## Раздел 2: Модели параллельных вычислений.

---

### Параграф 1: Модель PRAM.

Пусть есть таблица размера
![M * N](https://latex.codecogs.com/svg.latex?M&space;\times&space;N) (где
![N](https://latex.codecogs.com/svg.latex?N) &mdash; число потоков,
![M](https://latex.codecogs.com/svg.latex?M) &mdash; глубина инструкций).

|   | P[1] | P[2]  | .... | P[N] |
|---|------|-------|------|------|
| 1 | read | none  | ...  | 2+2  |
| 2 | 1+1  | write | ...  | none |
|...| ...  | ...   | ...  | ...  |
| M | none | 4+4   | ...  | 3+3  |

В каждой клетке таблицы записана ровно одна из инструкций (например: `none`, `read`, `write`, `sum`, и т.п.). Каждый
поток выполняет свою последовательность инструкций. Инструкции выполняются по одной за раз (сначала первая инструкция
выполняется у всех потоков одновременно, потом вторая, и т.д.).

***Запомним:***
> Варианты модели **PRAM**:
> + **EREW** (Exclusive Read Exclusive Write)
> + **CREW** (Concurrent Read Exclusive Write)
> + **CRCW** (Concurrent Read Concurrent Write <!---@formatter:off-->
>   + **common** (при одновременной записи все потоки обязаны писать одно и то же)
>   + **arbitrary** (при одновременной записи, гонку данных выигрывает случайный поток)
>   + **priority** (при одновременной записи, гонку данных выигрывает поток с наименьшим номером) <!---@formatter:on-->

***Запомним:***
> У алгоритмов в модели **PRAM** есть следующие асимптотические оценки:
> + ![O_work](https://latex.codecogs.com/svg.latex?O_{work}(...)) &mdash; общее количество инструкций (время работы на одном потоке)
> + ![O_depth](https://latex.codecogs.com/svg.latex?O_{depth}(...)) &mdash; глубина инструкций (время работы на бесконечности потоков)

Мы пытаемся сделать так, чтобы **_work_** &mdash; был приближен к последовательному алгоритму, а **_depth_** &mdash; был
чем быстрее, тем лучше.

***Утверждение:***
> Любой алгоритм в модели **CRCW** может быть записан в модели **EREW**, при этом его **_depth_** умножится на **_log(N)_**.

***Пример программы #1 (EREW):***
> Дан массив. Посчитать сумму элементов.

```haskell
-- INPUT:
N <- in
A[1..N] <- in

-- DECLARE:
B[1..N]

-- PARALLEL RUN ON N PROCESSES:
parallel run {
    processId -> {
        w <- A[processId]
        B[processId] <- w
        for (h = 1 .. log(N)):
            if (id <= N / 2^h):
                x <- B[2 * processId - 1]
                y <- B[2 * processId]
                z = x + y
                B[processId] <- z
            else:
                skip(4)
    }
}

-- OUTPUT:
out <- B[1]
```

> Асимптотика:
> + **_work_** = ![O(N*log(N))](https://latex.codecogs.com/svg.latex?O(N&space;\cdot&space;log(N)))
> + **_depth_** = ![O(log(N))](https://latex.codecogs.com/svg.latex?O(log(N)))

***Пример программы #2 (CREW):***
> Дано две матрицы. Перемножить их.

```haskell
-- INPUT:
N <- in
A[1..N][1..N] <- in
B[1..N][1..N] <- in

-- DECLARE:
C[1..N][1..N]
D[1..N][1..N][1..N]

-- PARALLEL RUN ON N^3 PROCESSES:
parallel run {
    processId -> {
        (i, j, k) <- processId
        
        x <- A[i][k]
        y <- B[k][j]
        z = x * y
        D[i][j][k] <- z
        
        for (h = 1 .. log(N)):
            if (k <= N / 2^h):
                x <- D[i][j][2 * k - 1]
                y <- D[i][j][2 * k]
                z = x + y
                D[i][j][k] <- z
            else:
                skip(4)
        
        if (k == 1):
            C[i][j] <- D[i][j][k]
        else:
            skip(1)
    }
}

-- OUTPUT:
out <- C[1..N][1..N]
```

> Асимптотика:
> + **_work_** = ![O(N^3 * log(N))](https://latex.codecogs.com/svg.latex?O(N^{3}&space;\cdot&space;log(N)))
> + **_depth_** = ![O(log(N))](https://latex.codecogs.com/svg.latex?O(log(N)))

***Вывод:***
> Одна из первых моделей.
> В этой модели есть большой недостаток: обязательные `none` инструкции.
> Реальная асимптотика в примере #1 (без `none` инструкций) была бы
> **_work_** = ![O(N)](https://latex.codecogs.com/svg.latex?O(N)),
> но мы вынуждены простаивать.

---

### Параграф 2: Модель `pfor`.

Считаем, что нам дан **`pfor`** (parallel for), который сам занимается вопросами `scheduling` (планирования исполнения).

***Пример программы #3 (CREW):***
> Дан массив. Посчитать сумму элементов.

```haskell
-- INPUT:
N <- in
A[1..N] <- in

-- DECLARE:
B[1..log(N)][1..N]

-- RUN:
run {
    pfor (i = 1 .. N):
        B[0][i] = A[i]
    for (h = 1 .. log(N)):
        pfor (i = 1 .. (N / 2^h)):
            B[h][i] = B[h - 1][2 * i - 1] + B[h - 1][2 * i]
}

-- OUTPUT:
out <- B[log(N)][1]
```

> Асимптотика:
> + **_work_** = ![O(N)](https://latex.codecogs.com/svg.latex?O(N))
> + **_depth_** = ![O(log(N))](https://latex.codecogs.com/svg.latex?O(log(N)))

---
