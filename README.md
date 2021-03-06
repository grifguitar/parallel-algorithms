# Parallel algorithms course

Лектор: Виталий Аксёнов

Конспект подготовил: студент группы М3437, Хлытин Григорий.

## Lecture 1

### Раздел 1: Concurrency is not Parallelism!

---

### &sect; 1: Многопоточные (concurrent) структуры данных и алгоритмы.

Допустим, есть
![N](https://latex.codecogs.com/svg.latex?N) потоков.

Мы хотим: реализовать структуру данных `vector`, чтобы можно было взаимодействовать с `vector` параллельно из
![N](https://latex.codecogs.com/svg.latex?N) потоков (например добавлять в `vector` элементы с помощью
операции `push_back`).

Какие проблемы при этом у нас могут возникнуть?

Нам нужна _синхронизация_!

**Пояснение:**
> Операция `push_back` сама по себе не является _атомарной_!
> Если два различных потока одновременно начнут делать эту операцию, то состояние `vector` может стать некорректным.

**Вывод:**
> В многопоточном программировании мы занимаемся синхронизацией нашего кода между потоками
> (например при помощи взятия _блокировок_ на объекты, или используя _lock-free_ стратегии).

---

### &sect; 2: Параллельные (parallel) алгоритмы.

Допустим, у нас в распоряжении есть
![N](https://latex.codecogs.com/svg.latex?N) потоков.

Мы хотим: использовать
![N](https://latex.codecogs.com/svg.latex?N) потоков так, чтобы ускорить решение какой-то "задачи".

Что это значит?

В отличие от предыдущего параграфа, наш код не будет использоваться в нескольких потоках, а наоборот, мы будем
использовать
![N](https://latex.codecogs.com/svg.latex?N) потоков внутри нашего кода. То есть, мы разбиваем всю нашу
"задачу" на несколько непересекающихся "подзадач", и выполняем их параллельно. При этом, так как "подзадачи" (почти) не
пересекаются, нам (почти) не нужно заниматься синхронизацией потоков между собой.

**Вывод:**
> В параллельных алгоритмах мы почти не занимаемся вопросами синхронизации,
> так как мы разбиваем общую "задачу" на несколько непересекающихся "подзадач" и выполняем их параллельно в
> ![N](https://latex.codecogs.com/svg.latex?N) потоков, для ускорения решения общей "задачи".

---

### Раздел 2: Модели параллельных вычислений.

---

### &sect; 1: Модель PRAM.

Пусть есть таблица размера
![M * N](https://latex.codecogs.com/svg.latex?M&space;\times&space;N) (где
![N](https://latex.codecogs.com/svg.latex?N) &mdash; число потоков,
![M](https://latex.codecogs.com/svg.latex?M) &mdash; глубина).

|   | P[1] | P[2]  | .... | P[N] |
|---|------|-------|------|------|
| 1 | read | none  | ...  | 2+2  |
| 2 | 1+1  | write | ...  | none |
|...| ...  | ...   | ...  | ...  |
| M | none | 4+4   | ...  | 3+3  |

В каждой клетке таблицы записана ровно одна из _инструкций_ (например: `none`, `read`, `write`, `sum`, и т.п.). Каждый
поток выполняет свою последовательность инструкций. Инструкции выполняются по одной за раз (сначала первая инструкция
выполняется у всех потоков одновременно, потом вторая, и т.д.).

**Запомним:**
> Варианты модели **PRAM**:
> + **EREW** (Exclusive Read Exclusive Write)
> + **CREW** (Concurrent Read Exclusive Write)
> + **CRCW** (Concurrent Read Concurrent Write) <!---@formatter:off-->
>   + **common** (при одновременной записи все потоки обязаны писать одно и то же)
>   + **arbitrary** (при одновременной записи, гонку данных выигрывает случайный поток)
>   + **priority** (при одновременной записи, гонку данных выигрывает поток с наименьшим номером) <!---@formatter:on-->

**Запомним:**
> У алгоритмов в модели **PRAM** есть следующие асимптотические оценки:
> + **work** &mdash; общее количество инструкций (время работы на одном потоке)
> + **depth** &mdash; глубина инструкций (время работы на бесконечности потоков)

Мы пытаемся сделать так, чтобы **work** &mdash; был приближен к последовательному алгоритму, а **depth** &mdash; был чем
быстрее, тем лучше.

**Утверждение:**
> Любой алгоритм в модели **CRCW** может быть записан в модели **EREW**,
> при этом его **depth** увеличится в
> ![O(log(N))](https://latex.codecogs.com/svg.latex?O(log(N))) раз.

**Пример #1 (EREW):**
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
> + **work:** ![O(N*log(N))](https://latex.codecogs.com/svg.latex?O(N&space;\cdot&space;log(N)))
> + **depth:** ![O(log(N))](https://latex.codecogs.com/svg.latex?O(log(N)))

**Пример #2 (CREW):**
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
> + **work:** ![O(N^3 * log(N))](https://latex.codecogs.com/svg.latex?O(N^{3}&space;\cdot&space;log(N)))
> + **depth:** ![O(log(N))](https://latex.codecogs.com/svg.latex?O(log(N)))

**Вывод:**
> Одна из первых моделей.
> В этой модели есть большой недостаток: обязательные `none` инструкции.
> Реальная асимптотика в примере #1 (без `none` инструкций) была бы
> **work:** ![O(N)](https://latex.codecogs.com/svg.latex?O(N)),
> но мы вынуждены простаивать.

---

### &sect; 2: Модель PFor.

Считаем, что нам дан цикл `pfor` (parallel for), который сам занимается вопросами _scheduling_
(планирования исполнения).

**Пример #3 (PFor):**
> Дан массив. Посчитать сумму элементов.

```haskell
-- INPUT:
N <- in
A[1..N] <- in

-- DECLARE:
B[1..log(N)][1..N]

-- RUN:
pfor (i = 1 .. N):
    B[0][i] = A[i]
for (h = 1 .. log(N)):
    pfor (i = 1 .. (N / 2^h)):
        B[h][i] = B[h - 1][2 * i - 1] + B[h - 1][2 * i]

-- OUTPUT:
out <- B[log(N)][1]
```

> Асимптотика:
> + **work:** ![O(N)](https://latex.codecogs.com/svg.latex?O(N))
> + **depth:** ![O(log(N))](https://latex.codecogs.com/svg.latex?O(log(N)))

---

### &sect; 3: Модель ForkJoin.

![graph1](https://github.com/grifguitar/parallel-algorithms/blob/main/graph1.svg)

Любую модель вычислений можно представить в виде **DAG** (Directed Acyclic Graph).

Вершина графа &mdash; "задача". Ребро графа &mdash; переход между "задачами".

**Запомним:**
> В модели **ForkJoin** у нас есть следующие операции:
> + **fork** &mdash; разделить текущую "задачу" на две "подзадачи"
> + **join** &mdash; соединить две "подзадачи" в одну "задачу" (для вычисления результата)

**Запомним:**
> У алгоритмов в модели **ForkJoin** есть следующие асимптотические оценки:
> + **work** &mdash; общее количество вершин в графе (время работы на одном потоке)
> + **span** &mdash; глубина графа (время работы на бесконечности потоков)

**Утверждение:**
> В модели **ForkJoin** реализация `pfor` обладает следующими асимптотиками:
> + **work:** ![O(N)](https://latex.codecogs.com/svg.latex?O(N))
> + **span:** ![O(log(N))](https://latex.codecogs.com/svg.latex?O(log(N)))

**Задача #1 (ForkJoin):**
> Дана матрица. Увеличить каждый элемент на 1.
>
> С какой асимптотикой работает код ниже?

```haskell
-- INPUT:
N <- in
M <- in
X[1..N][1..M] <- in

-- RUN:
pfor (i = 1 .. N):
    pfor (j = 1 .. M):
        X[i][j]++

-- OUTPUT:
out <- X[1..N][1..M]
```

> Ответ:
> + **work:** ![O(N * M)](https://latex.codecogs.com/svg.latex?O(N&space;\cdot&space;M))
> + **span:** ![O(log(N) + log(M))](https://latex.codecogs.com/svg.latex?O(log(N)+log(M)))

---

## Practice

### Launch 1:

`NUM_THREADS = 4`

```haskell
HELLO! LAUNCH:
test: SIMPLE_BFS_TEST; arg: 3; 0 ms
test: SIMPLE_BFS_TEST; arg: 10; 0 ms
test: SIMPLE_BFS_TEST; arg: 50; 1.1 ms
test: SIMPLE_BFS_TEST; arg: 100; 18.4 ms
test: SIMPLE_BFS_TEST; arg: 150; 108.8 ms
test: SIMPLE_BFS_TEST; arg: 170; 175.7 ms
test: PARALLEL_BFS_TEST; arg: 3; 0 ms
test: PARALLEL_BFS_TEST; arg: 10; 0 ms
test: PARALLEL_BFS_TEST; arg: 50; 1.5 ms
test: PARALLEL_BFS_TEST; arg: 100; 8.3 ms
test: PARALLEL_BFS_TEST; arg: 150; 38.1 ms
test: PARALLEL_BFS_TEST; arg: 170; 57.5 ms
```

### Launch 2:

`NUM_THREADS = 4`

```haskell
HELLO! LAUNCH:
test: SIMPLE_SCAN_TEST; arg: 1048576; 1.2 ms
test: SIMPLE_SCAN_TEST; arg: 4194304; 4.8 ms
test: SIMPLE_SCAN_TEST; arg: 16777216; 19.3 ms
test: SIMPLE_SCAN_TEST; arg: 67108864; 76.9 ms
test: PARALLEL_SCAN_TEST; arg: 1048576; 0.7 ms
test: PARALLEL_SCAN_TEST; arg: 4194304; 4 ms
test: PARALLEL_SCAN_TEST; arg: 16777216; 17.5 ms
test: PARALLEL_SCAN_TEST; arg: 67108864; 64 ms
test: SIMPLE_FILTER_TEST; arg: 1048576; 12.2 ms
test: SIMPLE_FILTER_TEST; arg: 4194304; 49.2 ms
test: SIMPLE_FILTER_TEST; arg: 16777216; 196.5 ms
test: SIMPLE_FILTER_TEST; arg: 67108864; 791 ms
test: PARALLEL_FILTER_TEST; arg: 1048576; 6.7 ms
test: PARALLEL_FILTER_TEST; arg: 4194304; 27.1 ms
test: PARALLEL_FILTER_TEST; arg: 16777216; 102.1 ms
test: PARALLEL_FILTER_TEST; arg: 67108864; 391.4 ms
```
