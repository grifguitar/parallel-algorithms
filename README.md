# Parallel algorithms course

Лектор: Виталий Аксёнов

Конспект подготовил: студент группы М3437, Хлытин Григорий.

## Lecture 1

### Раздел 1: Concurrency is not Parallelism!

#### Параграф 1: Многопоточные _(concurrency)_ структуры данных и алгоритмы.

Допустим, есть
![N](https://latex.codecogs.com/svg.latex?\Large&space;N) потоков.

Мы хотим &mdash; реализовать структуру данных `vector`, чтобы можно было взаимодействовать с `vector` параллельно из
![N](https://latex.codecogs.com/svg.latex?\Large&space;N) потоков (например добавлять в `vector` элементы с помощью
операции `push_back`).

Какие проблемы при этом у нас могут возникнуть?

Нам нужна _синхронизация_!

***Пояснение:***
> Операция `push_back` сама по себе не является _атомарной_!
> Если два различных потока одновременно начнут делать эту операцию, то состояние `vector` может стать некорректным.

***Вывод:***
> В многопоточном программировании мы занимаемся _синхронизацией_ нашего кода между потоками
> (например при помощи взятия _блокировок_ на объекты, или используя _lock-free_ стратегии).

#### Параграф 2: Параллельные _(parallel)_ алгоритмы.

Допустим, у нас в распоряжении есть
![N](https://latex.codecogs.com/svg.latex?\Large&space;N) потоков.

Мы хотим &mdash; использовать
![N](https://latex.codecogs.com/svg.latex?\Large&space;N) потоков так, чтобы ускорить решение какой-то "задачи".

Что это значит?

В отличие от предыдущего параграфа, наш код не будет использоваться в нескольких потоках, а наоборот, мы будем
использовать
![N](https://latex.codecogs.com/svg.latex?\Large&space;N) потоков внутри нашего кода. То есть, мы разбиваем всю нашу
"задачу" на несколько непересекающихся подзадач, и выполняем их параллельно. При этом, так как подзадачи (почти) не
пересекаются, нам (почти) не нужно заниматься _синхронизацией_
потоков между собой.

***Вывод:***
> В параллельных алгоритмах мы почти не занимаемся вопросами _синхронизации_,
> так как мы разбиваем общую "задачу" на несколько непересекающихся подзадач и выполняем их параллельно в
> ![N](https://latex.codecogs.com/svg.latex?\Large&space;N)
> потоков, для ускорения решения общей "задачи".

#### Параграф 3: Модель PRAM.

Пусть есть таблица размера
![M * N](https://latex.codecogs.com/svg.latex?M&space;\times&space;N) где
![N](https://latex.codecogs.com/svg.latex?N) &mdash; число потоков,
![M](https://latex.codecogs.com/svg.latex?M) &mdash; глубина инструкций.

|   | P[1] | P[2]  | .... | P[N] |
|---|------|-------|------|------|
| 1 | read | none  | 2+2  | none |
| 2 | 1+1  | write | none | none |
|...| ...  | ...   | ...  | ...  |

В каждой клетке таблицы записана ровно одна из инструкций (например: none, read, write, sum, и так далее). Каждый поток
выполняет свою последовательность инструкций. Инструкции выполняются по одной за раз (сначала первая инструкция
выполняется у всех потоков одновременно, потом вторая, и так далее).

Варианты модели PRAM:

+ EREW (Exclusive read exclusive write)
+ CREW (Concurrent read exclusive write)
+ CRCW (Concurrent read concurrent right)
    + common (при одновременной записи все потоки обязаны писать одно и то же)
    + arbitrary (при одновременной записи выигрывает гонку данных случайный поток)
    + priority (при одновременной записи выигрывает гонку данных поток с наименьшим номером)
