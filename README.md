# Parallel-algorithms course

Лектор: Виталий Аксёнов

Конспект подготовил: студент группы М3437, Хлытин Григорий.

## Lecture 1

### Раздел 1: Concurrency is not Parallelism!

#### Параграф 1: Многопоточные _(concurrency)_ структуры данных и алгоритмы.

Допустим, есть
_<img src="https://latex.codecogs.com/svg.latex?N">_
потоков.

Мы хотим &mdash; реализовать структуру данных `vector`, чтобы можно было взаимодействовать с `vector` параллельно из
<img src="https://latex.codecogs.com/svg.latex?N">
потоков (например добавлять в `vector` элементы с помощью операции `push_back`).

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
<img src="https://latex.codecogs.com/svg.latex?N">
потоков.

Мы хотим &mdash; использовать
<img src="https://latex.codecogs.com/svg.latex?N">
потоков так, чтобы ускорить решение какой-то "задачи".

Что это значит?

В отличие от предыдущего параграфа, наш код не будет использоваться в нескольких потоках, а наоборот, мы будем
использовать
<img src="https://latex.codecogs.com/svg.latex?N">
потоков внутри нашего кода. То есть, мы разбиваем всю нашу "задачу" на несколько непересекающихся подзадач, и выполняем
их параллельно. При этом, так как подзадачи (почти) не пересекаются, нам (почти) не нужно заниматься _синхронизацией_
потоков между собой.

***Вывод:***
> В параллельных алгоритмах мы почти не занимаемся вопросами _синхронизации_,
> так как мы разбиваем общую "задачу" на несколько непересекающихся подзадач и выполняем их параллельно в
> <img src="https://latex.codecogs.com/svg.latex?N">
> потоков, для ускорения решения общей "задачи".
