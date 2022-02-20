# Parallel-algorithms course

Лектор: Виталий Аксёнов

Конспект подготовил: студент группы М3437, Хлытин Григорий.

## Lecture 1

### Раздел 1: Concurrency is not Parallelism!

#### Параграф 1: Многопоточные _(concurrency)_ структуры данных и алгоритмы.

Допустим, есть <img src="https://latex.codecogs.com/gif.latex?N"> потоков.

Мы хотим &ndash; реализовать структуру данных `vector`, чтобы можно было взаимодействовать с `vector` параллельно из $N$
потоков
(например добавлять в `vector` элементы с помощью операции `push_back`).

Какие проблемы при этом у нас могут возникнуть?

Нам нужна _синхронизация_!

***Пояснение:***
> Операция `push_back` сама по себе не является _атомарной_!
> Если два различных потока одновременно начнут делать эту операцию, то состояние `vector` может стать некорректным.

***Вывод:***
> В многопоточном программировании мы занимаемся _синхронизацией_ нашего кода между потоками
> (например при помощи взятия _блокировок_ на объекты, или используя _lock-free_ стратегии).

#### Параграф 2: Параллельные _(parallel)_ алгоритмы.

Допустим, у нас в распоряжении есть $N$ потоков.

Мы хотим $-$ использовать $N$ потоков так, чтобы ускорить решение какой-то "задачи".

Что это значит?

В отличие от предыдущего параграфа, наш код не будет использоваться в нескольких потоках, а наоборот, мы будем
использовать $N$ потоков внутри нашего кода. То есть, мы разбиваем всю нашу "задачу" на несколько непересекающихся
подзадач, и выполняем их параллельно. При этом, так как подзадачи (почти) не пересекаются, нам (почти) не нужно
заниматься _синхронизацией_ потоков между собой.

***Вывод:***
> В параллельных алгоритмах мы почти не занимаемся вопросами _синхронизации_,
> так как мы разбиваем общую "задачу" на несколько непересекающихся подзадач и выполняем их параллельно в $N$ потоков,
> для ускорения решения общей "задачи".
