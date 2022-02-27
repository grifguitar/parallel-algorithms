#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <functional>
#include <random>
#include <chrono>
#include <set>
#include <atomic>
#include "omp.h"

// ------------------------------------------------------------ DEFINE:

using vec_t = std::vector<int>;
using vec_vec_t = std::vector<std::vector<int>>;
using fIntBool_t = std::function<bool(int)>;
using fVoidVoid_t = std::function<void(void)>;

// ------------------------------------------------------------ CONSTANTS:

static constexpr int BLOCK_FOR = (int) 1e3;
static constexpr int BLOCK_SCAN = (int) 1e6;

static constexpr int MAX_SIZE = (int) 1e8;

static constexpr int RND_SEED = 177013;
static constexpr int RND_FROM = 0;
static constexpr int RND_TO = (int) 1e5;

static constexpr int ITERATIONS = 10;

// ------------------------------------------------------------ UTILS:

std::ostream &operator<<(std::ostream &out, const vec_t &name) {
    for (auto elem: name) {
        out << elem << " ";
    }
    return out;
}

struct Utils {
    static int get_rand_int(
            int i
    );

    static void simpleScan(
            const vec_t &data, vec_t &res, int size
    );

    static int simpleFilter(
            const vec_t &data, vec_t &res, int size, const fIntBool_t &p
    );

    static bool isNotEquals(
            const vec_t &arr1, const vec_t &arr2, int size
    );

    static bool isNotEquals(
            const vec_t &arr1, const vec_t &arr2
    );

    static vec_vec_t getCubicGraph(
            int sideSize
    );
};

int Utils::get_rand_int(int i) {
    static vec_t rnd_values(MAX_SIZE, -1);
    static std::mt19937 gen{RND_SEED};
    static std::uniform_int_distribution<int> pick{RND_FROM, RND_TO};

    if (rnd_values[i] == -1) {
        rnd_values[i] = pick(gen);
    }

    return rnd_values[i];
}

void Utils::simpleScan(const vec_t &data, vec_t &res, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
        res[i] = sum;
    }
}

int Utils::simpleFilter(const vec_t &data, vec_t &res, int size, const fIntBool_t &p) {
    int len = 0;
    for (int i = 0; i < size; i++) {
        if (p(data[i])) {
            res[len] = data[i];
            len++;
        }
    }
    return len;
}

bool Utils::isNotEquals(const vec_t &arr1, const vec_t &arr2, int size) {
    for (int i = 0; i < size; i++) {
        if (arr1[i] != arr2[i]) {
            return true;
        }
    }
    return false;
}

bool Utils::isNotEquals(const vec_t &arr1, const vec_t &arr2) {
    if (arr1.size() != arr2.size()) {
        return true;
    }
    for (int i = 0; i < arr1.size(); i++) {
        if (arr1[i] != arr2[i]) {
            return true;
        }
    }
    return false;
}

vec_vec_t Utils::getCubicGraph(int sideSize) {
    auto check = [](int x, int size) {
        return (0 <= x && x < size);
    };
    auto getNum = [](int i, int j, int k, int base) {
        return ((i * base) + j) * base + k;
    };
    auto addEdge = [](int u, int v, vec_vec_t &graph, int &cnt) {
        graph[v].push_back(u);
        graph[u].push_back(v);
        cnt++;
    };

    int n = sideSize * sideSize * sideSize;
    int m = n * 3 - (sideSize * sideSize) * 3;

    vec_vec_t graph(n);

    int base = sideSize;
    int cnt = 0;
    for (int i = 0; i < sideSize; i++) {
        for (int j = 0; j < sideSize; j++) {
            for (int k = 0; k < sideSize; k++) {
                int v = getNum(i, j, k, base);
                if (check(i + 1, sideSize)) {
                    int u = getNum(i + 1, j, k, base);
                    addEdge(u, v, graph, cnt);
                }
                if (check(j + 1, sideSize)) {
                    int u = getNum(i, j + 1, k, base);
                    addEdge(u, v, graph, cnt);
                }
                if (check(k + 1, sideSize)) {
                    int u = getNum(i, j, k + 1, base);
                    addEdge(u, v, graph, cnt);
                }
            }
        }
    }

    if (cnt != m) {
        throw std::runtime_error("count of edges is wrong!");
    }

    return graph;
}

// ------------------------------------------------------------ PARALLEL UTILS:

struct ParallelUtils {
    static void parallelScanUp(
            int l, int r, int u, const vec_t &data, vec_t &tree
    );

    static void parallelScanDown(
            int delta, int l, int r, int u, const vec_t &data, vec_t &res,
            vec_t &tree
    );

    static void parallelScan(
            const vec_t &data, vec_t &res, int size, vec_t &tree
    );

    static int parallelFilter(
            const vec_t &data, vec_t &res, int size, const fIntBool_t &p,
            vec_t &mask, vec_t &tree
    );
};

void ParallelUtils::parallelScanUp(
        int l, int r, int u, const vec_t &data, vec_t &tree
) {
    if (r - l <= BLOCK_SCAN) {
        int sum = 0;
        for (int i = l; i < r; i++) {
            sum += data[i];
        }
        tree[u] = sum;
    } else {
        int m = (l + r) / 2;
#pragma omp parallel sections default(none) shared(l, m, r, u, data, tree)
        {
#pragma omp section
            {
                parallelScanUp(l, m, 2 * u + 1, data, tree);
            }
#pragma omp section
            {
                parallelScanUp(m, r, 2 * u + 2, data, tree);
            }
        }
        tree[u] = tree[2 * u + 1] + tree[2 * u + 2];
    }
}

void ParallelUtils::parallelScanDown(
        int delta, int l, int r, int u, const vec_t &data, vec_t &res,
        vec_t &tree
) {
    if (r - l <= BLOCK_SCAN) {
        int sum = delta;
        for (int i = l; i < r; i++) {
            sum += data[i];
            res[i] = sum;
        }
    } else {
        int m = (l + r) / 2;
#pragma omp parallel sections default(none) shared(delta, l, m, r, u, data, res, tree)
        {
#pragma omp section
            {
                parallelScanDown(
                        delta, l, m, 2 * u + 1, data, res, tree
                );
            }
#pragma omp section
            {
                parallelScanDown(
                        delta + tree[2 * u + 1], m, r, 2 * u + 2, data, res, tree
                );
            }
        }
    }
}

void ParallelUtils::parallelScan(
        const vec_t &data, vec_t &res, int size, vec_t &tree
) {
    parallelScanUp(0, size, 0, data, tree);
    parallelScanDown(0, 0, size, 0, data, res, tree);
}

int ParallelUtils::parallelFilter(
        const vec_t &data, vec_t &res, int size, const fIntBool_t &p,
        vec_t &mask, vec_t &tree
) {
#pragma omp parallel for default(none) schedule(static, BLOCK_FOR) shared(size, p, data, mask, BLOCK_FOR)
    for (int i = 0; i < size; i++) {
        mask[i] = p(data[i]) ? 1 : 0;
    }
    //Utils::simpleScan(mask, mask, size);
    parallelScan(mask, mask, size, tree);
#pragma omp parallel for default(none) schedule(static, BLOCK_FOR) shared(size, p, data, res, mask, BLOCK_FOR)
    for (int i = 0; i < size; i++) {
        if (p(data[i])) {
            res[mask[i] - 1] = data[i];
        }
    }
    return mask[size - 1];
}

// ------------------------------------------------------------ SIMPLE BFS:

struct SimpleBFS {
    vec_vec_t graph;

    std::vector<std::atomic_int> dist;

    int start;

    SimpleBFS(
            vec_vec_t &graph,
            int start
    );

    vec_t getAnswer() const;

    void run();
};

SimpleBFS::SimpleBFS(
        vec_vec_t &graph, int start
) : graph(graph), start(start) {
    if (graph.size() >= MAX_SIZE) {
        throw std::runtime_error("incorrect graph size!");
    }
    dist = std::vector<std::atomic_int>(MAX_SIZE);
}

vec_t SimpleBFS::getAnswer() const {
    vec_t result(graph.size());
    for (int i = 0; i < graph.size(); i++) {
        result[i] = dist[i].load();
    }
    return result;
}

void SimpleBFS::run() {
    std::deque<int> q;
    q.push_back(start);

    int expect = 0;
    dist[start].compare_exchange_strong(expect, 1);

    while (true) {
        int v = q.front();
        q.pop_front();

        for (int i = 0; i < graph[v].size(); i++) {
            int u = graph[v][i];

            expect = 0;
            if (dist[u].compare_exchange_strong(expect, dist[v].load() + 1)) {
                q.push_back(u);
            }
        }

        if (q.empty()) {
            return;
        }
    }

}

// ------------------------------------------------------------ PARALLEL BFS:

struct ParallelBFS {
    vec_vec_t graph;
    std::vector<std::atomic_int> dist;
    int start;

    int maxSize;
    vec_t front0;
    vec_t front1;
    vec_t degrees;
    vec_t mask;
    vec_t tree;

    ParallelBFS(
            vec_vec_t &graph,
            int start,
            int maxSize,
            vec_t &front0,
            vec_t &front1,
            vec_t &degrees,
            vec_t &mask,
            vec_t &tree
    );

    vec_t getAnswer() const;

    void run();
};

ParallelBFS::ParallelBFS(
        vec_vec_t &graph, int start, int maxSize, vec_t &front0, vec_t &front1, vec_t &degrees,
        vec_t &mask, vec_t &tree
) : graph(graph),
    start(start),
    maxSize(maxSize),
    front0(front0),
    front1(front1),
    degrees(degrees),
    mask(mask),
    tree(tree) {
    if (graph.size() >= MAX_SIZE) {
        throw std::runtime_error("incorrect graph size!");
    }
    dist = std::vector<std::atomic_int>(MAX_SIZE);
}

vec_t ParallelBFS::getAnswer() const {
    vec_t result(graph.size());
    for (int i = 0; i < graph.size(); i++) {
        result[i] = dist[i].load();
    }
    return result;
}

void ParallelBFS::run() {
#pragma omp parallel for default(none) schedule(static, BLOCK_FOR) shared(BLOCK_FOR)
    for (int i = 0; i < maxSize; i++) {
        front0[i] = -1;
        front1[i] = -1;
    }

    int size = 1;

    front0[0] = start;

    int expect = 0;
    dist[start].compare_exchange_strong(expect, 1);

    while (true) {

#pragma omp parallel for default(none) schedule(static, BLOCK_FOR) shared(size, BLOCK_FOR)
        for (int i = 0; i < size; i++) {
            int v = front0[i];
            degrees[i] = (int) graph[v].size();
        }

        //Utils::simpleScan(degrees, degrees, size);
        ParallelUtils::parallelScan(degrees, degrees, size, tree);

        std::atomic_int sizeNextFront{0};

#pragma omp parallel for default(none) schedule(static, BLOCK_FOR) shared(size, sizeNextFront, BLOCK_FOR)
        for (int i = 0; i < size; i++) {
            int v = front0[i];

            for (int j = 0; j < graph[v].size(); j++) {
                int u = graph[v][j];
                int ind = (i == 0) ? j : degrees[i - 1] + j;

                int ex = 0;
                int to = dist[v].load() + 1;
                if (dist[u].compare_exchange_strong(ex, to)) {
                    front1[ind] = u;
                    int old = sizeNextFront.load();
                    int nev = ind + 1;
                    while (true) {
                        if (old < nev) {
                            if (sizeNextFront.compare_exchange_weak(old, nev)) break;
                        } else {
                            break;
                        }
                    }
                }

            }
        }

        if (sizeNextFront == 0) {
            return;
        }

        int len = ParallelUtils::parallelFilter(
                front1,
                front0,
                sizeNextFront,
                [](int x) { return x >= 0; },
                mask,
                tree
        );

#pragma omp parallel for default(none) schedule(static, BLOCK_FOR) shared(sizeNextFront, BLOCK_FOR)
        for (int i = 0; i < sizeNextFront; i++) {
            front1[i] = -1;
        }

        size = len;
    }

}

// ------------------------------------------------------------ BENCHMARK:

struct Bench {
    static double run(const fVoidVoid_t &f, int iter);
};

double Bench::run(const fVoidVoid_t &f, int iter) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++)
        f();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    return (double) time.count() / (double) iter;
}

// ------------------------------------------------------------ TESTING:

struct Testing {
    enum Tests {
        SIMPLE_SCAN_TEST,
        PARALLEL_SCAN_TEST,
        SIMPLE_FILTER_TEST,
        PARALLEL_FILTER_TEST,
        SIMPLE_BFS_TEST,
        PARALLEL_BFS_TEST
    };

    struct FS_DTO {
        vec_t data;
        vec_t res1;
        vec_t res2;
        vec_t mask;
        vec_t tree;

        int size = 0;
        int outLength = 0;

        explicit FS_DTO(int size);
    };

    struct BFS_DTO {
        vec_t front0;
        vec_t front1;
        vec_t degrees;
        vec_t mask;
        vec_t tree;

        vec_vec_t graph;

        std::string mode;

        SimpleBFS simpleBfs;
        ParallelBFS parallelBfs;

        explicit BFS_DTO(int maxSize, vec_vec_t &graph);
    };

    static void scanPreTest(FS_DTO &dto);

    static void scanPostTest(FS_DTO &dto);

    static void filterPreTest(FS_DTO &dto);

    static void filterPostTest(FS_DTO &dto);

    static void bfsPreTest(BFS_DTO &dto);

    static void bfsPostTest(BFS_DTO &dto);

    static void simpleScanTest(FS_DTO &dto);

    static void parallelScanTest(FS_DTO &dto);

    static void simpleFilterTest(FS_DTO &dto);

    static void parallelFilterTest(FS_DTO &dto);

    static void simpleBfsTest(BFS_DTO &dto);

    static void parallelBfsTest(BFS_DTO &dto);

    static std::string getTestName(Testing::Tests i);

    static double launch(int arg, Tests test, int iter);

    static void main(
            const std::set<Testing::Tests> &tests, const std::set<int> &args, int num_threads
    );
};

Testing::FS_DTO::FS_DTO(
        int size
) : size(size) {
    data = vec_t(size);
    res1 = vec_t(size);
    res2 = vec_t(size);
    mask = vec_t(size);
    tree = vec_t(4 * size);

    for (int i = 0; i < size; i++) {
        data[i] = Utils::get_rand_int(i);
    }
}

Testing::BFS_DTO::BFS_DTO(
        int maxSize, vec_vec_t &graph
) : graph(graph),
    front0(vec_t(maxSize)),
    front1(vec_t(maxSize)),
    degrees(vec_t(maxSize)),
    mask(vec_t(maxSize)),
    tree(vec_t(4 * maxSize)),
    simpleBfs(SimpleBFS(graph, 0)),
    parallelBfs(ParallelBFS(graph, 0, maxSize, front0, front1, degrees, mask, tree)) {
    // nothing
}

void Testing::scanPreTest(Testing::FS_DTO &dto) {
    // nothing
}

void Testing::scanPostTest(Testing::FS_DTO &dto) {
    Utils::simpleScan(dto.data, dto.res2, dto.size);
    if (Utils::isNotEquals(dto.res1, dto.res2, dto.size)) {
        throw std::runtime_error("assert fail scan!");
    }
}

void Testing::filterPreTest(Testing::FS_DTO &dto) {
    // nothing
}

void Testing::filterPostTest(Testing::FS_DTO &dto) {
    int len = Utils::simpleFilter(
            dto.data, dto.res2, dto.size,
            [](int x) {
                return (x % 2 == 0);
            }
    );
    if (dto.outLength != len || Utils::isNotEquals(dto.res1, dto.res2, dto.size)) {
        throw std::runtime_error("assert fail filter!");
    }
}

void Testing::bfsPreTest(Testing::BFS_DTO &dto) {
    // nothing
}

void Testing::bfsPostTest(Testing::BFS_DTO &dto) {
    SimpleBFS simpleBfs = SimpleBFS(dto.graph, 0);
    simpleBfs.run();
    vec_t res2 = simpleBfs.getAnswer();
    vec_t res1;
    if (dto.mode == "parallel") {
        res1 = dto.parallelBfs.getAnswer();
    } else if (dto.mode == "simple") {
        res1 = dto.simpleBfs.getAnswer();
    } else {
        throw std::runtime_error("unexpected case!");
    }
    if (Utils::isNotEquals(res1, res2)) {
//        freopen("log.txt", "w", stdout);
        std::cout << "#1#" << res1.size() << "\n\n\n\n\n";
        std::cout << res1 << "\n\n\n\n\n";
        std::cout << "#2#" << res2.size() << "\n\n\n\n\n";
        std::cout << res2 << "\n\n\n\n\n";
        for (int i = 0; i < std::min(res1.size(), res2.size()); i++) {
            if (res1[i] != res2[i]) {
                std::cout << i << " : " << res1[i] << " : " << res2[i] << "\n";
            }
        }
        throw std::runtime_error("assert fail bfs!");
    }
}

void Testing::simpleScanTest(Testing::FS_DTO &dto) {
    Utils::simpleScan(
            dto.data, dto.res1, dto.size
    );
}

void Testing::parallelScanTest(Testing::FS_DTO &dto) {
    ParallelUtils::parallelScan(
            dto.data, dto.res1, dto.size, dto.tree
    );
}

void Testing::simpleFilterTest(Testing::FS_DTO &dto) {
    int len = Utils::simpleFilter(
            dto.data, dto.res1, dto.size,
            [](int x) {
                return (x % 2 == 0);
            }
    );
    dto.outLength = len;
}

void Testing::parallelFilterTest(Testing::FS_DTO &dto) {
    int len = ParallelUtils::parallelFilter(
            dto.data, dto.res1, dto.size,
            [](int x) {
                return (x % 2 == 0);
            },
            dto.mask, dto.tree
    );
    dto.outLength = len;
}

void Testing::simpleBfsTest(Testing::BFS_DTO &dto) {
    dto.mode = "simple";
    dto.simpleBfs.run();
}

void Testing::parallelBfsTest(Testing::BFS_DTO &dto) {
    dto.mode = "parallel";
    dto.parallelBfs.run();
}

std::string Testing::getTestName(Testing::Tests i) {
    if (i == SIMPLE_SCAN_TEST) return ("SIMPLE_SCAN_TEST");
    if (i == PARALLEL_SCAN_TEST) return ("PARALLEL_SCAN_TEST");
    if (i == SIMPLE_FILTER_TEST) return ("SIMPLE_FILTER_TEST");
    if (i == PARALLEL_FILTER_TEST) return ("PARALLEL_FILTER_TEST");
    if (i == SIMPLE_BFS_TEST) return ("SIMPLE_BFS_TEST");
    if (i == PARALLEL_BFS_TEST) return ("PARALLEL_BFS_TEST");
    return "null";
}

double Testing::launch(int arg, Testing::Tests test, int iter) {
    if (test == SIMPLE_SCAN_TEST || test == PARALLEL_SCAN_TEST) {
        FS_DTO dto(arg);
        scanPreTest(dto);
        double ans;
        if (test == SIMPLE_SCAN_TEST)
            ans = Bench::run([&dto]() { simpleScanTest(dto); }, iter);
        if (test == PARALLEL_SCAN_TEST)
            ans = Bench::run([&dto]() { parallelScanTest(dto); }, iter);
        scanPostTest(dto);
        return ans;
    }
    if (test == SIMPLE_FILTER_TEST || test == PARALLEL_FILTER_TEST) {
        FS_DTO dto(arg);
        filterPreTest(dto);
        double ans;
        if (test == SIMPLE_FILTER_TEST)
            ans = Bench::run([&dto]() { simpleFilterTest(dto); }, iter);
        if (test == PARALLEL_FILTER_TEST)
            ans = Bench::run([&dto]() { parallelFilterTest(dto); }, iter);
        filterPostTest(dto);
        return ans;
    }
    if (test == SIMPLE_BFS_TEST || test == PARALLEL_BFS_TEST) {
        int sideSize = arg;
        int maxSize = 5 * sideSize * sideSize;
        vec_vec_t graph = Utils::getCubicGraph(sideSize);
        BFS_DTO dto(maxSize, graph);
        bfsPreTest(dto);
        double ans;
        if (test == SIMPLE_BFS_TEST)
            ans = Bench::run([&dto]() { simpleBfsTest(dto); }, iter);
        if (test == PARALLEL_BFS_TEST)
            ans = Bench::run([&dto]() { parallelBfsTest(dto); }, iter);
        bfsPostTest(dto);
        return ans;
    }
    return INT32_MIN;
}

void Testing::main(
        const std::set<Testing::Tests> &tests, const std::set<int> &args, int num_threads
) {
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    std::cout << "HELLO! LAUNCH:\n";
    for (auto test: tests) {
        for (auto arg: args) {
            std::cout << "test: " << getTestName(test) << "; arg: " << arg << "; ";
            std::cout << launch(arg, test, ITERATIONS) << " ms\n";
        }
    }
}

// ------------------------------------------------------------ MAIN:

int main() {
    /*
     * AUTHOR: GRIGORIY KHLYTIN
     *
     * @COPYRIGHT 2022, ALL RIGHTS RESERVED
     */



    /*
     * ATTENTION:
     *
     * THE SIZE OF ANY ARRAY MUST NOT EXCEED THE CONSTANT: <MAX_SIZE> !!!
     */



    /*
     * BABY, SET ME UP HOW YOU LIKE:
     */
    int num_threads = 4;

    std::set<int> array_sizes = std::set<int>{(1 << 20), (1 << 22), (1 << 24), (1 << 26)};

    std::set<int> cube_sizes = std::set<int>{3, 10, 50, 100, 150, 170};

    bool FLAG = false;

    /*
     * RUN IT:
     */
    if (FLAG) {

        Testing::main(
                std::set<Testing::Tests>{
                        Testing::SIMPLE_SCAN_TEST,
                        Testing::PARALLEL_SCAN_TEST,
                        Testing::SIMPLE_FILTER_TEST,
                        Testing::PARALLEL_FILTER_TEST,
                },
                array_sizes,
                num_threads
        );

    } else {

        Testing::main(
                std::set<Testing::Tests>{
                        Testing::SIMPLE_BFS_TEST,
                        Testing::PARALLEL_BFS_TEST
                },
                cube_sizes,
                num_threads
        );

    }

    return 0;
}
