// ============================================================================
//  纯 C++ 并查集（Union-Find / Disjoint Set Union）+ 一个 CPU 三维连通域
//  标记参考实现，与 src/ccl.cu 里的 GPU 流水线一一对应。
//
//  只用标准库 —— 不依赖 CUDA，也不依赖任何第三方库。
//
//  构建：  g++ -O2 -std=c++17 reference/union_find.cpp -o bin/uf_demo
//  运行：  ./bin/uf_demo
//
//  演示要点：
//    * 路径压缩（完整压缩与 path-halving 两种写法）
//    * 按尺寸合并（以及按秩合并）
//    * 26 邻接的三维标记，前向偏移与 GPU 版本完全相同
//    * 基于尺寸的过滤，剔除孤立的假阳性
// ============================================================================
#include <algorithm>  // std::swap, std::max
#include <cstdint>
#include <cstdio>
#include <numeric>  // std::iota
#include <vector>

// ----------------------------------------------------------------------------
// 路径压缩 + 按尺寸合并的并查集。
// 单次操作的均摊开销：O(alpha(n)) —— 反 Ackermann 函数，实际不超过 5。
// 内存：每个元素 2 个 int。
// ----------------------------------------------------------------------------
class UnionFind {
public:
    explicit UnionFind(int n = 0) {
        init(n);
    }

    void init(int n) {
        parent_.resize(n);
        size_.assign(n, 1);
        std::iota(parent_.begin(), parent_.end(), 0);  // parent[i] = i
        sets_ = n;
    }

    // 带完整路径压缩的 find（迭代实现，两趟：先找到根，再把路径上的
    // 每个节点直接指到根）。
    int find(int x) {
        int root = x;
        while (parent_[root] != root) {
            root = parent_[root];
        }
        while (parent_[x] != root) {  // 第二趟：压缩
            int next   = parent_[x];
            parent_[x] = root;
            x          = next;
        }
        return root;
    }

    // 路径折半（path halving，单趟）：把每个节点指向它的祖父。
    // 比完整压缩更廉价，渐近行为相同。
    int findHalving(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x          = parent_[x];
        }
        return x;
    }

    // 按尺寸合并：把小树挂到大树下，保持树很浅。当两个不同的集合
    // 真的被合并时返回 true。
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return false;
        }
        if (size_[a] < size_[b]) {
            std::swap(a, b);  // a 是较大的根
        }
        parent_[b] = a;
        size_[a] += size_[b];
        --sets_;
        return true;
    }

    bool connected(int a, int b) {
        return find(a) == find(b);
    }

    int sizeOf(int x) {
        return size_[find(x)];
    }

    int sets() const {
        return sets_;
    }

    int parentOf(int x) const {
        return parent_[x];
    }

private:
    std::vector<int> parent_;
    std::vector<int> size_;  // 仅在根上有效
    int              sets_ = 0;
};

// ----------------------------------------------------------------------------
// 按秩（RANK）合并的变体（按尺寸合并的替代方案）。秩约等于树高。
// ----------------------------------------------------------------------------
class UnionFindByRank {
public:
    explicit UnionFindByRank(int n = 0) {
        init(n);
    }

    void init(int n) {
        parent_.resize(n);
        rank_.assign(n, 0);
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x          = parent_[x];
        }
        return x;
    }

    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return false;
        }
        if (rank_[a] < rank_[b]) {
            std::swap(a, b);
        }
        parent_[b] = a;
        if (rank_[a] == rank_[b]) {
            ++rank_[a];
        }
        return true;
    }

private:
    std::vector<int> parent_, rank_;
};

// ----------------------------------------------------------------------------
// CPU 参考实现：三维连通域标记 + 尺寸过滤。
// 对应 src/ccl.cu 里的 filterComponents()（26 邻接）。
//   mask：0/1，线性索引 i = (z*H + y)*W + x（z 最慢，x 最快）
//   minSize：体素数少于该值的连通域被清成 0
//   keepLargestOnly：为 true 时只保留最大的那一个连通域
// ----------------------------------------------------------------------------
struct CclInfo {
    int  numComponents = 0;  // 找到的连通域数
    int  kept          = 0;  // 通过过滤的连通域数
    int  largest       = 0;  // 最大连通域的体素数
    long fgBefore      = 0;  // 过滤前的前景体素数
    long fgAfter       = 0;  // 过滤后的前景体素数
};

CclInfo filterComponents3D(std::vector<uint8_t>& mask, int W, int H, int D, int minSize, bool keepLargestOnly) {
    const long N   = (long)W * H * D;
    auto       idx = [&](int x, int y, int z) { return (z * H + y) * W + x; };

    // 与 GPU merge 核完全相同的 13 个前向偏移：只访问这些偏移就能把
    // 26 邻域的每条无向边恰好覆盖一次（前 7 个同层，后 6 个下一层）。
    static const int off[13][3] = {
        {1,  0,  0},
        {1,  1,  0},
        {1,  -1, 0},
        {0,  1,  0},
        {0,  -1, 0},
        {-1, 1,  0},
        {-1, -1, 0},
        {1,  0,  1},
        {1,  1,  1},
        {1,  -1, 1},
        {0,  1,  1},
        {0,  -1, 1},
        {0,  0,  1}
    };

    UnionFind uf((int)N);

    // 第一趟：把每个前景体素与它的前向前景邻居做 union。
    long fg = 0;
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int i = idx(x, y, z);
                if (!mask[i]) {
                    continue;
                }
                ++fg;
                for (int k = 0; k < 13; ++k) {
                    int nx = x + off[k][0], ny = y + off[k][1], nz = z + off[k][2];
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= W || ny >= H || nz >= D) {
                        continue;
                    }
                    int j = idx(nx, ny, nz);
                    if (mask[j]) {
                        uf.unite(i, j);
                    }
                }
            }
        }
    }

    // 第二趟：按根统计的尺寸直方图。
    std::vector<int> compSize(N, 0);
    for (long i = 0; i < N; ++i) {
        if (mask[i]) {
            ++compSize[uf.find((int)i)];
        }
    }

    // 连通域统计 + 找出最大连通域的根。
    CclInfo info;
    int     largestRoot = -1;
    for (long r = 0; r < N; ++r) {
        if (compSize[r] <= 0) {
            continue;
        }
        ++info.numComponents;
        if (compSize[r] > info.largest) {
            info.largest = compSize[r];
            largestRoot  = (int)r;
        }
    }
    info.fgBefore = fg;

    // 第三趟：把没通过过滤的连通域所在体素清 0。
    long kept = 0;
    for (long i = 0; i < N; ++i) {
        if (!mask[i]) {
            continue;
        }
        int  r    = uf.find((int)i);
        bool keep = keepLargestOnly ? (r == largestRoot) : (compSize[r] >= minSize);
        if (keep) {
            ++kept;
        }
        else {
            mask[i] = 0;
        }
    }
    info.fgAfter = kept;
    info.kept    = keepLargestOnly
                       ? (info.numComponents > 0 ? 1 : 0)
                       : (int)std::count_if(compSize.begin(), compSize.end(), [&](int c) { return c >= minSize; });
    return info;
}

// ----------------------------------------------------------------------------
// 微型测试框架
// ----------------------------------------------------------------------------
static int  g_fail = 0;
static void check(bool cond, const char* msg) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", msg);
    if (!cond) {
        ++g_fail;
    }
}

static void testBasicUnionFind() {
    std::printf("[test] basic union-find semantics\n");
    UnionFind uf(10);
    check(uf.sets() == 10, "initially 10 disjoint sets");
    uf.unite(0, 1);
    uf.unite(1, 2);
    uf.unite(4, 5);
    check(uf.connected(0, 2), "0 and 2 connected after chained unions");
    check(!uf.connected(0, 4), "0 and 4 still separate");
    check(uf.sizeOf(0) == 3, "component {0,1,2} has size 3");
    check(uf.sets() == 7, "set count dropped to 7 (10 - 3 merges)");
    uf.unite(2, 4);  // 合并两个分组
    check(uf.connected(0, 5) && uf.sizeOf(0) == 5, "merging groups -> {0,1,2,4,5} size 5");
    check(uf.sets() == 6, "set count is 6 (10 - 4 merges)");

    // find() 必须幂等，且两种写法得到的根一致。
    int a = uf.find(5), b = uf.findHalving(5);
    check(a == b, "find() and findHalving() return the same root");
}

static void testCcl3D() {
    std::printf("[test] 3D CCL + size filter (mirror of src/ccl.cu)\n");
    const int            W = 8, H = 8, D = 8;
    std::vector<uint8_t> v((size_t)W * H * D, 0);
    auto                 idx = [&](int x, int y, int z) { return (z * H + y) * W + x; };

    // 4x4x4 实心块 = 64 个体素
    for (int z = 0; z < 4; ++z) {
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                v[idx(x, y, z)] = 1;
            }
        }
    }
    // 远离该块的孤立 2 体素碎块
    v[idx(6, 6, 6)] = 1;
    v[idx(7, 6, 6)] = 1;

    CclInfo r  = filterComponents3D(v, W, H, D, /*minSize=*/10, /*keepLargestOnly=*/false);
    long    fg = 0;
    for (auto b : v) {
        fg += b;
    }

    check(r.numComponents == 2, "found exactly 2 components");
    check(r.largest == 64, "largest component == 64 voxels");
    check(r.kept == 1, "kept exactly 1 component (minSize=10)");
    check(r.fgBefore == 66 && r.fgAfter == 64, "foreground 66 -> 64 (speck removed)");
    check(fg == 64, "resulting mask has 64 voxels");
    check(v[idx(6, 6, 6)] == 0 && v[idx(0, 0, 0)] == 1, "speck cleared, block kept");
}

static void testKeepLargest() {
    std::printf("[test] keep-largest-only mode\n");
    const int            W = 6, H = 6, D = 6;
    std::vector<uint8_t> v((size_t)W * H * D, 0);
    auto                 idx = [&](int x, int y, int z) { return (z * H + y) * W + x; };
    for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                v[idx(x, y, z)] = 1;  // 27 体素的块
            }
        }
    }
    v[idx(5, 5, 5)] = 1;
    v[idx(5, 5, 4)] = 1;
    v[idx(5, 4, 5)] = 1;  // 3 体素的小块

    CclInfo r  = filterComponents3D(v, W, H, D, /*minSize=*/1, /*keepLargestOnly=*/true);
    long    fg = 0;
    for (auto b : v) {
        fg += b;
    }
    check(r.numComponents == 2 && r.kept == 1, "2 found, 1 kept (largest only)");
    check(r.largest == 27 && fg == 27, "only the 27-voxel block remains");
    check(v[idx(5, 5, 5)] == 0, "smaller blob removed");
}

int main() {
    std::printf("=============================================================\n");
    std::printf(" Pure C++ Union-Find + 3D CCL reference (CPU mirror of GPU)\n");
    std::printf("=============================================================\n");
    testBasicUnionFind();
    testCcl3D();
    testKeepLargest();
    std::printf("-------------------------------------------------------------\n");
    if (g_fail == 0) {
        std::printf(" ALL TESTS PASSED\n");
    }
    else {
        std::printf(" %d CHECK(S) FAILED\n", g_fail);
    }
    std::printf("=============================================================\n");
    return g_fail == 0 ? 0 : 1;
}
