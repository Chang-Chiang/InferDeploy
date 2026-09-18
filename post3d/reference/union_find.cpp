// ============================================================================
//  Pure C++ Union-Find (Disjoint Set Union, 并查集) + a CPU 3D connected-
//  component labeling reference that mirrors the GPU pipeline in src/ccl.cu.
//
//  Standard library only — NO CUDA, NO third-party deps.
//
//  Build:  g++ -O2 -std=c++17 reference/union_find.cpp -o bin/uf_demo
//  Run:    ./bin/uf_demo
//
//  Demonstrates:
//    * path compression (two-pass and path-halving variants)
//    * union by size (and union by rank)
//    * 26-connectivity 3D labeling with the same 13 forward offsets as the GPU
//    * size-based filtering to drop isolated false positives
// ============================================================================
#include <cstdio>
#include <cstdint>
#include <vector>
#include <numeric>   // std::iota
#include <algorithm> // std::swap, std::max

// ----------------------------------------------------------------------------
// Union-Find with path compression + union by size.
// Amortized cost per operation: O(alpha(n)) — inverse Ackermann, <= 5 in
// practice. Memory: 2 ints per element.
// ----------------------------------------------------------------------------
class UnionFind {
public:
    explicit UnionFind(int n = 0) { init(n); }

    void init(int n) {
        parent_.resize(n);
        size_.assign(n, 1);
        std::iota(parent_.begin(), parent_.end(), 0);  // parent[i] = i
        sets_ = n;
    }

    // Find with full path compression (iterative, two passes: find root, then
    // re-point every node on the path directly to the root).
    int find(int x) {
        int root = x;
        while (parent_[root] != root) root = parent_[root];
        while (parent_[x] != root) {          // second pass: compress
            int next = parent_[x];
            parent_[x] = root;
            x = next;
        }
        return root;
    }

    // Path halving (single pass): point each node to its grandparent.
    // Cheaper than full compression, same asymptotic behaviour.
    int findHalving(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    // Union by size: attach the smaller tree under the larger one, keeping
    // trees shallow. Returns true when two distinct sets were merged.
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (size_[a] < size_[b]) std::swap(a, b);   // a = larger root
        parent_[b] = a;
        size_[a] += size_[b];
        --sets_;
        return true;
    }

    bool connected(int a, int b) { return find(a) == find(b); }
    int  sizeOf(int x)           { return size_[find(x)]; }
    int  sets() const            { return sets_; }
    int  parentOf(int x) const   { return parent_[x]; }

private:
    std::vector<int> parent_;
    std::vector<int> size_;   // valid only at roots
    int sets_ = 0;
};

// ----------------------------------------------------------------------------
// Union by RANK variant (alternative to union by size). Rank ~= tree height.
// ----------------------------------------------------------------------------
class UnionFindByRank {
public:
    explicit UnionFindByRank(int n = 0) { init(n); }
    void init(int n) { parent_.resize(n); rank_.assign(n, 0); std::iota(parent_.begin(), parent_.end(), 0); }
    int find(int x) { while (parent_[x] != x) { parent_[x] = parent_[parent_[x]]; x = parent_[x]; } return x; }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    }
private:
    std::vector<int> parent_, rank_;
};

// ----------------------------------------------------------------------------
// CPU reference: 3D connected-component labeling + size filtering.
// Mirrors filterComponents() in src/ccl.cu (26-connectivity).
//   mask: 0/1, linear index i = (z*H + y)*W + x  (z slowest, x fastest)
//   minSize: components with fewer voxels are cleared to 0
//   keepLargestOnly: if true, keep only the single largest component
// ----------------------------------------------------------------------------
struct CclInfo {
    int  numComponents = 0;   // connected components found
    int  kept          = 0;   // components surviving the filter
    int  largest       = 0;   // voxel count of the biggest component
    long fgBefore      = 0;   // foreground voxels before filtering
    long fgAfter       = 0;   // foreground voxels after filtering
};

CclInfo filterComponents3D(std::vector<uint8_t>& mask, int W, int H, int D,
                           int minSize, bool keepLargestOnly) {
    const long N = (long)W * H * D;
    auto idx = [&](int x, int y, int z) { return (z * H + y) * W + x; };

    // The same 13 forward offsets used by the GPU merge kernel: visiting only
    // these covers every undirected edge of the 26-neighbourhood exactly once.
    static const int off[13][3] = {
        {1,0,0},{1,1,0},{1,-1,0},{0,1,0},{0,-1,0},{-1,1,0},{-1,-1,0},
        {1,0,1},{1,1,1},{1,-1,1},{0,1,1},{0,-1,1},{0,0,1}
    };

    UnionFind uf((int)N);

    // Pass 1: union every foreground voxel with its forward foreground neighbours.
    long fg = 0;
    for (int z = 0; z < D; ++z)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                int i = idx(x, y, z);
                if (!mask[i]) continue;
                ++fg;
                for (int k = 0; k < 13; ++k) {
                    int nx = x + off[k][0], ny = y + off[k][1], nz = z + off[k][2];
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= W || ny >= H || nz >= D) continue;
                    int j = idx(nx, ny, nz);
                    if (mask[j]) uf.unite(i, j);
                }
            }

    // Pass 2: per-root size histogram.
    std::vector<int> compSize(N, 0);
    for (long i = 0; i < N; ++i)
        if (mask[i]) ++compSize[uf.find((int)i)];

    // Component statistics + identify the largest component's root.
    CclInfo info;
    int largestRoot = -1;
    for (long r = 0; r < N; ++r) {
        if (compSize[r] <= 0) continue;
        ++info.numComponents;
        if (compSize[r] > info.largest) { info.largest = compSize[r]; largestRoot = (int)r; }
    }
    info.fgBefore = fg;

    // Pass 3: clear voxels whose component does not survive the filter.
    long kept = 0;
    for (long i = 0; i < N; ++i) {
        if (!mask[i]) continue;
        int r = uf.find((int)i);
        bool keep = keepLargestOnly ? (r == largestRoot) : (compSize[r] >= minSize);
        if (keep) ++kept; else mask[i] = 0;
    }
    info.fgAfter = kept;
    info.kept = keepLargestOnly ? (info.numComponents > 0 ? 1 : 0)
                                : (int)std::count_if(compSize.begin(), compSize.end(),
                                                     [&](int c){ return c >= minSize; });
    return info;
}

// ----------------------------------------------------------------------------
// Tiny test harness
// ----------------------------------------------------------------------------
static int g_fail = 0;
static void check(bool cond, const char* msg) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", msg);
    if (!cond) ++g_fail;
}

static void testBasicUnionFind() {
    std::printf("[test] basic union-find semantics\n");
    UnionFind uf(10);
    check(uf.sets() == 10, "initially 10 disjoint sets");
    uf.unite(0, 1); uf.unite(1, 2); uf.unite(4, 5);
    check(uf.connected(0, 2), "0 and 2 connected after chained unions");
    check(!uf.connected(0, 4), "0 and 4 still separate");
    check(uf.sizeOf(0) == 3, "component {0,1,2} has size 3");
    check(uf.sets() == 7, "set count dropped to 7 (10 - 3 merges)");
    uf.unite(2, 4);                                  // merge the two groups
    check(uf.connected(0, 5) && uf.sizeOf(0) == 5, "merging groups -> {0,1,2,4,5} size 5");
    check(uf.sets() == 6, "set count is 6 (10 - 4 merges)");

    // find() must be idempotent and both variants agree on the root.
    int a = uf.find(5), b = uf.findHalving(5);
    check(a == b, "find() and findHalving() return the same root");
}

static void testCcl3D() {
    std::printf("[test] 3D CCL + size filter (mirror of src/ccl.cu)\n");
    const int W = 8, H = 8, D = 8;
    std::vector<uint8_t> v((size_t)W * H * D, 0);
    auto idx = [&](int x, int y, int z) { return (z * H + y) * W + x; };

    // 4x4x4 solid block = 64 voxels
    for (int z = 0; z < 4; ++z) for (int y = 0; y < 4; ++y) for (int x = 0; x < 4; ++x)
        v[idx(x, y, z)] = 1;
    // isolated 2-voxel speck, far from the block
    v[idx(6, 6, 6)] = 1;
    v[idx(7, 6, 6)] = 1;

    CclInfo r = filterComponents3D(v, W, H, D, /*minSize=*/10, /*keepLargestOnly=*/false);
    long fg = 0; for (auto b : v) fg += b;

    check(r.numComponents == 2, "found exactly 2 components");
    check(r.largest == 64, "largest component == 64 voxels");
    check(r.kept == 1, "kept exactly 1 component (minSize=10)");
    check(r.fgBefore == 66 && r.fgAfter == 64, "foreground 66 -> 64 (speck removed)");
    check(fg == 64, "resulting mask has 64 voxels");
    check(v[idx(6, 6, 6)] == 0 && v[idx(0, 0, 0)] == 1, "speck cleared, block kept");
}

static void testKeepLargest() {
    std::printf("[test] keep-largest-only mode\n");
    const int W = 6, H = 6, D = 6;
    std::vector<uint8_t> v((size_t)W * H * D, 0);
    auto idx = [&](int x, int y, int z) { return (z * H + y) * W + x; };
    for (int z = 0; z < 3; ++z) for (int y = 0; y < 3; ++y) for (int x = 0; x < 3; ++x)
        v[idx(x, y, z)] = 1;                 // 27-voxel block
    v[idx(5, 5, 5)] = 1; v[idx(5, 5, 4)] = 1; v[idx(5, 4, 5)] = 1;  // 3-voxel blob

    CclInfo r = filterComponents3D(v, W, H, D, /*minSize=*/1, /*keepLargestOnly=*/true);
    long fg = 0; for (auto b : v) fg += b;
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
    if (g_fail == 0) std::printf(" ALL TESTS PASSED\n");
    else             std::printf(" %d CHECK(S) FAILED\n", g_fail);
    std::printf("=============================================================\n");
    return g_fail == 0 ? 0 : 1;
}
