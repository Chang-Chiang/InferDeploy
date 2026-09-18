# 并查集（Union-Find / Disjoint Set Union）详解

> 配套代码：纯 C++ 实现 [`../reference/union_find.cpp`](../reference/union_find.cpp)
> （可编译运行、含自测）。项目中的 GPU 版连通域标记见 [`../src/ccl.cu`](../src/ccl.cu)，
> 二者互为“CPU 参考 / GPU 加速”对照。CUDA 相关知识点见
> [`CUDA_GUIDE.zh-CN.md`](CUDA_GUIDE.zh-CN.md)。

---

## 目录
1. 并查集是什么
2. 从朴素到高效：三种实现演进
3. 两大优化：路径压缩 + 按秩/按大小合并
4. 复杂度分析与反阿克曼函数
5. 纯 C++ 实现（逐段讲解）
6. 实战应用：三维连通域标记
7. CPU 实现 vs GPU 实现对照
8. 常见变体与扩展
9. 易错点与最佳实践
10. 构建、运行与测试
11. 练习题与延伸阅读

---

## 1. 并查集是什么

**并查集**（Union-Find，又称 **Disjoint Set Union, DSU**，不相交集合）维护一组**互不相交
的集合**，支持两种核心操作：

- **Find(x)**：返回 `x` 所属集合的代表元（“根”）。用于判断两个元素是否在同一集合。
- **Union(a, b)**：把 `a`、`b` 所在的两个集合合并为一个。

它回答的是**动态连通性**问题：在不断“加边/合并”的过程中，随时查询“两点是否连通”。

**典型应用**：
- 无向图连通分量、判环；
- **Kruskal 最小生成树**（按边权排序，能连通两端不同分量的边才选入）；
- **图像/体数据的连通域标记**（本项目就是三维版）；
- 等价类合并、账户合并、朋友圈/省份数量；
- 离线 LCA（Tarjan 算法）、二分图判定（扩展域）等。

**核心思想**：每个集合用一棵树表示，树根即代表元。`Find` 沿父指针上溯到根，`Union` 把一棵
树的根挂到另一棵树的根下。

---

## 2. 从朴素到高效：三种实现演进

设 `n` 为元素个数。

### 2.1 Quick-Find（快速查找）
用一个数组 `id[]`，同一集合的元素 `id` 相同。
- `Find(x)` = `id[x]`，**O(1)**；
- `Union(a,b)` 要把所有 `id==id[a]` 的元素改成 `id[b]`，**O(n)**。

查找快，但合并慢，做 `n` 次合并就是 `O(n^2)`。

### 2.2 Quick-Union（快速合并）
改用父指针数组 `parent[]`，`parent[x]` 指向父节点，根的 `parent[root]==root`。
- `Union(a,b)`：`parent[find(a)] = find(b)`，只需 **O(树高)**；
- `Find(x)`：沿父指针上溯到根，**O(树高)**。

合并快了，但**树可能退化成一条链**（例如依次 `union(1,2), union(2,3), …`），此时
`Find` 退化到 **O(n)**。

### 2.3 问题：树太高
退化的根因是树高不受控。于是有了下面两大优化，把树“压扁”并保持“矮”。

---

## 3. 两大优化

### 3.1 路径压缩（Path Compression）
在 `Find(x)` 上溯到根后，把**路径上的所有节点直接挂到根**，下次查询就近乎 O(1)。

```
压缩前:  x → p → q → root        压缩后:  x → root,  p → root,  q → root
```

常见写法：
- **完整压缩（两遍）**：第一遍找根，第二遍把路径上每个节点指到根；
- **路径减半（path halving）**：单遍，把每个节点指向它的**祖父**
  `parent[x] = parent[parent[x]]`；
- **路径分裂（path splitting）**：单遍，类似减半但更新方式略有不同。

三者的摊还复杂度同阶，减半/分裂常数更小、无需第二遍。

### 3.2 按秩合并 / 按大小合并（Union by Rank / Size）
合并时，把**矮树（秩小）挂到高树（秩大）**下，避免树变高：
- **按大小(size)**：`size[root]` 记录该集合元素数，小的挂到大的下面；
- **按秩(rank)**：`rank[root]` 记录树高的上界，秩相等时挂任意一边并把新根秩 +1。

只用“按秩合并”或只用“路径压缩”，摊还都是 `O(log n)`。

### 3.3 两者结合 → 反阿克曼
**路径压缩 + 按秩/按大小合并**同时使用时，单次操作的摊还代价降到
`O(α(n))`，`α` 是**反阿克曼函数**，增长极慢：对任何现实规模（哪怕 `n` 是宇宙原子数），
`α(n) ≤ 5`。因此可近似视为**常数**。

---

## 4. 复杂度分析

| 实现 | Find | Union | 空间 |
|---|---|---|---|
| Quick-Find | O(1) | O(n) | O(n) |
| Quick-Union（朴素） | O(n) 最坏 | O(n) 最坏 | O(n) |
| + 路径压缩 或 + 按秩 | O(log n) 摊还 | O(log n) | O(n) |
| **路径压缩 + 按秩/大小** | **O(α(n)) 摊还** | **O(α(n))** | O(n) |

- `α(n)`：反阿克曼函数，实际 ≤ 5。
- 空间：`parent[]` 必存；`size[]` 或 `rank[]` 二选一，各 `O(n)`。
- `m` 次操作在 `n` 元素上的总代价为 `O(m · α(n))`。

---

## 5. 纯 C++ 实现（逐段讲解）

完整可运行代码见 [`../reference/union_find.cpp`](../reference/union_find.cpp)。核心类：

```cpp
class UnionFind {
public:
    explicit UnionFind(int n = 0) { init(n); }

    void init(int n) {
        parent_.resize(n);
        size_.assign(n, 1);
        std::iota(parent_.begin(), parent_.end(), 0);  // parent[i] = i：每个元素自成集合
        sets_ = n;
    }

    // 完整路径压缩（迭代，两遍）：先找根，再把路径上节点直接挂到根
    int find(int x) {
        int root = x;
        while (parent_[root] != root) root = parent_[root];
        while (parent_[x] != root) {          // 第二遍：压缩
            int next = parent_[x];
            parent_[x] = root;
            x = next;
        }
        return root;
    }

    // 路径减半（单遍）：把节点指向祖父
    int findHalving(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    // 按大小合并：小树挂到大树下，保持树矮
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;                 // 已在同一集合，无需合并
        if (size_[a] < size_[b]) std::swap(a, b); // 保证 a 是较大的根
        parent_[b] = a;
        size_[a] += size_[b];
        --sets_;
        return true;
    }

    bool connected(int a, int b) { return find(a) == find(b); }
    int  sizeOf(int x)           { return size_[find(x)]; }
    int  sets() const            { return sets_; }

private:
    std::vector<int> parent_;
    std::vector<int> size_;   // 仅在“根”处有效
    int sets_ = 0;
};
```

**要点解释**
- `std::iota(parent_.begin(), parent_.end(), 0)`：把 `parent_[i]` 初始化为 `i`，即
  “每个元素是自己的根”，共 `n` 个集合。
- `find` 用**迭代**而非递归：深树时递归可能栈溢出，迭代更稳。
- `size_[a] += size_[b]`：**`size_` 只在根处有意义**；非根的 `size_` 值是过期的，绝不能读。
- `unite` 里先 `find` 拿到两个根，若相同则说明已连通，直接返回 `false`（避免自环、`sets_` 误减）。
- `connected(a,b)` 复用 `find`，顺带触发路径压缩。

**按秩合并变体**（与按大小二选一即可）：
```cpp
bool unite(int a, int b) {
    a = find(a); b = find(b);
    if (a == b) return false;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];   // 只有等高时才增高
    return true;
}
```

---

## 6. 实战应用：三维连通域标记

这正是本项目“去除层间孤立假阳性”的核心。给定一个二值体数据 `mask`（`1`=前景），把
**互相连通的前景体素**归为同一连通域，再按体积筛掉过小的域。

### 6.1 建模为并查集
- 每个前景体素 = 一个元素，线性索引 `i = (z*H + y)*W + x`；
- 对每个前景体素，与其**相邻的前景体素**做 `union`；
- 全部合并后，`find(i)` 相同的体素属于同一连通域；
- 统计每个根的元素数（直方图），删除 `< minSize` 的小域。

### 6.2 连通性与“13 前向偏移”
- **6 连通**：上下左右前后（面相邻）；
- **26 连通**：3×3×3 邻域去掉中心（面/棱/角都算相邻）。本项目用 26 连通。

若对每个体素都检查全部 26 个邻居，每条无向边会被处理两次。只检查 **13 个“前向”偏移**
即可让每条边恰好处理一次：
```cpp
static const int off[13][3] = {
    {1,0,0},{1,1,0},{1,-1,0},{0,1,0},{0,-1,0},{-1,1,0},{-1,-1,0},
    {1,0,1},{1,1,1},{1,-1,1},{0,1,1},{0,-1,1},{0,0,1}
};
```

### 6.3 CPU 参考实现（核心逻辑）
摘自 [`../reference/union_find.cpp`](../reference/union_find.cpp) 的 `filterComponents3D`：
```cpp
UnionFind uf((int)N);
// Pass 1: 与 13 个前向邻居中的前景体素合并
for (z...) for (y...) for (x...) {
    int i = idx(x,y,z);
    if (!mask[i]) { continue; }
    ++fg;
    for (int k = 0; k < 13; ++k) {
        int nx=x+off[k][0], ny=y+off[k][1], nz=z+off[k][2];
        if (越界) continue;
        int j = idx(nx,ny,nz);
        if (mask[j]) uf.unite(i, j);
    }
}
// Pass 2: 每个根的元素个数（连通域体积）
std::vector<int> compSize(N, 0);
for (i=0..N-1) if (mask[i]) ++compSize[uf.find(i)];
// Pass 3: 删除体积 < minSize 的小域（或只保留最大域）
for (i=0..N-1) {
    if (!mask[i]) continue;
    int r = uf.find(i);
    bool keep = keepLargestOnly ? (r == largestRoot) : (compSize[r] >= minSize);
    if (!keep) mask[i] = 0;
}
```

> 小知识：本项目把“连通域标签”直接取为**该域最小体素索引**（即根），天然稳定且无需
> 额外的标签重映射。若要让 `0` 专门表示背景，可把标签存成 `root + 1`。

---

## 7. CPU 实现 vs GPU 实现对照

同一套并查集思想，串行与并行的落地方式不同：

| 维度 | CPU（`reference/union_find.cpp`） | GPU（`src/ccl.cu`） |
|---|---|---|
| `find` | 只读上溯 + **就地路径压缩** | **只读**上溯，**不在其中压缩**（避免并发写竞争） |
| `union` | `parent[大] = 小`（直接写） | `atomicMin(&parent[hi], lo)`（原子，单调递减） |
| 压缩 | 融进 `find` | 独立的 `compressKernel` 做**指针跳跃** |
| 遍历/终止 | 顺序一遍即可 | `changed` 标志 + **不动点迭代**（merge↔compress 交替直到稳定） |
| 并发安全 | 单线程无竞争 | `atomicMin` 幂等 + 单调 ⇒ 多线程安全且必收敛 |
| 直方图 | `std::vector<int>` 计数 | `atomicAdd(&counts[root], 1)` |
| 标签 | 根索引 | 根索引（= 域内最小体素索引） |

关键差异在于**并发正确性**：GPU 上成千上万个线程同时 `find/union`，因此
1. `find` 必须只读（压缩另设一步），
2. `union` 必须用原子且保证**单调**（`atomicMin` 只会让根变小），从而迭代必然收敛。
这正是 GPU 版把它写成“merge + compress 交替的不动点迭代”的原因。详见
[`CUDA_GUIDE.zh-CN.md`](CUDA_GUIDE.zh-CN.md) 第 6 节。

---

## 8. 常见变体与扩展

- **带权并查集（Weighted DSU）**：`parent[x]` 之外再维护 `weight[x]` = `x` 到父节点的
  “权”（距离/异或/比值）。`find` 上溯时累加权重，可解“食物链”“除法求值”“向量偏移”等。
- **扩展域 / 种类并查集**：把“属于第 k 类”拆成 `n*k` 个节点，用普通并查集表达
  “敌对/朋友”等关系（如 2-SAT 式建模、二分图判定）。
- **可回滚并查集（Rollback DSU）**：不压缩路径、按秩合并，记录每次 `union` 的改动以便
  撤销；用于**分治/线段树分治**等离线算法。
- **可持久化并查集**：保留历史版本（用可持久化数组实现 `parent`），支持查询过去某时刻的
  连通性。
- **动态开点 / 离散化**：元素值域很大但实际用到的很少时，先离散化或用 `map` 动态开点。

---

## 9. 易错点与最佳实践

1. **忘记优化会退化**：不做路径压缩/按秩合并，`find` 最坏 `O(n)`；两者至少各选其一，
   最好都上。
2. **递归 `find` 可能栈溢出**：深树时改用**迭代**实现（本项目即迭代）。
3. **`size_/rank_` 只在根有效**：合并后非根的 `size_` 是过期值，别拿它做判断。
4. **`unite` 要先判 `find(a)==find(b)`**：否则会形成自环、`sets_` 误减、`size_` 翻倍。
5. **标签与“背景”冲突**：若用 `0` 表示背景，连通域标签记得整体 `+1`（或另设偏移）。
6. **并发场景（GPU）**：`find` 里边找边压缩会写竞争 —— 必须把“只读 find”和“统一压缩”
   分成两步；`union` 用原子且保证单调（如 `atomicMin`）以确保收敛。
7. **按大小 vs 按秩二选一**：无需同时使用；按大小还能顺带回答“集合多大”。
8. **路径压缩 + 按大小**：注意若同时用路径压缩，`rank` 不再是精确树高（只是上界），
   而 `size` 始终精确 —— 需要“集合元素数”时用 `size` 更方便。

---

## 10. 构建、运行与测试

```bash
# 方式一：Makefile 目标（纯 C++，无需 CUDA）
make uf            # -> bin/uf_demo
./bin/uf_demo

# 方式二：直接编译
g++ -O2 -std=c++17 -Wall -Wextra reference/union_find.cpp -o bin/uf_demo
./bin/uf_demo
```

程序内置自测（`main()`），覆盖：
- 基本语义：初始 `n` 个集合、链式合并后的连通性、`sizeOf`、`sets()` 计数、
  `find` 与 `findHalving` 结果一致；
- 三维 CCL：`4×4×4` 方块（64 体素）+ 孤立 2 体素斑点 → `minSize=10` 时找到 2 个域、
  保留 1 个、前景 `66 → 64`、斑点被清除、方块保留；
- `keepLargestOnly`：只保留最大域。

预期输出结尾为 `ALL TESTS PASSED`（返回码 0）。它与 GPU 版
`./bin/liver3d --selftest` 的用例一一对应，可作为交叉验证的“标准答案”。

---

## 11. 练习题与延伸阅读

**练习（LeetCode）**
- 547 省份数量、200 岛屿数量、684 冗余连接、721 账户合并、990 等式方程的可满足性；
- 399 除法求值（带权并查集）、839 相似字符串组、128（对比哈希/排序解法）。

**延伸阅读**
- *CLRS*（算法导论）第 21 章 “Data Structures for Disjoint Sets” —— 摊还分析与
  `α(n)` 的严格证明；
- Tarjan, “Efficiency of a Good But Not Linear Set Union Algorithm”（`α(n)` 界）；
- 本项目对照阅读：CPU 版 [`../reference/union_find.cpp`](../reference/union_find.cpp)
  ↔ GPU 版 [`../src/ccl.cu`](../src/ccl.cu)；CUDA 细节见
  [`CUDA_GUIDE.zh-CN.md`](CUDA_GUIDE.zh-CN.md)。
