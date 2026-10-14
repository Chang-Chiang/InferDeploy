/*
 * 体数据重建 + host 侧 I/O 的具体实现。
 */
#include <cstdio>
#include <cstring>

#include "common.cuh"
#include "volume.cuh"

bool loadMeta(const char* path, VolumeMeta& m) {
    FILE* f = fopen(path, "r");
    if (!f) {
        return false;
    }
    int got = fscanf(f, "%d %d %d %f %f %f", &m.W, &m.H, &m.D, &m.dx, &m.dy, &m.dz);
    fclose(f);
    if (got < 3) {
        return false;  // 间距字段可省略 -> 默认 1mm
    }
    if (got < 6) {
        m.dx = m.dy = m.dz = 1.f;
    }
    return m.W > 0 && m.H > 0 && m.D > 0;
}

bool loadStack(const char* path, size_t n, uint8_t* buf) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    size_t rd = fread(buf, 1, n, f);
    fclose(f);
    return rd == n;
}

bool saveStack(const char* path, const uint8_t* buf, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    size_t wr = fwrite(buf, 1, n, f);
    fclose(f);
    return wr == n;
}

bool loadSlicesDir(const char* dir, const VolumeMeta& m, uint8_t* hostBuf) {
    const size_t sliceBytes = (size_t)m.W * m.H;
    char         path[1024];
    for (int z = 0; z < m.D; ++z) {
        snprintf(path, sizeof(path), "%s/slice_%04d.u8", dir, z);
        FILE* f = fopen(path, "rb");
        if (!f) {
            fprintf(stderr, "[volume] missing slice file: %s\n", path);
            return false;
        }
        size_t rd = fread(hostBuf + (size_t)z * sliceBytes, 1, sliceBytes, f);
        fclose(f);
        if (rd != sliceBytes) {
            fprintf(stderr, "[volume] short read on %s (%zu/%zu)\n", path, rd, sliceBytes);
            return false;
        }
    }
    return true;
}

bool savePGM(const char* path, const uint8_t* slice, int W, int H) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    fprintf(f, "P5\n%d %d\n255\n", W, H);
    for (int i = 0; i < W * H; ++i) {
        unsigned char v = slice[i] ? 255 : 0;
        fwrite(&v, 1, 1, f);
    }
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// GPU 侧重建：上传裸字节栈，并二值化为干净的 0/1 uint8 体数据。
// 这一步就是「逐切面掩码 -> 三维体」的重建过程。
// ---------------------------------------------------------------------------
__global__ void binarizeKernel(const uint8_t* __restrict__ raw, uint8_t* __restrict__ out, size_t n, int thr) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    out[i] = (raw[i] > (uint8_t)thr) ? 1u : 0u;
}

uint8_t* reconstructOnDevice(const uint8_t* h_raw, const VolumeMeta& m, int binThreshold) {
    const size_t n      = m.voxels();
    uint8_t*     d_raw  = nullptr;
    uint8_t*     d_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw, n));
    CUDA_CHECK(cudaMalloc(&d_mask, n));
    CUDA_CHECK(cudaMemcpy(d_raw, h_raw, n, cudaMemcpyHostToDevice));

    int    threads = 256;
    size_t blocks  = (n + threads - 1) / threads;
    binarizeKernel<<<(unsigned)blocks, threads>>>(d_raw, d_mask, n, binThreshold);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_raw));
    return d_mask;
}

long countForeground(const uint8_t* buf, size_t n) {
    long c = 0;
    for (size_t i = 0; i < n; ++i) {
        c += buf[i] ? 1 : 0;
    }
    return c;
}

__global__ void countFgKernel(const uint8_t* __restrict__ mask, size_t n, unsigned long long* c) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i]) {
        atomicAdd(c, 1ULL);
    }
}

long countForegroundDevice(const uint8_t* d_mask, size_t n) {
    unsigned long long* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_c, 0, sizeof(unsigned long long)));
    int      threads = 256;
    unsigned blocks  = (unsigned)((n + threads - 1) / threads);
    countFgKernel<<<blocks, threads>>>(d_mask, n, d_c);
    unsigned long long h = 0;
    CUDA_CHECK(cudaMemcpy(&h, d_c, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_c));
    return (long)h;
}
