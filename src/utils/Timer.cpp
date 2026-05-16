#include "include/utils/Timer.h"
#include "include/utils/Logger.h"
#include "include/utils/CudaCheck.h"

Timer::Timer() {
    m_fTimeElasped = 0;
    m_pointCpuStart = std::chrono::high_resolution_clock::now();
    m_pointCpuStop = std::chrono::high_resolution_clock::now();
    cudaEventCreate(&m_eventGpuStart);
    cudaEventCreate(&m_eventGpuStop);
}

Timer::~Timer() {
    cudaFree(m_eventGpuStart);
    cudaFree(m_eventGpuStop);
    cudaEventDestroy(m_eventGpuStart);
    cudaEventDestroy(m_eventGpuStop);
}

void Timer::Init() {
    m_strTimeMsg.clear();
}

void Timer::Show() {
    for (int i = 0; i < m_strTimeMsg.size(); i++) {
        LOGV("%s", m_strTimeMsg[i].c_str());
    }
}

void Timer::StartGpu() {
    cudaEventRecord(m_eventGpuStart, 0);
}

void Timer::StopGpu() {
    cudaEventRecord(m_eventGpuStop, 0);
}

void Timer::StopGpu(std::string strMsg){
    cudaEventRecord(m_eventGpuStop, 0);
    char c_buff[100];

    CUDA_CHECK(cudaEventSynchronize(m_eventGpuStart));
    CUDA_CHECK(cudaEventSynchronize(m_eventGpuStop));
    cudaEventElapsedTime(&m_fTimeElasped, m_eventGpuStart, m_eventGpuStop);

    sprintf(c_buff, "\t%-60s uses %.6lf ms", strMsg.c_str(), m_fTimeElasped);
    this->m_strTimeMsg.emplace_back(c_buff);
}

void Timer::StartCpu() {
    m_pointCpuStart = std::chrono::high_resolution_clock::now();
}

void Timer::StopCpu() {
    m_pointCpuStop = std::chrono::high_resolution_clock::now();
}
