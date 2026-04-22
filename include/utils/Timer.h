#pragma once

#include <chrono>
#include <ratio>
#include <string>

#include <cuda_runtime.h>

class Timer {
public:
    using s  = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

public:
    Timer();
    ~Timer();

public:
    void Init();
    void Show();

    void StartCpu();
    void StopCpu();

    template <typename span>
    void StopCpu(std::string strMsg);

    void StartGpu();
    void StopGpu();
    void StopGpu(std::string strMsg);

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_pointCpuStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_pointCpuStop;

    cudaEvent_t m_eventGpuStart;
    cudaEvent_t m_eventGpuStop;

    float m_fTimeElasped;

    std::vector<std::string> m_strTimeMsg;
};

template <typename span>
void Timer::StopCpu(std::string strMsg) {
    m_pointCpuStop = std::chrono::high_resolution_clock::now();

    char c_buff[100];
    std::string str_time_unit;

    if(std::is_same<span, s>::value) { str_time_unit = "s"; }
    else if(std::is_same<span, ms>::value) { str_time_unit = "ms"; }
    else if(std::is_same<span, us>::value) { str_time_unit = "us"; }
    else if(std::is_same<span, ns>::value) { str_time_unit = "ns"; }

    std::chrono::duration<double, span> d_time = m_pointCpuStop - m_pointCpuStart;
    sprintf(c_buff, "\t%-60s uses %.6lf %s", strMsg.c_str(), d_time.count(), str_time_unit.c_str());
    m_strTimeMsg.emplace_back(c_buff);
}