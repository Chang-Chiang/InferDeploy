#pragma once

#include <NvInfer.h>
#include <string>
#include <stdarg.h>
#include <memory>

#define LOGF(...) Logger::__log_info(Level::FATAL, __VA_ARGS__)
#define LOGE(...) Logger::__log_info(Level::ERROR, __VA_ARGS__)
#define LOGW(...) Logger::__log_info(Level::WARN,  __VA_ARGS__)
#define LOG(...)  Logger::__log_info(Level::INFO,  __VA_ARGS__)
#define LOGV(...) Logger::__log_info(Level::VERB,  __VA_ARGS__)
#define LOGD(...) Logger::__log_info(Level::DEBUG, __VA_ARGS__)

#define DGREEN    "\033[1;36m"
#define BLUE      "\033[1;34m"
#define PURPLE    "\033[1;35m"
#define GREEN     "\033[1;32m"
#define YELLOW    "\033[1;33m"
#define RED       "\033[1;31m"
#define CLEAR     "\033[0m"

/// @brief 日志级别枚举类，定义了不同的日志级别
enum class Level : int32_t {
    FATAL = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    VERB  = 4,
    DEBUG = 5
};

/// @brief 日志类，继承自 TensorRT 的 ILogger 接口
class Logger : public nvinfer1::ILogger {

public:
    Logger();
    Logger(Level eLevel);

public:
    /// @brief 日志输出函数，根据日志级别和格式化字符串输出日志信息
    /// @param eLevel 日志级别
    /// @param pFormat 格式化字符串
    /// @param ... 可变参数列表
    static void __log_info(Level eLevel, const char* pFormat, ...);

public:
    /// @brief 重写 ILogger 接口的 log 函数，根据日志级别输出日志信息. noexcept 说明该函数不会抛出异常.
    /// @param nvSeverity TensorRT 日志级别
    /// @param pMsg 日志信息
    virtual void log(Severity nvSeverity, const char* pMsg) noexcept override;

    /// @brief 根据日志级别获取对应的 Severity
    /// @param eLevel 日志级别
    /// @return 对应的 Severity
    Severity GetSeverity(Level eLevel);

    /// @brief 根据 Severity 获取对应的日志级别
    /// @param nvSeverity TensorRT 日志级别
    /// @return 日志级别
    Level GetLevel(Severity nvSeverity);

private:
    static Level m_eLevel;  // 不同实例之间共享同一日志级别
    Severity m_nvSeverity;  // 当前日志级别对应的 TensorRT Severity
};

