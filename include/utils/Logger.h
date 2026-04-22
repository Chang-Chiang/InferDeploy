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

enum class Level : int32_t {
    FATAL = 0,
    ERROR = 1,
    WARN  = 2,
    INFO  = 3,
    VERB  = 4,
    DEBUG = 5
};

class Logger : public nvinfer1::ILogger {

public:
    Logger();
    Logger(Level iLevel);

public:
    static void __log_info(Level iLevel, const char* pFormat, ...);

public:
    virtual void log(Severity nvSeverity, const char* pMsg) noexcept override;
    Severity GetSeverity(Level iLevel);
    Level GetLevel(Severity nvSeverity);

private:
    static Level m_iLevel = Level::INFO;
    Severity m_nvSeverity;
};

std::shared_ptr<Logger> CreateLogger(Level iLevel);