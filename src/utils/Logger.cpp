#include "include/utils/Logger.h"

// 定义静态成员变量 m_eLevel 的初始值为 Level::INFO
Level Logger::m_eLevel = Level::INFO;

Logger::Logger(Level eLevel) {
    m_eLevel     = eLevel;
    m_nvSeverity = GetSeverity(eLevel);
}

Logger::Severity Logger::GetSeverity(Level eLevel) {
    switch (eLevel) {
        // TensorRT 内部发生了无法恢复的错误
        case Level::FATAL:
            return Severity::kINTERNAL_ERROR;

        // TensorRT 发生了错误，但不影响程序继续运行
        case Level::ERROR:
            return Severity::kERROR;

        // TensorRT 发生了警告，可能会影响性能或结果，但不影响程序继续运行
        case Level::WARN:
            return Severity::kWARNING;

        // TensorRT 普通信息，流程提示
        case Level::INFO:
            return Severity::kINFO;

        // TensorRT 详细信息，调试信息
        case Level::VERB:
            return Severity::kVERBOSE;

        default:
            return Severity::kVERBOSE;
    }
}

Level Logger::GetLevel(Severity nvSeverity) {
    switch (nvSeverity) {
        case Severity::kINTERNAL_ERROR:
            return Level::FATAL;

        case Severity::kERROR:
            return Level::ERROR;

        case Severity::kWARNING:
            return Level::WARN;

        case Severity::kINFO:
            return Level::INFO;

        case Severity::kVERBOSE:
            return Level::VERB;
    }
}

void Logger::log(Severity nvSeverity, const char* msg) noexcept {
    // TensorRT log 比较多，限制只输出 WARN 级别以上的日志
    // 或者当日志级别设置为 DEBUG 时输出所有日志
    if (nvSeverity <= GetSeverity(Level::WARN) || m_eLevel >= Level::DEBUG) {
        __log_info(GetLevel(nvSeverity), "%s", msg);
    }
}

void Logger::__log_info(Level eLevel, const char* pFormat, ...) {
    char    msg[1000];        // 缓存日志信息的最大长度，1000 字符长度的缓存
    va_list args;             // 定义可变参数列表
    va_start(args, pFormat);  // 初始化可变参数列表，pFormat 是最后一个固定参数
    int n = 0;

    switch (eLevel) {
        case Level::DEBUG:
            n += snprintf(msg + n, sizeof(msg) - n, DGREEN "[debug]" CLEAR);
            break;
        case Level::VERB:
            n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]" CLEAR);
            break;
        case Level::INFO:
            n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]" CLEAR);
            break;
        case Level::WARN:
            n += snprintf(msg + n, sizeof(msg) - n, BLUE "[warn]" CLEAR);
            break;
        case Level::ERROR:
            n += snprintf(msg + n, sizeof(msg) - n, RED "[error]" CLEAR);
            break;
        default:
            n += snprintf(msg + n, sizeof(msg) - n, RED "[fatal]" CLEAR);
            break;
    }

    // 格式化可变参数列表，并返回格式化后的长度，追加到 msg 中
    n += vsnprintf(msg + n, sizeof(msg) - n, pFormat, args);

    // 结束可变参数列表的处理
    va_end(args);

    // 如果日志级别小于等于当前设置的日志级别，则输出日志
    if (eLevel <= m_eLevel) {
        fprintf(stdout, "%s\n", msg);
    }

    // 出现 ERROR，输出日志并退出程序
    if (eLevel <= Level::ERROR) {
        fflush(stdout);
        exit(EXIT_FAILURE);
    }
}