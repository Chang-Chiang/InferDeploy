#include "include/utils/Logger.h"

Level Logger::m_eLevel = Level::INFO;

Logger::Logger(Level eLevel) {
    m_eLevel = eLevel;
    m_nvSeverity = GetSeverity(eLevel);
}

Logger::Severity Logger::GetSeverity(Level eLevel) {
    switch (eLevel) {
        case Level::FATAL: return Severity::kINTERNAL_ERROR;
        case Level::ERROR: return Severity::kERROR;
        case Level::WARN:  return Severity::kWARNING;
        case Level::INFO:  return Severity::kINFO;
        case Level::VERB:  return Severity::kVERBOSE;
        default:           return Severity::kVERBOSE;
    }
}

Level Logger::GetLevel(Severity nvSeverity) {
    switch (nvSeverity) {
        case Severity::kINTERNAL_ERROR: return Level::FATAL;
        case Severity::kERROR:          return Level::ERROR;
        case Severity::kWARNING:        return Level::WARN;
        case Severity::kINFO:           return Level::INFO;
        case Severity::kVERBOSE:        return Level::VERB;
    }
}

void Logger::log(Severity nvSeverity, const char* msg) noexcept {
    if (nvSeverity <= GetSeverity(Level::WARN)
        || m_eLevel >= Level::DEBUG)
        __log_info(GetLevel(nvSeverity), "%s", msg);
}

void Logger::__log_info(Level eLevel, const char* pFormat, ...) {
    char msg[1000];
    va_list args;
    va_start(args, pFormat);
    int n = 0;
    
    switch (eLevel) {
        case Level::DEBUG: n += snprintf(msg + n, sizeof(msg) - n, DGREEN "[debug]" CLEAR); break;
        case Level::VERB:  n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]" CLEAR); break;
        case Level::INFO:  n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]" CLEAR); break;
        case Level::WARN:  n += snprintf(msg + n, sizeof(msg) - n, BLUE "[warn]" CLEAR); break;
        case Level::ERROR: n += snprintf(msg + n, sizeof(msg) - n, RED "[error]" CLEAR); break;
        default:           n += snprintf(msg + n, sizeof(msg) - n, RED "[fatal]" CLEAR); break;
    }

    n += vsnprintf(msg + n, sizeof(msg) - n, pFormat, args);

    va_end(args);

    if (eLevel <= m_eLevel) {
        fprintf(stdout, "%s\n", msg);
    }

    if (eLevel <= Level::ERROR) {
        fflush(stdout);
        exit(0);
    }
}

std::shared_ptr<Logger> CreateLogger(Level eLevel) {
    return std::make_shared<Logger>(eLevel);
}