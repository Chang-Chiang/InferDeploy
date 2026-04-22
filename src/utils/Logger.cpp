#include "include/utils/Logger.h"

Logger::Logger(Level iLevel) {
    m_iLevel = iLevel;
    m_nvSeverity = GetSeverity(iLevel);
}

Logger::Severity Logger::GetSeverity(Level iLevel) {
    switch (iLevel) {
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
    if (nvSeverity <= get_severity(Level::WARN)
        || m_iLevel >= Level::DEBUG)
        __log_info(GetLevel(nvSeverity), "%s", msg);
}

void Logger::__log_info(Level iLevel, const char* pFormat, ...) {
    char msg[1000];
    va_list args;
    va_start(args, pFormat);
    int n = 0;
    
    switch (iLevel) {
        case Level::DEBUG: n += snprintf(msg + n, sizeof(msg) - n, DGREEN "[debug]" CLEAR); break;
        case Level::VERB:  n += snprintf(msg + n, sizeof(msg) - n, PURPLE "[verb]" CLEAR); break;
        case Level::INFO:  n += snprintf(msg + n, sizeof(msg) - n, YELLOW "[info]" CLEAR); break;
        case Level::WARN:  n += snprintf(msg + n, sizeof(msg) - n, BLUE "[warn]" CLEAR); break;
        case Level::ERROR: n += snprintf(msg + n, sizeof(msg) - n, RED "[error]" CLEAR); break;
        default:           n += snprintf(msg + n, sizeof(msg) - n, RED "[fatal]" CLEAR); break;
    }

    n += vsnprintf(msg + n, sizeof(msg) - n, pFormat, args);

    va_end(args);

    if (iLevel <= m_iLevel) {
        fprintf(stdout, "%s\n", msg);
    }

    if (iLevel <= Level::ERROR) {
        fflush(stdout);
        exit(0);
    }
}

shared_ptr<Logger> CreateLogger(Level iLevel) {
    return make_shared<Logger>(iLevel);
}