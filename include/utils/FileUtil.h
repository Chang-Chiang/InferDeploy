#include <string>
#include <vector>

bool IsFileExist(const std::string& strFileName);

bool CreateDir(const std::string& strDir);

bool IsFileReadSuccess(const std::string &strPath, std::vector<unsigned char> &vecData, size_t &iSize);

std::string GetFileType(std::string strFilePath);

std::string GetFileName(std::string strFilePath);

std::string ChangePath(std::string strOnnxPath, std::string strRelativePath, std::string strPostfix, std::string strTag);

std::vector<unsigned char> LoadFile(const std::string &strPath);

std::vector<std::string>   LoadDataList(const std::string &strPath);
