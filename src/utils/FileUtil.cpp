#include "include/utils/FileUtil.h"

#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "include/utils/Logger.h"

bool IsFileExist(const std::string strFileName) {
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(strFileName))) {
        return false;
    }
    else {
        return true;
    }
}

bool IsFileReadSuccess(const std::string &strPath, std::vector<unsigned char> &vecData, size_t &iSize){
    std::stringstream ss_model;
    std::ifstream cache(strPath);

    ss_model.seekg(0, ss_model.beg);
    ss_model << cache.rdbuf();
    cache.close();

    ss_model.seekg(0, std::ios::end);
    iSize = ss_model.tellg();

    std::vector<uint8_t> vec_tmp;
    ss_model.seekg(0, std::ios::beg);
    vec_tmp.resize(iSize);

    ss_model.read((char*)vecData[0], iSize);
    return true;
}

std::vector<unsigned char> LoadFile(const std::string &strFile) {
    std::ifstream in(strFile, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        return {};
    }

    in.seekg(0, std::ios::end);
    size_t i_length = in.tellg();

    std::vector<uint8_t> vecData;
    if (i_length > 0) {
        in.seekg(0, std::ios::beg);
        vecData.resize(i_length);
        in.read((char*)&vecData[0], i_length);
    }
    in.close();
    return vecData;
}

std::vector<std::string> LoadDataList(const std::string& strFile) {
    std::vector<std::string> vec_list;
    auto *pf = fopen(strFile.c_str(), "r");
    if (!pf) {
        LOGE("Failed to open %s", strFile.c_str());
    }

    char c_str[512];
    while (fgets(c_str, 512, pf) != NULL) {
        for (int i = 0; c_str[i] != '\0'; ++i) {
            if (c_str[i] == '\n'){
                c_str[i] = '\0';
                break;
            }
        }
        vec_list.push_back(c_str);
    }
    fclose(pf);
    return vec_list;
}

std::string ChangePath(std::string srcPath, std::string strRelativePath, std::string strPostfix, std::string strTag) {
    int i_name_l = srcPath.rfind("/");
    int i_name_r = srcPath.rfind(".");

    int i_dir_l  = 0;
    int i_dir_r  = srcPath.rfind("/");

    std::string str_new_path;

    str_new_path = srcPath.substr(i_dir_l, i_dir_r + 1);
    str_new_path += strRelativePath;
    str_new_path += srcPath.substr(i_name_l, i_name_r - i_name_l);

    if (!strTag.empty()) {
        str_new_path += "-" + strTag + strPostfix;
    }
    else {
        str_new_path += strPostfix;
    }

    return str_new_path;
}

std::string GetOutputPath(std::string srcPath, std::string strPostfix) {
    int i_pos = srcPath.rfind(".");
    std::string str_tar_path;
    str_tar_path = srcPath.substr(0, i_pos);
    str_tar_path += "_" + strPostfix + ".png";
    return str_tar_path;
}

std::string GetFileType(std::string strFilePath) {
    int i_pos = strFilePath.rfind(".");
    std::string str_suffix;
    str_suffix = strFilePath.substr(i_pos, strFilePath.i_length());
    return str_suffix;
}

std::string GetFileName(std::string strFilePath) {
    int i_pos = strFilePath.rfind("/");
    std::string str_suffix;
    str_suffix = strFilePath.substr(i_pos + 1, strFilePath.i_length());
    return str_suffix;
}