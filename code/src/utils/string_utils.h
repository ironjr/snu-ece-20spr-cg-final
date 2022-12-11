/**
 * Some string manipulation tool without using boost
 *
 * https://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring
 * https://stackoverflow.com/questions/44973435/stdptr-fun-replacement-for-c17/44973498#44973498
 */

#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <algorithm>
#include <cctype>
#include <locale>
#include <string>


static inline void ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

static inline void rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

static inline void trim(std::string &s)
{
    ltrim(s);
    rtrim(s);
}

static inline std::string ltrim_copy(std::string s)
{
    ltrim(s);
    return s;
}

static inline std::string rtrim_copy(std::string s)
{
    rtrim(s);
    return s;
}

static inline std::string trim_copy(std::string s)
{
    ltrim(s);
    rtrim(s);
    return s;
}

namespace jr
{
static inline int stoi(const std::string &s, size_t *idx = nullptr)
{
    try
    {
        return std::stoi(s, idx);
    }
    catch (std::invalid_argument const &e)
    {
        std::cout << "ERROR::STRING_UTILS::INTEGER_INVALID_ARGUMENT" << std::endl;
    }
    catch (std::out_of_range const &e)
    {
        std::cout << "ERROR::STRING_UTILS::INTEGER_OUT_OF_RANGE" << std::endl;
    }
    return 0;
}

static inline float stof(const std::string &s, size_t *idx = nullptr)
{
    try
    {
        return std::stof(s, idx);
    }
    catch (std::invalid_argument const &e)
    {
        std::cout << "ERROR::STRING_UTILS::FLOAT_INVALID_ARGUMENT" << std::endl;
    }
    catch (std::out_of_range const &e)
    {
        std::cout << "ERROR::STRING_UTILS::FLOAT_OUT_OF_RANGE" << std::endl;
    }
    return 0.0f;
}
}

#endif
