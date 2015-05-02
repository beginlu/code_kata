#ifndef TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_LOG_HPP_
#define TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_LOG_HPP_

#include <cstdio>
#include "test_gaussian_blur/impl_custom/config/config.hpp"

#define TGB_LOG(tag, fmt, ...)  fprintf(tag, fmt, ##__VA_ARGS__);
#define TGB_LOG_INFO(fmt, ...)  TGB_LOG(stdout, "Info: "fmt, ##__VA_ARGS__);
#define TGB_LOG_ERROR(fmt, ...) TGB_LOG(stderr, "Error: "fmt, ##__VA_ARGS__);

#endif // TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_LOG_HPP_
