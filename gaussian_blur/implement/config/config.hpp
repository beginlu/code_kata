#ifndef GAUSSIAN_BLUR_IMPL_CUSTOM_CONFIG_HPP_
#define GAUSSIAN_BLUR_IMPL_CUSTOM_CONFIG_HPP_

#include <cstdio>
#include "cuda_runtime.h"

#define TGB_SECTION_BEGIN(...) {;
#define TGB_SECTION_END(...)   } SEC_##__VA_ARGS__:;
#define TGB_SECTION_BREAK(...) goto SEC_##__VA_ARGS__;

#define TGB_LOG(tag, fmt, ...)  fprintf(tag, fmt, ##__VA_ARGS__);
#define TGB_LOG_INFO(fmt, ...)  TGB_LOG(stdout, "Info: "fmt, ##__VA_ARGS__);
#define TGB_LOG_ERROR(fmt, ...) TGB_LOG(stderr, "Error: "fmt, ##__VA_ARGS__);

#endif // GAUSSIAN_BLUR_IMPL_CUSTOM_CONFIG_HPP_
