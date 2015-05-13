#ifndef GAUSSIAN_BLUR_TEST_TEST_HPP_
#define GAUSSIAN_BLUR_TEST_TEST_HPP_

#include "utility/border_picker.hpp"

namespace test_gaussian_blur {

  struct ConfigSet {
    ConfigSet(void)
      : width_(0), height_(0), channels_(0),
        kernel_x_size_(0), kernel_y_size_(0),
        border_type_(BorderType::kReplicate) {}
    ConfigSet(
        const int width, const int height, const int channels,
        const int kernel_x_size, const int kernel_y_size,
        const BorderType border_type)
      : width_(width), height_(height), channels_(channels),
        kernel_x_size_(kernel_x_size), kernel_y_size_(kernel_y_size),
        border_type_(border_type) {}

    int width_;
    int height_;
    int channels_;
    int kernel_x_size_;
    int kernel_y_size_;
    BorderType border_type_;
  };

  int TestFunc1(void);
  int TestFunc2(void);
  int TestFunc3(void);

}

#endif // GAUSSIAN_BLUR_TEST_TEST_HPP_
