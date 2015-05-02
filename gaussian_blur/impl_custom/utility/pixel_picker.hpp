#ifndef TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_PIXEL_PICKER_HPP_
#define TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_PIXEL_PICKER_HPP_

#include "test_gaussian_blur/impl_custom/config/config.hpp"

namespace test_gaussian_blur {

  template <typename T>
  struct PixelPicker {
    typedef T value_type;

    __host__ __device__
    PixelPicker(
        const int width, const int height,
        const int stride, value_type *data_ptr)
        : width_(width), height_(height),
          stride_(stride), data_ptr_(data_ptr) {}

    __host__ __device__ __forceinline__
    int size(void) const {
      return height_ * stride_;
    }

    __host__ __device__ __forceinline__
    const value_type * get_row(const int y) const {
      return (value_type*)((unsigned char*)data_ptr_+y*stride_);
    }
    __host__ __device__ __forceinline__
    value_type * get_row(const int y) {
      return (value_type*)((unsigned char*)data_ptr_+y*stride_);
    }

    __host__ __device__ __forceinline__
    const value_type & get_pixel(const int y, const int x) const {
      return *(get_row(y) + x);
    }
    __host__ __device__ __forceinline__
    value_type & get_pixel(const int y, const int x) {
      return *(get_row(y) + x);
    }

    int width_;
    int height_;
    int stride_;
    value_type *data_ptr_;
  };

}

#endif // TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_PIXEL_PICKER_HPP_
