#ifndef GAUSSIAN_BLUR_UTILITY_PIXEL_PICKER_HPP_
#define GAUSSIAN_BLUR_UTILITY_PIXEL_PICKER_HPP_

#include "config/config.hpp"

namespace test_gaussian_blur {

  template <typename T>
  struct PixelPicker {
    typedef T value_type;

    __host__ __device__
    PixelPicker(void)
      : height_(0), width_(0),
        channels_(0), stride_(0),
        data_ptr_(nullptr) {}

    __host__ __device__
    PixelPicker(
        const int height, const int width,
        const int channels, const int stride,
        value_type *data_ptr) {
      set_fields(height, width, channels, stride, data_ptr);
      return;
    }

    __host__ __device__ __forceinline__
    int get_size(void) const {
      return height_ * width_ * channels_ * sizeof(value_type);
    }
    
    __host__ __device__ __forceinline__
    int get_capacity(void) const {
      return height_ * stride_;
    }

    __host__ __device__ __forceinline__
    const value_type * get_row(const int y) const {
      return 
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(
          data_ptr_)+y*stride_);
    }
    __host__ __device__ __forceinline__
    value_type * get_row(const int y) {
      return 
          reinterpret_cast<value_type*>(
          reinterpret_cast<unsigned char*>(
          data_ptr_)+y*stride_);
    }

    __host__ __device__ __forceinline__
    const value_type * get_pixel(const int y, const int x) const {
      return get_row(y)+(channels_*x);
    }
    __host__ __device__ __forceinline__
    value_type * get_pixel(const int y, const int x) {
      return get_row(y)+(channels_*x);
    }

    __host__ __device__ __forceinline__
    const value_type & get_value(
        const int y, const int x, const int c) const {
      return *(get_pixel(y, x)+c);
    }
    __host__ __device__ __forceinline__
    value_type & get_value(
        const int y, const int x, const int c) {
      return *(get_pixel(y, x)+c);
    }

    __host__ __device__ __forceinline__
    void clear(void) {
      height_   = 0;
      width_    = 0;
      channels_ = 0;
      stride_   = 0;
      data_ptr_ = nullptr;
      return;
    }

    __host__ __device__ __forceinline__
    void set_fields(
        const int height, const int width,
        const int channels, const int stride,
        value_type *data_ptr) {
      height_   = height;
      width_    = width;
      channels_ = channels;
      stride_   = stride;
      data_ptr_ = data_ptr;
      return;
    }

    __host__ __device__ __forceinline__
    bool check_validity(void) const {
      return (
          height_ > 0 && width_ > 0 && channels_ > 0 &&
          stride_ >= static_cast<int>(width_*channels_*sizeof(value_type)) &&
          data_ptr_ != nullptr);
    }
    
    int width_;
    int height_;
    int channels_;
    int stride_;
    value_type *data_ptr_;
  };

}

#endif // GAUSSIAN_BLUR_UTILITY_PIXEL_PICKER_HPP_
