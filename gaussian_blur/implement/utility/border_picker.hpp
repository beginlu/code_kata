#ifndef GAUSSIAN_BLUR_UTILITY_BORDER_PICKER_HPP_
#define GAUSSIAN_BLUR_UTILITY_BORDER_PICKER_HPP_

#include "utility/pixel_picker.hpp"

namespace test_gaussian_blur {

  enum BorderType {
    kReplicate,
    kReflect,
    kReflect101,
    kWrap,
    kConstant,
    kNumBorders
  };

  template <typename T, BorderType BT>
  struct RowBorderPicker {};

  template <typename T>
  struct RowBorderPicker<T, BorderType::kReplicate> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(
      const int row_id, const PixelPicker<T> &pixel_picker)
      : width_(pixel_picker.width_),
        channels_(pixel_picker.channels_),
        data_ptr_(pixel_picker.get_row(row_id)) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      return (i < 0 ? 0 : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      return (i >= width_ ? width_-1 : i);
    }

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return data_ptr_[get_lower_index(i)*channels_+c];
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return data_ptr_[get_higher_index(i)*channels_+c];
    }

    const int        &width_;
    const int        &channels_;
    const value_type *data_ptr_;
  };
  template <typename T>
  struct RowBorderPicker<T, BorderType::kReflect> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(
      const int row_id, const PixelPicker<T> &pixel_picker)
      : width_(pixel_picker.width_),
        channels_(pixel_picker.channels_),
        data_ptr_(pixel_picker.get_row(row_id)) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      return (i < 0 ? ::abs(i)-1 : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      return (i >= width_ ? 2*width_-i-1 : i);
    }

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return data_ptr_[get_lower_index(i)*channels_+c];
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return data_ptr_[get_higher_index(i)*channels_+c];
    }

    const int        &width_;
    const int        &channels_;
    const value_type *data_ptr_;
  };
  template <typename T>
  struct RowBorderPicker<T, BorderType::kReflect101> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(
      const int row_id, const PixelPicker<T> &pixel_picker)
      : width_(pixel_picker.width_),
        channels_(pixel_picker.channels_),
        data_ptr_(pixel_picker.get_row(row_id)) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      return ::abs(i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      return (i >= width_ ? 2*width_-i-2 : i);
    }

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return data_ptr_[get_lower_index(i)*channels_+c];
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return data_ptr_[get_higher_index(i)*channels_+c];
    }

    const int        &width_;
    const int        &channels_;
    const value_type *data_ptr_;
  };
  template <typename T>
  struct RowBorderPicker<T, BorderType::kWrap> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(
      const int row_id, const PixelPicker<T> &pixel_picker)
      : width_(pixel_picker.width_),
        channels_(pixel_picker.channels_),
        data_ptr_(pixel_picker.get_row(row_id)) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      return (i < 0 ? i+width_ : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      return (i >= width_ ? i-width_ : i);
    }

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return data_ptr_[get_lower_index(i)*channels_+c];
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return data_ptr_[get_higher_index(i)*channels_+c];
    }

    const int        &width_;
    const int        &channels_;
    const value_type *data_ptr_;
  };
  template <typename T>
  struct RowBorderPicker<T, BorderType::kConstant> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(
      const int row_id, const PixelPicker<T> &pixel_picker)
      : width_(pixel_picker.width_),
        channels_(pixel_picker.channels_),
        data_ptr_(pixel_picker.get_row(row_id)),
        const_val_(0) {}

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return i < 0 ? const_val_ : data_ptr_[i*channels_+c];
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return i >= width_ ? const_val_ : data_ptr_[i*channels_+c];
    }

    const int        &width_;
    const int        &channels_;
    const value_type *data_ptr_;
    const value_type  const_val_;
  };

  template <typename T, BorderType BT>
  struct ColBorderPicker {};

  template <typename T>
  struct ColBorderPicker<T, BorderType::kReplicate> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(
        const int col_id, const PixelPicker<T> &pixel_picker)
      : height_(pixel_picker.height_),
        stride_(pixel_picker.stride_),
        data_ptr_(pixel_picker.get_pixel(0, col_id)) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      return (i < 0 ? 0 : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      return (i >= height_ ? height_-1 : i);
    }

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          get_lower_index(i)*stride_)+c);
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          get_higher_index(i)*stride_)+c);
    }

    const int        &height_;
    const int        &stride_;
    const value_type *data_ptr_;
  };
  template <typename T>
  struct ColBorderPicker<T, BorderType::kReflect> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(
        const int col_id, const PixelPicker<T> &pixel_picker)
      : height_(pixel_picker.height_),
        stride_(pixel_picker.stride_),
        data_ptr_(pixel_picker.get_pixel(0, col_id)) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      return (i < 0 ? ::abs(i)-1 : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      return (i >= height_ ? 2*height_-i-1 : i);
    }

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          get_lower_index(i)*stride_)+c);
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          get_higher_index(i)*stride_)+c);
    }

    const int        &height_;
    const int        &stride_;
    const value_type *data_ptr_;
  };
  template <typename T>
  struct ColBorderPicker<T, BorderType::kReflect101> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(
        const int col_id, const PixelPicker<T> &pixel_picker)
      : height_(pixel_picker.height_),
        stride_(pixel_picker.stride_),
        data_ptr_(pixel_picker.get_pixel(0, col_id)) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      return ::abs(i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      return (i >= height_ ? 2*height_-i-2 : i);
    }

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          get_lower_index(i)*stride_)+c);
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          get_higher_index(i)*stride_)+c);
    }

    const int        &height_;
    const int        &stride_;
    const value_type *data_ptr_;
  };
  template <typename T>
  struct ColBorderPicker<T, BorderType::kWrap> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(
        const int col_id, const PixelPicker<T> &pixel_picker)
      : height_(pixel_picker.height_),
        stride_(pixel_picker.stride_),
        data_ptr_(pixel_picker.get_pixel(0, col_id)) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      return (i < 0 ? i+height_ : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      return (i >= height_ ? i-height_ : i);
    }

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          get_lower_index(i)*stride_)+c);
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          get_higher_index(i)*stride_)+c);
    }

    const int        &height_;
    const int        &stride_;
    const value_type *data_ptr_;
  };
  template <typename T>
  struct ColBorderPicker<T, BorderType::kConstant> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(
        const int col_id, const PixelPicker<T> &pixel_picker)
      : height_(pixel_picker.height_),
        stride_(pixel_picker.stride_),
        data_ptr_(pixel_picker.get_pixel(0, col_id)),
        const_val_(0) {}

    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const int c) const {
      return
          i < 0 ? const_val_ : *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          i*stride_)+c);
    }
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const int c) const {
      return
          i >= height_ ? const_val_ : *(
          reinterpret_cast<const value_type*>(
          reinterpret_cast<const unsigned char*>(data_ptr_)+
          i*stride_)+c);
    }

    const int        &height_;
    const int        &stride_;
    const value_type *data_ptr_;
    const value_type  const_val_;
  };
}

#endif // GAUSSIAN_BLUR_UTILITY_BORDER_PICKER_HPP_
