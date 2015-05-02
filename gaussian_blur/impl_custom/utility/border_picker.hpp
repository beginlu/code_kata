#ifndef TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_BORDER_PICKER_HPP_
#define TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_BORDER_PICKER_HPP_

#include "test_gaussian_blur/impl_custom/config/config.hpp"

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
    RowBorderPicker(const int length)
      : length_(length) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      //return ::abs(i < 0 ? 0 : i)%length_;
      return (i < 0 ? 0 : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      //return ::abs(i >= length_ ? length_-1 : i)%length_;
      return (i >= length_ ? length_-1 : i);
    }

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      return value_type(data_ptr[get_lower_index(i)]);
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      return value_type(data_ptr[get_higher_index(i)]);
    }

    int length_;
  };
  template <typename T>
  struct RowBorderPicker<T, BorderType::kReflect> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(const int length)
      : length_(length) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      //return ::abs(i < 0 ? ::abs(i)-1 : i)%length_;
      return (i < 0 ? ::abs(i)-1 : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      //return ::abs(i >= length_ ? 2*length_-i-1 : i)%length_;
      return (i >= length_ ? 2*length_-i-1 : i);
    }

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      return value_type(data_ptr[get_lower_index(i)]);
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      return value_type(data_ptr[get_higher_index(i)]);
    }

    int length_;
  };
  template <typename T>
  struct RowBorderPicker<T, BorderType::kReflect101> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(const int length)
      : length_(length) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      //return ::abs(i)%length_;
      return ::abs(i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      //return ::abs(i >= length_ ? 2*length_-i-2 : i)%length_;
      return (i >= length_ ? 2*length_-i-2 : i);
    }

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      return value_type(data_ptr[get_lower_index(i)]);
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      return value_type(data_ptr[get_higher_index(i)]);
    }

    int length_;
  };
  template <typename T>
  struct RowBorderPicker<T, BorderType::kWrap> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(const int length)
      : length_(length) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      //return ::abs(i < 0 ? i+length_ : i)%length_;
      return (i < 0 ? i+length_ : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      //return ::abs(i >= length_ ? i-length_ : i)%length_;
      return (i >= length_ ? i-length_ : i);
    }

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      return value_type(data_ptr[get_lower_index(i)]);
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      return value_type(data_ptr[get_higher_index(i)]);
    }

    int length_;
  };
  template <typename T>
  struct RowBorderPicker<T, BorderType::kConstant> {
    typedef T value_type;

    __host__ __device__
    RowBorderPicker(const int length)
      : length_(length), const_val_(0) {}

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      //return (i < 0 ? const_val_ : value_type(data_ptr[::abs(i)%length_]));
      return (i < 0 ? const_val_ : value_type(data_ptr[i]));
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      //return (i >= length_ ? const_val_ : value_type(data_ptr[::abs(i)%length_]));
      return (i >= length_ ? const_val_ : value_type(data_ptr[i]));
    }

    int        length_;
    value_type const_val_;
  };

  template <typename T, BorderType BT>
  struct ColBorderPicker {};

  template <typename T>
  struct ColBorderPicker<T, BorderType::kReplicate> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(const int length, const int stride)
      : length_(length), stride_(stride) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      //return ::abs(i < 0 ? 0 : i)%length_;
      return (i < 0 ? 0 : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      //return ::abs(i >= length_ ? length_-1 : i)%length_;
      return (i >= length_ ? length_-1 : i);
    }

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      return value_type(
        *((U*)((unsigned char*)data_ptr+get_lower_index(i)*stride_)));
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      return value_type(
        *((U*)((unsigned char*)data_ptr+get_higher_index(i)*stride_)));
    }

    int length_;
    int stride_;
  };
  template <typename T>
  struct ColBorderPicker<T, BorderType::kReflect> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(const int length, const int stride)
      : length_(length), stride_(stride) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      //return ::abs(i < 0 ? ::abs(i)-1 : i)%length_;
      return (i < 0 ? ::abs(i)-1 : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      //return ::abs(i >= length_ ? 2*length_-i-1 : i)%length_;
      return (i >= length_ ? 2*length_-i-1 : i);
    }

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      return value_type(
        *((U*)((unsigned char*)data_ptr+get_lower_index(i)*stride_)));
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      return value_type(
        *((U*)((unsigned char*)data_ptr+get_higher_index(i)*stride_)));
    }

    int length_;
    int stride_;
  };
  template <typename T>
  struct ColBorderPicker<T, BorderType::kReflect101> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(const int length, const int stride)
      : length_(length), stride_(stride) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      //return ::abs(i)%length_;
      return ::abs(i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      //return ::abs(i >= length_ ? 2*length_-i-2 : i)%length_;
      return (i >= length_ ? 2*length_-i-2 : i);
    }

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      return value_type(
        *((U*)((unsigned char*)data_ptr+get_lower_index(i)*stride_)));
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      return value_type(
        *((U*)((unsigned char*)data_ptr+get_higher_index(i)*stride_)));
    }

    int length_;
    int stride_;
  };
  template <typename T>
  struct ColBorderPicker<T, BorderType::kWrap> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(const int length, const int stride)
      : length_(length), stride_(stride) {}

    __host__ __device__ __forceinline__
    int get_lower_index(const int i) const {
      //return ::abs(i < 0 ? i+length_ : i)%length_;
      return (i < 0 ? i+length_ : i);
    }
    __host__ __device__ __forceinline__
    int get_higher_index(const int i) const {
      //return ::abs(i >= length_ ? i-length_ : i)%length_;
      return (i >= length_ ? i-length_ : i);
    }

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      return value_type(
        *((U*)((unsigned char*)data_ptr+get_lower_index(i)*stride_)));
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      return value_type(
        *((U*)((unsigned char*)data_ptr+get_higher_index(i)*stride_)));
    }

    int length_;
    int stride_;
  };
  template <typename T>
  struct ColBorderPicker<T, BorderType::kConstant> {
    typedef T value_type;

    __host__ __device__
    ColBorderPicker(const int length, const int stride)
      : length_(length), stride_(stride), const_val_(0) {}

    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_lower_value(const int i, const U *data_ptr) const {
      //return (
      //  i < 0 ? const_val_ : value_type(
      //  *((U*)((unsigned char*)data_ptr+(::abs(i)%length_)*stride_))));
      return (
        i < 0 ? const_val_ : value_type(
        *((U*)((unsigned char*)data_ptr+i*stride_))));
    }
    template <typename U>
    __host__ __device__ __forceinline__
    value_type get_higher_value(const int i, const U *data_ptr) const {
      //return (
      //  i >= length_ ? const_val_ : value_type(
      //  *((U*)((unsigned char*)data_ptr+(::abs(i)%length_)*stride_))));
      return (
        i >= length_ ? const_val_ : value_type(
        *((U*)((unsigned char*)data_ptr+i*stride_))));
    }

    int        length_;
    int        stride_;
    value_type const_val_;
  };

}

#endif // TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_BORDER_PICKER_HPP_
