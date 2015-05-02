#ifndef TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_TYPE_TRAITS_HPP_
#define TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_TYPE_TRAITS_HPP_

#include "test_gaussian_blur/impl_custom/numeric/vec2.hpp"
#include "test_gaussian_blur/impl_custom/numeric/vec3.hpp"
#include "test_gaussian_blur/impl_custom/numeric/vec4.hpp"

namespace test_gaussian_blur {

  template <typename T>
  struct ChannelType {
    typedef T value_type;
  };
  template <typename T>
  struct ChannelType<Vec2<T>> {
    typedef T value_type;
  };
  template <typename T>
  struct ChannelType<Vec3<T>> {
    typedef T value_type;
  };
  template <typename T>
  struct ChannelType<Vec4<T>> {
    typedef T value_type;
  };

  template <typename T>
  struct ChannelCount {
    enum { count = 1 };
  };
  template <typename T>
  struct ChannelCount<Vec2<T>> {
    enum { count = Vec2<T>::count };
  };
  template <typename T>
  struct ChannelCount<Vec3<T>> {
    enum { count = Vec3<T>::count };
  };
  template <typename T>
  struct ChannelCount<Vec4<T>> {
    enum { count = Vec4<T>::count };
  };

  template <typename T>
  struct UpgradeType {
    typedef T     original_type;
    typedef float upgraded_type;
  };
  template <>
  struct UpgradeType<long long> {
    typedef long long original_type;
    typedef double    upgraded_type;
  };
  template <>
  struct UpgradeType<unsigned long long> {
    typedef unsigned long long original_type;
    typedef double             upgraded_type;
  };
  template <>
  struct UpgradeType<double> {
    typedef double original_type;
    typedef double upgraded_type;
  };

  template <typename T>
  struct UpgradeType<Vec2<T>> {
    typedef Vec2<T> original_type;
    typedef Vec2<typename UpgradeType<T>::upgraded_type> upgraded_type;
  };
  template <typename T>
  struct UpgradeType<Vec3<T>> {
    typedef Vec3<T> original_type;
    typedef Vec3<typename UpgradeType<T>::upgraded_type> upgraded_type;
  };
  template <typename T>
  struct UpgradeType<Vec4<T>> {
    typedef Vec4<T> original_type;
    typedef Vec4<typename UpgradeType<T>::upgraded_type> upgraded_type;
  };

  template <typename T, typename U>
  __host__ __device__ __forceinline__
  T RoundType(const U &s) {
    return T(s);
  }

  template <>
  __host__ __device__ __forceinline__
  char RoundType(const float &s) {
    return char(s+0.5f);
  }
  template <>
  __host__ __device__ __forceinline__
  unsigned char RoundType(const float &s) {
    return unsigned char(s+0.5f);
  }
  template <>
  __host__ __device__ __forceinline__
  short RoundType(const float &s) {
    return short(s+0.5f);
  }
  template <>
  __host__ __device__ __forceinline__
  unsigned short RoundType(const float &s) {
    return unsigned short(s+0.5f);
  }
  template <>
  __host__ __device__ __forceinline__
  int RoundType(const float &s) {
    return int(s+0.5f);
  }
  template <>
  __host__ __device__ __forceinline__
  unsigned int RoundType(const float &s) {
    return unsigned int(s+0.5f);
  }
  template <>
  __host__ __device__ __forceinline__
  long long RoundType(const float &s) {
    return long long(s+0.5f);
  }
  template <>
  __host__ __device__ __forceinline__
  unsigned long long RoundType(const float &s) {
    return unsigned long long(s+0.5f);
  }

  template <>
  __host__ __device__ __forceinline__
  char RoundType(const double &s) {
    return char(s+0.5);
  }
  template <>
  __host__ __device__ __forceinline__
  unsigned char RoundType(const double &s) {
    return unsigned char(s+0.5);
  }
  template <>
  __host__ __device__ __forceinline__
  short RoundType(const double &s) {
    return short(s+0.5);
  }
  template <>
  __host__ __device__ __forceinline__
  unsigned short RoundType(const double &s) {
    return unsigned short(s+0.5);
  }
  template <>
  __host__ __device__ __forceinline__
  int RoundType(const double &s) {
    return int(s+0.5);
  }
  template <>
  __host__ __device__ __forceinline__
  unsigned int RoundType(const double &s) {
    return unsigned int(s+0.5);
  }
  template <>
  __host__ __device__ __forceinline__
  long long RoundType(const double &s) {
    return long long(s+0.5);
  }
  template <>
  __host__ __device__ __forceinline__
  unsigned long long RoundType(const double &s) {
    return unsigned long long(s+0.5);
  }

  template <typename T, typename U>
  __host__ __device__ __forceinline__
  Vec2<T> RoundType(const Vec2<U> &s) {
    return Vec2<T>(
      RoundType<T>(s(0)), RoundType<T>(s(1)));
  }
  template <typename T, typename U>
  __host__ __device__ __forceinline__
  Vec3<T> RoundType(const Vec3<U> &s) {
    return Vec3<T>(
      RoundType<T>(s(0)), RoundType<T>(s(1)),
      RoundType<T>(s(2)));
  }
  template <typename T, typename U>
  __host__ __device__ __forceinline__
  Vec4<T> RoundType(const Vec4<U> &s) {
    return Vec4<T>(
      RoundType<T>(s(0)), RoundType<T>(s(1)),
      RoundType<T>(s(2)), RoundType<T>(s(3)));
  }

}

#endif // TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_UTILITY_TYPE_TRAITS_HPP_
