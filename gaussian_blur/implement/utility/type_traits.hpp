#ifndef GAUSSIAN_BLUR_UTILITY_TYPE_TRAITS_HPP_
#define GAUSSIAN_BLUR_UTILITY_TYPE_TRAITS_HPP_

#include "config/config.hpp"

namespace test_gaussian_blur {

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

  template <typename ST, typename DT>
  __host__ __device__ __forceinline__
  DT RoundType(const ST &s) {
    return DT(s);
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

}

#endif // GAUSSIAN_BLUR_UTILITY_TYPE_TRAITS_HPP_
