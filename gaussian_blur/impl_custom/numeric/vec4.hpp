#ifndef TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_NUMERIC_VEC4_HPP_
#define TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_NUMERIC_VEC4_HPP_

#include "test_gaussian_blur/impl_custom/config/config.hpp"

namespace test_gaussian_blur {

  template <typename T>
  struct Vec4 {
    typedef T value_type;

    enum { count = 4, size = sizeof(value_type)*count };

    __host__ __device__
    Vec4(void) {
      vals[0] = vals[1] = vals[2] = vals[3] = value_type(0);
    }
    template <typename U>
    __host__ __device__
    Vec4(const Vec4<U> &v) {
      vals[0] = value_type(v(0)); vals[1] = value_type(v(1));
      vals[2] = value_type(v(2)); vals[3] = value_type(v(3));
    }
    template <typename U>
    __host__ __device__
    Vec4(const U &s) {
      vals[0] = vals[1] = vals[2] = vals[3] = value_type(s);
    }
    template <typename U>
    __host__ __device__
    Vec4(const U &s1, const U &s2, const U &s3, const U &s4) {
      vals[0] = value_type(s1); vals[1] = value_type(s2);
      vals[2] = value_type(s3); vals[3] = value_type(s4);
    }

    __host__ __device__ __forceinline__
    const value_type & operator () (const int i) const {
      return vals[i];
    }
    __host__ __device__ __forceinline__
    value_type & operator () (const int i) {
      return vals[i];
    }

    __host__ __device__ __forceinline__
    bool operator == (const Vec4 &v) const {
      return (vals[0] == v(0) && vals[1] == v(1) &&
              vals[2] == v(2) && vals[3] == v(3));
    }
    __host__ __device__ __forceinline__
    bool operator == (const value_type &s) const {
      return (vals[0] == s && vals[1] == s &&
              vals[2] == s && vals[3] == s);
    }
    __host__ __device__ __forceinline__
    bool operator != (const Vec4 &v) const {
      return !(*this == v);
    }
    __host__ __device__ __forceinline__
    bool operator != (const value_type &s) const {
      return !(*this != s);
    }

    __host__ __device__ __forceinline__
    Vec4 operator + (const Vec4 &v) const {
      return Vec4(vals[0]+v(0), vals[1]+v(1), vals[2]+v(2), vals[3]+v(3));
    }
    __host__ __device__ __forceinline__
    Vec4 operator - (const Vec4 &v) const {
      return Vec4(vals[0]-v(0), vals[1]-v(1), vals[2]-v(2), vals(3)-v(3));
    }
    __host__ __device__ __forceinline__
    Vec4 operator + (const value_type &s) const {
      return Vec4(vals[0]+s, vals[1]+s, vals[2]+s, vals[3]+s);
    }
    __host__ __device__ __forceinline__
    Vec4 operator - (const value_type &s) const {
      return Vec4(vals[0]-s, vals[1]-s, vals[2]-s, vals[3]-s);
    }
    __host__ __device__ __forceinline__
    Vec4 operator * (const value_type &s) const {
      return Vec4(vals[0]*s, vals[1]*s, vals[2]*s, vals[3]*s);
    }
    __host__ __device__ __forceinline__
    Vec4 operator / (const value_type &s) const {
      return Vec4(vals[0]/s, vals[1]/s, vals[2]/s, vals[3]/s);
    }
    
    __host__ __device__ __forceinline__
    Vec4 & operator = (const Vec4 &v) {
      vals[0] = v(0); vals[1] = v(1); vals[2] = v(2); vals[3] = v(3);
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec4 & operator += (const Vec4 &v) {
      vals[0] += v(0); vals[1] += v(1); vals[2] += v(2); vals[3] += v(3);
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec4 & operator -= (const Vec4 &v) {
      vals[0] -= v(0); vals[1] -= v(1); vals[2] -= v(2); vals[3] -= v(3);
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec4 & operator = (const value_type &s) {
      vals[0] = vals[1] = vals[2] = vals[3] = s;
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec4 & operator += (const value_type &s) {
      vals[0] += s; vals[1] += s; vals[2] += s; vals[3] += s;
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec4 & operator -= (const value_type &s) {
      vals[0] -= s; vals[1] -= s; vals[2] -= s; vals[3] -= s;
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec4 & operator *= (const value_type &s) {
      vals[0] *= s; vals[1] *= s; vals[2] *= s; vals[3] *= s;
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec4 & operator /= (const value_type &s) {
      vals[0] /= s; vals[1] /= s; vals[2] /= s; vals[3] /= s;
      return *this;
    }

    value_type vals[4];
  };

}

#endif // TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_NUMERIC_VEC4_HPP_
