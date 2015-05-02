#ifndef TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_NUMERIC_VEC3_HPP_
#define TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_NUMERIC_VEC3_HPP_

#include "test_gaussian_blur/impl_custom/config/config.hpp"

namespace test_gaussian_blur {

  template <typename T>
  struct Vec3 {
    typedef T value_type;

    enum { count = 3, size = sizeof(value_type)*count };

    __host__ __device__
    Vec3(void) {
      vals[0] = vals[1] = vals[2] = value_type(0);
    }
    template <typename U>
    __host__ __device__
    Vec3(const Vec3<U> &v) {
      vals[0] = value_type(v(0)); vals[1] = value_type(v(1));
      vals[2] = value_type(v(2));
    }
    template <typename U>
    __host__ __device__
    Vec3(const U &s) {
      vals[0] = vals[1] = vals[2] = value_type(s);
    }
    template <typename U>
    __host__ __device__
    Vec3(const U &s1, const U &s2, const U &s3) {
      vals[0] = value_type(s1); vals[1] = value_type(s2);
      vals[2] = value_type(s3);
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
    bool operator == (const Vec3 &v) const {
      return (vals[0] == v(0) && vals[1] == v(1) &&
              vals[2] == v(2));
    }
    __host__ __device__ __forceinline__
    bool operator == (const value_type &s) const {
      return (vals[0] == s && vals[1] == s &&
              vals[2] == s);
    }
    __host__ __device__ __forceinline__
    bool operator != (const Vec3 &v) const {
      return !(*this == v);
    }
    __host__ __device__ __forceinline__
    bool operator != (const value_type &s) const {
      return !(*this != s);
    }

    __host__ __device__ __forceinline__
    Vec3 operator + (const Vec3 &v) const {
      return Vec3(vals[0]+v(0), vals[1]+v(1), vals[2]+v(2));
    }
    __host__ __device__ __forceinline__
    Vec3 operator - (const Vec3 &v) const {
      return Vec3(vals[0]-v(0), vals[1]-v(1), vals[2]-v(2));
    }
    __host__ __device__ __forceinline__
    Vec3 operator + (const value_type &s) const {
      return Vec3(vals[0]+s, vals[1]+s, vals[2]+s);
    }
    __host__ __device__ __forceinline__
    Vec3 operator - (const value_type &s) const {
      return Vec3(vals[0]-s, vals[1]-s, vals[2]-s);
    }
    __host__ __device__ __forceinline__
    Vec3 operator * (const value_type &s) const {
      return Vec3(vals[0]*s, vals[1]*s, vals[2]*s);
    }
    __host__ __device__ __forceinline__
    Vec3 operator / (const value_type &s) const {
      return Vec3(vals[0]/s, vals[1]/s, vals[2]/s);
    }
    
    __host__ __device__ __forceinline__
    Vec3 & operator = (const Vec3 &v) {
      vals[0] = v(0); vals[1] = v(1); vals[2] = v(2);
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec3 & operator += (const Vec3 &v) {
      vals[0] += v(0); vals[1] += v(1); vals[2] += v(2);
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec3 & operator -= (const Vec3 &v) {
      vals[0] -= v(0); vals[1] -= v(1); vals[2] -= v(2);
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec3 & operator = (const value_type &s) {
      vals[0] = vals[1] = vals[2] = s;
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec3 & operator += (const value_type &s) {
      vals[0] += s; vals[1] += s; vals[2] += s;
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec3 & operator -= (const value_type &s) {
      vals[0] -= s; vals[1] -= s; vals[2] -= s;
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec3 & operator *= (const value_type &s) {
      vals[0] *= s; vals[1] *= s; vals[2] *= s;
      return *this;
    }
    __host__ __device__ __forceinline__
    Vec3 & operator /= (const value_type &s) {
      vals[0] /= s; vals[1] /= s; vals[2] /= s;
      return *this;
    }

    value_type vals[3];
  };

}

#endif // TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_NUMERIC_VEC3_HPP_
