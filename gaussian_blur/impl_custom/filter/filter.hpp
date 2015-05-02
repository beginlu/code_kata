#ifndef TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_FILTER_HPP_
#define TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_FILTER_HPP_

//#define OPENCV_VERFICATION

#include <algorithm>
#include <cassert>
#include <cmath>
#include <new>
#include "test_gaussian_blur/impl_custom/utility/border_picker.hpp"
#include "test_gaussian_blur/impl_custom/utility/pixel_picker.hpp"
#include "test_gaussian_blur/impl_custom/utility/type_traits.hpp"
#include "test_gaussian_blur/impl_custom/utility/log.hpp"

#if defined(OPENCV_VERFICATION)
# include "opencv2/imgproc/imgproc.hpp"
#endif // OPENCV_VERFICATION

namespace test_gaussian_blur {

  // constant memory use to cache kernel data
  __constant__ unsigned char c_const[TGB_MAX_CUDA_CONSTANT_SIZE];

  template <typename ST, typename DT, typename KT, BorderType BT>
  __global__
  void GaussianBlurRowFilterKernel(
      PixelPicker<ST> src_picker,
      PixelPicker<DT> dst_picker,
      int             kernel_size,
      int             pad_size,
      int             num_patches) {
    // shared memory use to cache block image data
    extern __shared__ unsigned char s_shared[];

    const int width                = src_picker.width_;
    const int half_kernel_size     = kernel_size/2;
    const int patched_block_size   = num_patches*blockDim.x;
    const int patched_block_offset = patched_block_size*blockIdx.x;

    PixelPicker<DT> cached_picker(
      2*pad_size+patched_block_size, blockDim.y,
      sizeof(DT)*(2*pad_size+patched_block_size),
      (DT*)s_shared);
    RowBorderPicker<DT, BT> border_picker(width);

    const int y_offset = blockDim.y*blockIdx.y+threadIdx.y;
    if (y_offset >= src_picker.height_) {
      return;
    }

    ST *src_data_ptr    = src_picker.get_row(y_offset);
    DT *cached_data_ptr = cached_picker.get_row(threadIdx.y);

    if (patched_block_offset-half_kernel_size < 0) {
      // load left-most blocks' left padding area
      for (int i = threadIdx.x; i < half_kernel_size; i += blockDim.x) {
        cached_data_ptr[pad_size-half_kernel_size+i] =
          border_picker.get_lower_value(
          patched_block_offset+i-half_kernel_size, src_data_ptr);
      }
    } else {
      // load other blocks' left padding area
      for (int i = threadIdx.x; i < half_kernel_size; i += blockDim.x) {
        cached_data_ptr[pad_size-half_kernel_size+i] =
          src_data_ptr[patched_block_offset+i-half_kernel_size];
      }
    }

    if (patched_block_offset+patched_block_size+half_kernel_size > width) {
      const int remain_size = ::min(
        patched_block_size, width-patched_block_offset)+half_kernel_size;
      // load right-most blocks' middle and right area
      for (int i = threadIdx.x; i < remain_size; i += blockDim.x) {
        cached_data_ptr[pad_size+i] =
          border_picker.get_higher_value(patched_block_offset+i, src_data_ptr);
      }
    } else {
      const int remain_size = patched_block_size+half_kernel_size;
      // loat other blocks' middle and right area
      for (int i = threadIdx.x; i < remain_size; i += blockDim.x) {
        cached_data_ptr[pad_size+i] = src_data_ptr[patched_block_offset+i];
      }
    }

    __syncthreads();

    // making convolution
    DT *dst_data_ptr    = dst_picker.get_row(y_offset);
    KT *kernel_data_ptr = (KT*)c_const;
    for (int i = threadIdx.x; i < patched_block_size; i += blockDim.x) {
      if (patched_block_offset+i < width) {
        DT  sum = 0;
        DT *temp_data_ptr = cached_data_ptr+pad_size+i-half_kernel_size;
        for (int j = 0; j < kernel_size; ++j) {
          sum += temp_data_ptr[j]*kernel_data_ptr[j];
        }
        dst_data_ptr[patched_block_offset+i] = sum;
      }
    }

    return;
  }

  template <typename ST, typename DT, typename KT, BorderType BT>
  __global__
  void GaussianBlurColFilterKernel(
      PixelPicker<ST> src_picker,
      PixelPicker<DT> dst_picker,
      int             kernel_size,
      int             pad_size,
      int             num_patches) {
    // shared memory use to cache block image data
    extern __shared__ unsigned char s_shared[];

    const int height               = src_picker.height_;
    const int half_kernel_size     = kernel_size/2;
    const int patched_block_size   = num_patches*blockDim.y;
    const int patched_block_offset = patched_block_size*blockIdx.y;

    PixelPicker<ST> cached_picker(
      blockDim.x, 2*pad_size+patched_block_size,
      sizeof(ST)*blockDim.x, (ST*)s_shared);
    ColBorderPicker<ST, BT> border_picker(
      height, src_picker.stride_);
    
    const int x_offset = blockDim.x*blockIdx.x+threadIdx.x;
    if (x_offset >= src_picker.width_) {
      return;
    }

    ST *src_data_ptr = src_picker.get_row(0) + x_offset;

    if (patched_block_offset-half_kernel_size < 0) {
      // load top-most blocks' top padding area
      for (int i = threadIdx.y; i < half_kernel_size; i += blockDim.y) {
        cached_picker.get_pixel(pad_size-half_kernel_size+i, threadIdx.x) =
          border_picker.get_lower_value(
          patched_block_offset+i-half_kernel_size, src_data_ptr);
      }
    } else {
      // load other blocks' top padding area
      for (int i = threadIdx.y; i < half_kernel_size; i += blockDim.y) {
        cached_picker.get_pixel(pad_size-half_kernel_size+i, threadIdx.x) =
          src_picker.get_pixel(patched_block_offset+i-half_kernel_size, x_offset);
      }
    }

    if (patched_block_offset+patched_block_size+half_kernel_size > height) {
      const int remain_size = ::min(
        patched_block_size, height-patched_block_offset)+half_kernel_size;
      // load bottom-most blocks' middle and right area
      for (int i = threadIdx.y; i < remain_size; i += blockDim.y) {
        cached_picker.get_pixel(pad_size+i, threadIdx.x) =
          border_picker.get_higher_value(patched_block_offset+i, src_data_ptr);
      }
    } else {
      const int remain_size = patched_block_size+half_kernel_size;
      // load other blocks' middle and bottom area
      for (int i = threadIdx.y; i < remain_size; i += blockDim.y) {
        cached_picker.get_pixel(pad_size+i, threadIdx.x) =
          src_picker.get_pixel(patched_block_offset+i, x_offset);
      }
    }

    __syncthreads();

    // making convolution
    KT *kernel_data_ptr = (KT*)c_const;
    for (int i = threadIdx.y; i < patched_block_size; i += blockDim.y) {
      if (patched_block_offset+i < height) {
        ST  sum      = 0;
        int y_offset = pad_size+i-half_kernel_size;
        for (int j = 0; j < kernel_size; ++j) {
          sum += cached_picker.get_pixel(
            y_offset+j, threadIdx.x)*kernel_data_ptr[j];
        }
        dst_picker.get_pixel(
          patched_block_offset+i, x_offset) = RoundType<DT, ST>(sum);
      }
    }

    return;
  }

  void NOOP(void) {}
  
  template <typename KT>
  int buildKernel(KT *kernel_ptr, int kernel_size, double kernel_sigma) {
    // use double type to gain most precision
    double *tmp_kernel_ptr = new (std::nothrow) double[kernel_size];
    if (!tmp_kernel_ptr) {
      return -1;
    }

    const int kernel_anchor = kernel_size/2;
    const double coeff = -0.5/(kernel_sigma*kernel_sigma);
    double sum = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
      sum += (tmp_kernel_ptr[i] = std::exp(
        (i-kernel_anchor)*(i-kernel_anchor)*coeff));
    }
    for (int i = 0; i < kernel_size; ++i) {
      kernel_ptr[i] = KT(tmp_kernel_ptr[i] / sum);
    }

    delete []tmp_kernel_ptr;
    tmp_kernel_ptr = nullptr;

    return 0;
  }

  template <typename T>
  int GetAdaptiveRowBlockSize(
      int  max_shared_mem_size_in_bytes,
      int  max_image_width,
      int  kernel_size,
      int &block_x_size,
      int &block_y_size,
      int &block_pad_size,
      int &block_num_patches) {
    // to get adaptive row block size
    // 1. find the mininum padding size which contain
    //    the half kernel size and align to 64 bytes
    // 2. fix the unit block size as (32x4)
    // 3. find the block patch number base on image width
    // 4. the total memory usage should not exceed the maximum shared memory size
    const int half_kernel_size = kernel_size/2;
    const int def_block_pad_size_in_bytes = 64;
    const int elem_size_in_bytes = sizeof(T);

    // find the 'greatest common divisor'
    int a = std::max(def_block_pad_size_in_bytes, elem_size_in_bytes);
    int b = std::min(def_block_pad_size_in_bytes, elem_size_in_bytes);
    for (int c = a % b; c != 0; a = b, b = c, c = a % b);
    // set pad size as 'least common multiple'
    const int block_unit_pad_size =
      def_block_pad_size_in_bytes*elem_size_in_bytes/b/elem_size_in_bytes;
    block_pad_size =
      (half_kernel_size+block_unit_pad_size-1)/
      block_unit_pad_size*block_unit_pad_size;

    block_x_size = 32;
    block_y_size = 4;
    block_num_patches =
      std::min(4, (max_image_width+block_x_size-1)/block_x_size);
    //block_num_patches =
    //  std::max(4, (block_x_size+half_kernel_size-1)/block_x_size);
    //block_num_patches =
    //  std::min(block_num_patches, (max_image_width+block_x_size-1)/block_x_size);
    const int total_size =
      elem_size_in_bytes*((2*block_pad_size+block_num_patches*block_x_size)*block_y_size);
    if (total_size > max_shared_mem_size_in_bytes) {
      return -1;
    }
    
    return 0;
  }

  template <typename T>
  int GetAdaptiveColBlockSize(
      int  max_shared_mem_size_in_bytes,
      int  max_image_height,
      int  kernel_size,
      int &block_x_size,
      int &block_y_size,
      int &block_pad_size,
      int &block_num_patches) {
    // to get adaptive column block size
    // 1. make pading size as half kernel size
    // 2. fix the unit block size as (16x8)
    // 3. find the block patch number base on image height
    // 4. the total memory usage should not exceed the maximum shared memory size
    const int half_kernel_size = kernel_size/2;
    const int elem_size_in_bytes = sizeof(T);

    block_pad_size = half_kernel_size;
    block_x_size   = 16;
    block_y_size   = 8;
    block_num_patches =
      std::min(4, (max_image_height+block_y_size-1)/block_y_size);
    //block_num_patches =
    //  std::max(4, (block_y_size+block_pad_size-1)/block_y_size);
    //block_num_patches =
    //  std::min(block_num_patches, (max_image_height+block_y_size-1)/block_y_size);
    const int total_size =
      elem_size_in_bytes*block_x_size*(2*block_pad_size+block_num_patches*block_y_size);
    if (total_size > max_shared_mem_size_in_bytes) {
      return -1;
    }

    return 0;
  }

  template <typename ST, typename DT, typename KT>
  int GaussianBlurRowFilter(
      PixelPicker<ST> &src_picker,
      PixelPicker<DT> &dst_picker,
      int              kernel_size,
      BorderType       border_type,
      int              max_shared_mem_size_in_bytes) {
    int ret = -1;

    TGB_SECTION_BEGIN();
    {
      // get adaptive block size
      int block_x_size = 0;
      int block_y_size = 0;
      int block_pad_size = 0;
      int block_num_patches = 0;
      if (GetAdaptiveRowBlockSize<DT>(
          max_shared_mem_size_in_bytes, src_picker.width_,
          kernel_size, block_x_size, block_y_size,
          block_pad_size, block_num_patches)) {
        TGB_LOG_ERROR("Failed to get adaptive row block size.\n");
        TGB_SECTION_BREAK();
      }
      
      const int patched_block_size =
        block_num_patches*block_x_size;
      const int shared_mem_size_in_bytes =
        sizeof(DT)*(2*block_pad_size+patched_block_size)*block_y_size;
      const dim3 blockDim(
        block_x_size, block_y_size);
      const dim3 gridDim(
        (src_picker.width_+patched_block_size-1)/patched_block_size,
        (src_picker.height_+block_y_size-1)/block_y_size);

      // make kernel call
      switch (border_type) {
        case BorderType::kReplicate: {
          GaussianBlurRowFilterKernel<ST, DT, KT, BorderType::kReplicate>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        case BorderType::kReflect: {
          GaussianBlurRowFilterKernel<ST, DT, KT, BorderType::kReflect>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        case BorderType::kReflect101: {
          GaussianBlurRowFilterKernel<ST, DT, KT, BorderType::kReflect101>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        case BorderType::kWrap: {
          GaussianBlurRowFilterKernel<ST, DT, KT, BorderType::kWrap>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        case BorderType::kConstant: {
          GaussianBlurRowFilterKernel<ST, DT, KT, BorderType::kConstant>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        default: {
          assert(0);
          break;
        }
      }

#   if defined(OPENCV_VERFICATION)
      // verify data correctness by compared with opencv result
      typedef cv::Vec<typename ChannelType<ST>::value_type, ChannelCount<ST>::count> ocv_src_t;
      typedef cv::Vec<typename ChannelType<DT>::value_type, ChannelCount<DT>::count> ocv_dst_t;
      typedef cv::Vec<typename ChannelType<KT>::value_type, ChannelCount<KT>::count> ocv_kernel_t;
      cv::Mat_<ocv_src_t>    ocv_src(src_picker.height_, src_picker.width_);
      cv::Mat_<ocv_dst_t>    ocv_dst1(dst_picker.height_, dst_picker.width_);
      cv::Mat_<ocv_dst_t>    ocv_dst2(dst_picker.height_, dst_picker.width_);
      cv::Mat_<ocv_kernel_t> ocv_kernel(1, kernel_size);

      cudaError_t cuda_error = cudaSuccess;
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          ocv_src.data, ocv_src.step,
          src_picker.data_ptr_, src_picker.stride_,
          sizeof(ST)*src_picker.width_, src_picker.height_,
          cudaMemcpyDeviceToHost)) ||
          cudaSuccess != (cuda_error = cudaMemcpy2D(
          ocv_dst1.data, ocv_dst1.step,
          dst_picker.data_ptr_, dst_picker.stride_,
          sizeof(DT)*dst_picker.width_, dst_picker.height_,
          cudaMemcpyDeviceToHost)) ||
          cudaSuccess != (cuda_error = cudaMemcpyFromSymbol(
          ocv_kernel.data, test_gaussian_blur::c_const,
          sizeof(KT)*kernel_size))) {
        TGB_LOG_ERROR(
          "Copy src and dst to host memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      int ocv_border_type = cv::BORDER_DEFAULT;
      switch (border_type) {
      case BorderType::kReplicate:
        ocv_border_type = cv::BORDER_REPLICATE;
        break;
      case BorderType::kReflect:
        ocv_border_type = cv::BORDER_REFLECT;
        break;
      case BorderType::kReflect101:
        ocv_border_type = cv::BORDER_REFLECT101;
        break;
      case BorderType::kWrap:
        ocv_border_type = cv::BORDER_WRAP;
        break;
      case BorderType::kConstant:
        ocv_border_type = cv::BORDER_CONSTANT;
        break;
      default:
        assert(0);
        break;
      }

      double sum = 0.0;
      cv::filter2D(
        ocv_src, ocv_dst2, ocv_dst2.type(), ocv_kernel,
        cv::Point(-1, -1), 0.0, ocv_border_type);
      for (int y = 0; y < ocv_dst1.rows; ++y) {
        for (int x = 0; x < ocv_dst1.cols; ++x) {
          ocv_dst_t val1 = ocv_dst1(y, x);
          ocv_dst_t val2 = ocv_dst2(y, x);
          sum += cv::norm(val1-val2);
          if (val1 != val2) {
            //TGB_LOG_ERROR("Row filter's diff %i and %i.\n", x, y);
          }
        }
      }
      sum = std::sqrt(sum/ocv_src.total());
      TGB_LOG_INFO("Row filter's avg error: %.6lf.\n", sum);

      NOOP();
#   endif // OPENCV_VERFICATION

      ret = 0;
    }
    TGB_SECTION_END();

    return ret;
  }

  template <typename ST, typename DT, typename KT>
  int GaussianBlurColFilter(
      PixelPicker<ST> &src_picker,
      PixelPicker<DT> &dst_picker,
      int              kernel_size,
      BorderType       border_type,
      int              max_shared_mem_size_in_bytes) {
    int ret = -1;

    TGB_SECTION_BEGIN();
    {
      // get adaptive block size
      int block_x_size = 0;
      int block_y_size = 0;
      int block_pad_size = 0;
      int block_num_patches = 0;
      if (GetAdaptiveColBlockSize<ST>(
          max_shared_mem_size_in_bytes, src_picker.height_,
          kernel_size, block_x_size, block_y_size,
          block_pad_size, block_num_patches)) {
        TGB_LOG_ERROR("Failed to get adaptive column block size.\n");
        TGB_SECTION_BREAK();
      }

      const int patched_block_size =
        block_num_patches*block_y_size;
      const int shared_mem_size_in_bytes =
        sizeof(ST)*block_x_size*(2*block_pad_size+patched_block_size);
      const dim3 blockDim(
        block_x_size, block_y_size);
      const dim3 gridDim(
        (src_picker.width_+block_x_size-1)/block_x_size,
        (src_picker.height_+patched_block_size-1)/patched_block_size);

      // make kernel call
      switch (border_type) {
        case BorderType::kReplicate: {
          GaussianBlurColFilterKernel<ST, DT, KT, BorderType::kReplicate>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        case BorderType::kReflect: {
          GaussianBlurColFilterKernel<ST, DT, KT, BorderType::kReflect>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        case BorderType::kReflect101: {
          GaussianBlurColFilterKernel<ST, DT, KT, BorderType::kReflect101>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        case BorderType::kWrap: {
          GaussianBlurColFilterKernel<ST, DT, KT, BorderType::kWrap>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        case BorderType::kConstant: {
          GaussianBlurColFilterKernel<ST, DT, KT, BorderType::kConstant>
            <<<gridDim, blockDim, shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, block_pad_size, block_num_patches);
          break;
        }
        default: {
          assert(0);
          break;
        }
      }

#   if defined(OPENCV_VERFICATION)
      // verify data correctness by compared with opencv result
      typedef cv::Vec<typename ChannelType<ST>::value_type, ChannelCount<ST>::count> ocv_src_t;
      typedef cv::Vec<typename ChannelType<DT>::value_type, ChannelCount<DT>::count> ocv_dst_t;
      typedef cv::Vec<typename ChannelType<KT>::value_type, ChannelCount<KT>::count> ocv_kernel_t;
      cv::Mat_<ocv_src_t>    ocv_src(src_picker.height_, src_picker.width_);
      cv::Mat_<ocv_dst_t>    ocv_dst1(dst_picker.height_, dst_picker.width_);
      cv::Mat_<ocv_src_t>    ocv_dst2(dst_picker.height_, dst_picker.width_);
      cv::Mat_<ocv_kernel_t> ocv_kernel(kernel_size, 1);

      cudaError_t cuda_error = cudaSuccess;
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          ocv_src.data, ocv_src.step,
          src_picker.data_ptr_, src_picker.stride_,
          sizeof(ST)*src_picker.width_, src_picker.height_,
          cudaMemcpyDeviceToHost)) ||
          cudaSuccess != (cuda_error = cudaMemcpy2D(
          ocv_dst1.data, ocv_dst1.step,
          dst_picker.data_ptr_, dst_picker.stride_,
          sizeof(DT)*dst_picker.width_, dst_picker.height_,
          cudaMemcpyDeviceToHost)) ||
          cudaSuccess != (cuda_error = cudaMemcpyFromSymbol(
          ocv_kernel.data, test_gaussian_blur::c_const,
          sizeof(KT)*kernel_size))) {
        TGB_LOG_ERROR(
          "Copy src and dst to host memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      int ocv_border_type = cv::BORDER_DEFAULT;
      switch (border_type) {
      case BorderType::kReplicate:
        ocv_border_type = cv::BORDER_REPLICATE;
        break;
      case BorderType::kReflect:
        ocv_border_type = cv::BORDER_REFLECT;
        break;
      case BorderType::kReflect101:
        ocv_border_type = cv::BORDER_REFLECT101;
        break;
      case BorderType::kWrap:
        ocv_border_type = cv::BORDER_WRAP;
        break;
      case BorderType::kConstant:
        ocv_border_type = cv::BORDER_CONSTANT;
        break;
      default:
        assert(0);
        break;
      }

      double sum = 0.0;
      cv::filter2D(
        ocv_src, ocv_dst2, ocv_dst2.type(), ocv_kernel,
        cv::Point(-1, -1), 0.0, ocv_border_type);
      for (int y = 0; y < ocv_dst1.rows; ++y) {
        for (int x = 0; x < ocv_dst1.cols; ++x) {
          ocv_dst_t val1 = ocv_dst1(y, x);
          ocv_src_t val2 = ocv_dst2(y, x);
          ocv_dst_t val3;
          for (int i = 0; i < val3.channels; ++i) {
            val3(i) = RoundType<ocv_dst_t::value_type, ocv_src_t::value_type>(val2(i));
          }
          sum += cv::norm(val1-val3);
          if (val1 != val3) {
            //TGB_LOG_ERROR("Col filter's diff %i and %i.\n", x, y);
          }
        }
      }
      sum = std::sqrt(sum/ocv_src.total());
      TGB_LOG_INFO("Col filter's avg error: %.6lf.\n", sum);

      NOOP();
#   endif // OPENCV_VERFICATION

      ret = 0;
    }
    TGB_SECTION_END();

    return ret;
  }

  template <typename ST, typename DT>
  int GaussianBlurWithDeviceBuffer(
      const ST  *d_src_data_ptr,
      DT        *d_dst_data_ptr,
      int        d_src_stride,
      int        d_dst_stride,
      int        width,
      int        height,
      int        kernel_x_size,
      int        kernel_y_size,
      double     kernel_x_sigma,
      double     kernel_y_sigma,
      BorderType border_type) {
    assert(width > 0 && height > 0);
    assert(size_t(d_src_stride) >= sizeof(ST)*width && size_t(d_dst_stride) >= sizeof(DT)*width);
    assert(kernel_x_size > 0 && kernel_x_size/2+1 < width && (kernel_x_size & 0x1) == 0x1);
    assert(kernel_y_size > 0 && kernel_y_size/2+1 < height && (kernel_y_size & 0x1) == 0x1);

    typedef typename UpgradeType<typename ChannelType<ST>::value_type>::upgraded_type KT;
    typedef typename UpgradeType<ST>::upgraded_type FT;
    KT *h_x_kernel_ptr = nullptr;
    KT *h_y_kernel_ptr = nullptr;
    FT *d_buf_data_ptr = nullptr;

    int ret = -1;
    cudaError_t cuda_error = cudaSuccess;

    TGB_SECTION_BEGIN();
    {
      // get cuda device
      int cuda_device = -1;
      if (cudaSuccess != (cuda_error = cudaGetDevice(&cuda_device)) ||
          cuda_device < 0) {
        TGB_LOG_ERROR(
          "Get cuda device id failed(%s).\n",
          cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      // get cuda device properties (shared memory size)
      cudaDeviceProp cuda_dev_prop;
      if (cudaSuccess != (cuda_error =
          cudaGetDeviceProperties(&cuda_dev_prop, cuda_device))) {
        TGB_LOG_ERROR(
          "Get cuda device properties failed.\n",
          cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      // setup kernel's sigma
      // formula come from opencv function 'getGaussianKernel' document
      if (kernel_x_sigma <= 0.0) {
        kernel_x_sigma = 0.3*((kernel_x_size-1)*0.5 - 1.0) + 0.8;
      }
      if (kernel_y_sigma <= 0.0) {
        kernel_y_sigma = 0.3*((kernel_y_size-1)*0.5 - 1.0) + 0.8;
      }
      if (kernel_x_size*sizeof(KT) > cuda_dev_prop.totalConstMem ||
        kernel_y_size*sizeof(KT) > cuda_dev_prop.totalConstMem) {
          TGB_LOG_ERROR(
            "Too large kernel size, max size is %d.\n",
            cuda_dev_prop.totalConstMem/sizeof(KT));
          TGB_SECTION_BREAK();
      }

      // create separable kernels
      if (!(h_x_kernel_ptr = new (std::nothrow) KT[kernel_x_size]) ||
          buildKernel(h_x_kernel_ptr, kernel_x_size, kernel_x_sigma)) {
        TGB_LOG_ERROR("Build x kernel failed.\n");
        TGB_SECTION_BREAK();
      };
      if (kernel_y_size == kernel_x_size && kernel_y_sigma == kernel_x_sigma) {
        h_y_kernel_ptr = h_x_kernel_ptr;
      } else if (
          !(h_y_kernel_ptr = new (std::nothrow) KT[kernel_y_size]) ||
          buildKernel(h_y_kernel_ptr, kernel_y_size, kernel_y_sigma)) {
        TGB_LOG_ERROR("Build y kernel failed.\n");
        TGB_SECTION_BREAK();
      }

      // allocate buffer device memory
      const int d_buf_stride = sizeof(FT)*width;
      if (cudaSuccess != (cuda_error = cudaMalloc(
          &d_buf_data_ptr, d_buf_stride*height))) {
        TGB_LOG_ERROR(
          "Allocate buffer device memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      // making x-direction convolution
      if (kernel_x_size > 0) {
        // copy kernel data to constant memory
        if (cudaSuccess != (cuda_error = cudaMemcpyToSymbol(
            test_gaussian_blur::c_const, h_x_kernel_ptr,
            kernel_x_size*sizeof(KT)))) {
          TGB_LOG_ERROR(
            "Copy x kernel to constant memory failed(%s).\n",
            cudaGetErrorString(cuda_error));
          TGB_SECTION_BREAK();
        }
        // determine x-dir block size and call cuda kernel function
        PixelPicker<ST> src_picker(
          width, height, d_src_stride, const_cast<ST*>(d_src_data_ptr));
        PixelPicker<FT> dst_picker(
          width, height, d_buf_stride, d_buf_data_ptr);
        if (GaussianBlurRowFilter<ST, FT, KT>(
            src_picker, dst_picker, kernel_x_size, border_type,
            (int)cuda_dev_prop.sharedMemPerBlock)) {
          TGB_LOG_ERROR("Perform column filter failed.\n");
          TGB_SECTION_BREAK();
        }
      }
      // making y-direction convolution
      if (kernel_y_size > 0) {
        // reuse the x-dir kernel data or copy y-dir data to constant memory
        if (h_y_kernel_ptr != h_x_kernel_ptr &&
            cudaSuccess != (cuda_error = cudaMemcpyToSymbol(
            test_gaussian_blur::c_const, h_y_kernel_ptr,
            kernel_y_size*sizeof(KT)))) {
          TGB_LOG_ERROR(
            "Copy y kernel to constant memroy failed(%s).\n",
            cudaGetErrorString(cuda_error));
          TGB_SECTION_BREAK();
        }
        // determine y-dir block size and call cuda kernel function
        PixelPicker<FT> src_picker(
          width, height, d_buf_stride, d_buf_data_ptr);
        PixelPicker<DT> dst_picker(
          width, height, d_dst_stride, d_dst_data_ptr);
        if (GaussianBlurColFilter<FT, DT, KT>(
            src_picker, dst_picker, kernel_y_size, border_type,
            (int)cuda_dev_prop.sharedMemPerBlock)) {
          TGB_LOG_ERROR("Perform column filter failed.\n");
          TGB_SECTION_BREAK();
        }
      }

      //// make sure all kernels are finished
      //if (cudaSuccess != (cuda_error = cudaDeviceSynchronize())) {
      //  TGB_LOG_ERROR(
      //    "Synchronize cuda device failed(%s).\n",
      //    cudaGetErrorString(cuda_error));
      //  TGB_SECTION_BREAK();
      //}

      ret = 0;
    }
    TGB_SECTION_END();

    // delete temporary memory buffers
    if (d_buf_data_ptr) {
      cudaFree(d_buf_data_ptr);
      d_buf_data_ptr = nullptr;
    }
    if (h_y_kernel_ptr && h_y_kernel_ptr != h_x_kernel_ptr) {
      delete []h_y_kernel_ptr;
      h_y_kernel_ptr = nullptr;
    }
    if (h_x_kernel_ptr) {
      delete []h_x_kernel_ptr;
      h_x_kernel_ptr = nullptr;
    }

    return ret;
  }

  template <typename ST, typename DT>
  int GaussianBlurWithHostBuffer(
      const ST  *h_src_data_ptr,
      DT        *h_dst_data_ptr,
      int        h_src_stride,
      int        h_dst_stride,
      int        width,
      int        height,
      int        kernel_x_size,
      int        kernel_y_size,
      double     kernel_x_sigma,
      double     kernel_y_sigma,
      BorderType border_type) {
    assert(width > 0 && height > 0);
    assert(size_t(h_src_stride) >= sizeof(ST)*width && size_t(h_dst_stride) >= sizeof(DT)*width);
    assert(kernel_x_size > 0 && kernel_x_size/2+1 < width && (kernel_x_size & 0x1) == 0x1);
    assert(kernel_y_size > 0 && kernel_y_size/2+1 < height && (kernel_y_size & 0x1) == 0x1);

    ST *d_src_data_ptr = nullptr;
    DT *d_dst_data_ptr = nullptr;

    int ret = -1;
    cudaError_t cuda_error = cudaSuccess;

    TGB_SECTION_BEGIN();
    {
      // allocate source and destination device memory
      const int d_src_stride = sizeof(ST)*width;
      const int d_dst_stride = sizeof(DT)*width;
      if (cudaSuccess != (cuda_error = cudaMalloc(
          &d_src_data_ptr, d_src_stride*height)) ||
          cudaSuccess != (cuda_error = cudaMalloc(
          &d_dst_data_ptr, d_dst_stride*height))) {
        TGB_LOG_ERROR(
          "Allocate source and destination device memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      // copy source data to device memory
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          d_src_data_ptr, d_src_stride, h_src_data_ptr, h_src_stride,
          width*sizeof(ST), height, cudaMemcpyHostToDevice))) {
        TGB_LOG_ERROR(
          "Copy src data to device memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      // call device buffer wrapper
      if (GaussianBlurWithDeviceBuffer(
          d_src_data_ptr, d_dst_data_ptr, d_src_stride, d_dst_stride,
          width, height, kernel_x_size, kernel_y_size,
          kernel_x_sigma, kernel_y_sigma, border_type)) {
        TGB_SECTION_BREAK();
      }

      // copy the result data from device memory
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          h_dst_data_ptr, h_dst_stride, d_dst_data_ptr, d_dst_stride,
          width*sizeof(DT), height, cudaMemcpyDeviceToHost))) {
        TGB_LOG_ERROR(
          "Copy dst data to host memory failed(%s).\n",
          cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      ret = 0;
    }
    TGB_SECTION_END();

    // delete temporary memory buffers 
    if (d_dst_data_ptr) {
      cudaFree(d_dst_data_ptr);
      d_dst_data_ptr = nullptr;
    }
    if (d_src_data_ptr) {
      cudaFree(d_src_data_ptr);
      d_src_data_ptr = nullptr;
    }

    return ret;
  }

}

#endif // TEST_GAUSSIAN_BLUR_IMPL_CUSTOM_FILTER_HPP_
