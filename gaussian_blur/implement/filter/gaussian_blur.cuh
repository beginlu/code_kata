#ifndef GAUSSIAN_BLUR_FILTER_GAUSSIAN_BLUR_HPP_
#define GAUSSIAN_BLUR_FILTER_GAUSSIAN_BLUR_HPP_

//#define OPENCV_VERFICATION

#include <algorithm>
#include <cassert>
#include <cmath>
#include <new>
#include "utility/border_picker.hpp"
#include "utility/type_traits.hpp"

#if defined(OPENCV_VERFICATION)
# include "opencv2/imgproc/imgproc.hpp"
#endif // OPENCV_VERFICATION

namespace test_gaussian_blur {

  extern __constant__ unsigned char c_const[];

  template <typename ST, typename DT, typename KT, BorderType BT>
  __global__
  void GaussianBlurRowFilterKernel(
      PixelPicker<ST> src_picker,
      PixelPicker<DT> dst_picker,
      int             kernel_size,
      int             region_size) {
    // shared memory use to cache block image data
    extern __shared__ unsigned char s_shared[];

    // unfold channels
    const int &channels        = src_picker.channels_;
    const int width            = src_picker.width_*channels;
    const int half_kernel_size = kernel_size/2*channels;
    const int region_offset    = region_size*blockIdx.x;

    const int y_offset = blockDim.y*blockIdx.y+threadIdx.y;
    if (y_offset >= src_picker.height_) {
      return;
    }

    PixelPicker<DT> cached_picker(
        blockDim.y, 2*half_kernel_size+region_size, 1,
        (2*half_kernel_size+region_size)*sizeof(DT),
        reinterpret_cast<DT*>(s_shared));
    RowBorderPicker<ST, BT> border_picker(
        y_offset, src_picker);

    ST *src_data_ptr =
        src_picker.get_pixel(
        y_offset, (region_offset-half_kernel_size)/channels)+
        region_offset%channels;
    DT *cached_data_ptr =
        cached_picker.get_row(threadIdx.y);

    if (region_offset-half_kernel_size < 0) {
      // load left-most blocks' left padding area
      for (int i = threadIdx.x; i < half_kernel_size; i += blockDim.x) {
        const int pos_offset = region_offset+i;
        cached_data_ptr[i] = static_cast<DT>(border_picker.get_lower_value(
            pos_offset/channels-half_kernel_size/channels, pos_offset%channels));
      }
    } else {
      // load other blocks' left padding area
      for (int i = threadIdx.x; i < half_kernel_size; i += blockDim.x) {
        cached_data_ptr[i] = src_data_ptr[i];
      }
    }

    //__syncthreads();

    src_data_ptr    += half_kernel_size;
    cached_data_ptr += half_kernel_size;

    if (region_offset+region_size+half_kernel_size > width) {
      const int remain_size = ::min(
          region_size, width-region_offset)+half_kernel_size;
      // load right-most blocks' middle and right padding area
      for (int i = threadIdx.x; i < remain_size; i += blockDim.x) {
        const int pos_offset = region_offset+i;
        cached_data_ptr[i] = static_cast<DT>(border_picker.get_higher_value(
            pos_offset/channels, pos_offset%channels));
      }
    } else {
      const int remain_size = region_size+half_kernel_size;
      // load other blocks' middle and right padding area
      for (int i = threadIdx.x; i < remain_size; i += blockDim.x) {
        cached_data_ptr[i] = src_data_ptr[i];
      }
    }

    __syncthreads();

    cached_data_ptr -= half_kernel_size;

    // making convolution
    DT *dst_data_ptr =
        dst_picker.get_pixel(y_offset, region_offset/channels)+
        region_offset%channels;
    const KT *kernel_data_ptr =
        reinterpret_cast<const KT*>(c_const);
    for (int i = threadIdx.x; i < region_size; i += blockDim.x) {
      if (region_offset+i < width) {
        DT sum = 0;
        for (int j = 0; j < kernel_size; ++j) {
          sum += cached_data_ptr[i+j*channels]*kernel_data_ptr[j];
        }
        dst_data_ptr[i] = sum;
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
      int             region_size) {
    // shared memory use to cache block image data
    extern __shared__ unsigned char s_shared[];

    const int &height          = src_picker.height_;
    const int &channels        = src_picker.channels_;
    const int half_kernel_size = kernel_size/2;
    const int region_offset    = region_size*blockIdx.y;

    const int chn_id   = (blockDim.x*blockIdx.x+threadIdx.x)%channels;
    const int x_offset = (blockDim.x*blockIdx.x+threadIdx.x)/channels;
    if (x_offset >= src_picker.width_) {
      return;
    }

    PixelPicker<ST> cached_picker(
        2*half_kernel_size+region_size, blockDim.x, 1,
        blockDim.x*sizeof(ST), reinterpret_cast<ST*>(s_shared));
    ColBorderPicker<ST, BT> border_picker(
        x_offset, src_picker);

    if (region_offset-half_kernel_size < 0) {
      // load top-most blocks' top padding area
      for (int i = threadIdx.y; i < half_kernel_size; i += blockDim.y) {
        cached_picker.get_value(i, threadIdx.x, 0) =
            static_cast<ST>(border_picker.get_lower_value(
            region_offset-half_kernel_size+i, chn_id));
      }
    } else {
      // load other blocks' top padding area
      for (int i = threadIdx.y; i < half_kernel_size; i += blockDim.y) {
        cached_picker.get_value(i, threadIdx.x, 0) =
            src_picker.get_value(
            region_offset-half_kernel_size+i, x_offset, chn_id);
      }
    }

    //__syncthreads();

    if (region_offset+region_size+half_kernel_size > height) {
      const int remain_size = ::min(
          region_size, height-region_offset)+half_kernel_size;
      // load bottom-most blocks' middle and right padding area
      for (int i = threadIdx.y; i < remain_size; i += blockDim.y) {
        cached_picker.get_value(half_kernel_size+i, threadIdx.x, 0) =
            static_cast<ST>(border_picker.get_higher_value(
            region_offset+i, chn_id));
      }
    } else {
      const int remain_size = region_size+half_kernel_size;
      // load other blocks' middle and right padding area
      for (int i = threadIdx.y; i < remain_size; i += blockDim.y) {
        cached_picker.get_value(half_kernel_size+i, threadIdx.x, 0) =
            src_picker.get_value(region_offset+i, x_offset, chn_id);
      }
    }

    __syncthreads();

    // making convolution
    const KT *kernel_data_ptr = reinterpret_cast<const KT*>(c_const);
    for (int i = threadIdx.y; i < region_size; i += blockDim.y) {
      if (region_offset+i < height) {
        ST sum = 0;
        for (int j = 0; j < kernel_size; ++j) {
          sum += cached_picker.get_value(
              i+j, threadIdx.x, 0)*kernel_data_ptr[j];
        }
        dst_picker.get_value(region_offset+i, x_offset, chn_id) =
            RoundType<ST, DT>(sum);
      }
    }

    return;
  }

  template <typename T>
  int buildKernel(T *kernel_ptr, int kernel_size, double kernel_sigma) {
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
      kernel_ptr[i] = static_cast<T>(tmp_kernel_ptr[i] / sum);
    }

    delete []tmp_kernel_ptr;
    tmp_kernel_ptr = nullptr;

    return 0;
  }

  template <typename T>
  int GetAdaptiveRowBlockSize(
      int  shared_mem_size_in_bytes,
      int  image_width,
      int  image_channels,
      int  kernel_size,
      int &region_size,
      int &block_x_size,
      int &block_y_size,
      int &alloc_shared_mem_size_in_bytes) {
    const int half_kernel_size = kernel_size/2;

    block_x_size = 32;
    block_y_size = 4;
    region_size  = std::min(block_x_size*4, image_width*image_channels);
    alloc_shared_mem_size_in_bytes =
        block_y_size*(2*half_kernel_size*image_channels+region_size)*sizeof(T);
    if (alloc_shared_mem_size_in_bytes > shared_mem_size_in_bytes) {
      return -1;
    }

    return 0;
  }

  template <typename T>
  int GetAdaptiveColBlockSize(
      int  shared_mem_size_in_bytes,
      int  image_height,
      int  image_channels,
      int  kernel_size,
      int &region_size,
      int &block_x_size,
      int &block_y_size,
      int &alloc_shared_mem_size_in_bytes) {
    const int half_kernel_size = kernel_size/2;

    block_x_size = 8;
    block_y_size = 16;
    region_size  = std::max(1, std::min(image_height, block_y_size*4));
    alloc_shared_mem_size_in_bytes =
        (2*half_kernel_size+region_size)*block_x_size*sizeof(T);
    if (alloc_shared_mem_size_in_bytes > shared_mem_size_in_bytes) {
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
      int              shared_mem_size_in_bytes) {
    int ret = -1;

    TGB_SECTION_BEGIN();
    {
      // get adaptive block size
      int region_size  = 0;
      int block_x_size = 0;
      int block_y_size = 0;
      int alloc_shared_mem_size_in_bytes = 0;
      if (GetAdaptiveRowBlockSize<DT>(
          shared_mem_size_in_bytes,
          src_picker.width_, src_picker.channels_, kernel_size,
          region_size, block_x_size, block_y_size,
          alloc_shared_mem_size_in_bytes)) {
        TGB_LOG_ERROR("Failed to get adaptive row block size.\n");
        TGB_SECTION_BREAK();
      }

      // make kernel call
      const dim3 blockDim(
          block_x_size, block_y_size);
      const dim3 gridDim(
          (src_picker.width_*src_picker.channels_+region_size-1)/region_size,
          (src_picker.height_+block_y_size-1)/block_y_size);
      switch (border_type) {
        case BorderType::kReplicate: {
          GaussianBlurRowFilterKernel
              <ST, DT, KT, BorderType::kReplicate>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>(
              src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        case BorderType::kReflect: {
          GaussianBlurRowFilterKernel
              <ST, DT, KT, BorderType::kReflect>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>(
              src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        case BorderType::kReflect101: {
          GaussianBlurRowFilterKernel
              <ST, DT, KT, BorderType::kReflect101>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>(
              src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        case BorderType::kWrap: {
          GaussianBlurRowFilterKernel
            <ST, DT, KT, BorderType::kWrap>
            <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>(
            src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        case BorderType::kConstant: {
          GaussianBlurRowFilterKernel
              <ST, DT, KT, BorderType::kConstant>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>(
              src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        default: {
          assert(0);
          break;
        }
      }

#   if defined(OPENCV_VERFICATION)
      // verify data correctness by compared with opencv result
      cv::Mat ocv_src(
          src_picker.height_, src_picker.width_,
          CV_MAKE_TYPE(cv::DataDepth<ST>::value, src_picker.channels_));
      cv::Mat ocv_dst1(
          dst_picker.height_, dst_picker.width_,
          CV_MAKE_TYPE(cv::DataDepth<DT>::value, dst_picker.channels_));
      cv::Mat ocv_dst2(
          dst_picker.height_, dst_picker.width_,
          CV_MAKE_TYPE(cv::DataDepth<DT>::value, dst_picker.channels_));
      cv::Mat ocv_kernel(
          1, kernel_size,
          CV_MAKE_TYPE(cv::DataDepth<KT>::value, 1));

      cudaError_t cuda_error = cudaSuccess;
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          ocv_src.data, ocv_src.step,
          src_picker.data_ptr_, src_picker.stride_,
          src_picker.width_*src_picker.channels_*sizeof(ST),
          src_picker.height_, cudaMemcpyDeviceToHost)) ||
          cudaSuccess != (cuda_error = cudaMemcpy2D(
          ocv_dst1.data, ocv_dst1.step,
          dst_picker.data_ptr_, dst_picker.stride_,
          dst_picker.width_*dst_picker.channels_*sizeof(DT),
          dst_picker.height_, cudaMemcpyDeviceToHost)) ||
          cudaSuccess != (cuda_error = cudaMemcpyFromSymbol(
          ocv_kernel.data, test_gaussian_blur::c_const,
          kernel_size*sizeof(KT)))) {
        TGB_LOG_ERROR(
            "Copy src, dst, kernel to host memory failed(%s).\n",
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
      for (int y = 0; y < dst_picker.height_; ++y) {
        const DT *ocv_dst1_data_ptr = ocv_dst1.ptr<DT>(y);
        const DT *ocv_dst2_data_ptr = ocv_dst2.ptr<DT>(y);
        for (int x = 0; x < dst_picker.width_; ++x) {
          for (int c = 0; c < dst_picker.channels_; ++c) {
            const DT val_dst1 = *(ocv_dst1_data_ptr++);
            const DT val_dst2 = *(ocv_dst2_data_ptr++);
            sum += (val_dst1-val_dst2)*(val_dst1-val_dst2);
            if (val_dst1 != val_dst2) {
              //TGB_LOG_ERROR("Row filter's diff (%d, %d, %d).\n", y, x, c);
            }
          }
        }
      }
      sum = std::sqrt(sum/ocv_dst1.total());
      TGB_LOG_INFO("Row filter's average error: %.6lf.\n", sum);
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
      int              shared_mem_size_in_bytes) {
    int ret = -1;

    TGB_SECTION_BEGIN();
    {
      // get adaptive block size
      int region_size  = 0;
      int block_x_size = 0;
      int block_y_size = 0;
      int alloc_shared_mem_size_in_bytes = 0;
      if (GetAdaptiveColBlockSize<ST>(
          shared_mem_size_in_bytes,
          src_picker.height_, src_picker.channels_, kernel_size,
          region_size, block_x_size, block_y_size,
          alloc_shared_mem_size_in_bytes)) {
        TGB_LOG_ERROR("Failed to get adaptive column block size.\n");
        TGB_SECTION_BREAK();
      }

      // make kernel call
      const dim3 blockDim(
          block_x_size, block_y_size);
      const dim3 gridDim(
          (src_picker.width_*src_picker.channels_+block_x_size-1)/block_x_size,
          (src_picker.height_+region_size-1)/region_size);
      switch (border_type) {
        case BorderType::kReplicate: {
          GaussianBlurColFilterKernel
              <ST, DT, KT, BorderType::kReplicate>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>
              (src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        case BorderType::kReflect: {
          GaussianBlurColFilterKernel
              <ST, DT, KT, BorderType::kReflect>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>
              (src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        case BorderType::kReflect101: {
          GaussianBlurColFilterKernel
              <ST, DT, KT, BorderType::kReflect101>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>
              (src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        case BorderType::kWrap: {
          GaussianBlurColFilterKernel
              <ST, DT, KT, BorderType::kWrap>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>
              (src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        case BorderType::kConstant: {
          GaussianBlurColFilterKernel
              <ST, DT, KT, BorderType::kConstant>
              <<<gridDim, blockDim, alloc_shared_mem_size_in_bytes>>>
              (src_picker, dst_picker, kernel_size, region_size);
          break;
        }
        default: {
          assert(0);
          break;
        }
      }

#   if defined(OPENCV_VERFICATION)
      // verify data correctness by compared with opencv result
      cv::Mat ocv_src(
          src_picker.height_, src_picker.width_,
          CV_MAKE_TYPE(cv::DataDepth<ST>::value, src_picker.channels_));
      cv::Mat ocv_dst1(
          dst_picker.height_, dst_picker.width_,
          CV_MAKE_TYPE(cv::DataDepth<DT>::value, dst_picker.channels_));
      cv::Mat ocv_dst2(
          dst_picker.height_, dst_picker.width_,
          CV_MAKE_TYPE(cv::DataDepth<ST>::value, dst_picker.channels_));
      cv::Mat ocv_kernel(
          kernel_size, 1, CV_MAKE_TYPE(cv::DataDepth<ST>::value, 1));

      cudaError_t cuda_error = cudaSuccess;
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          ocv_src.data, ocv_src.step,
          src_picker.data_ptr_, src_picker.stride_,
          src_picker.width_*src_picker.channels_*sizeof(ST),
          src_picker.height_, cudaMemcpyDeviceToHost)) ||
          cudaSuccess != (cuda_error = cudaMemcpy2D(
          ocv_dst1.data, ocv_dst1.step,
          dst_picker.data_ptr_, dst_picker.stride_,
          dst_picker.width_*dst_picker.channels_*sizeof(DT),
          dst_picker.height_, cudaMemcpyDeviceToHost)) ||
          cudaSuccess != (cuda_error = cudaMemcpyFromSymbol(
          ocv_kernel.data, test_gaussian_blur::c_const,
          kernel_size*sizeof(ST)))) {
        TGB_LOG_ERROR(
            "Copy src, dst, kernel to host memory failed(%s).\n",
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
      for (int y = 0; y < dst_picker.height_; ++y) {
        const DT *ocv_dst1_data_ptr = ocv_dst1.ptr<DT>(y);
        const ST *ocv_dst2_data_ptr = ocv_dst2.ptr<ST>(y);
        for (int x = 0; x < dst_picker.width_; ++x) {
          for (int c = 0; c < dst_picker.channels_; ++c) {
            const DT val_dst1 = *(ocv_dst1_data_ptr++);
            const DT val_dst2 = RoundType<ST, DT>(*(ocv_dst2_data_ptr++));
            sum += (val_dst1-val_dst2)*(val_dst1-val_dst2);
            if (val_dst1 != val_dst2) {
              //TGB_LOG_ERROR("Row filter's diff (%d, %d, %d).\n", y, x, c);
            }
          }
        }
      }
      sum = std::sqrt(sum/ocv_dst1.total());
      TGB_LOG_INFO("Column filter's average error: %.6lf.\n", sum);
#   endif // OPENCV_VERFICATION

      ret = 0;
    }
    TGB_SECTION_END();

    return ret;
  }

  template <typename ST, typename DT, typename FT>
  int GaussianBlurDeviceWithBuffer(
      PixelPicker<ST> &src_picker,
      PixelPicker<DT> &dst_picker,
      PixelPicker<FT> &buf_picker,
      int              kernel_x_size,
      int              kernel_y_size,
      double           kernel_x_sigma,
      double           kernel_y_sigma,
      BorderType       border_type) {
    static_assert(
        std::is_same<FT, typename UpgradeType<ST>::upgraded_type>::value,
        "Invalid buffer type, should be upgraded from source type.");
    assert(
        src_picker.check_validity() &&
        dst_picker.check_validity() &&
        buf_picker.check_validity());
    assert(
        src_picker.height_   == dst_picker.height_ &&
        src_picker.height_   == buf_picker.height_ &&
        src_picker.width_    == dst_picker.width_ &&
        src_picker.width_    == buf_picker.width_ &&
        src_picker.channels_ == dst_picker.channels_ &&
        src_picker.channels_ == buf_picker.channels_);
    assert(
        kernel_x_size > 0 && (kernel_x_size & 0x1) == 0x1 &&
        kernel_x_size/2+1 <= src_picker.width_ &&
        kernel_y_size > 0 && (kernel_y_size & 0x1) == 0x1 &&
        kernel_y_size/2+1 <= src_picker.height_);

    typedef FT KT;
    KT *h_kernel_x_ptr = nullptr;
    KT *h_kernel_y_ptr = nullptr;

    int ret = -1;

    TGB_SECTION_BEGIN();
    {
      cudaError_t cuda_error = cudaSuccess;

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
              "Too large kernel size, max size is %u.\n",
              cuda_dev_prop.totalConstMem/sizeof(KT));
          TGB_SECTION_BREAK();
      }
      // create separable kernels
      if (!(h_kernel_x_ptr = new (std::nothrow) KT[kernel_x_size]) ||
          buildKernel(h_kernel_x_ptr, kernel_x_size, kernel_x_sigma)) {
        TGB_LOG_ERROR("Build x kernel failed.\n");
        TGB_SECTION_BREAK();
      };
      if (kernel_y_size == kernel_x_size && kernel_y_sigma == kernel_x_sigma) {
          h_kernel_y_ptr = h_kernel_x_ptr;
      } else if (
          !(h_kernel_y_ptr = new (std::nothrow) KT[kernel_y_size]) ||
          buildKernel(h_kernel_y_ptr, kernel_y_size, kernel_y_sigma)) {
        TGB_LOG_ERROR("Build y kernel failed.\n");
        TGB_SECTION_BREAK();
      }

      // making x-direction convolution
      if (kernel_x_size > 0) {
        // copy kernel data to constant memory
        if (cudaSuccess != (cuda_error = cudaMemcpyToSymbol(
            test_gaussian_blur::c_const, h_kernel_x_ptr,
            kernel_x_size*sizeof(KT)))) {
          TGB_LOG_ERROR(
              "Copy x kernel to constant memory failed(%s).\n",
              cudaGetErrorString(cuda_error));
          TGB_SECTION_BREAK();
        }
        // determine x-dir block size and call cuda kernel function
        if (GaussianBlurRowFilter<ST, FT, KT>(
            src_picker, buf_picker, kernel_x_size, border_type,
            static_cast<int>(cuda_dev_prop.sharedMemPerBlock))) {
          TGB_LOG_ERROR("Perform column filter failed.\n");
          TGB_SECTION_BREAK();
        }
      }
      // making y-direction convolution
      if (kernel_y_size > 0) {
        // reuse the x-dir kernel data or copy y-dir data to constant memory
        if (h_kernel_y_ptr != h_kernel_x_ptr &&
            cudaSuccess != (cuda_error = cudaMemcpyToSymbol(
            test_gaussian_blur::c_const, h_kernel_y_ptr,
            kernel_y_size*sizeof(KT)))) {
          TGB_LOG_ERROR(
              "Copy y kernel to constant memroy failed(%s).\n",
              cudaGetErrorString(cuda_error));
          TGB_SECTION_BREAK();
        }
        // determine y-dir block size and call cuda kernel function
        if (GaussianBlurColFilter<FT, DT, KT>(
            buf_picker, dst_picker, kernel_y_size, border_type,
            static_cast<int>(cuda_dev_prop.sharedMemPerBlock))) {
          TGB_LOG_ERROR("Perform column filter failed.\n");
          TGB_SECTION_BREAK();
        }
      }

      //// make sure all kernels are finished
      //if (cudaSuccess != (cuda_error = cudaDeviceSynchronize())) {
      //  TGB_LOG_ERROR(
      //      "Synchronize cuda device failed(%s).\n",
      //      cudaGetErrorString(cuda_error));
      //  TGB_SECTION_BREAK();
      //}

      ret = 0;
    }
    TGB_SECTION_END();

    if (h_kernel_y_ptr && h_kernel_y_ptr != h_kernel_x_ptr) {
      delete []h_kernel_y_ptr;
      h_kernel_y_ptr = nullptr;
    }
    if (h_kernel_x_ptr) {
      delete []h_kernel_x_ptr;
      h_kernel_x_ptr = nullptr;
    }

    return ret;
  }

  template <typename ST, typename DT>
  int GaussianBlurDevice(
      PixelPicker<ST> &src_picker,
      PixelPicker<DT> &dst_picker,
      int              kernel_x_size,
      int              kernel_y_size,
      double           kernel_x_sigma,
      double           kernel_y_sigma,
      BorderType       border_type) {
    assert(src_picker.check_validity());

    typedef typename UpgradeType<ST>::upgraded_type FT;
    PixelPicker<FT> buf_picker(
        src_picker.height_, src_picker.width_, src_picker.channels_,
        src_picker.width_*src_picker.channels_*sizeof(FT),
        nullptr);

    int ret = -1;

    TGB_SECTION_BEGIN();
    {
      cudaError_t cuda_error = cudaSuccess;

      if (cudaSuccess != (cuda_error = cudaMalloc(
          &(buf_picker.data_ptr_),
          buf_picker.height_*buf_picker.stride_))) {
        TGB_LOG_ERROR(
            "Allocate device buffer memory failed(%s).\n",
            cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }
      if (GaussianBlurDeviceWithBuffer(
          src_picker, dst_picker, buf_picker,
          kernel_x_size, kernel_y_size,
          kernel_x_sigma, kernel_y_sigma, border_type)) {
        TGB_SECTION_BREAK();
      }

      ret = 0;
    }
    TGB_SECTION_END();

    if (buf_picker.data_ptr_) {
      cudaFree(buf_picker.data_ptr_);
    }
    buf_picker.clear();

    return ret;
  }

  template <typename ST, typename DT>
  int GaussianBlurHost(
      PixelPicker<ST> &src_picker,
      PixelPicker<DT> &dst_picker,
      int              kernel_x_size,
      int              kernel_y_size,
      double           kernel_x_sigma,
      double           kernel_y_sigma,
      BorderType       border_type) {
    assert(
        src_picker.check_validity() &&
        dst_picker.check_validity());
    assert(
        src_picker.height_   == dst_picker.height_ &&
        src_picker.width_    == dst_picker.width_ &&
        src_picker.channels_ == dst_picker.channels_);

    PixelPicker<ST> d_src_picker(
        src_picker.height_, src_picker.width_, src_picker.channels_,
        src_picker.width_*src_picker.channels_*sizeof(ST),
        nullptr);
    PixelPicker<DT> d_dst_picker(
        dst_picker.height_, dst_picker.width_, dst_picker.channels_,
        dst_picker.width_*dst_picker.channels_*sizeof(DT),
        nullptr);

    int ret = -1;

    TGB_SECTION_BEGIN();
    {
      cudaError_t cuda_error = cudaSuccess;

      // allocate source and destination device memory
      if (cudaSuccess != (cuda_error = cudaMalloc(
          &(d_src_picker.data_ptr_),
          d_src_picker.height_*d_src_picker.stride_)) ||
          cudaSuccess != (cuda_error = cudaMalloc(
          &(d_dst_picker.data_ptr_),
          d_dst_picker.height_*d_dst_picker.stride_))) {
        TGB_LOG_ERROR(
            "Allocate device src and dst memory failed(%s).\n",
            cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      // copy source data to device memory
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          d_src_picker.data_ptr_, d_src_picker.stride_,
          src_picker.data_ptr_, src_picker.stride_,
          src_picker.width_*src_picker.channels_*sizeof(ST),
          src_picker.height_, cudaMemcpyHostToDevice))) {
        TGB_LOG_ERROR(
            "Copy src data to device memory failed(%s).\n",
            cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      // call device function
      if (GaussianBlurDevice(
          d_src_picker, d_dst_picker,
          kernel_x_size, kernel_y_size,
          kernel_x_sigma, kernel_y_sigma, border_type)) {
        TGB_SECTION_BREAK();
      }

      // copy destination data from device memory
      if (cudaSuccess != (cuda_error = cudaMemcpy2D(
          dst_picker.data_ptr_, dst_picker.stride_,
          d_dst_picker.data_ptr_, d_dst_picker.stride_,
          d_dst_picker.width_*d_dst_picker.channels_*sizeof(DT),
          d_dst_picker.height_, cudaMemcpyDeviceToHost))) {
        TGB_LOG_ERROR(
            "Copy dst data to host memory failed(%s).\n",
            cudaGetErrorString(cuda_error));
        TGB_SECTION_BREAK();
      }

      ret = 0;
    }
    TGB_SECTION_END();

    if (d_dst_picker.data_ptr_) {
      cudaFree(d_dst_picker.data_ptr_);
    }
    if (d_src_picker.data_ptr_) {
      cudaFree(d_src_picker.data_ptr_);
    }
    d_dst_picker.clear();
    d_src_picker.clear();

    return ret;
  }

}

#endif // GAUSSIAN_BLUR_FILTER_GAUSSIAN_BLUR_HPP_
