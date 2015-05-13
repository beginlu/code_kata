#include "config/config.hpp"

#if __CUDACC__ >= 200
# define TGB_MAX_CUDA_CONSTANT_SIZE 65536
#else
# define TGB_MAX_CUDA_CONSTANT_SIZE 65536
#endif

namespace test_gaussian_blur {

  // constant memory use to cache kernel data
  // define in *.cu file to avoid duplicate definition
  __constant__ unsigned char c_const[TGB_MAX_CUDA_CONSTANT_SIZE];

}
