//
// Created by liupeng on 2021/3/23.
//

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <torch/torch.h>
#include <iostream>
#include <mutex>
#include <queue>
#include <vector>
#include <thread>
#include <algorithm>

#include <torch/csrc/cuda/Stream.h>
#include <torch/extension.h>


//  less than cuda10.1
#ifndef NVJPEG_FLAGS_DEFAULT
#define NVJPEG_FLAGS_DEFAULT 0
#endif

namespace torchnvjpeg {
#define CUDA(a)              \
  {                          \
    cudaError_t _e = a;      \
    if (_e != cudaSuccess) { \
      throw cuda_error(_e);  \
    }                        \
  }
#define NVJPEG(a)                      \
  {                                    \
    nvjpegStatus_t _e = a;             \
    if (_e != NVJPEG_STATUS_SUCCESS) { \
      throw nvjpeg_error(_e);          \
    }                                  \
  }

class cuda_error : std::exception {
 public:
  /**
  Create exception from error code.
  */
  explicit cuda_error(cudaError_t error) : code(error) {}

  /**
  Text description of error.
  */
  const char* what() const noexcept override {
    return cudaGetErrorString(code);
  }

 private:
  cudaError_t code;
};

inline bool SupportedSubsampling(const nvjpegChromaSubsampling_t& subsampling) {
  switch (subsampling) {
    case NVJPEG_CSS_444:
    case NVJPEG_CSS_440:
    case NVJPEG_CSS_422:
    case NVJPEG_CSS_420:
    case NVJPEG_CSS_411:
    case NVJPEG_CSS_410:
    case NVJPEG_CSS_GRAY:
      return true;
      //        case NVJPEG_CSS_UNKNOWN:
      //            return false;
    default:
      return false;
  }
}

// C++ exception for nvjpeg errors.
class nvjpeg_error : std::exception {
 public:
  explicit nvjpeg_error(nvjpegStatus_t error) : code(error) {}

  const char* what() const noexcept override {
    switch (code) {
      case NVJPEG_STATUS_NOT_INITIALIZED:
        return "nvjpeg returned \'NVJPEG_STATUS_NOT_INITIALIZED\'";
      case NVJPEG_STATUS_INVALID_PARAMETER:
        return "nvjpeg returned \'NVJPEG_STATUS_INVALID_PARAMETER\'";
      case NVJPEG_STATUS_BAD_JPEG:
        return "nvjpeg returned \'NVJPEG_STATUS_BAD_JPEG\'";
      case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return "nvjpeg returned \'NVJPEG_STATUS_JPEG_NOT_SUPPORTED\'";
      case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        return "nvjpeg returned \'NVJPEG_STATUS_ALLOCATOR_FAILURE\'";
      case NVJPEG_STATUS_EXECUTION_FAILED:
        return "nvjpeg returned \'NVJPEG_STATUS_EXECUTION_FAILED\'";
      case NVJPEG_STATUS_ARCH_MISMATCH:
        return "nvjpeg returned \'NVJPEG_STATUS_ARCH_MISMATCH\'";
      case NVJPEG_STATUS_INTERNAL_ERROR:
        return "nvjpeg returned \'NVJPEG_STATUS_INTERNAL_ERROR\'";
#if CUDA_VERSION >= 10020
      case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return "nvjpeg returned \'NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED\'";
#endif
      case NVJPEG_STATUS_SUCCESS:
        return "nvjpeg returned no error \'NVJPEG_STATUS_SUCCESS\' THIS IS A BUG";
      default:
        return "UNKNOWN NVJPEG ERROR! THIS IS A BUG!";
    }
  }

 private:
  nvjpegStatus_t code;
};

int dev_malloc(void** p, size_t s) {
  return (int)cudaMalloc(p, s);
}

int dev_free(void* p) {
  return (int)cudaFree(p);
}

int host_malloc(void** p, size_t s, unsigned int f) {
  return (int)cudaHostAlloc(p, s, f);
}

int host_free(void* p) {
  return (int)cudaFreeHost(p);
}


class NvJpeg {
 public:
  NvJpeg() = delete;

  NvJpeg(
      int device_id=0,
      cudaStream_t stream=nullptr,
      size_t max_image_size=3840*2160*3,
      size_t batch_size=2,
      int max_cpu_threads=1,
      size_t device_padding=0,
      size_t host_padding=0,
      bool gpu_huffman=true);
  NvJpeg(
      int device_id=0,
      const py::object& py_cuda_stream=py::none(),
      size_t max_image_size=3840*2160*3,
      size_t batch_size=2,
      int max_cpu_threads=1,
      size_t device_padding=0,
      size_t host_padding=0,
      bool gpu_huffman=true);


  ~NvJpeg();

  int get_device_id() const;

  torch::Tensor decode(const std::string& data, bool stream_sync);
  py::bytes encode(const torch::Tensor& img, int quality=75, std::string fromat="RGB");

  std::vector<torch::Tensor> batch_decode(const std::vector<std::string>& data_list, bool stream_sync);

 private:
  nvjpegDevAllocator_t device_allocator;
  nvjpegPinnedAllocator_t pinned_allocator;

  nvjpegHandle_t handle;
  nvjpegJpegState_t state_dec;
  nvjpegEncoderState_t state_enc;

  int device_id=0;
  cudaStream_t cuda_stream{};
  size_t max_image_size = std::numeric_limits<size_t>::max();
  int max_cpu_threads=1;
  size_t batchsize_g=2;
};
} // namespace torchnvjpeg
