//
// Created by liupeng on 2021/3/23.
//
#include "torchnvjpeg.h"

namespace torchnvjpeg {

static void* ctypes_void_ptr(const py::object& object) {
  PyObject* p_ptr = object.ptr();
  if (!PyObject_HasAttr(p_ptr, PyUnicode_FromString("value"))) {
    return nullptr;
  }
  PyObject* ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
  if (ptr_as_int == Py_None) {
    return nullptr;
  }
  void* ptr = PyLong_AsVoidPtr(ptr_as_int);
  return ptr;
}

NvJpeg::NvJpeg(
    int device_id,
    cudaStream_t stream,
    size_t max_image_size,
    size_t batch_size,
    int max_cpu_threads,
    size_t device_padding,
    size_t host_padding,
    bool gpu_huffman)
    : device_allocator{&dev_malloc, &dev_free},
      pinned_allocator{&host_malloc, &host_free},
      device_id(device_id),
      cuda_stream(stream),
      max_image_size(max_image_size),
      batchsize_g(batch_size),
      max_cpu_threads(max_cpu_threads) {
  /**
   * using pytorch:  torch.cuda.set_device
   * torch version 1.8
   * https://github.com/Quansight/pytorch/commit/3788a42f5e4e16f86fc3d5b2062b20262d71a051
   *  torch::cuda::set_device(device_id);
   */

  CUDA(cudaSetDevice(device_id));

  // nvjpegBackend_t backend = NVJPEG_BACKEND_DEFAULT;
  nvjpegBackend_t backend = NVJPEG_BACKEND_HYBRID;
  if (gpu_huffman) {
    backend = NVJPEG_BACKEND_GPU_HYBRID;
  }
  NVJPEG(nvjpegCreateEx(backend, &device_allocator, &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &handle))
  // NVJPEG(nvjpegCreateSimple(&handle));

  NVJPEG(nvjpegJpegStateCreate(handle, &state_dec))
  NVJPEG(nvjpegEncoderStateCreate(handle, &state_enc, NULL));

  NVJPEG(nvjpegSetDeviceMemoryPadding(device_padding, handle))
  NVJPEG(nvjpegSetPinnedMemoryPadding(host_padding, handle))
  // std::cout<<device_id<<" "<<max_image_size<<" "<<batchsize_g<<" "<<max_cpu_threads<<std::endl;
  NVJPEG(nvjpegDecodeBatchedInitialize(handle, state_dec, batchsize_g, max_cpu_threads, NVJPEG_OUTPUT_RGBI));
}

NvJpeg::NvJpeg(
    int device_id,
    const py::object& py_cuda_stream,
    size_t max_image_size,
    size_t batch_size,
    int max_cpu_threads,
    size_t device_padding,
    size_t host_padding,
    bool gpu_huffman) {
  cudaStream_t stream = py_cuda_stream.is_none() ? c10::cuda::getDefaultCUDAStream(device_id).stream()
                                                 : static_cast<cudaStream_t>(ctypes_void_ptr(py_cuda_stream));

  new (this) NvJpeg(device_id, stream, max_image_size, batch_size, max_cpu_threads, device_padding, host_padding, gpu_huffman);
}

NvJpeg::~NvJpeg() {
  nvjpegJpegStateDestroy(state_dec);
  nvjpegDestroy(handle);
}

int NvJpeg::get_device_id() const {
  return device_id;
}


torch::Tensor NvJpeg::decode(const std::string& data, bool stream_sync = true) {
  auto orig_device = c10::cuda::current_device();
  c10::cuda::set_device(device_id);

  const auto* blob = (const unsigned char*)data.data();
  int nComponents;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  NVJPEG(nvjpegGetImageInfo(handle, blob, data.length(), &nComponents, &subsampling, widths, heights));
  
  if (!SupportedSubsampling(subsampling)) {
    throw std::invalid_argument("nvjpeg: not supported subsampling");
  }

  int h = heights[0];
  int w = widths[0];

  size_t image_size = h * w * 3;
  if (max_image_size < image_size) {
    std::ostringstream ss;
    ss << "image too large: " << image_size << " > max image size " << max_image_size;
    throw std::invalid_argument(ss.str());
  }

  auto options = at::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kUInt8)
                     .layout(torch::kStrided).requires_grad(false);
  auto image_tensor = at::empty({h, w, 3}, options, at::MemoryFormat::Contiguous);
  auto* image = image_tensor.data_ptr<unsigned char>();

  nvjpegImage_t nv_image;
  for (size_t i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
    nv_image.channel[i] = nullptr;
    nv_image.pitch[i] = 0;
  }
  nv_image.channel[0] = image;
  nv_image.pitch[0] = 3 * w;

  NVJPEG(nvjpegDecode(handle, state_dec, blob, data.length(), NVJPEG_OUTPUT_RGBI, &nv_image, cuda_stream))
  if (stream_sync) {
    cudaStreamSynchronize(cuda_stream);
  }
  c10::cuda::set_device(orig_device);
  return image_tensor;
}

py::bytes NvJpeg::encode(const torch::Tensor& image, int quality, std::string format)
{
  
  nvjpegEncoderParams_t encoder_params;
  nvjpegInputFormat_t input_format = NVJPEG_INPUT_RGBI;
  if(format=="BGR" || format=="bgr"){
    input_format = NVJPEG_INPUT_BGRI;
  }
  NVJPEG(nvjpegEncoderParamsCreate(handle, &encoder_params, NULL));
  NVJPEG(nvjpegEncoderParamsSetQuality(encoder_params, quality, NULL));
  NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encoder_params, 1, NULL));
  // NVJPEG(nvjpegEncoderParamsSetEncoding(encoder_params, nvjpegJpegEncoding_t::NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN, NULL));
  NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encoder_params, nvjpegChromaSubsampling_t::NVJPEG_CSS_444, NULL));

  int h = image.size(0);
  int w = image.size(1);
  auto* imageptr = image.data_ptr<unsigned char>();
  nvjpegImage_t nv_image;
  for (size_t i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
    nv_image.channel[i] = nullptr;
    nv_image.pitch[i] = 0;
  }
  nv_image.channel[0] = imageptr;
  nv_image.pitch[0] = 3 * w;
  // encode
  NVJPEG(nvjpegEncodeImage(handle, state_enc, encoder_params, &nv_image, input_format, w, h, NULL));
  
  size_t length;
  NVJPEG(nvjpegEncodeRetrieveBitstream(handle, state_enc, NULL, &length, NULL));

  char *obuffer = new char[length];
  NVJPEG(nvjpegEncodeRetrieveBitstream(handle, state_enc, (unsigned char*)obuffer, &length, NULL));
  NVJPEG(nvjpegEncoderParamsDestroy(encoder_params));
  // in function have copy
  return py::bytes((const char *)obuffer, length);
}


std::vector<torch::Tensor> NvJpeg::batch_decode(const std::vector<std::string>& data_list, bool stream_sync = true) {
  auto orig_device = c10::cuda::current_device();
  c10::cuda::set_device(device_id);
  // not implement
  int batch_size = data_list.size();
  if(batch_size!=batchsize_g){
    batchsize_g = batch_size;
    NVJPEG(nvjpegDecodeBatchedInitialize(handle, state_dec, batchsize_g, max_cpu_threads, NVJPEG_OUTPUT_RGBI));
  }
  std::vector<const unsigned char*> raw_inputs;
  std::vector<size_t> image_len_list;
  std::vector<torch::Tensor> tensor_list;
  std::vector<nvjpegImage_t> nv_image_list;
  raw_inputs.reserve(batch_size);
  image_len_list.reserve(batch_size);
  tensor_list.reserve(batch_size);
  nv_image_list.reserve(batch_size);
  int nComponents;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  for (const auto& data : data_list) {
    const auto* blob = (const unsigned char*)data.data();
    raw_inputs.emplace_back(blob);
    image_len_list.emplace_back(data.length());
    NVJPEG(nvjpegGetImageInfo(handle, blob, data.length(), &nComponents, &subsampling, widths, heights));
    if (!SupportedSubsampling(subsampling)) {
      throw std::invalid_argument("nvjpeg: not supported subsampling");
    }
    int h = heights[0];
    int w = widths[0];
    size_t image_size = h * w * 3;
    if (max_image_size < image_size) {
      std::ostringstream ss;
      ss << "image too large: " << image_size << " > max image size " << max_image_size;
      throw std::invalid_argument(ss.str());
    }
    auto image_tensor = torch::empty({h, w, 3},
        torch::TensorOptions().device(torch::kCUDA, device_id).dtype(torch::kUInt8)
          .layout(torch::kStrided).requires_grad(false));
    tensor_list.emplace_back(image_tensor);
    auto* image = image_tensor.data_ptr<unsigned char>();
    nvjpegImage_t nv_image;
    for (size_t i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
      nv_image.channel[i] = nullptr;
      nv_image.pitch[i] = 0;
    }
    nv_image.channel[0] = image;
    nv_image.pitch[0] = 3 * w;
    nv_image_list.emplace_back(nv_image);
  }
  NVJPEG(nvjpegDecodeBatched(handle, state_dec, raw_inputs.data(), image_len_list.data(), nv_image_list.data(), cuda_stream));
  if (stream_sync) {
    cudaStreamSynchronize(cuda_stream);
  }
  c10::cuda::set_device(orig_device);
  return tensor_list;
}

#ifdef PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<NvJpeg> nvjpeg(m, "NvJpeg");
  nvjpeg
      .def(
          py::init<int, py::object, size_t, size_t, int,  size_t, size_t, bool>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    Initialize nvjpeg.
                    Parameters:
                        device_id: int
                        stream: torch.cuda.Stream
                        max_image_size: int
                        batch_size: int
                        max_cpu_threads: int
                        device_padding: int
                        host_padding: int
                        gpu_huffman: bool
                )docdelimiter",
          py::arg("device_id") = 0,
          py::arg("stream") = py::none(),
          py::arg("max_image_size") = 3840 * 2160 * 3,
          py::arg("batch_size") = 2,
          py::arg("max_cpu_threads") = 1,
          py::arg("device_padding") = 0,
          py::arg("host_padding") = 0,
          py::arg("gpu_huffman") = true)
      .def(
          "decode",
          &NvJpeg::decode,
          py::call_guard<py::gil_scoped_release>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    Decode image to torch cuda tensor.
                    Parameters:
                        data: string, image bytes
                        stream_sync: bool, whether to do steam.synchronize()
                    Returns:
                        image cuda tensor in HWC foramt.
                )docdelimiter",
          py::arg("data"),
          py::arg("stream_sync") = true)
      .def(
          "batch_decode",
          &NvJpeg::batch_decode,
          py::call_guard<py::gil_scoped_release>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    Decode list of images to list of torch cuda tensor.
                    Parameters:
                        data: List[string], list of image bytes
                        stream_sync: bool, whether to do steam.synchronize()
                    Returns:
                        list of image cuda tensor in HWC foramt.
                )docdelimiter",
          py::arg("data"),
          py::arg("stream_sync") = true)
      .def(
          "encode",
          &NvJpeg::encode,
          py::call_guard<py::gil_scoped_release>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    encode torch cuda image to bytes.
                    Parameters:
                        img: torch.cuda.Tensor, image 
                        quality: int, encode quality 
                        format: string, 'RGB' or 'BGR'
                        stream_sync: bool, whether to do steam.synchronize()
                    Returns:
                        image cuda tensor in HWC foramt.
                )docdelimiter",
          py::arg("img"),
          py::arg("quality") = 75,
          py::arg("format") = "RGB")
      .def("get_device_id", &NvJpeg::get_device_id, py::return_value_policy::take_ownership);
}
#endif
} // namespace torchnvjpeg

inline std::string read_image(const std::string& image_path) {
  std::ifstream instream(image_path, std::ios::in | std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(instream)), std::istreambuf_iterator<char>());
  return data;
}



int main(int argc, const char** argv) 
{
  std::string image_path = "images/cat.jpg";
  if (argc > 1) {
    image_path = argv[1];
  }
  std::ifstream fin(image_path);
  if(!fin){
    std::cout<<"imgage is not exists\n";
    return 0;
  }
  fin.close();

  int device_id = 0;
  size_t max_size = 1920 * 1080 * 3;
  int max_cpu_threads = 4;
  auto image_data = read_image(image_path);

  int batch_size = 4;

  std::vector<std::string> data_list;
  data_list.reserve(batch_size);
  for (int i = 0; i < batch_size; i++) {
    data_list.emplace_back(image_data);
  }

  auto nvjpg = torchnvjpeg::NvJpeg(device_id, c10::cuda::getDefaultCUDAStream(), max_size, 8, 8);
  torch::Tensor t = nvjpg.decode(image_data);
  std::cout << "single deocde: " << t.device() << "\t\t" << t.sizes() << std::endl;

  auto tensor_list = nvjpg.batch_decode(data_list);
  std::cout << "batch decode:" << std::endl;
  for (auto& t : tensor_list) {
    std::cout << t.sizes() << std::endl;
  }

  // encode
  auto endata = nvjpg.encode(t);
  std::string outfile = "out.jpg";
  if (argc > 2) {
    outfile = argv[2];
  }
  std::string str(endata);
  std::cout << "single encode: " << str.length() << std::endl;  // 106306
  // dec->enc->dec
  torch::Tensor img = nvjpg.decode(str);
  std::cout << "dec->enc->dec: " << t.device() << "\t\t" << t.sizes() << std::endl;

  std::ofstream fout(outfile, std::ios::out|std::ios::binary);
  fout.write((const char*)str.c_str(), str.length());
  fout.close();
  printf("Done\n");
  return 0;
}