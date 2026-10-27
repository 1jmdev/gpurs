#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA(call)                                                             \
  do {                                                                         \
    cudaError_t result = (call);                                               \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(result));          \
      abort();                                                                 \
    }                                                                          \
  } while (0)

template <int operation>
__global__ void reference(const uint32_t *left, size_t left_count,
                          const uint32_t *right, size_t right_count,
                          uint32_t *output, size_t count) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  uint32_t a = left[index];
  uint32_t b = right[index];
  uint32_t value = 0;
  if constexpr (operation == 0)
    value = a;
  if constexpr (operation == 1)
    value = a + b;
  if constexpr (operation == 2)
    value = a - b;
  if constexpr (operation == 3)
    value = a * b;
  if constexpr (operation == 4)
    value = a / 7;
  if constexpr (operation == 5)
    value = a % 7;
  if constexpr (operation == 6)
    value = a & b;
  if constexpr (operation == 7)
    value = a | b;
  if constexpr (operation == 8)
    value = a ^ b;
  if constexpr (operation == 9)
    value = a << 3;
  if constexpr (operation == 10)
    value = a >> 3;
  if constexpr (operation == 11)
    value = a < b ? a : b;
  if constexpr (operation == 12)
    value = a > b ? a : b;
  if constexpr (operation == 13)
    value = a < b ? a + 3 : b + 5;
  if constexpr (operation == 14)
    value = (a & 1) ? a * 3 : b * 5;
  if constexpr (operation == 15)
    value = ((a + b) ^ (a >> 2)) * 3;
  if constexpr (operation == 16)
    value = (a == b) ? 1 : 0;
  if constexpr (operation == 17)
    value = ~a;
  if constexpr (operation == 18)
    value = left[left_count - 1 - index];
  if constexpr (operation == 19)
    value = left[(index + 1) % left_count];
  if constexpr (operation == 20) {
    value = a;
    for (int step = 0; step < 4; ++step)
      value = value * 3 + b;
  }
  if constexpr (operation == 21)
    value = (a & 255) + (b & 255);
  if constexpr (operation == 22)
    value = __float_as_uint(static_cast<float>(a) + static_cast<float>(b));
  if constexpr (operation == 23)
    value = __float_as_uint(static_cast<float>(a) * static_cast<float>(b));
  if constexpr (operation == 24)
    value = __float_as_uint(static_cast<float>(a) / static_cast<float>(b + 1));
  if constexpr (operation == 25)
    value = __float_as_uint((static_cast<float>(a) + 0.5f) *
                            (static_cast<float>(b) + 0.25f));
  output[index] = value;
}

struct BenchmarkGraph {
  cudaGraphExec_t executable;
  cudaEvent_t start, end;
};

extern "C" void *benchmark_capture_begin(void *stream) {
  auto graph = new BenchmarkGraph;
  CUDA(cudaEventCreate(&graph->start));
  CUDA(cudaEventCreate(&graph->end));
  CUDA(cudaStreamBeginCapture(static_cast<cudaStream_t>(stream),
                              cudaStreamCaptureModeRelaxed));
  CUDA(cudaEventRecordWithFlags(graph->start, static_cast<cudaStream_t>(stream),
                               cudaEventRecordExternal));
  return graph;
}

extern "C" void benchmark_capture_end(void *graph, void *stream) {
  auto capture = static_cast<BenchmarkGraph *>(graph);
  cudaGraph_t definition;
  CUDA(cudaEventRecordWithFlags(capture->end, static_cast<cudaStream_t>(stream),
                               cudaEventRecordExternal));
  CUDA(cudaStreamEndCapture(static_cast<cudaStream_t>(stream), &definition));
  CUDA(cudaGraphInstantiate(&capture->executable, definition, 0));
  CUDA(cudaGraphDestroy(definition));
  CUDA(cudaGraphUpload(capture->executable, static_cast<cudaStream_t>(stream)));
}

extern "C" float benchmark_graph_time(void *graph, void *stream,
                                       unsigned iterations) {
  auto capture = static_cast<BenchmarkGraph *>(graph);
  auto queue = static_cast<cudaStream_t>(stream);
  CUDA(cudaGraphLaunch(capture->executable, queue));
  CUDA(cudaStreamSynchronize(queue));
  CUDA(cudaGraphLaunch(capture->executable, queue));
  CUDA(cudaStreamSynchronize(queue));
  float milliseconds;
  CUDA(cudaEventElapsedTime(&milliseconds, capture->start, capture->end));
  return milliseconds / iterations;
}

extern "C" void benchmark_graph_destroy(void *graph) {
  auto capture = static_cast<BenchmarkGraph *>(graph);
  CUDA(cudaGraphExecDestroy(capture->executable));
  CUDA(cudaEventDestroy(capture->start));
  CUDA(cudaEventDestroy(capture->end));
  delete capture;
}

extern "C" float benchmark_reference(int operation, const uint32_t *left,
                                     const uint32_t *right, uint32_t *output,
                                     size_t count, unsigned block,
                                     unsigned iterations) {
  uint32_t *device_left, *device_right, *device_output;
  cudaStream_t stream;
  CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA(cudaMalloc(&device_left, count * sizeof(uint32_t)));
  CUDA(cudaMalloc(&device_right, count * sizeof(uint32_t)));
  CUDA(cudaMalloc(&device_output, count * sizeof(uint32_t)));
  CUDA(cudaMemcpyAsync(device_left, left, count * sizeof(uint32_t),
                       cudaMemcpyHostToDevice, stream));
  CUDA(cudaMemcpyAsync(device_right, right, count * sizeof(uint32_t),
                       cudaMemcpyHostToDevice, stream));
  unsigned grid = (count + block - 1) / block;
  auto launch = [&] {
    switch (operation) {
#define CASE(n)                                                                \
  case n:                                                                      \
    reference<n><<<grid, block, 0, stream>>>(device_left, count, device_right,    \
                                            count, device_output, count);       \
    break;
      CASE(0)
      CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9)
          CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16)
              CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23)
                  CASE(24) CASE(25)
#undef CASE
    }
    CUDA(cudaGetLastError());
  };
  for (int warmup = 0; warmup < 10; ++warmup)
    launch();
  CUDA(cudaStreamSynchronize(stream));
  void *graph = benchmark_capture_begin(stream);
  for (unsigned iteration = 0; iteration < iterations; ++iteration)
    launch();
  benchmark_capture_end(graph, stream);
  float milliseconds = benchmark_graph_time(graph, stream, iterations);
  benchmark_graph_destroy(graph);
  CUDA(cudaMemcpyAsync(output, device_output, count * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  CUDA(cudaFree(device_left));
  CUDA(cudaFree(device_right));
  CUDA(cudaFree(device_output));
  CUDA(cudaStreamDestroy(stream));
  return milliseconds;
}

__global__ void workload_matrix(const float *input, size_t input_count,
                                const float *weights, size_t weight_count,
                                float *output, size_t output_count, size_t size) {
  __shared__ float input_tile[256];
  __shared__ float weight_tile[256];
  size_t row = blockIdx.y * 16 + threadIdx.y;
  size_t column = blockIdx.x * 16 + threadIdx.x;
  size_t position = threadIdx.y * 16 + threadIdx.x;
  float total = 0.0f;
  for (size_t tile = 0; tile < (size + 15) / 16; ++tile) {
    size_t input_column = tile * 16 + threadIdx.x;
    size_t weight_row = tile * 16 + threadIdx.y;
    input_tile[position] = row < size && input_column < size
                               ? input[row * size + input_column]
                               : 0.0f;
    weight_tile[position] = weight_row < size && column < size
                                ? weights[weight_row * size + column]
                                : 0.0f;
    __syncthreads();
    for (size_t index = 0; index < 16; ++index)
      total += input_tile[threadIdx.y * 16 + index] *
               weight_tile[index * 16 + threadIdx.x];
    __syncthreads();
  }
  if (row < size && column < size)
    output[row * size + column] = total;
}

__global__ void workload_convolution(const float *input, size_t input_count,
                                     const float *weights, size_t weight_count,
                                     float *output, size_t output_count,
                                     size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < output_count) {
    size_t row = index / size;
    size_t column = index % size;
    float total = 0.0f;
    for (size_t y = 0; y < 5; ++y) {
      for (size_t x = 0; x < 5; ++x) {
        if (row + y >= 2 && row + y < size + 2 && column + x >= 2 &&
            column + x < size + 2)
          total += input[(row + y - 2) * size + column + x - 2] *
                   weights[y * 5 + x];
      }
    }
    output[index] = total;
  }
}

__global__ void workload_reduction(const float *input, size_t count,
                                   float *output, size_t output_count) {
  __shared__ float values[8];
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t thread = threadIdx.x;
  size_t lane = thread % 32;
  size_t warp = thread / 32;
  float value = index < count ? input[index] : 0.0f;
  for (int mask = 16; mask > 0; mask /= 2)
    value += __shfl_xor_sync(0xffffffff, value, mask);
  if (lane == 0)
    values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < 8 ? values[lane] : 0.0f;
    for (int mask = 16; mask > 0; mask /= 2)
      value += __shfl_xor_sync(0xffffffff, value, mask);
    if (lane == 0)
      output[blockIdx.x] = value;
  }
}

__global__ void workload_normalization(const float *input, size_t count,
                                       float *output, size_t output_count) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < output_count) {
    size_t start = index / 32 * 32;
    float sum = 0.0f;
    float squares = 0.0f;
    for (size_t offset = 0; offset < 32; ++offset) {
      if (start + offset < count) {
        float value = input[start + offset];
        sum += value;
        squares += value * value;
      }
    }
    float length = static_cast<float>(min(count - start, size_t(32)));
    float mean = sum / length;
    output[index] = (input[index] - mean) /
                    sqrtf(squares / length - mean * mean + 0.0001f);
  }
}

extern "C" float benchmark_workload(unsigned operation, const float *input,
                                    const float *weights, float *output,
                                    size_t size, unsigned iterations) {
  size_t count = size * size;
  size_t weight_count = max(count, size_t(25));
  size_t output_count = operation == 2 ? (count + 255) / 256 : count;
  float *device_input, *device_weights, *device_output;
  cudaStream_t stream;
  CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA(cudaMalloc(&device_input, count * sizeof(float)));
  CUDA(cudaMalloc(&device_weights, max(count, size_t(25)) * sizeof(float)));
  CUDA(cudaMalloc(&device_output, output_count * sizeof(float)));
  CUDA(cudaMemcpyAsync(device_input, input, count * sizeof(float),
                       cudaMemcpyHostToDevice, stream));
  CUDA(cudaMemcpyAsync(device_weights, weights, max(count, size_t(25)) * sizeof(float),
                       cudaMemcpyHostToDevice, stream));
  auto launch = [&] {
    if (operation == 0)
      workload_matrix<<<dim3((size + 15) / 16, (size + 15) / 16),
                        dim3(16, 16), 0, stream>>>(device_input, count, device_weights,
                                                   weight_count, device_output,
                                                   output_count, size);
    if (operation == 1)
      workload_convolution<<<(count + 255) / 256, 256, 0, stream>>>(
          device_input, count, device_weights, weight_count, device_output,
          output_count, size);
    if (operation == 2)
      workload_reduction<<<(count + 255) / 256, 256, 0, stream>>>(
          device_input, count, device_output, output_count);
    if (operation == 3)
      workload_normalization<<<(count + 255) / 256, 256, 0, stream>>>(
          device_input, count, device_output, output_count);
    CUDA(cudaGetLastError());
  };
  for (int warmup = 0; warmup < 5; ++warmup)
    launch();
  CUDA(cudaStreamSynchronize(stream));
  void *graph = benchmark_capture_begin(stream);
  for (unsigned iteration = 0; iteration < iterations; ++iteration)
    launch();
  benchmark_capture_end(graph, stream);
  float milliseconds = benchmark_graph_time(graph, stream, iterations);
  benchmark_graph_destroy(graph);
  CUDA(cudaMemcpyAsync(output, device_output, output_count * sizeof(float),
                       cudaMemcpyDeviceToHost, stream));
  CUDA(cudaStreamSynchronize(stream));
  CUDA(cudaFree(device_input));
  CUDA(cudaFree(device_weights));
  CUDA(cudaFree(device_output));
  CUDA(cudaStreamDestroy(stream));
  return milliseconds;
}
