#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

enum { OPENCL_UNAVAILABLE = 77 };
static const rlim_t MEMORY_LIMIT_BYTES = (rlim_t)4 * 1024 * 1024 * 1024;

static void enforce_memory_limit(void) {
  const struct rlimit limit = {MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES};
  if (setrlimit(RLIMIT_AS, &limit) != 0) {
    fprintf(stderr, "setrlimit(RLIMIT_AS): %s\n", strerror(errno));
    exit(2);
  }
}

static void fail(const char *message, cl_int status) {
  fprintf(stderr, "%s: OpenCL status %d\n", message, status);
  exit(2);
}

static void *read_exact(const char *path, size_t bytes) {
  FILE *stream = fopen(path, "rb");
  void *data = malloc(bytes == 0 ? 1 : bytes);
  if (stream == NULL || data == NULL ||
      fread(data, 1, bytes, stream) != bytes || fgetc(stream) != EOF) {
    fprintf(stderr, "cannot read exactly %zu bytes from %s\n", bytes, path);
    exit(2);
  }
  fclose(stream);
  return data;
}

static char *read_source(const char *path, size_t *bytes_out) {
  FILE *stream = fopen(path, "rb");
  if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) {
    fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
    exit(2);
  }
  long length = ftell(stream);
  if (length < 0 || fseek(stream, 0, SEEK_SET) != 0) exit(2);
  char *source = malloc((size_t)length + 1);
  if (source == NULL ||
      fread(source, 1, (size_t)length, stream) != (size_t)length) exit(2);
  fclose(stream);
  source[length] = '\0';
  *bytes_out = (size_t)length;
  return source;
}

static unsigned long parse_positive(const char *text, const char *name) {
  char *end = NULL;
  unsigned long value = strtoul(text, &end, 10);
  if (end == text || *end != '\0' || value == 0 || value > INT32_MAX) {
    fprintf(stderr, "invalid %s: %s\n", name, text);
    exit(2);
  }
  return value;
}

static float float_from_word(const char *text) {
  char *end = NULL;
  unsigned long raw = strtoul(text, &end, 16);
  if (end == text || *end != '\0' || raw > UINT32_MAX) {
    fprintf(stderr, "invalid float word: %s\n", text);
    exit(2);
  }
  uint32_t word = (uint32_t)raw;
  float value;
  memcpy(&value, &word, sizeof(value));
  return value;
}

static void write_step(const char *prefix, unsigned step, const float *data,
                       size_t bytes) {
  char path[4096];
  int length = snprintf(path, sizeof(path), "%s-step-%u.bin", prefix, step);
  if (length < 0 || (size_t)length >= sizeof(path)) {
    fprintf(stderr, "output path is too long\n");
    exit(2);
  }
  FILE *stream = fopen(path, "wb");
  if (stream == NULL || fwrite(data, 1, bytes, stream) != bytes ||
      fclose(stream) != 0) {
    fprintf(stderr, "cannot write %s\n", path);
    exit(2);
  }
}

static void print_device_field(cl_device_id device, cl_device_info field,
                               const char *label) {
  size_t bytes = 0;
  if (clGetDeviceInfo(device, field, 0, NULL, &bytes) != CL_SUCCESS ||
      bytes == 0) return;
  char *value = calloc(bytes + 1, 1);
  if (value == NULL) return;
  if (clGetDeviceInfo(device, field, bytes, value, NULL) == CL_SUCCESS)
    printf("%s=%s\n", label, value);
  free(value);
}

int main(int argc, char **argv) {
  enforce_memory_limit();
  if (argc != 17) {
    fprintf(stderr,
            "usage: %s KERNEL POWER INITIAL OUTPUT_PREFIX NX NY NZ ITERATIONS "
            "SDC CE CW CN CS CT CB CC\n", argv[0]);
    return 2;
  }
  const int nx = (int)parse_positive(argv[5], "nx");
  const int ny = (int)parse_positive(argv[6], "ny");
  const int nz = (int)parse_positive(argv[7], "nz");
  const int iterations = (int)parse_positive(argv[8], "iterations");
  if ((nx % 4) != 0 || (ny % 4) != 0 || nz < 2) {
    fprintf(stderr, "fixture dimensions violate upstream launch constraints\n");
    return 2;
  }
  const size_t elements = (size_t)nx * (size_t)ny * (size_t)nz;
  if (elements > SIZE_MAX / sizeof(float)) return 2;
  const size_t bytes = elements * sizeof(float);
  float *power = read_exact(argv[2], bytes);
  float *initial = read_exact(argv[3], bytes);
  float *stage = malloc(bytes);
  if (stage == NULL) return 2;
  const float sdc = float_from_word(argv[9]);
  const float ce = float_from_word(argv[10]);
  const float cw = float_from_word(argv[11]);
  const float cn = float_from_word(argv[12]);
  const float cs = float_from_word(argv[13]);
  const float ct = float_from_word(argv[14]);
  const float cb = float_from_word(argv[15]);
  const float cc = float_from_word(argv[16]);

  cl_uint platform_count = 0;
  cl_int status = clGetPlatformIDs(0, NULL, &platform_count);
  if (status == CL_PLATFORM_NOT_FOUND_KHR ||
      (status == CL_SUCCESS && platform_count == 0)) {
    fprintf(stderr, "OpenCL unavailable: no platform (status %d)\n", status);
    return OPENCL_UNAVAILABLE;
  }
  if (status != CL_SUCCESS) fail("clGetPlatformIDs", status);
  cl_platform_id *platforms = malloc(platform_count * sizeof(*platforms));
  if (platforms == NULL) return 2;
  status = clGetPlatformIDs(platform_count, platforms, NULL);
  if (status != CL_SUCCESS) fail("clGetPlatformIDs(list)", status);
  cl_device_id device = NULL;
  for (cl_uint index = 0; index < platform_count && device == NULL; ++index) {
    status = clGetDeviceIDs(platforms[index], CL_DEVICE_TYPE_CPU, 1,
                            &device, NULL);
    if (status == CL_DEVICE_NOT_FOUND) device = NULL;
    else if (status != CL_SUCCESS) fail("clGetDeviceIDs", status);
  }
  if (device == NULL) {
    fprintf(stderr, "OpenCL unavailable: platforms expose no CPU device\n");
    return OPENCL_UNAVAILABLE;
  }
  print_device_field(device, CL_DEVICE_NAME, "OPENCL_DEVICE_NAME");
  print_device_field(device, CL_DEVICE_VENDOR, "OPENCL_DEVICE_VENDOR");
  print_device_field(device, CL_DEVICE_VERSION, "OPENCL_DEVICE_VERSION");

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  if (status != CL_SUCCESS) fail("clCreateContext", status);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
  if (status != CL_SUCCESS) fail("clCreateCommandQueue", status);
  size_t source_bytes = 0;
  char *source = read_source(argv[1], &source_bytes);
  const char *source_pointer = source;
  cl_program program = clCreateProgramWithSource(
      context, 1, &source_pointer, &source_bytes, &status);
  if (status != CL_SUCCESS) fail("clCreateProgramWithSource", status);
  status = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
  if (status != CL_SUCCESS) {
    size_t log_bytes = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_bytes);
    char *log = calloc(log_bytes + 1, 1);
    if (log != NULL) {
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_bytes,
                            log, NULL);
      fprintf(stderr, "OpenCL build log:\n%s\n", log);
    }
    fail("clBuildProgram", status);
  }
  cl_kernel kernel = clCreateKernel(program, "hotspotOpt1", &status);
  if (status != CL_SUCCESS) fail("clCreateKernel", status);
  cl_mem power_buffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, power, &status);
  if (status != CL_SUCCESS) fail("clCreateBuffer(power)", status);
  cl_mem buffers[2];
  buffers[0] = clCreateBuffer(
      context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, initial,
      &status);
  if (status != CL_SUCCESS) fail("clCreateBuffer(input)", status);
  buffers[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &status);
  if (status != CL_SUCCESS) fail("clCreateBuffer(output)", status);

  const size_t global[2] = {(size_t)nx, (size_t)ny};
  const size_t local[2] = {4, 4};
  int input_index = 0;
  for (int step = 1; step <= iterations; ++step) {
    int output_index = 1 - input_index;
    status = clSetKernelArg(kernel, 0, sizeof(power_buffer), &power_buffer);
    if (status == CL_SUCCESS)
      status = clSetKernelArg(kernel, 1, sizeof(buffers[input_index]),
                              &buffers[input_index]);
    if (status == CL_SUCCESS)
      status = clSetKernelArg(kernel, 2, sizeof(buffers[output_index]),
                              &buffers[output_index]);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 3, sizeof(sdc), &sdc);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 4, sizeof(nx), &nx);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 5, sizeof(ny), &ny);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 6, sizeof(nz), &nz);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 7, sizeof(ce), &ce);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 8, sizeof(cw), &cw);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 9, sizeof(cn), &cn);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 10, sizeof(cs), &cs);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 11, sizeof(ct), &ct);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 12, sizeof(cb), &cb);
    if (status == CL_SUCCESS) status = clSetKernelArg(kernel, 13, sizeof(cc), &cc);
    if (status != CL_SUCCESS) fail("clSetKernelArg", status);
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local,
                                    0, NULL, NULL);
    if (status != CL_SUCCESS) fail("clEnqueueNDRangeKernel", status);
    status = clEnqueueReadBuffer(queue, buffers[output_index], CL_TRUE, 0,
                                 bytes, stage, 0, NULL, NULL);
    if (status != CL_SUCCESS) fail("clEnqueueReadBuffer", status);
    write_step(argv[4], (unsigned)step, stage, bytes);
    input_index = output_index;
  }

  clReleaseMemObject(buffers[1]);
  clReleaseMemObject(buffers[0]);
  clReleaseMemObject(power_buffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(source);
  free(platforms);
  free(stage);
  free(initial);
  free(power);
  return 0;
}
