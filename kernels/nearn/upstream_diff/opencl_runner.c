#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { OPENCL_UNAVAILABLE = 77 };

typedef struct {
  float lat;
  float lng;
} LatLong;

static void fail(const char *message, cl_int status) {
  fprintf(stderr, "%s: OpenCL status %d\n", message, status);
  exit(2);
}

static void *read_exact(const char *path, size_t bytes) {
  FILE *stream = fopen(path, "rb");
  if (stream == NULL) {
    fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
    exit(2);
  }
  void *data = malloc(bytes == 0 ? 1 : bytes);
  if (data == NULL || fread(data, 1, bytes, stream) != bytes ||
      fgetc(stream) != EOF) {
    fprintf(stderr, "cannot read exactly %zu bytes from %s\n", bytes, path);
    exit(2);
  }
  fclose(stream);
  return data;
}

static char *read_source(const char *path, size_t *bytes_out) {
  FILE *stream = fopen(path, "rb");
  if (stream == NULL) {
    fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
    exit(2);
  }
  if (fseek(stream, 0, SEEK_END) != 0) exit(2);
  long length = ftell(stream);
  if (length < 0 || fseek(stream, 0, SEEK_SET) != 0) exit(2);
  char *source = malloc((size_t)length + 1);
  if (source == NULL || fread(source, 1, (size_t)length, stream) !=
                            (size_t)length) exit(2);
  fclose(stream);
  source[length] = '\0';
  *bytes_out = (size_t)length;
  return source;
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

int main(int argc, char **argv) {
  if (argc != 7) {
    fprintf(stderr,
            "usage: %s KERNEL INPUT OUTPUT RECORDS LAT_WORD LNG_WORD\n",
            argv[0]);
    return 2;
  }
  char *end = NULL;
  unsigned long parsed_records = strtoul(argv[4], &end, 10);
  if (end == argv[4] || *end != '\0' || parsed_records == 0 ||
      parsed_records > INT32_MAX) {
    fprintf(stderr, "invalid record count: %s\n", argv[4]);
    return 2;
  }
  const int records = (int)parsed_records;
  const size_t location_bytes = (size_t)records * sizeof(LatLong);
  const size_t output_bytes = (size_t)records * sizeof(float);
  LatLong *locations = read_exact(argv[2], location_bytes);
  float *distances = malloc(output_bytes);
  if (distances == NULL) return 2;
  const float query_lat = float_from_word(argv[5]);
  const float query_lng = float_from_word(argv[6]);

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
  if ((status = clGetPlatformIDs(platform_count, platforms, NULL)) !=
      CL_SUCCESS) fail("clGetPlatformIDs(list)", status);

  cl_device_id device = NULL;
  for (cl_uint i = 0; i < platform_count && device == NULL; ++i) {
    cl_uint count = 0;
    status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1, &device,
                            &count);
    if (status == CL_DEVICE_NOT_FOUND) device = NULL;
    else if (status != CL_SUCCESS) fail("clGetDeviceIDs", status);
  }
  if (device == NULL) {
    fprintf(stderr, "OpenCL unavailable: platforms expose no devices\n");
    return OPENCL_UNAVAILABLE;
  }

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
  cl_kernel kernel = clCreateKernel(program, "NearestNeighbor", &status);
  if (status != CL_SUCCESS) fail("clCreateKernel", status);
  cl_mem locations_buffer = clCreateBuffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, location_bytes,
      locations, &status);
  if (status != CL_SUCCESS) fail("clCreateBuffer(locations)", status);
  cl_mem distances_buffer = clCreateBuffer(
      context, CL_MEM_WRITE_ONLY, output_bytes, NULL, &status);
  if (status != CL_SUCCESS) fail("clCreateBuffer(distances)", status);
  status = clSetKernelArg(kernel, 0, sizeof(locations_buffer),
                          &locations_buffer);
  status |= clSetKernelArg(kernel, 1, sizeof(distances_buffer),
                           &distances_buffer);
  status |= clSetKernelArg(kernel, 2, sizeof(records), &records);
  status |= clSetKernelArg(kernel, 3, sizeof(query_lat), &query_lat);
  status |= clSetKernelArg(kernel, 4, sizeof(query_lng), &query_lng);
  if (status != CL_SUCCESS) fail("clSetKernelArg", status);
  const size_t global_size = (size_t)records;
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL,
                                  0, NULL, NULL);
  if (status != CL_SUCCESS) fail("clEnqueueNDRangeKernel", status);
  status = clEnqueueReadBuffer(queue, distances_buffer, CL_TRUE, 0,
                               output_bytes, distances, 0, NULL, NULL);
  if (status != CL_SUCCESS) fail("clEnqueueReadBuffer", status);

  FILE *output = fopen(argv[3], "wb");
  if (output == NULL || fwrite(distances, 1, output_bytes, output) !=
                            output_bytes || fclose(output) != 0) {
    fprintf(stderr, "cannot write %s\n", argv[3]);
    return 2;
  }
  clReleaseMemObject(distances_buffer);
  clReleaseMemObject(locations_buffer);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(source);
  free(platforms);
  free(distances);
  free(locations);
  return 0;
}
