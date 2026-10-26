#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdatomic.h>
#include <stddef.h>
#include <string.h>
#include <sys/uio.h>
#include <unistd.h>

static _Atomic int stopped = 0;

static void stop_if_gmem_pass(const void* buffer, size_t bytes) {
  static const char marker[] = "Cyclotron: gmem check";
  static const char pass[] = ": PASS";
  if (atomic_load(&stopped) || buffer == NULL) return;
  if (memmem(buffer, bytes, marker, sizeof(marker) - 1) != NULL &&
      memmem(buffer, bytes, pass, sizeof(pass) - 1) != NULL &&
      atomic_exchange(&stopped, 1) == 0) {
    raise(SIGSTOP);
  }
}

ssize_t write(int fd, const void* buffer, size_t bytes) {
  static ssize_t (*real_write)(int, const void*, size_t) = NULL;
  if (real_write == NULL) real_write = dlsym(RTLD_NEXT, "write");
  const ssize_t result = real_write(fd, buffer, bytes);
  if (fd == STDOUT_FILENO && result > 0)
    stop_if_gmem_pass(buffer, (size_t)result);
  return result;
}
ssize_t writev(int fd, const struct iovec* vectors, int count) {
  static ssize_t (*real_writev)(int, const struct iovec*, int) = NULL;
  if (real_writev == NULL) real_writev = dlsym(RTLD_NEXT, "writev");
  const ssize_t result = real_writev(fd, vectors, count);
  if (fd == STDOUT_FILENO && result > 0) {
    for (int i = 0; i < count; ++i)
      stop_if_gmem_pass(vectors[i].iov_base, vectors[i].iov_len);
  }
  return result;
}
