// softmax/include/records.hpp
// Benchmark Record Generators

#include <string>
#include <cstdio>

static inline std::string make_csv_record(
  const char* name,
  int M, int N, int iters,
  double ms_avg, double gflops,
  double max_abs_err, double rel_err
) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
    "%s,%d,%d,%d,%.5f,%.3f,%.3e,%.3e",
    name, M, N, iters, ms_avg, gflops, max_abs_err, rel_err);
  return std::string(buf);
}

static inline std::string make_json_record(
  const char* name,
  int M, int N, int iters,
  double ms_avg, double gflops,
  double max_abs_err, double rel_err
) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
    "{\"name\":\"%s\",\"M\":%d,\"N\":%d,\"iters\":%d,"
    "\"ms_avg\":%.5f,\"gflops\":%.3f,\"max_abs_err\":%.3e,\"rel_err\":%.3e}",
    name, M, N, iters, ms_avg, gflops, max_abs_err, rel_err);
  return std::string(buf);
}