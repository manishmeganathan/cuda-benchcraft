// gemm/include/records.hpp
// Benchmark Record Generators 

#include <string>

// Creates a CSV benchmark record
std::string make_csv_record(
  const char* name,
  int M, int N, int K, int iters,
  double ms_avg, double gflops,
  double max_abs_err, double rel_err
) {
  char buf[512];

  std::snprintf(buf, sizeof(buf),
    "%s,%d,%d,%d,%d,%.5f,%.3f,%.3e,%.3e",
    name, M, N, K, iters, ms_avg, gflops, max_abs_err, rel_err);

  return std::string(buf);
}

// Creates a JSON benchmark record
std::string make_json_record(
  const char* name,
  int M, int N, int K, int iters,
  double ms_avg, double gflops,
  double max_abs_err, double rel_err
) {
  char buf[512];

  std::snprintf(buf, sizeof(buf),
    "{\"name\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"iters\":%d,"
    "\"ms_avg\":%.5f,\"gflops\":%.3f,\"max_abs_err\":%.3e,\"rel_err\":%.3e}",
    name, M, N, K, iters, ms_avg, gflops, max_abs_err, rel_err);

  return std::string(buf);
}