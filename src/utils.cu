// src/util.cu
// Utilities: deterministic vector init + CLI utilities

#include <random>
#include <vector>
#include <string>
#include <cstring>
#include <cerrno>
#include <cstdio>

#include "utilities.hpp" 

#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>

// Fills a vector with deterministic Uniform(-1.0, 1.0) floats (mt19937 + seed).
// Note: same seed ⇒ same values; use different seeds for A and B to avoid structure.
void fill_vector(std::vector<float>& v, unsigned seed=1234) {
  // Create random distribution between -1.0 and 1.0 with given seed
  std::mt19937 gen(seed); 
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Create random value for the vector from the distribution
  for (float& x : v) x = dist(gen); 
}

// Returns true if an exact flag (e.g., "--check") exists in argv.
bool flag_present(int argc, char** argv, const char* flag) {
  for (int i = 0; i < argc; i++) {
    if (std::strcmp(argv[i], flag)==0) 
      return true;
  }

  return false;
}

// Returns the value string following a key (e.g., "--M 4096"), or def if absent.
const char* flag_opt(int argc, char** argv, const char* key, const char* def) {
  for (int i = 0; i < argc-1; i++) {
    if (std::strcmp(argv[i], key)==0) 
      return argv[i+1];
  } 

  return def;
}

// Recursively create directories for a given POSIX-style path (like `mkdir -p`)
static bool mkdir_p(const std::string& dir) {
  // Root or empty path
  if (dir.empty() || dir == "/") return true;

  // If path ends with '/', normalize and recurse
  if (dir.back() == '/') return mkdir_p(dir.substr(0, dir.size()-1));

  struct stat st{};
  // Check if directory already exists
  if (stat(dir.c_str(), &st) == 0) 
    return S_ISDIR(st.st_mode);

  // Isolate parent path and recurse path creation
  auto slash = dir.find_last_of('/');
  if (slash != std::string::npos) {
    if (!mkdir_p(dir.substr(0, slash))) 
      return false;
  }

  // Attempt directory creation
  if (mkdir(dir.c_str(), 0755) == 0) return true;
  if (errno == EEXIST) return true; // Creation failed because it already exists

  return false;
}

// Ensures parent directories for 'path' exist (best-effort)
bool ensure_parent_dirs(const std::string& path) {
  // Null or empty string paths
  if (path.empty()) return false;

  std::string p(path);
  // Isolate parent path
  auto slash = p.find_last_of('/');
  // No parent -> current path, nothing to do
  if (slash == std::string::npos) return true;

  // Create parent directories
  return mkdir_p(p.substr(0, slash));
}

// Appends one text line to 'path' (adds '\n' if missing). 
// Creates parent dirs if needed.
bool append_text_line(const std::string& path, const std::string& line) {
  // Null or empty string paths
  if (path.empty()) return false;

  // Ensure that parent path exists
  if (!ensure_parent_dirs(path)) return false;

  // Open the file in append mode
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) return false; // fail

  // Empty line
  if (line.empty()) {
    std::fputc('\n', f);
    std::fclose(f);

    return true;
  }

  // Write the text to file
  std::fwrite(line.data(), 1, line.size(), f);
  // Append newline if not included in line
  if (line[line.size() - 1] != '\n') 
    std::fputc('\n', f);

  std::fclose(f);
  return true;
}

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