// src/util.cu
// Utilities: deterministic vector init + CLI utilities

#include <random>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

// Fills a vector with deterministic Uniform(-1.0, 1.0) floats (mt19937 + seed).
// Note: same seed ⇒ same values; use different seeds for A and B to avoid structure.
void fill_vector(std::vector<float>& v, unsigned seed=1234) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float x : v) x = dist(gen);
}

// Returns true if an exact flag (e.g., "--check") exists in argv.
bool flag_present(int argc, char** argv, const char* flag) {
  for (int i = 0; i < argc; i++) if (std::strcmp(argv[i], flag)==0) return true;
  return false;
}

// Returns the value string following a key (e.g., "--M 4096"), or def if absent.
const char* opt_value(int argc, char** argv, const char* key, const char* def) {
  for (int i = 0; i < argc-1; i++) if (std::strcmp(argv[i], key)==0) return argv[i+1];
  return def;
}