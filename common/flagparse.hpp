// common/flagparse.hpp
// CLI Flag Parsing Utilities

#pragma once
#include <cstring>

// Returns true if an exact flag (e.g., "--check") exists in argv.
bool flag_present(int argc, char** argv, const char* flag) {
  for (int i = 0; i < argc; i++) {
    if (std::strcmp(argv[i], flag)==0) 
      return true;
  }

  return false;
}

// Returns the value string following a key (e.g., "--M 4096"), or default if absent.
const char* flag_opt(int argc, char** argv, const char* key, const char* def) {
  for (int i = 0; i < argc-1; i++) {
    if (std::strcmp(argv[i], key)==0) 
      return argv[i+1];
  } 

  return def;
}