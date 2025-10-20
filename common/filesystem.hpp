// common/filesystem.hpp
// File Handling Utilities

#pragma once
#include <string>
#include <sys/stat.h>

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
bool ensure_path_dirs(const std::string& path) {
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

// Appends one text line to file at path (adds '\n' if missing). 
// Creates parent directories if needed.
bool append_file_line(const std::string& path, const std::string& line) {
  // Null or empty string paths
  if (path.empty()) return false;

  // Ensure that parent path exists
  if (!ensure_path_dirs(path)) return false;

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