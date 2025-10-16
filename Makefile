BUILD_DIR ?= build
CONFIG    ?= Release
ARCHS     ?= 120;90
TARGET    ?= gemm_bench
CMAKE     ?= cmake

# CMake configure (runs if build has been cleaned)
$(BUILD_DIR)/CMakeCache.txt:
	$(CMAKE) -B $(BUILD_DIR) -S . \
	  -DCMAKE_BUILD_TYPE=$(CONFIG) \
	  -DCMAKE_CUDA_ARCHITECTURES="$(ARCHS)"

.PHONY: build
build: $(BUILD_DIR)/CMakeCache.txt
	$(CMAKE) --build $(BUILD_DIR) -j

.PHONY: clean-build
clean-build:
	rm -rf $(BUILD_DIR)

.PHONY: clean-results
clean-results:
	rm -rf results

.PHONY: clean
clean: clean-build clean-results

# Rebuild from scratch
.PHONY: rebuild
rebuild: clean-build build
