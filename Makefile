BUILD_DIR ?= build
CONFIG    ?= Release
ARCHS     ?= 120;90
TARGET    ?= gemm_bench
CMAKE     ?= cmake

# Default run params (used only if no flags are passed after 'make run')
M      ?= 1024
N      ?= 1024
K      ?= 1024
ITERS  ?= 10
KIND   ?= all

.PHONY: help
help:
	@echo "Targets:"
	@echo "  make build    [ARCHS=120;90 CONFIG=Release]"
	@echo "  make run      usage: make run -- --M 1024 --N 1024 --K 1024 --iters 10 --kind all | make run -- --help"
	@echo "  make list"
	@echo "  make rebuild  (clean + build)"
	@echo "  make clean"

$(BUILD_DIR)/CMakeCache.txt:
	$(CMAKE) -B $(BUILD_DIR) -S . \
	  -DCMAKE_BUILD_TYPE=$(CONFIG) \
	  -DCMAKE_CUDA_ARCHITECTURES="$(ARCHS)"

.PHONY: build
build: $(BUILD_DIR)/CMakeCache.txt
	$(CMAKE) --build $(BUILD_DIR) -j

# Capture extra words after 'make run' and treat them as args
RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
# Create no-op rules for those words so make doesn't look for targets
$(eval $(RUN_ARGS):;@:)
# If no extra args were passed, fall back to defaults
RUN_CMD = $(if $(RUN_ARGS),$(RUN_ARGS),--M $(M) --N $(N) --K $(K) --iters $(ITERS) --kind $(KIND))

.PHONY: run
run: build
	./$(BUILD_DIR)/$(TARGET) $(RUN_CMD)

.PHONY: list
list: build
	./$(BUILD_DIR)/$(TARGET) --list

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: rebuild
rebuild: clean build




