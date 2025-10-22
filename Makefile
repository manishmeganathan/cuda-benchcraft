BUILD_DIR ?= build
CONFIG    ?= Release
ARCHS     ?= 86;90 # Ampere & Ada
TARGET    ?= gemm_bench
VENV      ?= .venv
PIP       := $(VENV)/bin/pip
PYTHON    := $(VENV)/bin/python

.PHONY: build rebuild clean clean-build clean-results clean-venv venv deps craft

# CMake configure (runs if build has been cleaned)
$(BUILD_DIR)/CMakeCache.txt:
	cmake -B $(BUILD_DIR) -S . \
	  -DCMAKE_BUILD_TYPE=$(CONFIG) \
	  -DCMAKE_CUDA_ARCHITECTURES="$(ARCHS)"

build: $(BUILD_DIR)/CMakeCache.txt deps
	cmake --build $(BUILD_DIR) -j

clean-build:
	rm -rf $(BUILD_DIR)

clean-results:
	rm -rf results

clean-venv:
	rm -rf "$(VENV)"

clean: clean-build clean-results clean-venv

rebuild: clean-build build

venv:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"

deps: venv
	if [ -f requirements.txt ]; then \
	  "$(PIP)" install --upgrade pip && \
	  "$(PIP)" install -r requirements.txt ; \
	else \
	  echo "requirements.txt not found; skipping Python deps."; \
	fi

craft: venv
	@"$(PYTHON)" benchcraft.py
