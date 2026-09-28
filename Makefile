# ClickRank developer targets.
#
# These are shorthands for commands that already work. Nothing here is required
# and every target prints or is a single readable command, so anything can be
# copied and run by hand instead.
#
# Run make with no target for the list.

PYTHON ?= python3
COMPOSE ?= docker compose -f docker/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help install install-gpu test configs bench-cpu bench-cpu-sweep bench-gpu engines \
        docker-build docker-check docker-bench docker-engines docker-shell clean

help:
	@echo "ClickRank targets"
	@echo ""
	@echo "  install          Install the CPU dependencies"
	@echo "  install-gpu      Install the GPU extras on top, in the right order"
	@echo "  test             Run the test suite"
	@echo "  configs          Parse the sweep configs and the compose file"
	@echo ""
	@echo "  bench-cpu        Run the CPU inference benchmark with the cpu_only settings"
	@echo "  bench-cpu-sweep  The full cpu_only matrix. Needs the sweep flags in the script"
	@echo "  bench-gpu        The full gpu_full matrix. Needs a CUDA box"
	@echo "  engines          Build the TensorRT engines. Needs a CUDA box"
	@echo ""
	@echo "  docker-build     Build the pinned TensorRT image"
	@echo "  docker-check     Lint the Dockerfile without building it"
	@echo "  docker-engines   Build the engines inside the container"
	@echo "  docker-bench     Run the sweep inside the container"
	@echo "  docker-shell     Interactive shell inside the container"
	@echo ""
	@echo "  clean            Remove generated reports, charts, engines, and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

# The uninstall in the middle is not optional. onnxruntime and onnxruntime-gpu
# install the same module and leaving both on disk gives a session that
# advertises CUDA and then runs on the CPU. See requirements-gpu.txt.
install-gpu:
	$(PYTHON) -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip uninstall -y onnxruntime
	$(PYTHON) -m pip install -r requirements-gpu.txt

test:
	$(PYTHON) -m pytest -q

# Parses the sweep configs and the compose file. This is the same check CI runs.
# Needs pyyaml, which is a dev convenience and is deliberately not in
# requirements.txt because nothing in the library imports it.
configs:
	$(PYTHON) -c "import glob, yaml; [print('OK  ', p, yaml.safe_load(open(p))['name']) for p in sorted(glob.glob('benchmarks/*.yaml'))]"
	$(COMPOSE) config --quiet && echo "OK   docker/docker-compose.yml"

# The CPU run that works today, using only the flags the script already has.
# Settings come from benchmarks/cpu_only.yaml. Keep the two in step.
bench-cpu:
	$(PYTHON) scripts/run_inference_benchmark.py \
	  --data-path data/criteo.csv \
	  --sample-size 100000 \
	  --batch-size 1024 \
	  --repeats 20 \
	  --warmup 10 \
	  --checkpoint results/DeepFM.pt \
	  --output results/cpu_only

# The full cpu_only matrix. This needs the sweep flags, which are being added to
# the benchmark script. Until they land, use bench-cpu above.
bench-cpu-sweep:
	$(PYTHON) scripts/run_inference_benchmark.py \
	  --data-path data/criteo.csv \
	  --sample-size 100000 \
	  --repeats 20 \
	  --warmup 10 \
	  --checkpoint results/DeepFM.pt \
	  --output results/cpu_only \
	  --models deepfm \
	  --runtimes pytorch,onnxruntime-cpu,openvino \
	  --precisions fp32 \
	  --batch-sizes 1,256,1024,4096

# The full gpu_full matrix, transcribed from benchmarks/gpu_full.yaml. Run
# engines first so the engine build cost is measured on its own.
bench-gpu:
	$(PYTHON) scripts/run_inference_benchmark.py \
	  --data-path data/criteo.csv \
	  --sample-size 2000000 \
	  --repeats 50 \
	  --warmup 50 \
	  --checkpoint results/DeepFM.pt \
	  --output results/gpu_full \
	  --models deepfm \
	  --runtimes pytorch,onnxruntime-cuda,onnxruntime-trt,tensorrt,openvino \
	  --precisions fp32,fp16,int8 \
	  --batch-sizes 1,64,1024,4096,16384

engines:
	$(PYTHON) scripts/build_trt_engines.py

docker-build:
	$(COMPOSE) build

# Parses the Dockerfile, runs the BuildKit lint rules, and resolves the pinned
# base image metadata without pulling any layers. Runs anywhere, including on a
# machine with no GPU.
docker-check:
	docker buildx build --check -f docker/Dockerfile.tensorrt .

docker-engines:
	$(COMPOSE) run --rm engines

docker-bench:
	$(COMPOSE) run --rm bench

docker-shell:
	$(COMPOSE) run --rm shell

# Leaves the trained checkpoint alone, since retraining it is the slow part.
clean:
	rm -rf results/cpu_only results/gpu_full
	rm -f results/deepfm.onnx results/inference_benchmark.md results/inference_latency.png
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
