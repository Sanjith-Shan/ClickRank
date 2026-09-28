# Running the GPU Inference Benchmark

Everything needed to reproduce the ClickRank GPU inference numbers on a CUDA machine, starting from a box you have just rented and ending with the results back on your laptop. Commands are meant to be copied as written.

The image pins `nvcr.io/nvidia/tensorrt:25.01-py3`, which ships TensorRT 10.8.0.43 on CUDA 12.8.0.038 and expects driver 570.86.10 or newer. The pin is exact on purpose. A TensorRT engine is tied to the TensorRT version, the CUDA version, and the GPU architecture that built it, so floating the tag would change the builder under the benchmark and make two runs of the same command produce numbers that cannot be compared. If your host driver is older than 570, read the driver section near the bottom before building anything.

## What You Need

An x86_64 Linux host with an NVIDIA GPU, a driver of 570 or newer, Docker, and the NVIDIA Container Toolkit. About 40 GB of free disk for the image and the layers. No CUDA toolkit on the host, since it lives inside the image.

## Check the Host First

Two commands, in this order. The first confirms the host driver, the second confirms Docker can hand a GPU to a container at all.

```bash
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/tensorrt:25.01-py3 nvidia-smi
```

The first must print a driver version of 570 or newer. The second must print the same GPU from inside a container. If the first works and the second does not, the problem is the NVIDIA Container Toolkit and not the driver, and the fix is on the host rather than in this repository.

```bash
# Install the container toolkit on Ubuntu if that second command failed.
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Clone and Build

```bash
git clone https://github.com/Sanjith-Shan/ClickRank.git
cd ClickRank

docker compose -f docker/docker-compose.yml build
```

The build takes roughly fifteen to thirty minutes on a first run, most of which is pulling the base image and the CUDA build of PyTorch. It ends with a smoke check that prints the torch, TensorRT, and ONNX Runtime versions and the available ONNX Runtime providers, so a broken install fails at build time rather than forty minutes into a rented hour.

To build without compose, which is the same thing with the arguments spelled out.

```bash
docker build -f docker/Dockerfile.tensorrt -t clickrank-trt:25.01 .
```

## Verify the GPU Is Visible Inside the Image

Do this before running anything long. It costs ten seconds and it catches the failure mode where the benchmark runs to completion on the CPU and produces a table of GPU numbers that are not GPU numbers.

```bash
docker compose -f docker/docker-compose.yml run --rm shell -lc '
  nvidia-smi
  python3 -c "import torch; print(\"torch cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python3 -c "import tensorrt; print(\"tensorrt\", tensorrt.__version__)"
  python3 -c "import onnxruntime; print(\"providers\", onnxruntime.get_available_providers())"
  python3 -c "import pynvml; pynvml.nvmlInit(); print(\"nvml driver\", pynvml.nvmlSystemGetDriverVersion())"
'
```

Four things have to be true. `torch.cuda.is_available()` prints True and names your GPU. `tensorrt.__version__` prints 10.8.0.43. The provider list contains both `CUDAExecutionProvider` and `TensorrtExecutionProvider`. NVML reports the host driver. If the provider list is missing `TensorrtExecutionProvider` while `import tensorrt` works, the `onnxruntime-gpu` pin in `requirements-gpu.txt` was built against a different TensorRT minor version than the container ships, and one of the two pins has to move.

## Get the Data

The benchmark falls back to the synthetic generator when no real Criteo file is present, which is fine for a smoke test and wrong for a published number. For the real thing, pull the data on the host before running, because the download helper needs network and the results directory is bind mounted anyway.

```bash
bash scripts/download_data.sh 2000000
```

That streams a public mirror and stops the transfer once it has the rows it needs, so it costs about 1 GB rather than the full 11 GB. The file lands at `data/criteo.csv`, which the compose file mounts into the container.

## Build the Engines

Engines are built in their own step, before any timing run. Engine build time is a real cost and it is reported on its own, so it must never land inside an inference measurement.

```bash
docker compose -f docker/docker-compose.yml run --rm engines
```

This is the slow step. TensorRT benchmarks candidate kernels on the actual device to pick the fastest one for every layer, so a build takes minutes per precision. The INT8 build also runs a calibration pass over validation rows, which is the reason it takes longest. Engines land in `results/` on the host through the bind mount, so a second sweep can reuse them.

## Run the Sweep

```bash
docker compose -f docker/docker-compose.yml run --rm bench
```

The defaults reproduce the default configuration. To run the full matrix described by `benchmarks/gpu_full.yaml`, pass the flags that config records.

```bash
docker compose -f docker/docker-compose.yml run --rm bench \
  python3 scripts/run_inference_benchmark.py \
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
```

Reports, charts, and engines all land in `results/` on the host.

## Pull the Results Back

Everything worth keeping is already on the host filesystem because of the bind mount, so this is an ordinary copy off the box. From your laptop, not from the remote.

```bash
# Results only, without the multi hundred megabyte engines and checkpoints.
rsync -avz --progress -e "ssh -p <PORT>" \
  --include='*/' --include='*.md' --include='*.png' --include='*.json' --exclude='*' \
  root@<HOST>:/workspace/ClickRank/results/ ./results/

# Or everything, engines included.
rsync -avz --progress -e "ssh -p <PORT>" root@<HOST>:/workspace/ClickRank/results/ ./results/
```

The engines are worth leaving behind. They are large, they are specific to the GPU that built them, and they will not load on a different architecture.

## RunPod, Start to Finish

The pod facts below were measured on a real pod on 2026-08-28, not taken from documentation. Where this section contradicts a vendor page, trust this section.

**Pick the card, and do not reach for an A100 by reflex.** The TensorRT numbers are new numbers with nothing they have to stay comparable against, so they do not need the biggest card. A cheaper GPU produces an equally valid runtime comparison at a fraction of the burn rate. Reserve the expensive cards for work that genuinely needs them.

Choose **Secure Cloud rather than Community Cloud**. Community hosts are shared and shared hosts make noisy latency numbers, which is the one thing a benchmark cannot tolerate.

**If you do use an A100, it will be SXM and not PCIe.** As of 2026-08-28 the PCIe variant shows low availability and does not offer CUDA 12.8, only 12.5. The SXM4 80GB does offer 12.8. These are different parts with different memory bandwidth, so an SXM number is labeled SXM and is never presented as comparable to a PCIe number from another project.

**The verified image.** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, which was confirmed to carry driver 570.172.08 and `torch 2.8.0+cu128` with `cuda.is_available()` returning True. Because torch is already present and already built against CUDA 12.8, do not reinstall it. The torch pin in `requirements-gpu.txt` exists for the container path and reinstalling it here would be a slow downgrade for no benefit.

**TensorRT is not on that image.** This is the one thing that will stop the run. The image ships torch and CUDA but no TensorRT, so it has to be installed before anything in this repository will build an engine.

**Three environment facts that cost time if you do not know them.**

CUDA is not on `PATH`. Both `nvcc` and `ncu` exist under `/usr/local/cuda/bin` but nothing points at them, so any check of the form `which nvcc` will wrongly conclude the toolkit is missing. Export the path first.

The image marks its Python externally managed under PEP 668, so a plain `pip install` refuses to run. Use `--break-system-packages` or work inside a virtual environment.

The sshd inside the pod lags the RUNNING status by roughly two to three minutes. The pod will report itself running with a live uptime counter while SSH is still refusing connections, so poll the actual `ssh` command rather than the status field.

**Nsight Compute hardware counters are blocked and this is not fixable from inside the pod.** The `ncu` binary is present and it will attach to the process, then fail at the counter read with `ERR_NVGPUCTRPERM`. RunPod containers are unprivileged and are not granted `CAP_SYS_ADMIN`. Nothing in this repository depends on those counters. Memory traffic is derived analytically from tensor shapes in `src/inference/analysis.py` instead, which is reproducible and needs no elevated permission. Nsight Systems does not need the counter permission and is expected to work.

**Do not develop on the pod.** GPU time bills by the second and most of a working session is thinking rather than computing. Write code on your own machine, sync it up, run the measurement, pull the results back. The scripts under `deploy/runpod/` do exactly this and are the supported path.

```bash
# On the pod, once. Installs TensorRT and the telemetry packages the image lacks.
bash deploy/runpod/bootstrap.sh

# From your laptop. Syncs, runs the sweep, pulls results back.
bash deploy/runpod/sync_and_run.sh --host <IP> --port <PORT>
```

Note that this path installs `tensorrt` from PyPI rather than inheriting it from the NGC container, which is the one real difference from the containerized run. A result produced this way is labeled with the host driver and the TensorRT version that `bootstrap.sh` prints, rather than with the container tag, because the container was not involved.

**Terminate the pod, do not stop it.** A stopped pod can still bill for storage. Anything that must survive termination belongs on the network volume at `/workspace`, because the container disk is wiped.

```bash
# From your laptop. The port and host come from the pod's SSH panel.
rsync -avz -e "ssh -p <PORT>" root@<HOST>:/workspace/ClickRank/results/ ./results/
```

## When the Driver Is Too Old for the Pinned Image

The base image expects driver 570.86.10 or newer. An older driver produces a startup message from the NVIDIA entrypoint saying the detected driver is insufficient for the CUDA version in the image, and then either a hard failure or, worse, a fallback that silently does not use the GPU. There are three ways out, in order of preference.

**Upgrade the host driver.** This is the correct fix when you control the machine. The driver is backward compatible with older CUDA versions, so upgrading it does not break anything else on the box.

**Use CUDA forward compatibility.** The image already carries the forward compatibility libraries at `/usr/local/cuda/compat` and they are already first on `LD_LIBRARY_PATH`, so on a supported configuration this works with no changes. The catch is that forward compatibility only works on data center GPUs such as the A100, H100, L40S, and the Tesla line, and it does not work on GeForce or workstation cards. Confirm it took effect from inside the container.

```bash
docker run --rm --gpus all clickrank-trt:25.01 \
  python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Move the pin down to a tag your driver supports.** This is the honest fallback and it is one build argument, not a code change. Older NGC tags target older CUDA versions and therefore older drivers.

```bash
docker compose -f docker/docker-compose.yml build \
  --build-arg TRT_IMAGE=nvcr.io/nvidia/tensorrt:24.08-py3 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
  --build-arg TORCH_VERSION=2.4.1
```

Two things have to happen if you do this. The TensorRT assertion in the Dockerfile will fail, because it checks for the 10.8 line, so update that check to the version the new tag ships. And every number produced from a moved pin has to be labeled with the new TensorRT and CUDA versions rather than the ones in this README, because engines from different TensorRT versions are different artifacts and their timings are not interchangeable. Read the real versions off the new tag rather than guessing.

```bash
docker run --rm nvcr.io/nvidia/tensorrt:24.08-py3 env | grep -E 'TENSORRT_VERSION|CUDA_VERSION|CUDA_DRIVER_VERSION|CUDNN_VERSION'
```

## Common Failures

**A bus error or a crash inside a DataLoader at large batch sizes.** The Docker default of 64 MB for `/dev/shm` is too small. The compose file already sets `shm_size` to 8 GB. If you are running `docker run` by hand, add `--shm-size=8g`.

**`CUDAExecutionProvider` appears in `get_available_providers()` but the session runs on the CPU.** Both `onnxruntime` and `onnxruntime-gpu` are installed and one is shadowing the other. The Dockerfile removes the CPU package on purpose. Outside the container, run the uninstall step from `requirements-gpu.txt`.

**Latency numbers that look impossibly good on the GPU.** Almost always a missing device synchronize before the timer stops, which measures the kernel launch rather than the kernel. See the benchmarking section of `docs/INFERENCE.md`.

**A TensorRT engine that fails to deserialize.** Engines are not portable across TensorRT versions or GPU architectures. Delete the cached engine in `results/` and rebuild it on the machine you are running on.
