FROM nvcr.io/nvidia/pytorch:25.11-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        coinor-libipopt-dev \
        default-jdk-headless \
        gfortran \
        git \
        graphviz \
        libfftw3-dev \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# PAL_SRC: name of the pal checkout directory inside the build context.
ARG PAL_SRC=pal
COPY ${PAL_SRC} /workspace/pal

RUN pip install --no-cache-dir --upgrade "pip<25.2" setuptools wheel && \
    python - <<'PY'
import pathlib
import tomllib

pyproject = pathlib.Path("/workspace/pal/pyproject.toml")
deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
filtered = []
for dep in deps:
    if dep.startswith("torch>="):
        continue
    if dep.startswith("jax[cpu]"):
        continue
    if dep == "torchvision":
        continue
    filtered.append(dep)
pathlib.Path("/tmp/pal_ipopt.requirements.txt").write_text(
    "".join(f"{dep}\n" for dep in filtered)
)
PY

RUN pip install --no-cache-dir -r /tmp/pal_ipopt.requirements.txt

RUN pip install --no-cache-dir -e /workspace/pal --no-deps

# CUDA JAX keeps the e2 surrogate on GPU while IPOPT runs on CPU.
RUN pip install --no-cache-dir --upgrade "jax[cuda12_local]" && \
    pip install --no-cache-dir cyipopt==1.6.1

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/pal

WORKDIR /workspace/pal
