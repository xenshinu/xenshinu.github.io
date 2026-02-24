---
layout: single
title: "Claude Code in a Docker sandbox (kept alive with tmux)"
date: 2026-02-23
author_profile: true
comments: true
tags: [Docker, tmux, Claude]
---

I wanted a **YOLO sandbox** for Claude Code: isolated dependencies, optional CUDA/PyTorch, and a way to keep the agent running even when my SSH session drops. My setup is: **Docker for isolation + tmux inside the container** for persistence.

## Prereqs

- Docker installed
- If you want GPU: Linux + NVIDIA drivers + NVIDIA Container Toolkit
- If you don’t have GPU access: drop `--gpus all` and use a non-CUDA base image (e.g. `ubuntu:24.04`)

Quick GPU sanity check:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## 1) Docker image (CUDA + uv + Claude Code + tmux)

This Dockerfile creates an `ubuntu` user, installs `tmux`, installs `uv` + Claude Code, and builds a Python venv (including CUDA PyTorch as an example).

```dockerfile
# CUDA dev image (includes compilers + headers). If you don’t need GPU, switch to e.g. ubuntu:24.04.
FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive

# Create a non-root user so the container behaves more like a real dev machine.
# UID=1000 matches the common host user id (helps avoid permission pain on bind mounts).
ARG USER=ubuntu
ARG UID=1000
RUN useradd -m -u "${UID}" -s /bin/bash "${USER}"

# Minimal set of tools I tend to want in a sandbox: build tooling, git, tmux, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      git \
      nano \
      ninja-build \
      numactl \
      tmux && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu
# Put the venv in $HOME and ensure both venv + uv are on PATH for interactive shells.
ENV VIRTUAL_ENV="/home/ubuntu/.venv" \
  PATH="/home/ubuntu/.venv/bin:/home/ubuntu/.local/bin:${PATH}" \
  TERM=xterm-256color \
  COLORTERM=truecolor \
  LANG=en_US.UTF-8

# Install uv (fast Python package manager) and Claude Code CLI.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    curl -fsSL https://claude.ai/install.sh | bash

# Example Python env. Keep/remove depending on whether you need PyTorch in this sandbox.
RUN uv venv "${VIRTUAL_ENV}" --python 3.12 && \
    uv pip install cmake && \
    uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
      --index-url https://download.pytorch.org/whl/cu128

# Optional: a nicer tmux UX for long agent runs.
COPY --chown=ubuntu:ubuntu .tmux.conf /home/ubuntu/.tmux.conf

WORKDIR /workspace
CMD ["/bin/bash"]
```

## 2) Build + run helper script

Drop this into `docker/run.sh`, then `chmod +x docker/run.sh`.

Assuming you use the same layout, put `Dockerfile` and `.tmux.conf` under `docker/` as well (the script builds using `docker/` as the build context).

Replace `IMAGE_NAME` / `CONTAINER_NAME` if you want.

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile"

IMAGE_NAME="claude-sandbox:latest"
CONTAINER_NAME="claude-sandbox"

build() {
  docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE_PATH}" "${SCRIPT_DIR}"
}

run() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker run -dit \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --shm-size=64g \
    -v "${HOST_DIR}:/workspace:rw" \
    -w /workspace \
    "${IMAGE_NAME}"
  echo "Container started: ${CONTAINER_NAME}"
}

attach() {
  local user="${1:-ubuntu}"
  if [[ "${user}" == "root" ]]; then
    docker exec -it -u root -w /workspace "${CONTAINER_NAME}" /bin/bash
  else
    docker exec -it -u ubuntu -w /workspace "${CONTAINER_NAME}" /bin/bash -lc \
      "source /home/ubuntu/.venv/bin/activate && exec /bin/bash"
  fi
}

case "${1:-all}" in
  build)  build ;;
  run)    run ;;
  attach) attach "${2:-ubuntu}" ;;
  all)    build && run ;;
  *)      echo "Usage: $0 {build|run|attach [root]|all}" ;;
esac
```

## 3) tmux: let Claude keep running

Attach, start a tmux session, run Claude, detach, and reattach later:

```bash
./docker/run.sh
./docker/run.sh attach

tmux new -s claude
claude
```

On the first run you’ll likely need to authenticate:

```bash
claude login
```

Detach with `Ctrl-b d`, then later:

```bash
./docker/run.sh attach
tmux attach -t claude
```

If you’re intentionally sandboxing and accept the risk, you can also run:

```bash
claude --dangerously-skip-permissions
```

## Appendix: `.tmux.conf` I use

```bash
# Apply with: tmux source-file ~/.tmux.conf

set -g default-terminal "xterm-256color"
set-option -ga terminal-overrides ",xterm-256color:Tc"

set -g mouse on
set -g history-limit 10000

setw -q -g utf8 on

# Catppuccin-ish Mocha status bar
set -g status-style "bg=#1e1e2e,fg=#cdd6f4"
set -g window-status-current-style "bg=#89b4fa,fg=#1e1e2e,bold"
set -g pane-border-style "fg=#313244"
set -g pane-active-border-style "fg=#89b4fa"
set -g message-style "bg=#313244,fg=#cdd6f4"

set -g status-left "#[bg=#89b4fa,fg=#1e1e2e,bold] #S #[bg=#1e1e2e,fg=#89b4fa]"
set -g status-right "#[fg=#f5c2e7]#(whoami) #[fg=#89b4fa]│ #[fg=#cba6f7]%Y-%m-%d %H:%M #[bg=#89b4fa,fg=#1e1e2e,bold] #h "
set -g status-left-length 50
set -g status-right-length 100
```