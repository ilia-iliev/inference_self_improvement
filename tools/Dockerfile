FROM gemma4-challenge:latest

# Suppress NVIDIA banner / license splash on every container start.
RUN rm -f /opt/nvidia/entrypoint.d/10-banner.sh \
          /opt/nvidia/entrypoint.d/12-banner.sh \
          /opt/nvidia/entrypoint.d/15-container-copyright.txt \
          /opt/nvidia/entrypoint.d/30-container-license.txt

# Profilers for the agent to use against the solution container.
# Version-pinned to the CUDA 12.8 toolkit (matches the base image). py-spy
# is pip-only (pinned below).
RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-nsight-systems-12-8 cuda-nsight-compute-12-8 \
    && rm -rf /var/lib/apt/lists/*

# uv for in-sandbox experimentation; openai SDK for the LLM endpoint;
# httpx for evaluate.py; py-spy for profiling. Versions pinned for reproducibility.
RUN pip install --no-cache-dir \
      uv==0.11.7 \
      openai==2.32.0 \
      httpx==0.28.1 \
      py-spy==0.4.0

# Docker CLI (static binary) so the agent can drive the host daemon via the
# mounted /var/run/docker.sock to build and run the solution container.
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz \
      | tar -xzC /usr/local/bin --strip-components=1 docker/docker \
    && docker --version

# Agent scripts live inside the image (not under any mount path so they can't
# be shadowed).
COPY runner.py agent_tools.py system_prompt.md /workspace/agent/

# Harness scripts referenced by the system prompt. These live at /workspace/
# (not /workspace/solution/) so the agent cannot clobber them.
COPY start_inference.py evaluate.py /workspace/

WORKDIR /workspace
CMD ["bash"]
