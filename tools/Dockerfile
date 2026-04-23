FROM gemma4-challenge:latest

# Suppress NVIDIA banner / license splash on every container start.
RUN rm -f /opt/nvidia/entrypoint.d/10-banner.sh \
          /opt/nvidia/entrypoint.d/12-banner.sh \
          /opt/nvidia/entrypoint.d/15-container-copyright.txt \
          /opt/nvidia/entrypoint.d/30-container-license.txt

# uv for in-sandbox experimentation; openai SDK so runner.py can talk to the
# local LLM serving endpoint.
RUN pip install --no-cache-dir uv openai

# Agent scripts live inside the image (not under any mount path so they can't
# be shadowed). The `agent` subcommand uses the default `bash` CMD for an
# interactive shell; `agent-run` overrides it with runner.py.
COPY runner.py agent_tools.py system_prompt.md /workspace/agent/

WORKDIR /workspace
CMD ["bash"]
