# Sandbox image for the claude-worker plugin.
#
# Built and tagged OUT-OF-BAND as ``policy.SANDBOX_IMAGE_TAG``
# ("claude-worker-sandbox:2.1.237").  Activation records the resulting
# immutable image ID in ``policy.SANDBOX_IMAGE_ID``; runner.py executes only
# that ID and merely checks that the human-readable tag still resolves to it.
# If either is absent or mismatched, every spawn refuses closed.
#
# The image pins the Claude Code CLI to the version baked into the tag, so
# a host-side CLI upgrade can never silently change what actually runs
# inside the sandbox.  Runtime identity is the resulting image sha256, not
# the mutable tag.

FROM node:22-slim@sha256:d649c27dae7ba0137b3cef5dd75baa422c08dc3d9e3fc0c23dfb172dc3cc6436

ARG CLAUDE_CLI_VERSION=2.1.237

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CLI_VERSION}" \
    && npm cache clean --force

# Matches policy.CONTAINER_HOME — runner.py mounts the operator's OAuth
# credentials read-only at $HOME/.claude/.credentials.json inside this home.
RUN useradd --create-home --home-dir /home/worker --uid 10001 --shell /usr/sbin/nologin worker

USER worker
WORKDIR /workspace
ENV HOME=/home/worker

# runner.py always passes --entrypoint explicitly (claude for the worker
# spawn, the fixed operator verification argv for a verification run), so
# this default is never actually relied on — it exists only so the image is
# self-describing when run manually.
ENTRYPOINT ["claude"]
