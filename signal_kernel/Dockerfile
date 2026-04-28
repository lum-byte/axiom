# =============================================================================
# Dockerfile — signal_kernel execution container
# =============================================================================
#
# Base: Alpine 3.19 (pinned by digest for reproducible builds).
# Role: Run one grep/sed/awk pipeline (Process 1) spawned by pipeline.py.
#       Run mft_checkpoint.sh every 15 minutes (Process 2, crond).
#
# What this image deliberately EXCLUDES:
#   - No Python, Node, build tools, compilers, shells beyond BusyBox /bin/sh
#   - No COPY of recipe files  — mounted at runtime via volume bind at /recipe/
#   - No COPY of Python source — pipeline.py lives outside the container
#   - No EXPOSE               — network_mode=none; no ports, ever
#   - No ENV                  — subprocess environment injected by pipeline.py
#   - No USER                 — Alpine default; isolation model is the boundary
#   - No WORKDIR              — entrypoint owns its own working directory
#
# Runtime volume requirements (enforced by entrypoint, not here):
#   /recipe/run.sh   — recipe script, bind-mounted read-only by pipeline.py
#   /store/          — TAG index files + checkpoint output, bind-mounted rw
#   /var/log/        — checkpoint and entrypoint log output
#
# Build:
#   docker build --no-cache -t signal-kernel:$(git rev-parse --short HEAD) .
#
# The image is intentionally narrow. Every added layer is an attack surface.
# =============================================================================

# Pin to digest for byte-for-byte reproducible builds.
# To refresh: docker pull alpine:3.19 && docker inspect --format='{{index .RepoDigests 0}}' alpine:3.19
FROM alpine:3.19@sha256:7144f7bab3d4c2648d7e59409f15ec52a18006a128c733fcff20d3a4a54ba44a

# ── OCI image labels ──────────────────────────────────────────────────────────
# These are read by Docker, container registries, and audit tooling.
# org.opencontainers.image.* is the standard namespace.
LABEL org.opencontainers.image.title="signal-kernel" \
      org.opencontainers.image.description="Stateless grep/sed/awk pipeline container for TAG signal extraction" \
      org.opencontainers.image.base.name="alpine:3.19" \
      org.opencontainers.image.licenses="proprietary" \
      com.axiom.component="signal-kernel" \
      com.axiom.process.1="recipe pipeline (grep/sed/awk, stdin → stdout)" \
      com.axiom.process.2="mft checkpoint (crond, /etc/periodic/15min)"

# ── Exactly three packages ────────────────────────────────────────────────────
# grep  — POSIX-extended regex line filtering (GNU grep for -P support)
# sed   — stream editor; in-place and pipe transformations
# gawk  — GNU awk; recipes may use gawk-specific features (match arrays, etc.)
#
# crond is BusyBox built-in. No additional install required.
# --no-cache avoids the apk index layer being committed to the image.
# --virtual is not used because these are runtime deps, not build-time.
RUN apk add --no-cache \
      grep \
      sed \
      gawk

# ── Checkpoint script ─────────────────────────────────────────────────────────
# BusyBox crond runs /etc/periodic/15min/* on every 15-minute mark.
# The script must be executable and have no file extension — crond on
# BusyBox skips files whose names contain a dot (e.g. mft_checkpoint.sh
# would be silently ignored). The name mft_checkpoint has no dot.
#
# Source path matches the build context layout:
#   checkpoint/mft_checkpoint.sh  →  /etc/periodic/15min/mft_checkpoint
COPY checkpoint/mft_checkpoint.sh /etc/periodic/15min/mft_checkpoint
RUN chmod 0755 /etc/periodic/15min/mft_checkpoint \
 && chown root:root /etc/periodic/15min/mft_checkpoint

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Copied from build context root. Not inside checkpoint/ — it is the
# container's own init, not a checkpoint concern.
COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh \
 && chown root:root /entrypoint.sh

# ── Health check ─────────────────────────────────────────────────────────────
# Not HTTP — network_mode=none. Instead: verify crond is alive and the
# recipe mount point exists. Fails open (exit 1) rather than crashing
# the container — health failure is a signal for the orchestrator.
#
# --interval: check every 30 seconds (recipe runs are sub-5-second; crond
#   is the slow signal). --timeout: 5 seconds is more than enough for a
#   process check. --start-period: give crond 10 seconds to start before
#   health is evaluated. --retries: 3 consecutive failures = unhealthy.
HEALTHCHECK \
  --interval=30s \
  --timeout=5s \
  --start-period=10s \
  --retries=3 \
  CMD [ "/bin/sh", "-c", \
        "pgrep -x crond > /dev/null && test -e /recipe/run.sh" ]

# ── Entrypoint (exec form, not shell form) ────────────────────────────────────
# Exec form: PID 1 is /entrypoint.sh directly. No shell wrapper.
# This means SIGTERM from `docker stop` reaches the entrypoint immediately
# without going through a shell that might absorb or delay it.
ENTRYPOINT ["/entrypoint.sh"]


# cd ~/Axiom/signal_kernel
# cp ~/Axiom/signal_kernel/recipes/hardcoded/saas_docs.sh ~/Axiom/signal_kernel/recipes/active/run.sh
# cp /mnt/c/Users/Kanishk/PycharmProjects/Axiom/signal_kernel/contracts.py ~/Axiom/signal_kernel
# cp -r /mnt/c/Users/Kanishk/PycharmProjects/Axiom/tag/ ~/Axiom
# docker compose up -d