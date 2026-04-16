---
name: croq-env-fix
description: Auto-fix GPU profiling environment for croq-tune. Fixes perf_event_paranoid, CUDA PATH, and nvidia profiling restrictions. Load from croq-baseline when environment validation fails.
---

# Croq-Env-Fix

Automatically remediate known environment blockers for `croq-tune` sessions.

## When to Load

Load this skill when `croq-baseline` environment validation fails on any of:

1. `kernel.perf_event_paranoid > 2`
2. `nvcc` not on PATH
3. `RmProfilingAdminOnly != 0`

## Usage

Run the fix script:

```bash
bash .cursor/skills/croq-env-fix/fix-env.sh
```

The script:

1. Detects and fixes `perf_event_paranoid` (requires sudo — will prompt once)
2. Installs a sudoers drop-in so future sessions fix it without a password
3. Finds CUDA toolkit and exports `PATH` and `LD_LIBRARY_PATH`
4. Validates `RmProfilingAdminOnly` and fixes if needed
5. Prints a sourceable env block for the current shell

## Integration with croq-baseline

When `croq-baseline` environment validation fails, it should:

1. Load this skill
2. Run `fix-env.sh`
3. Source the env output
4. Re-run validation

## Fix Details

### perf_event_paranoid

- Checks current value via `/proc/sys/kernel/perf_event_paranoid`
- If > 2: runs `sudo sysctl -w kernel.perf_event_paranoid=2`
- Installs `/etc/sudoers.d/croq-perf` so `sysctl` for this param is passwordless going forward
- Installs `/etc/sysctl.d/99-croq-perf.conf` so the fix persists across reboots

### CUDA PATH

- Searches for nvcc in: `/usr/local/cuda/bin`, `/usr/local/cuda-*/bin`
- Picks the newest CUDA version found
- Exports `PATH` and `LD_LIBRARY_PATH` additions

### RmProfilingAdminOnly

- Reads `/proc/driver/nvidia/params`
- If `RmProfilingAdminOnly != 0`: creates modprobe config and reloads nvidia modules
- This fix requires sudo and a brief GPU interruption

## Output

The script exits 0 on success with all fixes applied. Exit 1 if any fix requires
user intervention (e.g., password prompt failed). The script prints machine-readable
status lines prefixed with `[croq-env-fix]`.

## Idempotency

Safe to run multiple times. Each fix checks current state before acting.
