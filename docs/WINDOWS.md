# Windows notes — large models, mmap, and pread

Rabbit's expert path prefers `mmap` for speed. On Windows (and on flaky
FUSE/network mounts under WSL), full-file mapping of multi-GB `.rabbit`
files can fail or later raise I/O errors.

## Recommended settings

```bat
set RABBIT_NO_MMAP=1
set RABBIT_CACHE_SLOTS=64
rabbit.exe chat --model models\olmoe-v2.rabbit --tokenizer models\olmoe.tok
```

Or PowerShell:

```powershell
$env:RABBIT_NO_MMAP = "1"
$env:RABBIT_CACHE_SLOTS = "64"
.\rabbit.exe chat --model models\olmoe-v2.rabbit --tokenizer models\olmoe.tok
```

## What `RABBIT_NO_MMAP=1` does

- Skips `mmap` of the model file.
- Keeps an open file descriptor and loads experts with **`pread`** into the
  expert cache on miss (see v0.4.24).
- Shared-expert pinning still works.

## WSL / network drives

1. Prefer copying the `.rabbit` file to a local NTFS or ext4 path (`/tmp`,
   `C:\Models\...`) before chat.
2. If you must keep the file on a network share, always set `RABBIT_NO_MMAP=1`.
3. Warm the cache once so subsequent tokens hit RAM:

```bat
set RABBIT_NO_MMAP=1
rabbit.exe warm --model models\olmoe-v2.rabbit --tokens 16
rabbit.exe chat --model models\olmoe-v2.rabbit
```

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Instant crash or `mmap` error on load | Large model on network/FUSE drive | `RABBIT_NO_MMAP=1` + copy to local disk |
| Slow first tokens, then normal | Cold expert cache | Run `rabbit warm` once |
| High RAM use / OOM on 4 GB machine | Too many cache slots + large KV | Lower `RABBIT_CACHE_SLOTS` and `RABBIT_REF_V2_CTX` |
| `rabbit.bat` fails to find binary | Not built yet | Run `make` under WSL or rebuild |

## Build

Use the included `rabbit.bat` / `rabbit.ps1` helpers, or build under WSL with
the normal `Makefile` (`make`). The pread path is POSIX (`pread`); on native
Win32 without WSL, map or copy experts through the C runtime read path as
needed — WSL is the supported large-model path today.

## Related env vars

| Variable | Purpose |
|----------|---------|
| `RABBIT_NO_MMAP=1` | Expert load via pread + cache |
| `RABBIT_CACHE_SLOTS=N` | Expert cache capacity (max 64) |
| `RABBIT_REF_V2_CTX=N` | KV window (16–256) for low-RAM decode |
| `RABBIT_MEM_BUDGET_MB` | Dense + cache budget |
