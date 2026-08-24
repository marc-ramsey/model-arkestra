"""GPU and CPU hardware detection for ModelArkestra init.

Probes the system for available accelerators (AMD/NVIDIA GPUs, CPU)
and determines which backends are supported. Called during:
    model-arkestra init

Detection methods use only stdlib + common CLI tools (lspci, nvidia-smi,
rocm-smi, vulkaninfo, lscpu). No external deps.
"""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

# Display-friendly vendor names
_VENDOR_NAMES = {
    "nvidia": "NVIDIA",
    "amd": "AMD",
    "intel": "Intel",
    "unknown": "Unknown",
}


# ── Runtime availability checks ────────────────────────────────────


def has_vulkan() -> bool:
    """Check if Vulkan runtime is available."""
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, timeout=5, text=True,
        )
        return result.returncode == 0 and "Vulkan Instance Version" in result.stdout
    except (FileNotFoundError, OSError):
        return False


def has_rocm() -> bool:
    """Check if ROCm runtime is available."""
    # Check for rocm-smi first (most reliable indicator)
    try:
        subprocess.run(["rocm-smi", "--showconfig"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, OSError):
        pass

    # Fallback: check for ROCm lib directory
    rocm_paths = [
        Path("/opt/rocm/lib"),
        Path("/usr/lib64/rocm"),
        Path("/usr/lib/x86_64-linux-gnu/rocm"),
    ]
    return any(p.exists() for p in rocm_paths)


def has_nvidia() -> bool:
    """Check if NVIDIA CUDA runtime is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"],
            capture_output=True, timeout=5, text=True,
        )
        return result.returncode == 0 and "NVIDIA" in result.stdout
    except (FileNotFoundError, OSError):
        return False


def has_cpu_binary() -> bool:
    """Check if a llama-server CPU binary exists or can be downloaded.

    Returns True if the binary is already on PATH or in a known location.
    During init, this always returns True because init will download it.
    """
    # Check common locations where a pre-built binary might live
    for candidate in [
        Path("/usr/local/bin/llama-server"),
        Path.home() / ".local" / "bin" / "llama-server",
    ]:
        if candidate.exists():
            return True
    return False  # Will be downloaded during model start via backends.yaml


# ── ROCm GFX architecture detection ────────────────────────────────

# Maps exact gfx IDs (from rocm-smi rocminfo) to lemonade-sdk build families.
# lemonade-sdk ships per-family binaries: gfx110X, gfx1151, etc.
_GFX_TO_FAMILY = {
    # RDNA 4 — Strix Halo desktop / mobile
    "gfx1151": "gfx1151",
    "gfx1152": "gfx1151",   # some SKUs share the build
    # RDNA 3 — Strix Point iGPU (Ryzen AI 9 HX 370)
    "gfx1150": "gfx1150",
    # RDNA 3 — Desktop: 7800XT/7900XTX, Laptop: 7600M/7700M
    "gfx1102": "gfx110X",
    "gfx1103": "gfx110X",
    # RDNA 4 — Desktop: Radeon 9070 series
    "gfx1200": "gfx120X",
    "gfx1201": "gfx120X",
    # RDNA 3 — Desktop: RX 6500/6600/6700/7600 (Navi-3)
    "gfx1032": "gfx103X",
    # CDNA / GCN legacy
    "gfx906": "gfx906",      # Vega
    "gfx908": "gfx908",      # MI50/MI25
    "gfx90a": "gfx90a",      # MI300X / MI250X
    "gfx940": "gfx942",      # CDNA 2 (MI250X) — use latest available
    "gfx941": "gfx942",
    "gfx942": "gfx942",
}


def detect_gfx_version() -> str | None:
    """Detect the exact GFX microarchitecture via rocm-smi.

    Returns e.g. 'gfx1151', or None if ROCm tools unavailable / no GPU found.
    Falls back to lspci PCI device ID for AMD GPUs as a secondary method.
    """
    # Primary: rocm-smi --showproductname → GFX Version
    result = subprocess.run(
        ["rocm-smi", "--showproductname"],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "GFX Version:" in line:
                raw = line.split(":")[-1].strip()
                return _resolve_gfx_family(raw)

    # Secondary: lspci + PCI device ID lookup
    result = _run_cmd(["lspci", "-nnk"])
    if result and result.stdout:
        for line in result.stdout.splitlines():
            if not any(kw in line.lower() for kw in ("vga", "3d controller", "display controller")):
                continue
            if "amd" not in line.lower() and "ati" not in line.lower():
                continue
            # Extract device ID like [1002:1586]
            import re
            m = re.search(r"\[1002:([0-9a-fA-F]{4})\]", line)
            if m:
                dev_id = m.group(1).lower()
                return _pci_to_gfx_family(dev_id)

    return None


def _resolve_gfx_family(raw_version: str) -> str | None:
    """Map a raw gfx version string (e.g. 'gfx1151') to the build family."""
    ver = raw_version.strip().lower()
    if ver not in _GFX_TO_FAMILY:
        return None
    return _GFX_TO_FAMILY[ver]


def _pci_to_gfx_family(dev_id: str) -> str | None:
    """Map an AMD PCI device ID to a GFX family for the lemonade-sdk build."""
    _PCI_TO_GFX = {
        # RDNA 4 — Strix Halo
        "1586": "gfx1151",   # Radeon 8050S / 8060S
        # RDNA 3 — Desktop
        "743c": "gfx110X",   # RX 7900 XTX
        "743f": "gfx110X",   # RX 7900 XT
        "741f": "gfx110X",   # RX 7800 XT
        "742f": "gfx110X",   # RX 7700 XT
        "747f": "gfx110X",   # RX 7600 XT
        "739f": "gfx110X",   # RX 7600
        "73ef": "gfx110X",   # RX 7500 XT
        # RDNA 2 — Desktop / Laptop
        "164e": "gfx103X",   # RX 6600/6600XT
        "1638": "gfx103X",   # RX 6700 XT / 6800 XT
        "164c": "gfx103X",   # RX 6900 XT
        # RDNA 4 — Laptop
        "155f": "gfx1150",   # Ryzen AI 9 HX 370 (Strix Point)
    }
    return _PCI_TO_GFX.get(dev_id)


def _run_cmd(cmd: list[str], timeout: int = 5) -> subprocess.CompletedProcess[str] | None:
    """Run a command and return result, or None on failure."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (FileNotFoundError, OSError):
        return None


def _detect_gpus() -> list[dict[str, Any]]:
    """Discover all GPUs via lspci and classify them.

    Returns list of dicts with keys: vendor, pci_id, name, backend_hint
    """
    result = _run_cmd(["lspci"])
    if not result or not result.stdout.strip():
        return []

    gpus = []
    for line in result.stdout.splitlines():
        # Match VGA/3D/Display controllers (case-insensitive)
        if not any(kw in line.lower() for kw in ("vga", "3d controller", "display controller")):
            continue

        parts = line.split(" ", 1)
        pci_id = parts[0].strip() if parts else ""
        desc = parts[1].strip() if len(parts) > 1 else line.strip()

        # Classify vendor → backend hint
        lower = desc.lower()
        if "nvidia" in lower:
            vendor = "nvidia"
            backend = "cuda"
        elif "advanced micro devices" in lower or "amd/ati" in lower or "raadeon" in lower:
            vendor = "amd"
            # AMD Strix Halo / iGPU: rocm is preferred, vulkan-radv works too
            if "strix halo" in lower or "ryzen ai" in lower:
                backend = "rocm"  # prefer ROCm for iGPU
            else:
                backend = "vulkan-radv"  # safe default for discrete AMD GPUs
        elif "intel" in lower:
            vendor = "intel"
            backend = "vulkan-radv"
        else:
            vendor = "unknown"
            backend = "vulkan-radv"  # safest fallback

        gpus.append({
            "vendor": vendor,
            "pci_id": pci_id,
            "name": desc,
            "backend": backend,
        })

    return gpus


# ── CPU detection ──────────────────────────────────────────────────


def _detect_cpu() -> dict[str, Any]:
    """Probe CPU capabilities and architecture.

    Returns dict with: arch, vendor, cores, features, compatible_backends
    """
    info: dict[str, Any] = {
        "arch": platform.machine(),
        "vendor": "unknown",
        "cores": 0,
        "features": [],
        "compatible_backends": ["cpu"],
    }

    # Get CPU features from /proc/cpuinfo
    try:
        cpuinfo = (Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else "")
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                info["vendor"] = "Intel" if "intel" in line.lower() else "AMD"
                # Try to extract core/thread count
                if "@" in line:
                    parts = line.split("@")
                    # model name often contains core counts, e.g. "8-core"
                    for kw in ("core",):
                        if kw in parts[0].lower():
                            try:
                                num = int(parts[0].split(kw)[0].strip().split()[-1])
                                info["cores"] = num * 2  # assume hyperthreading
                            except (ValueError, IndexError):
                                pass

    except (OSError, IOError):
        pass

    # Use lscpu as fallback for core count
    if info["cores"] == 0:
        result = _run_cmd(["lscpu"])
        if result and result.stdout:
            for line in result.stdout.splitlines():
                if "CPU(s):" in line or "Core(s) per socket:" in line:
                    try:
                        info["cores"] = int(line.split(":")[1].strip().replace(",", ""))
                    except (ValueError, IndexError):
                        pass

    # Detect CPU instruction set features
    if cpuinfo:
        for feat in ("avx2", "avx512", "avx512vnni", "bmi2", "fma3", "aes", "sha"):
            if f" {feat}" in cpuinfo or f"\n{feat}" in cpuinfo:
                info["features"].append(feat)

    # Add basic features that are nearly universal on modern x86_64
    if info["arch"] == "x86_64":
        for required in ("aes", "fma3"):
            if required not in info["features"]:
                info["features"].append(required)

    return info


# ── Public API ─────────────────────────────────────────────────────


def detect_all() -> dict[str, Any]:
    """Full hardware detection: GPUs + CPU.

    Returns a dict with:
        - gpus: list[dict] detected GPUs (sorted: NVIDIA first, then AMD/Intel)
        - primary_gpu: first GPU or None
        - primary_backend: recommended backend for primary GPU
        - multi_gpu_warn: True if >1 GPU found
        - cpu: CPU info dict
        - has_runtime: dict of which runtimes are available
        - recommendation: (backend, reason) tuple for init to write
    """
    gpus = _detect_gpus()
    cpu = _detect_cpu()

    # Sort GPUs: NVIDIA first (best performance), then AMD/Intel
    gpu_order = {"nvidia": 0, "amd": 1, "intel": 1, "unknown": 2}
    gpus.sort(key=lambda g: gpu_order.get(g["vendor"], 3))

    # Pick primary GPU and its recommended backend
    primary_gpu = gpus[0] if gpus else None
    multi_gpu_warn = len(gpus) > 1

    # Detect exact GFX architecture for ROCm binary selection
    gfx_family = None
    if primary_gpu and primary_gpu["vendor"] == "amd":
        gfx_family = detect_gfx_version()
    runtimes = {
        "vulkan": has_vulkan(),
        "rocm": has_rocm(),
        "nvidia": has_nvidia(),
        "cpu": True,  # always available if binary can be downloaded
    }

    recommendation: tuple[str, str] | None = None

    if primary_gpu:
        backend = primary_gpu["backend"]
        # Check if the recommended runtime exists
        runtime_map = {
            "cuda": "nvidia",
            "rocm": "rocm",
            "vulkan-radv": "vulkan",
        }
        rt_key = runtime_map.get(backend)
        if rt_key and runtimes[rt_key]:
            recommendation = (backend, f"{_VENDOR_NAMES.get(primary_gpu['vendor'], primary_gpu['vendor'].title())} GPU detected")
        elif backend == "rocm" and not runtimes["rocm"]:
            # AMD GPU but no ROCm — fall back to vulkan-radv
            if runtimes["vulkan"]:
                recommendation = ("vulkan-radv", f"{_VENDOR_NAMES.get(primary_gpu['vendor'], primary_gpu['vendor'].title())} GPU detected (ROCm unavailable, using Vulkan)")
            else:
                recommendation = ("cpu", f"GPU detected but no GPU runtime available")
        elif backend == "cuda" and not runtimes["nvidia"]:
            # NVIDIA GPU but no CUDA — try vulkan as fallback
            if runtimes["vulkan"]:
                recommendation = ("vulkan-radv", "NVIDIA GPU detected (CUDA unavailable, using Vulkan)")
            else:
                recommendation = ("cpu", f"GPU detected but no runtime available")
        elif backend == "vulkan-radv" and not runtimes["vulkan"]:
            recommendation = ("cpu", f"{_VENDOR_NAMES.get(primary_gpu['vendor'], primary_gpu['vendor'].title())} GPU detected but no Vulkan runtime")
    else:
        # No GPU — use CPU
        if cpu["arch"] == "x86_64":
            recommendation = ("cpu", "No GPU found — using CPU with compatible pre-built binary")
        else:
            recommendation = ("cpu", f"No GPU found — CPU detected ({cpu['arch']})")

    # Build warnings for special cases
    warnings: list[str] = []
    if gpus and primary_gpu and primary_gpu["vendor"] == "amd":
        if runtimes["rocm"]:
            warnings.append("ROCm also available — recommended over vulkan-radv for AMD iGPU/Strix Halo")

    return {
        "gpus": gpus,
        "primary_gpu": primary_gpu,
        "primary_backend": primary_gpu["backend"] if primary_gpu else None,
        "gfx_family": gfx_family,          # e.g. 'gfx1151', 'gfx110X'
        "multi_gpu_warn": multi_gpu_warn,
        "cpu": cpu,
        "has_runtime": runtimes,
        "recommendation": recommendation or ("cpu", "Defaulting to CPU"),
        "warnings": warnings,
    }
