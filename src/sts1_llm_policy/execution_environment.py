"""Execution requirements and observations, separate from experiment identity."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import platform
import subprocess
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version


def validate_execution_profile(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "dependencies", "hardware",
    } or value.get("schema_version") not in {"training_execution_v1", "training_execution_v2"}:
        raise ValueError("Invalid training execution profile")
    hardware = value["hardware"]
    portable = value["schema_version"] == "training_execution_v2"
    if portable:
        if (hardware != {"backend": "cuda", "bf16_supported": True}
                or hardware["bf16_supported"] is not True):
            raise ValueError("Execution requires CUDA and BF16 support")
    elif not isinstance(hardware, dict) or set(hardware) != {
        "platform", "architecture", "compute_capabilities", "cuda_runtime", "bf16_supported",
    }:
        raise ValueError("Execution hardware requirements are incomplete")
    capabilities = hardware.get("compute_capabilities", [])
    if not portable and (
        hardware["platform"] not in {"linux", "windows"}
        or hardware["architecture"] != "x86_64"
        or not isinstance(capabilities, list) or not capabilities
        or any(not isinstance(item, str) or len(item.split(".")) != 2
               or not all(part.isdigit() for part in item.split(".")) for item in capabilities)
        or len(set(capabilities)) != len(capabilities)
        or not isinstance(hardware["cuda_runtime"], str) or not hardware["cuda_runtime"]
        or hardware["bf16_supported"] is not True
    ):
        raise ValueError("Unsupported execution hardware requirements")
    dependencies = value["dependencies"]
    if not isinstance(dependencies, dict) or set(dependencies) != {
        "python", "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors",
    } or any(not isinstance(v, str) or not v for v in dependencies.values()):
        raise ValueError("Execution dependencies must contain the complete version set")
    if portable:
        try:
            for requirement in dependencies.values():
                SpecifierSet(requirement)
        except InvalidSpecifier as error:
            raise ValueError("Execution dependencies must use version specifiers") from error


def observe_hardware(torch_module: Any, device: str) -> dict[str, Any]:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise ValueError("Recovery requires an available CUDA device")
    index = int(device.split(":")[1])
    machine = platform.machine().lower()
    with cuda.device(index):
        bf16 = cuda.is_bf16_supported(including_emulation=False)
    return {
        "platform": platform.system().lower(),
        "architecture": "x86_64" if machine in {"amd64", "x86_64"} else machine,
        "device_name": cuda.get_device_name(index),
        "compute_capability": ".".join(map(str, cuda.get_device_capability(index))),
        "cuda_runtime": torch_module.version.cuda,
        "bf16_supported": bf16,
        "total_memory_bytes": cuda.get_device_properties(index).total_memory,
    }


def verify_execution_hardware(profile: dict, torch_module: Any, device: str) -> dict:
    validate_execution_profile(profile)
    observed = observe_hardware(torch_module, device)
    required = profile["hardware"]
    if profile["schema_version"] == "training_execution_v2":
        if not observed["cuda_runtime"]:
            raise ValueError("Execution requires a CUDA build of PyTorch")
        if not observed["bf16_supported"]:
            raise ValueError("Execution requires BF16 support on the selected CUDA device")
        return observed
    for key in ("platform", "architecture", "cuda_runtime", "bf16_supported"):
        if observed[key] != required[key]:
            raise ValueError(f"Execution hardware mismatch: {key}={observed[key]!r}, required={required[key]!r}")
    if observed["compute_capability"] not in required["compute_capabilities"]:
        raise ValueError("Execution compute capability is not allowed by the profile")
    return observed


def observe_environment(runtime: Any, torch_module: Any) -> dict:
    dependencies = verify_runtime_dependencies(runtime)
    if runtime.execution_requirements is not None:
        hardware = verify_execution_hardware(runtime.execution_requirements, torch_module, runtime.device)
    else:
        verify_torch_hardware(runtime, torch_module)
        hardware = observe_hardware(torch_module, runtime.device)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        drivers = sorted(set(result.stdout.split())) or None
    except (OSError, subprocess.SubprocessError):
        drivers = None
    if not drivers:
        raise ValueError("Cannot observe NVIDIA driver version; check nvidia-smi before recovery")
    return {
        **hardware,
        "dependencies": dependencies,
        "driver_versions": drivers,
        "os_release": platform.release(),
        "libc": list(platform.libc_ver()),
    }


def installed_runtime_dependency_versions() -> dict[str, str]:
    installed = {"python": platform.python_version()}
    for package in (
        "torch",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "safetensors",
    ):
        try:
            installed[package] = version(package)
        except PackageNotFoundError as error:
            raise ValueError(f"Required package is not installed: {package}") from error
    return installed


def verify_runtime_dependencies(config: Any) -> dict[str, str]:
    observed = installed_runtime_dependency_versions()
    expected = dict(config.dependencies)
    portable = (config.execution_requirements or {}).get("schema_version") == "training_execution_v2"
    matches = (all(Version(observed[key]) in SpecifierSet(requirement)
                   for key, requirement in expected.items()) if portable else observed == expected)
    if not matches:
        raise ValueError(
            f"Runtime dependency mismatch: observed={observed!r}, expected={expected!r}"
        )
    return observed


def verify_torch_hardware(config: Any, torch_module: object) -> None:
    if config.execution_requirements is not None:
        verify_execution_hardware(config.execution_requirements, torch_module, config.device)
        return
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise ValueError("The frozen Base runtime requires an available CUDA device")
    device_index = int(config.device.split(":", maxsplit=1)[1])
    observed = {
        "device_name": cuda.get_device_name(device_index),
        "compute_capability": ".".join(
            str(value) for value in cuda.get_device_capability(device_index)
        ),
        "cuda_runtime": getattr(getattr(torch_module, "version", None), "cuda", None),
        "bf16_supported": cuda.is_bf16_supported(),
    }
    expected = {
        "device_name": config.validated_device_name,
        "compute_capability": config.validated_compute_capability,
        "cuda_runtime": config.validated_cuda_runtime,
        "bf16_supported": config.validated_bf16_supported,
    }
    if observed != expected:
        raise ValueError(
            f"Frozen hardware mismatch: observed={observed!r}, expected={expected!r}"
        )
