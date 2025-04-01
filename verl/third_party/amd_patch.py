import importlib.util
import os
import subprocess

def patch_line(filepath, lineno, new_line):
    """Patch a line in a file, using direct write or sudo sed if necessary."""
    if os.access(filepath, os.W_OK):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if len(lines) >= lineno:
            lines[lineno - 1] = new_line + '\n'
            with open(filepath, 'w') as f:
                f.writelines(lines)
            print(f"Line {lineno} modified successfully (direct write).")
        else:
            print(f"{filepath} has fewer than {lineno} lines.")
    else:
        print(f"{filepath} not writable. Attempting modification with sudo and sed.")
        escaped_line = new_line.replace('"', '\\"')
        sed_cmd = f'sudo sed -i \'{lineno}s/.*/{escaped_line}/\' "{filepath}"'
        subprocess.run(sed_cmd, shell=True, check=True)
        print(f"Line {lineno} modified successfully (via sudo sed).")

def find_module_file(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec.origin if spec and spec.origin else None

def main():
    # Patch ray._private.accelerators.amd_gpu
    amd_gpu_path = find_module_file("ray._private.accelerators.amd_gpu")
    if amd_gpu_path:
        print(f"Located amd_gpu: {amd_gpu_path}")
        patch_line(amd_gpu_path, 9, 'ROCR_VISIBLE_DEVICES_ENV_VAR = "CUDA_VISIBLE_DEVICES"')
    else:
        print("Could not locate ray._private.accelerators.amd_gpu")

    # Patch aiter.jit.core
    aiter_core_path = find_module_file("aiter.jit.core")
    if aiter_core_path:
        print(f"Located aiter.jit.core: {aiter_core_path}")
        patch_line(
            aiter_core_path,
            74,
            'if multiprocessing.current_process().name == \'MainProcess\' and os.environ.get(\'RANK\', 0) == 0:'
        )
    else:
        print("Could not locate aiter.jit.core")

if __name__ == "__main__":
    main()
