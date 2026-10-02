"""Prove that PyTorch really runs on the Orin GPU.

"torch.cuda.is_available() == True" is not enough: a wheel without sm_87
kernels still reports the GPU and only fails on the first real operation.
This script runs real BF16 math, a convolution and memory transfers.
"""

import sys

import torch

EXPECTED_CAPABILITY = (8, 7)  # Jetson AGX Orin / Orin NX / Orin Nano


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        sys.exit(1)


print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
check("CUDA available", torch.cuda.is_available())

capability = torch.cuda.get_device_capability()
check("device capability", capability == EXPECTED_CAPABILITY, f"{capability}")

arch_list = torch.cuda.get_arch_list()
check("wheel contains sm_87", "sm_87" in arch_list, f"{arch_list}")

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
check("BF16 matmul", y.shape == (1024, 1024))

conv = torch.nn.Conv2d(3, 16, 3).cuda().half()
out = conv(torch.randn(1, 3, 64, 64, device="cuda", dtype=torch.half))
torch.cuda.synchronize()
check("FP16 convolution", out.shape == (1, 16, 62, 62))

cpu = torch.arange(1_000_000, dtype=torch.float32)
back = cpu.cuda().mul(2).cpu()
check("host <-> device transfer", bool(torch.equal(back, cpu * 2)))

print("\nPyTorch is a valid base for compiling vLLM.")
