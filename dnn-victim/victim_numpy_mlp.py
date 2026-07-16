import time
import ctypes
import numpy as np

np.random.seed(0)

gpa = ctypes.CDLL("./libgpa_helper.so")
gpa.virt_to_phys.argtypes = [ctypes.c_void_p]
gpa.virt_to_phys.restype = ctypes.c_ulong


def print_gpa(name, arr):
    addr = arr.ctypes.data
    phys = gpa.virt_to_phys(ctypes.c_void_p(addr))
    print(f"{name} GPA: 0x{phys:x}", flush=True)

def print_func_gpa(name, func):
    addr = ctypes.cast(func, ctypes.c_void_p).value
    phys = gpa.virt_to_phys(ctypes.c_void_p(addr))
    print(f"{name} VA:  0x{addr:x}", flush=True)
    print(f"{name} GPA: 0x{phys:x}", flush=True)

_marker_keepalive = []


def make_marker():
    marker = np.zeros(8 * 1024 * 1024, dtype=np.uint8)

    offset = 4 * 1024 * 1024
    marker[offset] = 1

    _marker_keepalive.append(marker)
    return marker[offset:]

def touch_marker(marker):
    marker[0] ^= 1

W1 = np.random.randn(4096, 2048).astype(np.float32)
b1 = np.random.randn(2048).astype(np.float32)

W2 = np.random.randn(2048, 1024).astype(np.float32)
b2 = np.random.randn(1024).astype(np.float32)

W3 = np.random.randn(1024, 128).astype(np.float32)
b3 = np.random.randn(128).astype(np.float32)

x = np.random.randn(1, 4096).astype(np.float32)

inference_start_marker = make_marker()

layer1_start_marker = make_marker()
layer1_end_marker = make_marker()

layer2_start_marker = make_marker()
layer2_end_marker = make_marker()

layer3_start_marker = make_marker()
layer3_end_marker = make_marker()

termination_marker = make_marker()


def relu(z):
    return np.maximum(z, 0)


print("DNN victim started inside TD", flush=True)

blas = ctypes.CDLL("/lib/x86_64-linux-gnu/libblas.so.3")

print_func_gpa("cblas_sgemm", blas.cblas_sgemm)

libm = ctypes.CDLL("/lib/x86_64-linux-gnu/libm.so.6")
print_func_gpa("fmaxf", libm.fmaxf)

print_func_gpa("sgemm_", blas.sgemm_)

print_gpa("Input x", x)

print_gpa("W1", W1)
print_gpa("W2", W2)
print_gpa("W3", W3)

print_gpa("Inference start marker", inference_start_marker)

print_gpa("Layer 1 start marker", layer1_start_marker)
print_gpa("Layer 1 end marker", layer1_end_marker)

print_gpa("Layer 2 start marker", layer2_start_marker)
print_gpa("Layer 2 end marker", layer2_end_marker)

print_gpa("Layer 3 start marker", layer3_start_marker)
print_gpa("Layer 3 end marker", layer3_end_marker)

print_gpa("Termination marker", termination_marker)

i = 0
while True:
    print(f"INFERENCE {i} START", flush=True)
    touch_marker(inference_start_marker)
    time.sleep(0.05)

    touch_marker(layer1_start_marker)
    print("LAYER 1 START: Linear 4096 -> 2048 + ReLU", flush=True)
    h1 = relu(x @ W1 + b1)
    print("LAYER 1 END", flush=True)
    touch_marker(layer1_end_marker)
    time.sleep(0.05)

    touch_marker(layer2_start_marker)
    print("LAYER 2 START: Linear 2048 -> 1024 + ReLU", flush=True)
    h2 = relu(h1 @ W2 + b2)
    print("LAYER 2 END", flush=True)
    touch_marker(layer2_end_marker)
    time.sleep(0.05)

    touch_marker(layer3_start_marker)
    print("LAYER 3 START: Linear 1024 -> 128", flush=True)
    y = h2 @ W3 + b3
    print("LAYER 3 END", flush=True)
    touch_marker(layer3_end_marker)
    time.sleep(0.05)

    print(f"INFERENCE {i} END", flush=True)
    touch_marker(termination_marker)
    time.sleep(0.5)

    i += 1
