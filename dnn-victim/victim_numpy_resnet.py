import time
import ctypes
import numpy as np
import mmap
import ctypes

np.random.seed(2)
gpa = ctypes.CDLL("./libgpa_helper.so")
gpa.virt_to_phys.argtypes = [ctypes.c_void_p]
gpa.virt_to_phys.restype = ctypes.c_ulong

def print_gpa(name, arr):
    addr = arr.ctypes.data
    phys = gpa.virt_to_phys(ctypes.c_void_p(addr))
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

def relu(x):
    return np.maximum(x, 0)


X = np.random.randn(1, 1024).astype(np.float32)

W1 = np.random.randn(1024, 1024).astype(np.float32)
b1 = np.random.randn(1024).astype(np.float32)

W2 = np.random.randn(1024, 1024).astype(np.float32)
b2 = np.random.randn(1024).astype(np.float32)

W3 = np.random.randn(1024, 1024).astype(np.float32)
b3 = np.random.randn(1024).astype(np.float32)

W4 = np.random.randn(1024, 1024).astype(np.float32)
b4 = np.random.randn(1024).astype(np.float32)

W5 = np.random.randn(1024, 10).astype(np.float32)
b5 = np.random.randn(10).astype(np.float32)

inf_start = make_marker()
block1_start = make_marker()
block1_end = make_marker()
block2_start = make_marker()
block2_end = make_marker()
classifier_start = make_marker()
classifier_end = make_marker()
termination = make_marker()

print("MODEL: NUMPY_TINY_RESNET", flush=True)
print_gpa("INPUT_X", X)

print_gpa("W1", W1)
print_gpa("W2", W2)
print_gpa("W3", W3)
print_gpa("W4", W4)
print_gpa("W5", W5)

print_gpa("INFERENCE_START", inf_start)
print_gpa("BLOCK1_START", block1_start)
print_gpa("BLOCK1_END", block1_end)
print_gpa("BLOCK2_START", block2_start)
print_gpa("BLOCK2_END", block2_end)
print_gpa("CLASSIFIER_START", classifier_start)
print_gpa("CLASSIFIER_END", classifier_end)
print_gpa("TERMINATION", termination)

i = 0
while True:
    print(f"INFERENCE {i} START", flush=True)
    touch_marker(inf_start)
    time.sleep(0.05)

    print("BLOCK1 START", flush=True)
    touch_marker(block1_start)
    residual = X
    h = relu(X @ W1 + b1)
    h = h @ W2 + b2
    h = relu(h + residual)
    print("BLOCK1 END", flush=True)
    touch_marker(block1_end)
    time.sleep(0.05)

    print("BLOCK2 START", flush=True)
    touch_marker(block2_start)
    residual = h
    h2 = relu(h @ W3 + b3)
    h2 = h2 @ W4 + b4
    h = relu(h2 + residual)
    print("BLOCK2 END", flush=True)
    touch_marker(block2_end)
    time.sleep(0.05)

    print("CLASSIFIER START", flush=True)
    touch_marker(classifier_start)
    y = h @ W5 + b5
    print("CLASSIFIER END", flush=True)
    touch_marker(classifier_end)
    time.sleep(0.05)

    print(f"INFERENCE {i} END", flush=True)
    touch_marker(termination)
    time.sleep(0.5)

    i += 1

