import time
import ctypes
import numpy as np

np.random.seed(1)

gpa = ctypes.CDLL("./libgpa_helper.so")
gpa.virt_to_phys.argtypes = [ctypes.c_void_p]
gpa.virt_to_phys.restype = ctypes.c_ulong

def print_gpa(name, arr):
    addr = arr.ctypes.data
    phys = gpa.virt_to_phys(ctypes.c_void_p(addr))
    print(f"{name} GPA: 0x{phys:x}", flush=True)

_marker_keepalive = []

_marker_keepalive = []
_marker_count = [0]

def make_marker():
    # Allocate 4MB, touch at a unique 2MB-aligned offset per marker
    # Each marker gets its own 4MB array so they land on different 2MB pages
    size = 4 * 1024 * 1024
    marker = np.zeros(size, dtype=np.uint8)
    # Force a unique physical page by writing to different offsets
    # and keeping all arrays alive
    marker[0] = _marker_count[0] + 1
    _marker_count[0] += 1
    _marker_keepalive.append(marker)
    # Touch the middle to ensure physical allocation
    mid = size // 2
    marker[mid] = 1
    return marker[mid:]

def touch_marker(marker):
    marker[0] ^= 1

def relu(x):
    return np.maximum(x, 0)

def conv2d(x, kernels, bias):
    out_channels = kernels.shape[0]
    _, h, w = x.shape
    kh, kw = kernels.shape[2], kernels.shape[3]

    out_h = h - kh + 1
    out_w = w - kw + 1

    out = np.zeros((out_channels, out_h, out_w), dtype=np.float32)

    for oc in range(out_channels):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i:i+kh, j:j+kw]
                out[oc, i, j] = np.sum(patch * kernels[oc]) + bias[oc]

    return out

def maxpool2d(x):
    channels, h, w = x.shape
    out = np.zeros((channels, h // 2, w // 2), dtype=np.float32)

    for c in range(channels):
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                out[c, i // 2, j // 2] = np.max(x[c, i:i+2, j:j+2])

    return out

X = np.random.randn(1, 64, 64).astype(np.float32)

K1 = np.random.randn(32, 1, 3, 3).astype(np.float32)
b1 = np.random.randn(32).astype(np.float32)

W1 = np.random.randn(32 * 31 * 31, 512).astype(np.float32)
b2 = np.random.randn(512).astype(np.float32)

W2 = np.random.randn(512, 10).astype(np.float32)
b3 = np.random.randn(10).astype(np.float32)

inf_start = make_marker()
conv_start = make_marker()
conv_end = make_marker()
pool_start = make_marker()
pool_end = make_marker()
fc1_start = make_marker()
fc1_end = make_marker()
fc2_start = make_marker()
fc2_end = make_marker()
termination = make_marker()

print("MODEL: NUMPY_CNN", flush=True)
print_gpa("INPUT_X", X)

print_gpa("W1", W1)
print_gpa("W2", W2)

print_gpa("INFERENCE_START", inf_start)
print_gpa("CONV_START", conv_start)
print_gpa("CONV_END", conv_end)
print_gpa("POOL_START", pool_start)
print_gpa("POOL_END", pool_end)
print_gpa("FC1_START", fc1_start)
print_gpa("FC1_END", fc1_end)
print_gpa("FC2_START", fc2_start)
print_gpa("FC2_END", fc2_end)
print_gpa("TERMINATION", termination)

i = 0

while True:
    print(f"INFERENCE {i} START", flush=True)
    touch_marker(inf_start)
    time.sleep(0.05)

    print("CONV START", flush=True)
    touch_marker(conv_start)
    h = relu(conv2d(X, K1, b1))
    print("CONV END", flush=True)
    touch_marker(conv_end)
    time.sleep(0.05)

    print("POOL START", flush=True)
    touch_marker(pool_start)
    h = maxpool2d(h)
    print("POOL END", flush=True)
    touch_marker(pool_end)
    time.sleep(0.05)

    print("FC1 START", flush=True)
    touch_marker(fc1_start)
    h = h.reshape(1, -1)
    h = relu(h @ W1 + b2)
    print("FC1 END", flush=True)
    touch_marker(fc1_end)
    time.sleep(0.05)

    print("FC2 START", flush=True)
    touch_marker(fc2_start)
    y = h @ W2 + b3
    print("FC2 END", flush=True)
    touch_marker(fc2_end)
    time.sleep(0.05)

    print(f"INFERENCE {i} END", flush=True)
    touch_marker(termination)
    time.sleep(0.5)
    i += 1
