import time
import ctypes
import numpy as np

np.random.seed(3)

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

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)

X = np.random.randn(256, 512).astype(np.float32)

Wq = np.random.randn(512, 512).astype(np.float32)
Wk = np.random.randn(512, 512).astype(np.float32)
Wv = np.random.randn(512, 512).astype(np.float32)


Wff1 = np.random.randn(512, 2048).astype(np.float32)
bff1 = np.random.randn(2048).astype(np.float32)

Wff2 = np.random.randn(2048, 512).astype(np.float32)
bff2 = np.random.randn(512).astype(np.float32)

Wout = np.random.randn(512, 10).astype(np.float32)
bout = np.random.randn(10).astype(np.float32)

inf_start = make_marker()
qkv_start = make_marker()
qkv_end = make_marker()
attention_start = make_marker()
attention_end = make_marker()
ffn_start = make_marker()
ffn_end = make_marker()
classifier_start = make_marker()
classifier_end = make_marker()
termination = make_marker()

print("MODEL: NUMPY_MINI_TRANSFORMER", flush=True)
print_gpa("INPUT_X", X)

print_gpa("Wq", Wq)
print_gpa("Wk", Wk)
print_gpa("Wv", Wv)
print_gpa("Wff1", Wff1)
print_gpa("Wff2", Wff2)
print_gpa("Wout", Wout)

print_gpa("INFERENCE_START", inf_start)
print_gpa("QKV_START", qkv_start)
print_gpa("QKV_END", qkv_end)
print_gpa("ATTENTION_START", attention_start)
print_gpa("ATTENTION_END", attention_end)
print_gpa("FFN_START", ffn_start)
print_gpa("FFN_END", ffn_end)
print_gpa("CLASSIFIER_START", classifier_start)
print_gpa("CLASSIFIER_END", classifier_end)
print_gpa("TERMINATION", termination)

i = 0
while True:
    print(f"INFERENCE {i} START", flush=True)
    touch_marker(inf_start)
    time.sleep(0.05)

    print("QKV START", flush=True)
    touch_marker(qkv_start)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    print("QKV END", flush=True)
    touch_marker(qkv_end)
    time.sleep(0.05)

    print("ATTENTION START", flush=True)
    touch_marker(attention_start)
    scores = Q @ K.T / np.sqrt(64)
    attn = softmax(scores)
    h = attn @ V
    print("ATTENTION END", flush=True)
    touch_marker(attention_end)
    time.sleep(0.05)

    print("FFN START", flush=True)
    touch_marker(ffn_start)
    h = relu(h @ Wff1 + bff1)
    h = h @ Wff2 + bff2
    print("FFN END", flush=True)
    touch_marker(ffn_end)
    time.sleep(0.05)

    print("CLASSIFIER START", flush=True)
    touch_marker(classifier_start)
    pooled = np.mean(h, axis=0)
    y = pooled @ Wout + bout
    print("CLASSIFIER END", flush=True)
    touch_marker(classifier_end)
    time.sleep(0.05)

    print(f"INFERENCE {i} END", flush=True)
    touch_marker(termination)
    time.sleep(0.05)
    time.sleep(0.5)
    i += 1
