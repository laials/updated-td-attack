import sys
import time
import ctypes
import numpy as np

helper = ctypes.CDLL("./libgpa_helper.so")
helper.virt_to_phys.argtypes = [ctypes.c_void_p]
helper.virt_to_phys.restype = ctypes.c_uint64

def gpa(arr):
    return helper.virt_to_phys(arr.ctypes.data_as(ctypes.c_void_p))

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def conv2d_simple(x, kernels):
    # x: 1 x 28 x 28
    # kernels: out_channels x 1 x 3 x 3
    out_c = kernels.shape[0]
    out = np.zeros((out_c, 26, 26), dtype=np.float32)

    for oc in range(out_c):
        for i in range(26):
            for j in range(26):
                patch = x[:, i:i+3, j:j+3]
                out[oc, i, j] = np.sum(patch * kernels[oc])
    return out

def maxpool2d(x):
    c, h, w = x.shape
    out = np.zeros((c, h // 2, w // 2), dtype=np.float32)

    for ch in range(c):
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                out[ch, i//2, j//2] = np.max(x[ch, i:i+2, j:j+2])
    return out

def run_mlp():
    x = np.random.randn(1, 784).astype(np.float32)
    w1 = np.random.randn(784, 512).astype(np.float32)
    w2 = np.random.randn(512, 256).astype(np.float32)
    w3 = np.random.randn(256, 10).astype(np.float32)

    markers = [np.zeros(4096, dtype=np.uint8) for _ in range(8)]

    print("MODEL MLP")
    print(f"INPUT_GPA 0x{gpa(x):x}")
    for i, m in enumerate(markers):
        print(f"MARKER_{i}_GPA 0x{gpa(m):x}")

    for r in range(10):
        print(f"INFERENCE {r} START")
        markers[0][0] += 1

        print("LAYER 1 START")
        markers[1][0] += 1
        h1 = relu(x @ w1)
        print("LAYER 1 END")
        markers[2][0] += 1

        print("LAYER 2 START")
        markers[3][0] += 1
        h2 = relu(h1 @ w2)
        print("LAYER 2 END")
        markers[4][0] += 1

        print("LAYER 3 START")
        markers[5][0] += 1
        y = h2 @ w3
        print("LAYER 3 END")
        markers[6][0] += 1

        markers[7][0] += 1
        print(f"INFERENCE {r} END")
        time.sleep(0.5)

def run_cnn():
    x = np.random.randn(1, 28, 28).astype(np.float32)
    k1 = np.random.randn(8, 1, 3, 3).astype(np.float32)
    w1 = np.random.randn(8 * 13 * 13, 64).astype(np.float32)
    w2 = np.random.randn(64, 10).astype(np.float32)

    markers = [np.zeros(4096, dtype=np.uint8) for _ in range(10)]

    print("MODEL SIMPLE_CNN")
    print(f"INPUT_GPA 0x{gpa(x):x}")
    for i, m in enumerate(markers):
        print(f"MARKER_{i}_GPA 0x{gpa(m):x}")

    for r in range(10):
        print(f"INFERENCE {r} START")
        markers[0][0] += 1

        print("CONV START")
        markers[1][0] += 1
        h = relu(conv2d_simple(x, k1))
        print("CONV END")
        markers[2][0] += 1

        print("POOL START")
        markers[3][0] += 1
        h = maxpool2d(h)
        print("POOL END")
        markers[4][0] += 1

        print("FC1 START")
        markers[5][0] += 1
        h = h.reshape(1, -1)
        h = relu(h @ w1)
        print("FC1 END")
        markers[6][0] += 1

        print("FC2 START")
        markers[7][0] += 1
        y = h @ w2
        print("FC2 END")
        markers[8][0] += 1

        markers[9][0] += 1
        print(f"INFERENCE {r} END")
        time.sleep(0.5)

def run_resnet():
    x = np.random.randn(1, 128).astype(np.float32)
    w1 = np.random.randn(128, 128).astype(np.float32)
    w2 = np.random.randn(128, 128).astype(np.float32)
    w3 = np.random.randn(128, 10).astype(np.float32)

    markers = [np.zeros(4096, dtype=np.uint8) for _ in range(8)]

    print("MODEL TINY_RESNET")
    print(f"INPUT_GPA 0x{gpa(x):x}")
    for i, m in enumerate(markers):
        print(f"MARKER_{i}_GPA 0x{gpa(m):x}")

    for r in range(10):
        print(f"INFERENCE {r} START")
        markers[0][0] += 1

        print("BLOCK 1 START")
        markers[1][0] += 1
        residual = x
        h = relu(x @ w1)
        h = h @ w2
        h = relu(h + residual)
        print("BLOCK 1 END")
        markers[2][0] += 1

        print("BLOCK 2 START")
        markers[3][0] += 1
        residual = h
        h2 = relu(h @ w1)
        h2 = h2 @ w2
        h = relu(h2 + residual)
        print("BLOCK 2 END")
        markers[4][0] += 1

        print("CLASSIFIER START")
        markers[5][0] += 1
        y = h @ w3
        print("CLASSIFIER END")
        markers[6][0] += 1

        markers[7][0] += 1
        print(f"INFERENCE {r} END")
        time.sleep(0.5)

def run_transformer():
    x = np.random.randn(16, 64).astype(np.float32)
    wq = np.random.randn(64, 64).astype(np.float32)
    wk = np.random.randn(64, 64).astype(np.float32)
    wv = np.random.randn(64, 64).astype(np.float32)
    wo = np.random.randn(64, 10).astype(np.float32)

    markers = [np.zeros(4096, dtype=np.uint8) for _ in range(8)]

    print("MODEL MINI_TRANSFORMER")
    print(f"INPUT_GPA 0x{gpa(x):x}")
    for i, m in enumerate(markers):
        print(f"MARKER_{i}_GPA 0x{gpa(m):x}")

    for r in range(10):
        print(f"INFERENCE {r} START")
        markers[0][0] += 1

        print("QKV START")
        markers[1][0] += 1
        q = x @ wq
        k = x @ wk
        v = x @ wv
        print("QKV END")
        markers[2][0] += 1

        print("ATTENTION START")
        markers[3][0] += 1
        scores = q @ k.T / np.sqrt(64)
        attn = softmax(scores)
        h = attn @ v
        print("ATTENTION END")
        markers[4][0] += 1

        print("CLASSIFIER START")
        markers[5][0] += 1
        pooled = np.mean(h, axis=0)
        y = pooled @ wo
        print("CLASSIFIER END")
        markers[6][0] += 1

        markers[7][0] += 1
        print(f"INFERENCE {r} END")
        time.sleep(0.5)

model = sys.argv[1] if len(sys.argv) > 1 else "mlp"

if model == "mlp":
    run_mlp()
elif model == "cnn":
    run_cnn()
elif model == "resnet":
    run_resnet()
elif model == "transformer":
    run_transformer()
else:
    print("Use: mlp, cnn, resnet, or transformer")
