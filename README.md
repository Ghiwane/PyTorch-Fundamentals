# PyTorch Fundamentals

Self-directed practice repository covering the core building blocks of PyTorch — tensors, autograd, `nn.Module`, training loops, GPU acceleration, and a full regression mini-project.

> 📌 This repo is a learning log, not a polished library — code favors clarity and comments over abstraction, since the goal was to understand every line and build stronger ML/DL skills.

---

## Repository Structure

PyTorch-Fundamentals/
├── data/
│   └── advertising.csv              # Advertising dataset (TV/radio/newspaper spend → sales)
├── advertising2.0.py                # Regression mini-project (PyTorch, from scratch)
├── fashion_mnist_classification.py  # Multi-class image classification (MLP)
├── gpu_cpu_benchmark.py             # CPU vs GPU training speed benchmark
└── README.md


---

## What This Repo Covers

The main areas practiced here, each building on the previous one:

| # | Module | Where it shows up |
|---|--------|-------------------|
| 1 | Tensors: creation, shapes, dtypes, device placement | all scripts |
| 2 | Autograd: `loss.backward()`, gradient tracking | all scripts |
| 3 | `nn.Module` (custom class) and `nn.Sequential` | `gpu_cpu_benchmark.py`, `fashion_mnist_classification.py` |
| 4 | Full training loop: forward → loss → backward → step | all scripts |
| 5 | GPU usage (`.to(device)`, `torch.cuda`) | `gpu_cpu_benchmark.py` |
| 6 | Mini-project: recoding a known regression problem in pure PyTorch | `advertising2.0.py` |

---

## Script Details

### 1. `gpu_cpu_benchmark.py` — CPU vs GPU Training Benchmark

A small `nn.Sequential`-based MLP (`4 → 32 → ReLU → 2`) trained on 1,000,000 randomly generated samples, once on CPU and once on GPU (if CUDA is available), to directly compare wall-clock training time for the same architecture and data.

**Key implementation details:**
- Same seed (`torch.manual_seed(7)`) so both runs start from identical weights.
- `torch.cuda.synchronize()` is called before/after timing the GPU run — necessary because CUDA operations are asynchronous; without this, the timer would stop before the GPU actually finishes computing, giving a falsely optimistic time.
- Falls back gracefully to CPU-only timing if no GPU is detected.

**Observed on this machine (RTX 3050 Laptop GPU, 4GB VRAM):**

| Device | Time (100 epochs, 1M samples) |
|--------|-------------------------------|
| CPU    | 28.50 s |
| GPU    | 1.40 s  |

**≈ 20x speedup** on GPU.

**What this shows:** even though the model itself is tiny (`4 → 32 → 2`, three layers), the dataset here is large (1,000,000 samples per forward/backward pass), so the workload is dominated by raw matrix-multiplication throughput rather than by kernel-launch or data-transfer overhead. That's exactly the regime where a GPU's parallelism pays off — a ~20x speedup is a clean, textbook result. It's a useful contrast to keep in mind for future scripts: GPU gains scale with *data volume and model size together*, not with model size alone. A tiny model on a tiny batch would likely show a much smaller (or even negative) speedup, since transfer/launch overhead would dominate instead.

---

### 2. `advertising2.0.py` — Regression Mini-Project (PyTorch from scratch)

Reimplementation of the advertising sales regression problem (previously solved with scikit-learn) using a pure PyTorch `nn.Module` — a single `nn.Linear` layer, i.e. plain linear regression trained via gradient descent instead of a closed-form / `sklearn` solver.

**Pipeline:**
1. Feature engineering: adds a `tv_x_radio` interaction term to capture the fact that TV and radio spend have a combined (synergy) effect on sales, not just an additive one.
2. `train_test_split` (80/20) — same split logic as the sklearn version, done in NumPy before converting to tensors.
3. `StandardScaler` fit on the training set only, then applied to both sets — avoids data leakage from the test set into the scaling statistics.
4. Tensors converted to `float32`, targets reshaped to `(N, 1)` to match the model's output shape.
5. Training loop: `MSELoss` + `Adam` (`lr=0.1`), 500 epochs, loss logged every 20 epochs.
6. Evaluation in `model.eval()` + `torch.no_grad()` mode: RMSE and R² computed on the test set.

**Observed results:** RMSE ≈ 0.90, R² ≈ 0.97 — matching the scikit-learn baseline almost exactly.

**Key debugging insight worth documenting:** an earlier version of this script (without the `tv_x_radio` interaction feature) produced a noticeably worse RMSE (≈2.12) than the sklearn model. Rather than assuming a bug in the training loop, the discrepancy was traced systematically — checking the loss curve, the scaling, and the features — before concluding the model was underfitting because it was missing the interaction term. Adding `tv_x_radio` immediately closed the gap. This is the same "don't blame the model before ruling out the data" habit that transfers directly from supervised ML into RL debugging later on.

---

### 3. `fashion_mnist_classification.py` — Multi-Class Image Classification

First multi-class classification project in PyTorch, using the FashionMNIST dataset (10 clothing categories, 28×28 grayscale images).

**Architecture — MLP (not a CNN):**
Flatten (784) → Linear(784, 128) → ReLU → Dropout(0.2)
→ Linear(128, 64)  → ReLU
→ Linear(64, 10)


**Training setup:**
- Loss: `CrossEntropyLoss` (applies softmax internally — the model outputs raw logits, no manual softmax needed).
- Optimizer: Adam, `lr = 0.001`.
- 10 epochs, batch size 64, via `DataLoader` (training set shuffled, test set not).
- `model.train()` / `model.eval()` correctly toggled between phases — this matters here specifically because of the `Dropout` layer, which is active during training and disabled during evaluation.
- Accuracy tracked manually per epoch via `torch.argmax` on the logits, for both train and test sets.

**Observed results (epoch 10/10):**

| | Loss | Accuracy |
|---|------|----------|
| Train | 0.2893 | 89.18% |
| Test  | 0.3424 | 87.82% |

Training took **1.74 minutes** for 10 epochs.

**Things worth noting from these numbers:**
- **Train/test gap is small and healthy.** Only ~1.4 points of accuracy and ~0.05 of loss separate train from test — a mild, expected gap rather than the kind of widening divergence that signals real overfitting. That suggests `Dropout(0.2)` is doing its job at this depth/epoch count: the model is generalizing reasonably well rather than memorizing training examples. If more epochs were added, the thing to watch for is whether that gap *stays* roughly this size (good) or starts widening (a sign to add more dropout, weight decay, or early stopping).
- **MLP vs CNN ceiling.** ~88% test accuracy is a solid, expected result for a flatten-based MLP on FashionMNIST — this architecture throws away all spatial structure (a pixel's neighbors carry no special meaning to a `Linear` layer). Published CNN benchmarks on the same dataset typically reach ~92–93%+ by exploiting that spatial locality directly. This ~88% ceiling is a natural, concrete motivator for *why* convolutional layers exist, rather than an abstract argument read somewhere.
- **CrossEntropyLoss expects raw logits.** A common beginner mistake is applying `softmax` manually before `CrossEntropyLoss`, which double-applies the normalization and silently hurts training — this script correctly avoids that by feeding raw logits directly from the last `Linear` layer.

---

## Requirements

torch
torchvision
pandas
numpy
matplotlib
scikit-learn


Install with:
```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn
How to Run
Bash
# CPU vs GPU benchmark
python gpu_cpu_benchmark.py

# Advertising regression mini-project
python advertising2.0.py

# FashionMNIST multi-class classification (downloads dataset on first run)
python fashion_mnist_classification.py
Environment
GPU: NVIDIA RTX 3050 Laptop (4GB VRAM), CUDA confirmed available via torch.cuda.is_available()

CPU: Intel i5-10300H, 8GB RAM

Python: 3.12.10 (venv)

OS: Windows
```
---

## Learning Notes

Code here is deliberately verbose and heavily commented — the priority was understanding why each line is needed (e.g., why scale after splitting, why synchronize() before timing GPU code, why .eval() matters with Dropout) rather than writing the shortest possible implementation. This repo exists to build a solid, hands-on understanding of PyTorch and deep learning fundamentals through practice.