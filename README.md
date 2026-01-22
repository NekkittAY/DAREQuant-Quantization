# DARE-Quantization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Entropy-aware low-bit weight quantization for neural networks based on Dynamic Adaptive Residual Entropy (DARE).

DARE-Quantization introduces a custom linear layer `DAREQuantLinear` that performs **low-bit weight quantization** by jointly minimizing **reconstruction error** and **residual entropy**.  
Unlike standard uniform or MSE-only quantization methods, the proposed approach explicitly controls the information content of quantization residuals, leading to better stability and accuracy in low-bit regimes (e.g. 4-bit).

---

## Overview

`DAREQuantLinear` is a drop-in replacement for `torch.nn.Linear` that:
1. Quantizes weights to a fixed number of bits
2. Automatically selects an optimal scaling factor per output channel
3. Balances **L2 reconstruction error** and **entropy of quantization residuals**
4. Stores quantized weights as buffers for efficient inference

This makes the layer particularly useful for:
- Model compression
- Efficient inference
- Research on entropy-aware and information-theoretic quantization

---

## Table of Contents

- [Motivation](#motivation)
- [Method](#method)
- [Algorithm](#algorithm)
- [Technology Stack](#technology-stack)
- [Usage](#usage)
- [Parameters](#parameters)
- [Limitations](#limitations)

---

## Motivation

Standard post-training quantization typically minimizes only weight reconstruction error (e.g. MSE).  
However, two quantizers with similar MSE may produce residuals with very different statistical properties.

DARE-Quantization addresses this by:
- Penalizing **high-entropy residuals**
- Encouraging structured, low-information noise
- Improving robustness in aggressive low-bit settings

---

## Method

Given a weight vector ($w$), we search for an optimal scaling factor ($s$) that minimizes:
```math
\mathcal{L}(s) =
\lambda_{w} \cdot \| w - \hat{w}(s) \|_2^2
+
\lambda_{H} \cdot H(w - \hat{w}(s))
```

where:
- $\( \hat{w}(s) = \text{clip}(\text{round}(w / s)) \cdot s \)$
- $\( H(\cdot) \)$ is the entropy of quantization residuals
- Optimization is performed **per output channel**

---

## Algorithm

1. Split weight matrix by output channels  
2. For each channel:
   - Enumerate candidate scaling factors
   - Quantize weights using fixed bit-width
   - Compute:
     - L2 reconstruction error
     - Entropy of residual histogram
   - Minimize combined loss
3. Store quantized weights as a non-trainable buffer
4. Use standard linear operation at inference time

---

## Technology Stack

- **Core**: Python 3.8+
- **Deep Learning**: PyTorch
- **Numerical Computing**: NumPy
- **Quantization**: Custom entropy-aware algorithm

---

## Usage

```python
import torch.nn as nn
from dare_quant import DAREQuantLinear


def apply_dareq(model, bits=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, DAREQuantLinear(module, bits=bits))
        else:
            apply_dareq(module, bits)

model_dareq = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
apply_dareq(model_dareq, bits=4)
model_dareq.eval()
```

## Parameters

| Parameter          | Description |
|--------------------|-------------|
| `bits`             | Number of quantization bits |
| `bins`             | Number of bins used to estimate residual entropy |
| `lambda_weight`    | Weight reconstruction (L2) loss coefficient |
| `lambda_entropy`   | Residual entropy regularization coefficient |

---

## Limitations

- Quantization is performed **once at initialization time**
- Scaling factor search is based on brute-force enumeration
- Entropy estimation relies on histogram approximation
- Not optimized for extremely large layers without further acceleration
