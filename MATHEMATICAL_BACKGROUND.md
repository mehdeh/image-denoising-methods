# Mathematical Background: Ideal Denoiser

This document explains the mathematical theory behind the ideal denoiser implementation.

## 1. Problem Setup

### Noise Model

Given a clean image $ x' \sim p_{\text{data}} $, we observe a noisy version:

$$
x = x' + n, \quad n \sim \mathcal{N}(0, \sigma^2 I)
$$

where:
- $ x' $: clean image from the data distribution
- $ x $: noisy observation
- $ n $: Gaussian noise with standard deviation $ \sigma $

### Denoising Goal

The ideal denoiser aims to recover the clean image by computing:

$$
D(x; \sigma) = \mathbb{E}[x' \mid x] = \int x' \cdot p(x' \mid x) \, dx'
$$

This is the **posterior mean** - the expected value of the clean image given the noisy observation.

## 2. Derivation of Equation 57

### Step 1: Bayes' Rule

Using Bayes' rule:

$$
p(x' \mid x) = \frac{p(x \mid x') \cdot p(x')}{p(x)}
$$

### Step 2: Likelihood

The likelihood of observing $ x $ given $ x' $ follows from the Gaussian noise model:

$$
p(x \mid x') = \mathcal{N}(x; x', \sigma^2 I) = \frac{1}{Z} \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)
$$

where $ Z $ is the normalization constant.

### Step 3: Empirical Distribution

For a finite dataset $ \{x_1, x_2, \ldots, x_N\} $, we approximate:

$$
p(x') \approx \frac{1}{N} \sum_{i=1}^{N} \delta(x' - x_i)
$$

### Step 4: Posterior Mean

Combining the above:

$$
D(x; \sigma) = \int x' \cdot p(x' \mid x) \, dx'
$$

$$
= \frac{\int x' \cdot p(x \mid x') \cdot p(x') \, dx'}{p(x)}
$$

$$
= \frac{\sum_{i=1}^{N} x_i \cdot \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{i=1}^{N} \exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}
$$

This is **Equation 57** from the EDM paper.

## 3. Intuitive Understanding

### Weighted Average

The ideal denoiser computes a **weighted average** of all training images:

$$
D(x; \sigma) = \sum_{i=1}^{N} w_i \cdot x_i
$$

where the weights are:

$$
w_i = \frac{\exp\left(-\frac{\|x - x_i\|^2}{2\sigma^2}\right)}{\sum_{j=1}^{N} \exp\left(-\frac{\|x - x_j\|^2}{2\sigma^2}\right)}
$$

### Interpretation

- **Similar images get higher weight**: If $ x_i $ is close to the noisy input $ x $, then $ \|x - x_i\|^2 $ is small, and $ w_i $ is large.
- **Dissimilar images get lower weight**: If $ x_i $ is far from $ x $, then $ w_i $ is small.
- **Sigma controls similarity**: Larger $ \sigma $ makes the weights more uniform; smaller $ \sigma $ makes the weights more peaked.

## 4. Special Cases

### Case 1: Zero Noise ($ \sigma \to 0 $)

When $ \sigma \to 0 $:

$$
D(x; \sigma) \to x_{\text{nearest}}
$$

The denoiser returns the **nearest neighbor** from the training set.

**Proof**: As $ \sigma \to 0 $, the weight $ w_i $ for the closest training image dominates all others.

### Case 2: Infinite Noise ($ \sigma \to \infty $)

When $ \sigma \to \infty $:

$$
D(x; \sigma) \to \frac{1}{N} \sum_{i=1}^{N} x_i = \bar{x}
$$

The denoiser returns the **mean** of all training images.

**Proof**: As $ \sigma \to \infty $, all weights become equal: $ w_i \to 1/N $.

### Case 3: Exact Match ($ x = x_k $ for some $ k $)

If the noisy input exactly matches a training image:

$$
D(x; \sigma) \approx x_k
$$

for small $ \sigma $.

## 5. Connection to Score Matching

### Score Function

The **score function** is the gradient of the log-density:

$$
\nabla_x \log p(x; \sigma) = -\frac{1}{\sigma^2}(x - D(x; \sigma))
$$

This relates the ideal denoiser to the **score** of the noisy distribution.

### Tweedie's Formula

The ideal denoiser can be written as:

$$
D(x; \sigma) = x + \sigma^2 \nabla_x \log p(x; \sigma)
$$

This is known as **Tweedie's formula** in statistics.

## 6. Computational Considerations

### Time Complexity

Computing $ D(x; \sigma) $ requires:

1. **Distance computation**: $ O(N \times d) $ where $ d = C \times H \times W $
2. **Weight computation**: $ O(N) $
3. **Weighted sum**: $ O(N \times d) $

**Total**: $ O(N \times d) $ per query

For CIFAR-10:
- $ N = 50,000 $ training images
- $ d = 3 \times 32 \times 32 = 3,072 $ dimensions
- **Total**: ~150 million operations per denoising

### Memory Complexity

Storing the training set requires:

$$
\text{Memory} = N \times C \times H \times W \times \text{bytes per pixel}
$$

For CIFAR-10 with float32:
- $ 50,000 \times 3 \times 32 \times 32 \times 4 = 600 \text{ MB} $

### Numerical Stability

The direct computation of weights:

$$
w_i = \frac{\exp(-\|x - x_i\|^2 / (2\sigma^2))}{\sum_j \exp(-\|x - x_j\|^2 / (2\sigma^2))}
$$

can cause **numerical overflow** when $ \sigma $ is small or distances are large.

#### Log-Sum-Exp Trick

To improve stability, we use:

$$
\log w_i = -\frac{\|x - x_i\|^2}{2\sigma^2} - \log \sum_j \exp\left(-\frac{\|x - x_j\|^2}{2\sigma^2}\right)
$$

Let $ M = \max_j \left(-\frac{\|x - x_j\|^2}{2\sigma^2}\right) $, then:

$$
\log \sum_j \exp\left(-\frac{\|x - x_j\|^2}{2\sigma^2}\right) = M + \log \sum_j \exp\left(-\frac{\|x - x_j\|^2}{2\sigma^2} - M\right)
$$

This ensures all exponentials are in the range $ [0, 1] $.

## 7. Comparison with Neural Denoisers

| Aspect | Ideal Denoiser | Neural Denoiser |
|--------|----------------|-----------------|
| **Knowledge** | Full training set | Learned parameters |
| **Computation** | $ O(N \times d) $ | $ O(d) $ |
| **Memory** | Stores entire dataset | Stores network weights |
| **Performance** | Optimal (theoretical upper bound) | Approximates optimal |
| **Scalability** | Poor (increases with dataset size) | Good (fixed size) |

### Why Use Neural Networks?

Neural networks **approximate** the ideal denoiser but with:
- **Constant computation**: $ O(d) $ regardless of dataset size
- **Compact representation**: Store only network weights
- **Generalization**: Can denoise images outside the training set

## 8. Practical Implementation

### PyTorch Implementation

```python
def ideal_denoiser(x_noisy, sigma, x_train):
    # Compute distances
    norm2 = ((x_train[:, None] - x_noisy[None, :]) ** 2).sum(dim=(2,3,4))
    
    # Compute log weights (with numerical stability)
    log_weights = -norm2 / (2 * sigma ** 2)
    delta = log_weights.max(dim=0, keepdim=True)[0]
    weights = (log_weights - delta).exp()
    
    # Weighted average
    numerator = (weights[:, :, None, None, None] * x_train[:, None]).sum(dim=0)
    denominator = weights.sum(dim=0)
    
    return numerator / denominator[:, None, None, None]
```

### Batch Processing

For efficiency, process multiple noisy images at once:

```python
# x_noisy: (B, C, H, W) - batch of B noisy images
# x_train: (N, C, H, W) - N training images
denoised = ideal_denoiser(x_noisy, sigma, x_train)  # (B, C, H, W)
```

## 9. Extensions and Variations

### 1. Kernel Denoising

Replace Gaussian kernel with other kernels:

$$
D(x; \sigma) = \frac{\sum_i K(x, x_i) \cdot x_i}{\sum_i K(x, x_i)}
$$

where $ K(x, x_i) $ is any positive kernel function.

### 2. Local Denoising

Use only $ k $ nearest neighbors:

$$
D(x; \sigma) = \frac{\sum_{i \in \mathcal{N}_k(x)} \exp(-\|x - x_i\|^2 / (2\sigma^2)) \cdot x_i}{\sum_{i \in \mathcal{N}_k(x)} \exp(-\|x - x_i\|^2 / (2\sigma^2))}
$$

where $ \mathcal{N}_k(x) $ are the $ k $ nearest neighbors.

### 3. Anisotropic Denoising

Use different noise levels for different dimensions or channels.

## 10. References

1. **EDM Paper**: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022
   - Equation 57 in Appendix B.3
   
2. **Score Matching**: Hyvärinen, "Estimation of Non-Normalized Statistical Models by Score Matching", JMLR 2005

3. **Tweedie's Formula**: Efron, "Tweedie's Formula and Selection Bias", JASA 2011

4. **Diffusion Models**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020

5. **Non-parametric Regression**: Nadaraya-Watson kernel regression

---

**Note**: This mathematical background provides the theoretical foundation for understanding the ideal denoiser implementation in this repository.

