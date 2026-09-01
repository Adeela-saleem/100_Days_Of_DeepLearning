# MLP Notation — Weights, Biases, Outputs & Trainable Parameters

These notes are designed to make **Multilayer Perceptron (MLP) notation** easy to read before learning **backpropagation**.

The part that confuses many beginners is not only the backpropagation algorithm itself — it is identifying **which weight or bias belongs to which layer and neuron**. If the notation is clear, forward propagation and backpropagation become much easier to follow.

---

## 1. Why notation matters before backpropagation

When we create a neural network, it contains:

- **Weights**
- **Biases**
- **Neuron outputs / activations**

During training, backpropagation computes gradients for the trainable parameters, and the optimizer updates them.

So, before studying backpropagation, we should be able to answer:

1. **How many trainable parameters are in the network?**
2. **How many are weights and how many are biases?**
3. **Which layer does a particular weight or bias belong to?**
4. **Which two neurons does a weight connect?**
5. **How do we denote neuron outputs consistently?**

---

# 2. Example MLP Architecture

Consider the following network:

```text
Input Layer        Hidden Layer 1       Hidden Layer 2       Output Layer
   4 nodes      →       3 nodes      →      2 nodes       →      1 node

   L0                    L1                    L2                   L3
```

Architecture:

```text
4 → 3 → 2 → 1
```

Where:

- Input features = **4**
- Hidden Layer 1 neurons = **3**
- Hidden Layer 2 neurons = **2**
- Output neurons = **1**

---

# 3. Dataset notation

Suppose the dataset is:

```
X ∈ R^(m × n)
```

Where:

- `m` = number of rows / training examples
- `n` = number of input features

For this example:

```
n = 4
```

So each sample contains four features.

A single row/sample can be denoted as:

```
x_i
```

and its individual features can be written as:

```
x_i1, x_i2, x_i3, x_i4
```

Here:

- `i` = sample number
- `1,2,3,4` = feature number

Example:

```
x_i = [x_i1, x_i2, x_i3, x_i4]
```

---

# 4. What are trainable parameters?

The values learned during training are mainly:

> **Weights + Biases**

The input values themselves are **not trainable parameters**.

For a fully connected layer:

```
No. of weights = (neurons in previous layer) × (neurons in current layer)
```

and:

```
No. of biases = neurons in current layer
```

Therefore:

> **Parameters in layer = n_(l-1) × n_l + n_l**

or equivalently:

> **Parameters = (n_(l-1) + 1) × n_l**

where:

- `n_(l-1)` = number of neurons in the previous layer
- `n_l` = number of neurons in the current layer

---

# 5. Parameter calculation for the 4 → 3 → 2 → 1 network

## Layer 0 → Layer 1

There are:

- 4 input nodes
- 3 neurons in Hidden Layer 1

Every input connects to every neuron.

### Weights

```
4 × 3 = 12
```

### Biases

Each neuron in Hidden Layer 1 has one bias:

```
3
```

### Total

```
12 + 3 = 15
```

---

## Layer 1 → Layer 2

There are:

- 3 neurons in Hidden Layer 1
- 2 neurons in Hidden Layer 2

### Weights

```
3 × 2 = 6
```

### Biases

```
2
```

### Total

```
6 + 2 = 8
```

---

## Layer 2 → Output Layer

There are:

- 2 neurons in Hidden Layer 2
- 1 output neuron

### Weights

```
2 × 1 = 2
```

### Biases

```
1
```

### Total

```
2 + 1 = 3
```

---

# 6. Total trainable parameters

Adding all layers:

```
15 + 8 + 3 = 26
```

Therefore:

> **Total trainable parameters = 26**

Breakdown:

| Connection | Weights | Biases | Total |
|---|---:|---:|---:|
| Input → Hidden 1 | 12 | 3 | 15 |
| Hidden 1 → Hidden 2 | 6 | 2 | 8 |
| Hidden 2 → Output | 2 | 1 | 3 |
| **Total** | **20** | **6** | **26** |

So the network contains:

> **20 weights + 6 biases = 26**

---

# 7. What does backpropagation do with these 26 parameters?

When the network is trained:

1. **Forward propagation** produces a prediction.
2. A **loss function** measures the prediction error.
3. **Backpropagation** computes the gradient of the loss with respect to each trainable parameter.
4. An optimizer such as Gradient Descent updates the weights and biases.

So for this architecture, gradients are required for all:

> **26 trainable parameters**

Backpropagation does not randomly search for 26 values. It efficiently computes how each parameter contributes to the loss using the **chain rule**.

---

# 8. Standard layer notation

A very common convention is:

```
L0 = input layer
L1 = first hidden layer
L2 = second hidden layer
L3 = output layer
```

For this network:

```text
L0       L1       L2       L3
4   →    3   →    2   →    1
```

---

# 9. Bias notation

A clean notation for the bias of neuron `j` in layer `l` is:

> **b[l]_j**

Where:

- `l` = layer number
- `j` = neuron/node number inside that layer

For Hidden Layer 1:

```
b[1]_1,  b[1]_2,  b[1]_3
```

For Hidden Layer 2:

```
b[2]_1,  b[2]_2
```

For the output layer:

```
b[3]_1
```

### Meaning

```
b[1]_2
```

means:

> bias of neuron 2 in layer 1.

---

# 10. Output / activation notation

A neuron's output is commonly written as:

> **a[l]_j**

Some notes or lectures may use:

```
o[l]_j
```

Both can represent the output of a neuron, but `a` is especially common because it means **activation**.

For Hidden Layer 1:

```
a[1]_1,  a[1]_2,  a[1]_3
```

For Hidden Layer 2:

```
a[2]_1,  a[2]_2
```

Output layer:

```
a[3]_1 = ŷ
```

A key idea is:

> The output of one layer becomes the input to the next layer.

---

# 11. Weight notation

A weight needs to tell us three things:

1. **Which layer the weight belongs to**
2. **Which neuron it comes from**
3. **Which neuron it goes to**

A widely used notation is:

> **W[l]_ji**

where:

- `l` = destination/current layer
- `i` = neuron in the previous layer
- `j` = neuron in the current layer

Therefore:

```
W[l]_ji
```

means:

> weight connecting neuron `i` in layer `l-1` to neuron `j` in layer `l`.

### Example

```
W[2]_21
```

means:

- destination layer = L2
- source neuron = neuron 1 of L1
- destination neuron = neuron 2 of L2

In words:

> Weight from neuron 1 of Hidden Layer 1 to neuron 2 of Hidden Layer 2.

---

# 12. Why you may see different index orders

Different books and instructors sometimes use notation such as:

```
W[l]_ij
```

instead of:

```
W[l]_ji
```

The symbols are less important than the **definition being used consistently**.

For these notes, we use:

> **W[l]_ji = weight from previous-layer neuron `i` to current-layer neuron `j`**

Always check the convention used in your lecture or textbook.

---

# 13. Weight matrices

Instead of treating every weight separately, neural networks store the weights of a layer in a matrix.

For layer `l`:

> **W[l] ∈ R^(n_l × n_(l-1))**

and:

> **b[l] ∈ R^(n_l × 1)**

---

## Layer 1

```
W[1] ∈ R^(3×4)
```

because:

- current layer has 3 neurons
- previous layer has 4 inputs

Total entries:

```
3 × 4 = 12
```

Bias:

```
b[1] ∈ R^(3×1)
```

---

## Layer 2

```
W[2] ∈ R^(2×3)
```

Total weights:

```
2 × 3 = 6
```

Bias:

```
b[2] ∈ R^(2×1)
```

---

## Output layer

```
W[3] ∈ R^(1×2)
```

Total weights:

```
1 × 2 = 2
```

Bias:

```
b[3] ∈ R^(1×1)
```

---

# 14. Forward propagation notation

For each layer:

> **z[l] = W[l]·a[l-1] + b[l]**

Then an activation function is applied:

> **a[l] = g[l](z[l])**

For the input layer:

```
a[0] = x
```

So the complete flow is:

```text
x
↓
z[1] = W[1]x + b[1]
↓
a[1] = g(z[1])
↓
z[2] = W[2]a[1] + b[2]
↓
a[2] = g(z[2])
↓
z[3] = W[3]a[2] + b[3]
↓
a[3] = ŷ
```

---

# 15. One neuron equation

For neuron `j` in layer `l`:

```
z[l]_j = Σᵢ ( W[l]_ji × a[l-1]_i ) + b[l]_j
```

Then:

```
a[l]_j = g(z[l]_j)
```

This equation explains exactly why the indices matter.

Each current neuron receives:

- outputs from previous-layer neurons
- one weight for each incoming connection
- one bias

---

# 16. Example: first neuron of Hidden Layer 1

The first neuron receives all four input features.

Its linear combination is:

```
z[1]_1 = W[1]_11·x1 + W[1]_12·x2 + W[1]_13·x3 + W[1]_14·x4 + b[1]_1
```

Then:

```
a[1]_1 = g(z[1]_1)
```

The same process happens for neurons 2 and 3.

---

# 17. Activation functions

Common activation functions include:

### ReLU

```
g(z) = max(0, z)
```

Frequently used in hidden layers.

### Sigmoid

```
g(z) = 1 / (1 + e^(-z))
```

Often used for binary classification output.

### Tanh

```
g(z) = tanh(z)
```

Outputs values between -1 and 1.

### Linear

```
g(z) = z
```

Often used in the output layer for regression.

---

# 18. Backpropagation notation preview

During backpropagation, we are interested in quantities such as:

```
∂L / ∂W[l]
```

and:

```
∂L / ∂b[l]
```

where `L` is the loss.

These tell us:

> If this weight or bias changes slightly, how much will the loss change?

The optimizer then performs an update such as:

```
W[l] ← W[l] − α · (∂L / ∂W[l])
b[l] ← b[l] − α · (∂L / ∂b[l])
```

where:

```
α = learning rate
```

---

# 19. Quick architecture trick

For any dense network:

```text
n0 → n1 → n2 → ... → nL
```

The total trainable parameters are:

> **Σ (from l=1 to L) of ( n_(l-1)·n_l + n_l )**

### For 4 → 3 → 2 → 1

```
(4×3 + 3) + (3×2 + 2) + (2×1 + 1)
= 15 + 8 + 3
= 26
```

---

# 20. Quick notation cheat sheet

| Symbol | Meaning |
|---|---|
| `m` | Number of samples / rows |
| `n` | Number of input features |
| `x_i` | i-th training sample |
| `x_ij` | Feature j of sample i |
| `L_l` | Layer l |
| `n_l` | Number of neurons in layer l |
| `W[l]` | Weight matrix of layer l |
| `W[l]_ji` | Weight from previous neuron i to current neuron j |
| `b[l]_j` | Bias of neuron j in layer l |
| `z[l]_j` | Linear/pre-activation value |
| `a[l]_j` | Output/activation of neuron j |
| `g` | Activation function |
| `ŷ` | Predicted output |
| `L` | Loss / cost (context-dependent; also sometimes total layer count, so define it clearly) |
| `α` | Learning rate |

---

# 21. Most important points to remember

- The **input layer has no trainable weights of its own**; weights belong to the connections leading into the next layer.
- Every neuron in a standard dense layer usually has **one bias**.
- If a layer has `p` incoming nodes and `q` neurons:

```
weights = p × q
biases  = q
```

- For the architecture 4 → 3 → 2 → 1:

> **20 weights + 6 biases = 26 trainable parameters**

- Standard notation used here:

> **W[l]_ji** — the weight from neuron `i` in layer `l-1` to neuron `j` in layer `l`.

- Bias:

> **b[l]_j**

- Neuron activation/output:

> **a[l]_j**

- Forward propagation:

> **z[l] = W[l]·a[l-1] + b[l]**
> **a[l] = g(z[l])**

- Backpropagation computes gradients for these trainable parameters so that an optimizer can update them.

---

## Final picture

```text
Data sample x
      │
      ▼
┌──────────┐
│ Input L0 │  4 nodes
└──────────┘
      │
      │ 12 weights + 3 biases
      ▼
┌──────────┐
│Hidden L1 │  3 neurons
└──────────┘
      │
      │ 6 weights + 2 biases
      ▼
┌──────────┐
│Hidden L2 │  2 neurons
└──────────┘
      │
      │ 2 weights + 1 bias
      ▼
┌──────────┐
│Output L3 │  1 neuron
└──────────┘
      │
      ▼
     ŷ

Total = 20 weights + 6 biases = 26 trainable parameters
```

Once this notation is comfortable, **backpropagation becomes much easier**, because every derivative can be linked to a specific layer, neuron, weight, and bias.
