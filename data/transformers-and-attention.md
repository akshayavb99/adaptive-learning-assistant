# Transformers and Attention — Interview Study Notes

> **Goal:** A comprehensive interview-oriented guide covering the core ideas behind Attention and Transformer models.

---

# Table of Contents

1. Motivation
2. Problems with RNNs and LSTMs
3. What is Attention?
4. Query, Key and Value
5. Self-Attention
6. Scaled Dot-Product Attention
7. Multi-Head Attention
8. Positional Encoding
9. Transformer Architecture
10. Encoder
11. Decoder
12. Residual Connections
13. Layer Normalization
14. Feed Forward Network
15. Masking
16. Training Transformers
17. Time Complexity
18. Transformer Variants
19. Advantages
20. Limitations
21. Common Interview Questions
22. Formula Cheat Sheet

---

# 1. Motivation

Traditional sequence models include:

- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

These process tokens **one at a time**, making them difficult to parallelize.

Example:

```
I → love → machine → learning
```

Each word waits for the previous one.

Problems:

- Slow training
- Difficult to capture long-range dependencies
- Vanishing gradients
- Sequential computation

The Transformer removes recurrence completely.

---

# 2. Why Attention?

Suppose we want to translate

```
The animal didn't cross the street because it was tired.
```

When predicting **it**, the model should focus on

```
animal
```

instead of

```
street
```

Attention allows the model to dynamically decide which words are most relevant.

---

# 3. What is Attention?

Attention computes a weighted combination of values.

General equation

$$
Output=\sum_i \alpha_iV_i
$$

where

- $V_i$ = Value vectors
- $\alpha_i$ = Attention weights

Weights satisfy

$$
\sum_i\alpha_i=1
$$

---

# 4. Query, Key and Value

Every token is projected into three vectors.

Input

$$
X
$$

Linear projections

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

Meaning

| Vector | Purpose                        |
| ------ | ------------------------------ |
| Query  | What am I looking for?         |
| Key    | What information do I contain? |
| Value  | Information passed forward     |

Think of a library:

- Query = search request
- Key = book title
- Value = book contents

---

# 5. Self-Attention

Self-attention allows each token to attend to every other token.

Sentence

```
Cats chase mice
```

When processing

```
chase
```

the model attends to

- Cats
- chase
- mice

Each word influences every other word.

---

## Self-Attention Steps

### Step 1

Create Q,K,V.

### Step 2

Similarity scores

$$
QK^T
$$

### Step 3

Scale

$$
\frac{QK^T}{\sqrt{d_k}}
$$

### Step 4

Softmax

$$
Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
$$

### Step 5

Multiply by values

$$
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
V
$$

This is the most important Transformer equation.

---

# 6. Why Divide by √dk?

Large dot products produce huge numbers.

Large numbers cause

- Softmax saturation
- Tiny gradients
- Slow learning

Scaling prevents instability.

$$
\frac{QK^T}{\sqrt{d_k}}
$$

---

# 7. Multi-Head Attention

Instead of one attention mechanism, use several.

Example

8 heads

Each head learns different relationships.

Examples

Head 1

Grammar

Head 2

Verb-object

Head 3

Pronouns

Head 4

Long-distance dependency

Equation

$$
head_i=
Attention(Q_i,K_i,V_i)
$$

Concatenate

$$
Concat(head_1,...,head_h)
$$

Final projection

$$
MultiHead=
Concat(...)
W^O
$$

Advantages

- Multiple representations
- Richer context
- Better performance

---

# 8. Positional Encoding

Transformers process all words simultaneously.

Without extra information

```
Dog bites man
```

and

```
Man bites dog
```

look identical.

Need position information.

Final embedding

$$
Input=
Embedding
+
Position
$$

---

## Sinusoidal Encoding

Even dimensions

$$
PE(pos,2i)
=
\sin
\left(
\frac{pos}{10000^{2i/d}}
\right)
$$

Odd dimensions

$$
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}{10000^{2i/d}}
\right)
$$

Advantages

- Infinite sequence length
- Relative positions emerge naturally

---

# 9. Transformer Architecture

```
Input

↓

Embedding

↓

Positional Encoding

↓

Encoder × N

↓

Decoder × N

↓

Linear

↓

Softmax

↓

Prediction
```

Original model

- Encoder layers = 6
- Decoder layers = 6

---

# 10. Encoder Block

Each encoder consists of

```
Multi-Head Attention

↓

Add & LayerNorm

↓

Feed Forward

↓

Add & LayerNorm
```

Every layer produces contextual representations.

---

# 11. Decoder Block

Decoder contains

1. Masked self-attention

2. Cross attention

3. Feed forward network

Difference

Encoder:

Can see entire sentence.

Decoder:

Cannot see future words.

---

# 12. Residual Connections

Instead of learning

$$
F(x)
$$

learn

$$
x+F(x)
$$

Benefits

- Better gradient flow
- Easier optimization
- Deeper networks

---

# 13. Layer Normalization

Normalizes activations.

Equation

$$
LayerNorm(x)
=
\gamma
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

Benefits

- Stable gradients
- Faster convergence
- Less sensitive to initialization

---

# 14. Feed Forward Network

Applied independently to every token.

Equation

$$
FFN(x)
=
W_2ReLU(W_1x+b_1)+b_2
$$

Hidden dimension usually

$$
4d_{model}
$$

---

# 15. Masking

## Padding Mask

Ignore padding tokens.

Example

```
I like AI PAD PAD PAD
```

Padding should not influence attention.

---

## Causal Mask

During generation

```
The cat sat
```

Predicting

```
sat
```

cannot use

future words.

Mask

```
✓
✓ ✓
✓ ✓ ✓
```

Upper triangle blocked.

---

# 16. Training

Objective

Predict next token.

Loss

Cross entropy

$$
L
=
-\sum_i
y_i
\log(\hat y_i)
$$

Optimizer

Adam

Learning rate

Warmup followed by decay.

Teacher forcing

Ground-truth previous token is used during training.

---

# 17. Complexity

For sequence length

$$
n
$$

Self-attention

$$
O(n^2)
$$

Memory

$$
O(n^2)
$$

RNN

Time

$$
O(n)
$$

Parallelism

No

Transformer

Parallelism

Yes

---

# 18. Popular Transformer Models

## BERT

- Encoder only
- Bidirectional
- Masked Language Modeling
- Classification
- Question answering

---

## GPT

- Decoder only
- Autoregressive
- Text generation
- Chatbots

---

## T5

Encoder-decoder

Everything converted into text-to-text tasks.

---

## ViT

Vision Transformer

Image split into patches.

---

# 19. Advantages

- Parallel training
- Long-range dependency modelling
- State-of-the-art accuracy
- Scales well
- Flexible architecture

---

# 20. Limitations

- Quadratic attention
- Large memory usage
- Expensive training
- Requires massive datasets
- Computationally intensive

---

# 21. Common Interview Questions

### Why not use RNN?

Sequential computation and poor long-distance memory.

---

### Why Q,K,V?

Separates searching, matching and information retrieval.

---

### Why divide by √dk?

Prevent softmax saturation.

---

### Why multiple heads?

Different heads learn different relationships.

---

### Why positional encoding?

Attention alone has no notion of order.

---

### Encoder vs Decoder?

Encoder

- Reads input.

Decoder

- Generates output.

---

### Self-attention vs Cross-attention?

Self-attention

Queries, Keys and Values come from the same sequence.

Cross-attention

Queries come from decoder.

Keys and Values come from encoder.

---

### Why residual connections?

Prevent vanishing gradients.

Enable deep networks.

---

### Why LayerNorm instead of BatchNorm?

LayerNorm works independently for each sequence and batch size.

---

### Why masking?

Prevent information leakage during training.

---

### What is teacher forcing?

Use the correct previous token during training instead of the model's prediction.

---

# 22. Formula Cheat Sheet

### Query

$$
Q=XW_Q
$$

### Key

$$
K=XW_K
$$

### Value

$$
V=XW_V
$$

### Attention

$$
Attention(Q,K,V)=
Softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)
V
$$

### Multi-head

$$
MultiHead=
Concat(head_1,\ldots,head_h)
W^O
$$

### Feed Forward

$$
FFN(x)
=
W_2ReLU(W_1x+b_1)+b_2
$$

### LayerNorm

$$
LayerNorm(x)
=
\gamma
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

### Cross Entropy

$$
L=
-\sum_i
y_i
\log(\hat y_i)
$$

### Positional Encoding

$$
PE(pos,2i)
=
\sin
\left(
\frac{pos}
{10000^{2i/d}}
\right)
$$

$$
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}
{10000^{2i/d}}
\right)
$$

---

# Interview Quick Facts

| Topic                   | Key Point                                                                               |
| ----------------------- | --------------------------------------------------------------------------------------- |
| Transformer             | Removes recurrence                                                                      |
| Attention               | Weighted average of values                                                              |
| Self-Attention          | Every token attends to every token                                                      |
| Multi-Head              | Learns multiple relationships simultaneously                                            |
| Positional Encoding     | Injects word order information                                                          |
| Encoder                 | Understands input                                                                       |
| Decoder                 | Generates output                                                                        |
| BERT                    | Encoder-only, bidirectional                                                             |
| GPT                     | Decoder-only, autoregressive                                                            |
| T5                      | Encoder–decoder                                                                         |
| ViT                     | Transformer for images                                                                  |
| Complexity              | Self-attention is O(n²)                                                                 |
| Optimizer               | Adam                                                                                    |
| Loss                    | Cross-entropy                                                                           |
| Original Transformer    | 6 encoder + 6 decoder layers                                                            |
| Most Important Equation | $$\text{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$ |

---

# Last-Minute Interview Tips

- Explain **why** attention is needed before describing how it works.
- Remember the roles of **Query, Key, and Value** using the "library search" analogy.
- Be ready to derive the scaled dot-product attention equation.
- Understand the difference between **self-attention** and **cross-attention**.
- Know why **multi-head attention**, **residual connections**, **layer normalization**, and **positional encoding** are essential.
- Be able to compare **BERT**, **GPT**, and **encoder–decoder** architectures.
- Mention the quadratic **O(n²)** complexity of standard self-attention and be aware that many modern variants aim to reduce this cost.
