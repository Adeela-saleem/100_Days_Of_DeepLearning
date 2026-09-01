# Perceptron

Implementation and mathematical intuition of the **Perceptron** — the fundamental building block of Artificial Neural Networks (ANNs).

## What is a Perceptron?

The Perceptron is one of the earliest and simplest supervised learning algorithms, in the same family as Linear Regression, Logistic Regression, and SVM. Its design is what made it the foundation for Deep Learning — it's often called a **mathematical model / function** that maps inputs to an output.

## Mathematical Intuition

Given inputs `x1, x2` with weights `w1, w2` and a bias `b`, the perceptron first computes a weighted sum:

```
z = w1*x1 + w2*x2 + b
```

This `z` is then passed through an **activation function**, whose job is to squash the value into a defined range (e.g. `0` to `1`, or `-1` to `1`).

The classic perceptron uses a **step function**:

```
output = 1   if z >= 0
output = 0   if z < 0
```

Other commonly used activation functions include **tanh**, **ReLU**, etc.

During **training**, the model learns from the training data and calculates the optimal `w1, w2, b`. These learned values are exactly what's required at **prediction** time — once trained, the perceptron uses them to classify new inputs.

## Biological Inspiration

Deep Learning is inspired by the nervous system. A single perceptron is loosely analogous to one **neuron**:

| Biological Neuron | Perceptron |
|---|---|
| Dendrites receive input | Inputs (`x1, x2, ...`) |
| Nucleus processes it | Weighted sum + activation |
| Axon produces output | Output |

It's not an exact replica — real neurons are far more complex — but the inspiration holds. In the perceptron, **weights** represent connection strength: a higher weight means that input is more important; a weight near zero means it barely matters.

## Geometric Intuition

The activation function separates outputs into two levels (0 and 1). Geometrically, a perceptron is **nothing but a line**:

```
ax + by + c = 0
```

- If `ax + by + c > 0` → point lies on one side (say, class 1)
- If `ax + by + c < 0` → point lies on the other side (class 0)

Because a perceptron draws one line to split the space into two regions, it's called a **classifier** — it creates a decision boundary. In 3D, this generalizes to a **plane**, and in higher dimensions, a **hyperplane**.

## Code

This notebook trains a perceptron using `sklearn.linear_model.Perceptron` on a placement dataset (`cgpa`, `resume_score` → `placed`):

```python
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv('placement.csv')
sns.scatterplot(data=df, x='cgpa', y='resume_score', hue='placed')

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]

p = Perceptron()
p.fit(X, y)

p.coef_        # learned weights
p.intercept_   # learned bias

plot_decision_regions(X.values, y.values, clf=p, legend=2)
```

The scatter plot visualizes the data, and `plot_decision_regions` draws the learned decision boundary (the line discussed above) that separates placed vs. not-placed students based on `cgpa` and `resume_score`.

## Files

- `Perceptron.ipynb` — full notebook: data loading, EDA, training, and decision boundary visualization
- `placement.csv` — dataset used (cgpa, resume_score, placed)