# TABLINER

Interactive AI Foundation Lab for Open-Vocabulary Text Entity Extraction (GLiNER) and Prior-Data Fitted Tabular Inference (TabPFN).

---

## Live Interactive Web Application

Explore the interactive playground directly in your browser:
**[https://kushagrakushwah.github.io/TABLINER/](https://kushagrakushwah.github.io/TABLINER/)**

---

## Overview

Modern machine learning is undergoing a major paradigm shift toward foundation architectures that replace slow iterative training loops with direct, generalized in-context inference. **TABLINER** is an educational, interactive web laboratory designed to demonstrate and benchmark two foundational model architectures:

1. **GLiNER (Generalist and Lightweight Model for Named Entity Recognition):** Eliminates the rigid classification heads of legacy BERT models, enabling arbitrary open-vocabulary entity extraction in sub-120ms without retraining.
2. **TabPFN (Prior-data Fitted Networks for Tabular Data):** Replaces iterative tree building (XGBoost, LightGBM) and hyperparameter grid searches with instant Bayesian posterior prediction in a single transformer forward pass (<1 second).

---

## Key Architectures

### 1. GLiNER: Bidirectional Span-Label Matching

Traditional named entity recognition (spaCy, BERT token classifiers) maps subwords to a fixed, closed set of labels (`PERSON`, `ORG`, `LOC`) using a linear projection layer. Adapting these legacy models to new domain categories (such as `penalty_rate`, `liability_cap`, or `vulnerability_cve`) requires architectural modification and thousands of newly annotated training examples.

GLiNER reformulates entity recognition into a bidirectional representation matching task:

```text
[Input Text Tokens] ──┐
                      ├──> [Bidirectional DeBERTa Encoder] ──> [Span Representations h_span] ──┐
[Target Label Strings] ──┘                                                                     │
                                                                                               ├──> [Bilinear Dot Product] ──> [Sigmoid Score > Threshold]
                      ───> [Label Projection MLP]          ──> [Label Vectors v_label]       ──┘
```

#### How GLiNER Operates in 4 Steps:
1. **Joint Tokenization:** Document tokens and arbitrary target label strings are concatenated and jointly encoded through a bidirectional DeBERTa backbone.
2. **Span Representation Construction:** All candidate spans of length $k \le 12$ are computed by concatenating start and end token representations with learned span length embeddings.
3. **Dynamic Label Vectors:** Labels are not rigid class indices; they are natural language words projected into the shared latent space.
4. **Bilinear Dot-Product Scoring:** Every candidate span is scored against every target label vector via dot product followed by a sigmoid activation function, natively supporting overlapping and multi-label entities.

---

### 2. TabPFN: Prior-Data Fitted Networks for Tabular Data

Traditional tabular machine learning requires extensive iterative workflows: feature scaling, one-hot encoding, tree depth optimization, learning rate tuning, and cross-validation over dozens of trials with XGBoost, LightGBM, or CatBoost.

TabPFN (Hollmann et al., Nature / ICLR) introduces a foundation model for tabular data:

```text
[Synthetic Causal Priors & Structural Models] ──> [Offline Transformer Pre-Training (Millions of Datasets)]
                                                                      │
[User Tabular Dataset (X_train, y_train, X_test)] ─────────────> [In-Context Attention Layers]
                                                                      │
                                                [Single Forward Pass (<1s on CPU)]
                                                                      │
                                                [Exact Bayesian Posterior: P(y_test | D_train, X_test)]
```

#### Why TabPFN Outperforms Gradient Boosted Trees on Small Data (<10,000 samples):
* **Zero Training Loops:** The model weights are fixed; the entire training dataset $(X_{train}, y_{train})$ and query features $X_{test}$ are passed together through attention layers as an in-context prompt.
* **Instant Inference:** Computes exact posterior predictive distributions in under 1 second on CPU without GPU requirements.
* **Zero Hyperparameter Tuning:** Eliminates learning rates, tree depth limits, and regularization searches entirely.
* **Continuous Bayesian Prior Ensemble:** Pre-trained on millions of synthetic causal structural equation models (SEMs) and Gaussian processes, providing robust probabilistic calibration.

---

## Comparative Architecture Matrix

| Dimension | Legacy Token Classifiers (BERT/RoBERTa) | Generative LLMs (7B - 70B) | GLiNER (DeBERTa Small) | Gradient Boosted Trees (XGBoost) | TabPFN Foundation Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Domain** | Text (NER) | Text (Reasoning) | Text (Span Extraction) | Tabular (Classification) | Tabular (Classification) |
| **Zero-Shot Adaptability** | None (Closed Schema) | High (Prompt-based) | Native Open-Vocabulary | None (Requires Training) | Instant In-Context |
| **Training / Tuning Time** | Hours (Fine-Tuning) | Days (RLHF / SFT) | Minutes (<150 samples) | 30 - 300s (Grid Search) | **0.0 Seconds (No Training)** |
| **Inference Latency** | ~30 ms | 1,500 - 8,000 ms | **110 - 180 ms (CPU)** | ~10 ms | **500 - 900 ms (CPU)** |
| **Span Precision / Hallucination** | Low Hallucination | High Hallucination Risk | **Zero Hallucination** | N/A | N/A |
| **Small-Data Performance (<10k)** | Poor (<100 samples) | Moderate | **High (81.3% F1)** | Baseline | **State of the Art** |

---

## Interactive Lab Features

The web application includes live, client-side interactive sandboxes designed with clean institutional typography and strictly zero emojis:

### GLiNER Interactive Lab:
* **Preset Domains:** One-click presets for Financial Agreements, Clinical Healthcare Notes, Cybersecurity Threat Reports, and Executive M&A Announcements.
* **Arbitrary Open Labels:** Enter any comma-separated entity category (e.g. `symptom`, `vulnerability_cve`, `acquirer`, `closing_date`) to test zero-shot extraction in real-time.
* **Span Visualizer & Offset Table:** Interactive text highlighter with color-coded entity pills, confidence tooltips, and token start/end offset tables.
* **Confidence Threshold Slider:** Dynamically filter candidate spans between 0.10 and 0.90.

### TabPFN Interactive Lab:
* **Simulation Scenarios:** Loan Default Risk, SaaS Customer Churn, and Industrial Pump Failure.
* **Dynamic Feature Controls:** Adjust sliders (e.g. FICO Credit Score, Debt-to-Income, Runtime Hours) to see the Bayesian posterior class probability update dynamically.
* **Posterior Confidence Gauges:** Live probability bar charts and factor insight breakdowns grounded in synthetic Bayesian prior distributions.

---

## Local Setup & Quickstart

TABLINER is built as a zero-dependency Single Page Application (SPA). It requires no Node.js build step or package installations to run the UI.

### Option 1: Direct Browser Launch
Simply double-click `index.html` or open it in any web browser.

### Option 2: Python Local Server
```bash
# Clone the repository
git clone https://github.com/kushagrakushwah/TABLINER.git
cd TABLINER

# Start local server
python -m http.server 8080
```
Then navigate to `http://localhost:8080` in your browser.

---

## Python Integration Examples

### Using GLiNER in Python:
```python
from gliner import GLiNER

# 1. Load model checkpoint (base zero-shot or fine-tuned)
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# 2. Define arbitrary target categories dynamically
labels = ["contracting_party", "liability_cap", "penalty_rate", "effective_date"]

# 3. Extract exact substring spans with offsets
text = "Acme Corp shall pay a 2.0% late penalty. Liability cap is $500,000 USD."
entities = model.predict_entities(text, labels, threshold=0.45)

for ent in entities:
    print(f"[{ent['label'].upper()}]: '{ent['text']}' (Confidence: {ent['score']:.2f}, Span: {ent['start']}:{ent['end']})")
```

### Using TabPFN in Python:
```python
from tabpfn import TabPFNClassifier
import numpy as np

# 1. Initialize foundation model (no training loop required)
classifier = TabPFNClassifier(device='cpu')

# 2. Provide small tabular dataset as in-context prompt
X_train = np.array([[740, 22, 110000], [580, 48, 42000], [690, 31, 85000]])
y_train = np.array([0, 1, 0])  # 0: Approved, 1: Default

classifier.fit(X_train, y_train)

# 3. Predict posterior class probabilities instantly
X_test = np.array([[710, 25, 95000]])
probabilities = classifier.predict_proba(X_test)
print("Predicted Class Probabilities:", probabilities)
```

---

## Technical Citations

* **GLiNER:** Urchade Zaratiana, Nianlong Gu, Ni Lao. *"GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"* (2023). [arXiv:2311.08526](https://arxiv.org/abs/2311.08526)
* **TabPFN:** Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter. *"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"* (Nature / ICLR 2023). [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)

---

## License

Apache-2.0 License.
