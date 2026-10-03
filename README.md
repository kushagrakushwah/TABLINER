# TABLINER

Interactive AI Lab for Text & Tables.

Test two fast AI models directly in your browser:
* **GLiNER:** Finds custom words and tags in text without training a new model.
* **TabPFN:** Delivers instant predictions on small tables (<10,000 rows) in under 1 second.

Live Demo: **[https://kushagrakushwah.github.io/TABLINER/](https://kushagrakushwah.github.io/TABLINER/)**

---

## What is TABLINER?

Most machine learning models require hours of training, huge datasets, and complex settings.

TABLINER shows two modern models that change this:
1. **GLiNER (for Text):** Instead of fixed tags like standard BERT, you can type any tag you want (e.g. `penalty_rate`, `medication`, `cve_id`) and GLiNER finds it immediately.
2. **TabPFN (for Tables):** Instead of training decision trees (like XGBoost) from scratch every time, TabPFN reads your table all at once and gives predictions in less than a second on CPU.

---

## How They Work

### 1. GLiNER: Match Words to Tags

```text
[Your Text] ──┐
              ├──> [DeBERTa Model] ──> [Word Meanings] ──┐
[Custom Tags] ┘                                          ├──> [Compare] ──> [Highlight Matches]
                                       [Tag Meanings]  ──┘
```

* **Step 1: Read Text & Tags:** Pass the text and custom tags into the model together.
* **Step 2: Check Phrases:** The model looks at each word phrase and where it begins and ends.
* **Step 3: Convert Meanings:** Phrases and tags are turned into math vectors.
* **Step 4: Match & Highlight:** If a phrase closely matches a tag, it is highlighted.

---

### 2. TabPFN: Instant Table Predictions

```text
[Millions of Practice Tables] ──> [Pre-Trained Transformer]
                                          │
[Your Table Data (Train + Test)] ─────────> [Read All At Once]
                                          │
                                   [Answer in < 1 Second]
```

* **No Training Needed:** Pre-trained on millions of practice tables so it already knows number patterns.
* **Reads All at Once:** Feeds training rows and new rows together into attention layers, just like an LLM prompt.
* **Under 1 Second:** Delivers predictions on a normal laptop CPU with zero hyperparameter tuning.
* **Best for Small Data:** Gives higher accuracy than XGBoost on tables under 10,000 rows.

---

## Quick Comparison

| Model | Data Type | Custom Tags Without Retraining? | Training Time | Speed | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Traditional BERT** | Text | No (Fixed tags) | Hours | ~30 ms | Fixed tags with lots of data |
| **Large LLM (7B-70B)** | Text | Yes (via prompts) | Days | 1,500 - 8,000 ms | Summaries and conversation |
| **GLiNER** | Text | Yes (Any tag) | None or Minutes | ~150 ms (CPU) | Exact word extraction, no hallucinations |
| **XGBoost** | Tables | No | Minutes | ~10 ms | Big tables (>100,000 rows) |
| **TabPFN** | Tables | Yes (Instant) | 0.0 seconds | < 1 second (CPU) | Small tables (<10,000 rows) |

---

## Run Locally

TABLINER is a single-file web app with zero dependencies.

### Option 1: Browser
Double-click `index.html` to open it in any browser.

### Option 2: Python Server
```bash
git clone https://github.com/kushagrakushwah/TABLINER.git
cd TABLINER
python -m http.server 8080
```
Open `http://localhost:8080`.

---

## Python Code Examples

### 1. Run GLiNER in Python

```python
from gliner import GLiNER

# Load model
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Pick any tags you want
labels = ["contracting_party", "liability_cap", "penalty_rate"]

# Extract matching words
text = "Acme Corp shall pay a 2.0% late penalty. Liability cap is $500,000 USD."
entities = model.predict_entities(text, labels, threshold=0.45)

for ent in entities:
    print(f"{ent['label']}: '{ent['text']}' (Confidence: {ent['score']:.2f})")
```

### 2. Run TabPFN in Python

```python
from tabpfn import TabPFNClassifier
import numpy as np

# Load model (ready to predict immediately)
classifier = TabPFNClassifier(device='cpu')

# Small training table: [Credit Score, Debt Ratio, Income]
X_train = np.array([[740, 22, 110000], [580, 48, 42000], [690, 31, 85000]])
y_train = np.array([0, 1, 0])  # 0: Approved, 1: Default

classifier.fit(X_train, y_train)

# Predict new customer in under 1 second
X_test = np.array([[710, 25, 95000]])
probabilities = classifier.predict_proba(X_test)
print("Probabilities:", probabilities)
```

---

## References

* **GLiNER:** Urchade Zaratiana, Nianlong Gu, Ni Lao. *"GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer"* (2023). [arXiv:2311.08526](https://arxiv.org/abs/2311.08526)
* **TabPFN:** Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter. *"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"* (Nature / ICLR 2023). [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)

---

## License

Apache-2.0 License.
