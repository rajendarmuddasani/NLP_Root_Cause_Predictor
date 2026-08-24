# 📖 FA Failure Classifier — User Guide

> **App Version:** v0.1.2 &nbsp;|&nbsp; **Model:** MultinomialNB α=0.1 &nbsp;|&nbsp; **Last Updated:** February 2026

---

## Table of Contents

1. [What This App Does](#1-what-this-app-does)
2. [Page Layout — Top to Bottom](#2-page-layout--top-to-bottom)
3. [How to Use the App](#3-how-to-use-the-app)
4. [Understanding the Results](#4-understanding-the-results)
   - [Route Badge (Specialist Lab / Standard Lab)](#41-route-badge)
   - [Timestamp Bar](#42-timestamp-bar)
   - [Table 1 — Class Probabilities](#43-table-1--class-probabilities)
   - [Table 2 — Positive Keywords Found](#44-table-2--positive-keywords-found)
   - [Table 3 — SHAP Explainability](#45-table-3--shap-explainability)
   - [Table 4 — Keyword Combination Co-occurrence](#46-table-4--keyword-combination-co-occurrence)
5. [How the Model Predicts](#5-how-the-model-predicts)
6. [Understanding SHAP — Deep Dive](#6-understanding-shap--deep-dive)
7. [Understanding Keywords — Deep Dive](#7-understanding-keywords--deep-dive)
8. [FAQ / Common Questions](#8-faq--common-questions)
9. [Technical Reference](#9-technical-reference)

---

## 1. What This App Does

The **FA Failure Classifier** predicts which type of failure a sample belongs to, based on free-text failure descriptions written by engineers. It classifies each description into one of **3 classes**:

| Class | Full Name | Meaning | Routing |
|-------|-----------|---------|---------|
| **EOS** | Customer EOS | Electrical Overstress caused by customer usage | ⚠️ **Route → Specialist Lab** |
| **MOS** | Customer MOS | Mechanical Overstress (physical damage, bent leads) | ✅ Route → Standard Lab |
| **TOS** | Customer TOS | Thermal Overstress (heat damage, thermal failure) | ✅ Route → Standard Lab |

The app provides:
- A **class probability** for each of the 3 classes
- **Keyword matching** against a known dictionary of 461 domain keywords
- **SHAP explainability** showing which words in your input influenced the prediction
- **Keyword combination co-occurrence** rates from the training data

---

## 2. Page Layout — Top to Bottom

```
┌─────────────────────────────────────────────────────────────────┐
│  🤖 FA Failure Classifier                          [☰ Menu]    │  ← Title + Streamlit menu
├─────────────────────────────────────────────────────────────────┤
│  Model: MultinomialNB α=0.1 | Accuracy: 90.5% | Macro F1: ... │  ← Stats bar
├───────────────────────────┬──────────┬──────────────────────────┤
│  [Text input area]        │ 🔮Predict│  ⚠️ Route → Specialist Lab        │  ← Input row
├───────────────────────────┴──────────┴──────────────────────────┤
│  🕐 Predicted at 14:32:05 | Input: short circuit between ...   │  ← Timestamp
├──────────────────┬──────────────────┬───────────────────────────┤
│ 📊 Class Probs   │ 🔑 Keywords      │ 🧪 SHAP Explainability   │  ← 3 tables
├──────────────────┴──────────────────┴───────────────────────────┤
│ 🔗 Keyword Combination Co-occurrence (in Customer EOS data)    │  ← Combos table
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. How to Use the App

### Step 1 — Enter Text
Type or paste a **failure description** into the text box. Example inputs:

- `"short circuit at vddc pin"`
- `"bent lead on package corner"`
- `"MCU communication failure after 0km field return"`
- `"low impedance between vss and gnd"`

### Step 2 — Click Predict
Click the **🔮 Predict** button. The app will:
1. Clean your text (lowercase, remove special characters)
2. Run the ML model to produce class probabilities
3. Search for known keywords in your text
4. Compute SHAP explanations for each word
5. Calculate keyword combination co-occurrence rates

### Step 3 — Read the Results
All results appear below the input row. See [Section 4](#4-understanding-the-results) for detailed explanations of each table.

### Important Behaviours

| Behaviour | Explanation |
|-----------|-------------|
| **Results persist** | After clicking Predict, results stay on screen even if you click elsewhere on the page. They only clear/update when you click **Predict** again. |
| **Timestamp** | A timestamp line shows when the current prediction was made and what input was used, so you can confirm whether results are fresh or stale. |
| **First click is slow** | The very first Predict click may take ~1–2 minutes because the SHAP library needs to load. All subsequent predictions are instant (< 1 second). |
| **All tables are copyable** | You can select and copy text from any table (Ctrl+C). |

---

## 4. Understanding the Results

### 4.1 Route Badge

The coloured badge on the right side of the input row shows the **routing decision**:

| Badge | Meaning |
|-------|---------|
| **⚠️ Route → Specialist Lab** (red) | Predicted class is **EOS** → sample should be routed to the Specialist Lab site for analysis |
| **✅ Route → Standard Lab** (green) | Predicted class is **MOS** or **TOS** → sample does NOT go to Specialist Lab |
| **Specialist Lab / Standard Lab** (grey, dashed) | No prediction yet — click Predict to classify |

**Routing rule:** Only `Customer EOS` routes to Specialist Lab. Both `Customer MOS` and `Customer TOS` route to Standard Lab.

---

### 4.2 Timestamp Bar

```
🕐 Predicted at 14:32:05 | Input: short circuit between vdd and gnd pin
```

| Element | Meaning |
|---------|---------|
| **14:32:05** | The time (HH:MM:SS) when you last clicked Predict |
| **Input:** | The first 80 characters of your cleaned input text, so you can confirm which text was classified |

This helps you know whether the displayed results correspond to your **current** input or a **previous** prediction.

---

### 4.3 Table 1 — Class Probabilities

| Column | Meaning |
|--------|---------|
| **Class** | The failure class: `EOS`, `MOS`, or `TOS` |
| **Prob %** | The model's predicted probability for this class (0–100%) |
| **◀** | An arrow marker on the row with the **highest probability** — this is the predicted class |

**Example:**

| Class | Prob % |
|-------|--------|
| EOS ◀ | 92.3 |
| MOS | 5.1 |
| TOS | 2.6 |

**How to read:** The model is 92.3% confident this is an EOS failure. The three probabilities always sum to 100%.

**How probabilities are calculated:**

The model uses Naïve Bayes, which computes:

$$P(\text{class} \mid \text{words}) = \frac{P(\text{words} \mid \text{class}) \times P(\text{class})}{P(\text{words})}$$

Where:
- $P(\text{class})$ is the **prior probability** (base rate from training data: EOS=83.4%, MOS=12.9%, TOS=3.7%)
- $P(\text{words} \mid \text{class})$ is how likely these words are given the class
- The result is normalised so all 3 probabilities sum to 100%

---

### 4.4 Table 2 — Positive Keywords Found

| Column | Meaning |
|--------|---------|
| **Keyword** | A domain keyword from the 461-keyword dictionary that was found in your input text |
| **Rate %** | How often this keyword appears in training samples of the **predicted class** |

**Example** (if predicted class is EOS):

| Keyword | Rate % |
|---------|--------|
| short | 40.4 |
| gnd | 16.6 |
| circuit | 5.3 |
| vdd | 4.6 |

**How to read Rate %:**

The Rate % is a **pre-computed** statistic from the training data:

$$\text{Rate \%} = \frac{\text{Number of [predicted class] samples containing this keyword}}{\text{Total number of [predicted class] samples}} \times 100$$

For example, `short | 40.4` for EOS means:
- Out of **922** EOS training samples, **372** contain the word "short"
- 372 ÷ 922 × 100 = **40.35%**

**Key points:**
- Rate % is **class-specific** — the same keyword has different rates for EOS, MOS, and TOS
- A **high rate** means this keyword is common in the predicted class
- A **low rate** means the keyword is rare in that class (but still present in your input)
- Only **whole-word matches** are used — "short" will NOT match "shor", and "pin" will NOT match "pi"
- Up to **10 keywords** are shown, sorted by Rate % descending

**"No domain keywords detected"** means none of the 461 dictionary keywords were found (whole-word) in your input text.

---

### 4.5 Table 3 — SHAP Explainability

| Column | Meaning |
|--------|---------|
| **Feature** | A word (TF-IDF feature) from your input that influenced the prediction |
| **SHAP** | The SHAP value showing how much this word **pushed** the prediction toward the predicted class |

**Example:**

| Feature | SHAP |
|---------|------|
| short | +0.0063 |
| gnd | +0.0032 |
| circuit | +0.0018 |

#### How to Read SHAP Values

SHAP (SHapley Additive exPlanations) decomposes the prediction into contributions from each word:

$$P(\text{predicted class}) = \underbrace{\text{Base Rate}}_{\text{prior}} + \underbrace{\text{SHAP}_{\text{word}_1} + \text{SHAP}_{\text{word}_2} + \cdots}_{\text{word contributions}}$$

For **EOS**, the base rate (prior) is **83.4%**. Each SHAP value tells you how much a word shifts the probability:

| SHAP Value | Interpretation |
|------------|----------------|
| **+0.0063** | This word pushes EOS probability **up by 0.63 percentage points** |
| **+0.0032** | This word pushes EOS probability **up by 0.32 percentage points** |
| (negative — not shown) | The word pushes probability **down** — we only show positive contributions |

**Why are SHAP values small?**

Because the EOS base rate is already 83.4%, individual words don't need to push much. The model is essentially saying: *"The baseline says EOS, and these words confirm it."*

#### "No significant SHAP features for this input"

This message appears when **all SHAP values are ≤ 0** for the predicted class. This does **NOT** mean the prediction is wrong. It means:

- The model predicted the class based on its **prior probability** (base rate)
- EOS has an 83.4% prior — so with generic/short inputs, EOS wins by default
- No specific word in your text pushed the prediction **above** the baseline
- The words may have SHAP values near zero (neutral) rather than positive

**When you WILL see SHAP features:**
- Longer, more descriptive inputs: `"short circuit between vdd and gnd pin, low impedance"`
- Inputs with class-specific vocabulary: `"bent lead"` (MOS), `"thermal damage"` (TOS)

**When you WON'T see SHAP features:**
- Very short inputs: `"failure"`, `"abnormal"`
- Generic inputs that don't contain discriminative words

---

### 4.6 Table 4 — Keyword Combination Co-occurrence

This table appears only when **2 or more keywords** are detected. It shows how often combinations of keywords appear **together** in training data.

| Column | Meaning |
|--------|---------|
| **Combination** | Two or three keywords joined with `+` |
| **Count** | `X/Y` format — see explanation below |
| **Rate %** | Percentage of the predicted class's training samples containing ALL keywords in the combination |

#### Understanding the Count Column: `X/Y`

The Count column uses the format **`X/Y`** where:

- **X** = Number of training samples (of the predicted class) that contain **ALL** keywords in the combination appearing **together in the same description**
- **Y** = Total number of training samples in the predicted class

**Examples (for predicted class = EOS, where Y = 922):**

| Combination | Count | Rate % | Interpretation |
|-------------|-------|--------|----------------|
| short + gnd | 120/922 | 13.0 | 120 out of 922 EOS samples have BOTH "short" AND "gnd" |
| short + circuit | 45/922 | 4.9 | 45 out of 922 EOS samples have BOTH "short" AND "circuit" |
| short + gnd + circuit | 38/922 | 4.1 | 38 EOS samples have ALL THREE words together |
| vdd + circuit | 0/922 | 0.0 | No EOS sample has both "vdd" and "circuit" together |

**How the Rate % is calculated:**

$$\text{Co-occurrence Rate \%} = \frac{|\text{samples containing ALL keywords in combo}|}{|\text{total samples in predicted class}|} \times 100$$

**How to interpret:**
- **High Rate %** (>5%) → This combination of words is a **common pattern** in that failure class
- **Low Rate %** (1–5%) → The combination occurs but is **uncommon**
- **0.0%** (`0/Y`) → These keywords **never appear together** in the training data for this class, even though each keyword individually may be common. This is interesting — it may indicate your input describes an unusual or novel failure pattern

**What about Y values?**

| Predicted Class | Y (denominator) |
|----------------|-----------------|
| Customer EOS | 922 |
| Customer MOS | 142 |
| Customer TOS | 41 |

So `3/41` for TOS means 3 out of only 41 TOS training samples — the percentage is higher (7.3%) but the absolute count is small.

---

## 5. How the Model Predicts

### The Pipeline

```
Your Text → Clean → TF-IDF Vectorize → Naïve Bayes → Probabilities → Argmax → Class
```

1. **Clean:** Lowercase, remove special characters, collapse whitespace
2. **TF-IDF:** Convert text to a vector of 846 numerical features (one per known word). Each feature = word importance weighted by term frequency × inverse document frequency
3. **Naïve Bayes (MultinomialNB α=0.1):** Computes probability of each class given the word features
4. **Argmax:** The class with the highest probability wins

### Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| **Accuracy** | 90.5% | 90.5% of test samples were correctly classified |
| **Macro F1** | 62.1% | Average F1 score across all 3 classes (weighted equally) |
| **F1 EOS** | 94.6% | F1 for the EOS class (excellent — because 83.4% of data is EOS) |
| **F1 MOS** | 69.6% | F1 for MOS (good) |
| **F1 TOS** | 22.2% | F1 for TOS (poor — only 41 training samples available) |

### Why is TOS F1 Low?

TOS only has **41 samples** (3.7% of data). With so few examples, the model struggles to learn distinctive patterns. This is a **data limitation**, not a model limitation — more TOS training data would improve it.

---

## 6. Understanding SHAP — Deep Dive

### What is SHAP?

SHAP is a method from game theory that fairly distributes the "credit" for a prediction among all input features (words). Each word gets a SHAP value that represents its contribution.

### The SHAP Formula

For a single prediction, SHAP decomposes it as:

$$f(x) = \phi_0 + \phi_1 + \phi_2 + \cdots + \phi_n$$

Where:
- $f(x)$ = predicted probability for a class
- $\phi_0$ = **base value** (average prediction across all training data ≈ class prior)
- $\phi_i$ = SHAP value for the $i$-th word

### Visual Explanation

```
Base rate (EOS = 83.4%)
  + "short"    → +0.63%   ▓▓▓▓
  + "gnd"      → +0.32%   ▓▓
  + "circuit"  → +0.18%   ▓
  + "between"  → +0.05%   ▏
  - "pin"      → -0.10%   (negative — hidden from table)
  ─────────────────────────
  = Final prob   84.5%
```

### Why We Only Show Positive SHAP

We show only words with **positive SHAP values** — words that push **toward** the predicted class. This answers the question: *"What evidence in my text supports this prediction?"*

Negative SHAP values (words pushing against the predicted class) exist but would confuse the interpretation. The model still predicted this class despite those negative pushes.

### SHAP vs Keywords — What's the Difference?

| | Keywords (Table 2) | SHAP (Table 3) |
|---|---|---|
| **Source** | Pre-computed dictionary of 461 known keywords | Computed live for your specific input |
| **What it shows** | How common the word is in the training data for that class | How much the word influenced THIS specific prediction |
| **Rate %** | Percentage of class samples containing this word | — |
| **SHAP value** | — | Contribution to the predicted probability |
| **Scope** | Matches any word from the dictionary | Only words in the TF-IDF vocabulary (846 words) |

Both provide different lenses into the prediction. Keywords tell you *"this word is common in EOS failures"*. SHAP tells you *"this word made the model more confident about EOS for your specific input"*.

---

## 7. Understanding Keywords — Deep Dive

### How the Keyword Dictionary Was Built

The 461-keyword dictionary was created by analysing all 1,105 training samples:

1. **Tokenise** each failure description into words (lowercase, ≥2 characters, stopwords removed)
2. **Count** how many samples of each class contain each word (deduplicated per sample — a word is counted once even if it appears multiple times in the same description)
3. **Calculate per-class rates:**

$$\text{EOS Rate \%} = \frac{\text{EOS samples containing keyword}}{\text{Total EOS samples (922)}} \times 100$$

$$\text{MOS Rate \%} = \frac{\text{MOS samples containing keyword}}{\text{Total MOS samples (142)}} \times 100$$

$$\text{TOS Rate \%} = \frac{\text{TOS samples containing keyword}}{\text{Total TOS samples (41)}} \times 100$$

4. **Filter** — only keywords appearing in ≥2 samples are kept

### Example Keyword Rates

| Keyword | EOS Rate % | MOS Rate % | TOS Rate % | Interpretation |
|---------|-----------|-----------|-----------|----------------|
| short | 40.4 | 8.5 | 14.6 | Very common in EOS, less in MOS/TOS |
| lead | 0.0 | 19.0 | 0.0 | Exclusively MOS — "bent lead" |
| bent | 0.0 | 18.3 | 0.0 | Exclusively MOS — physical damage |
| vss | 10.3 | 2.8 | 22.0 | Highest in TOS — thermal/voltage |

### Keyword Combination Co-occurrence

When multiple keywords are detected, the app computes **co-occurrence rates** — how often the keywords appear **together** in the same training sample:

$$\text{Co-occurrence Rate \%} = \frac{|\{samples \text{ containing } kw_1 \textbf{ AND } kw_2 \textbf{ AND } \cdots\}|}{|\text{total class samples}|} \times 100$$

This uses the **intersection** (AND logic), not the union. All keywords in the combination must be present in the same sample.

- **Pairs:** All 2-word combinations from the top 5 detected keywords (up to 10 pairs)
- **Triples:** All 3-word combinations from the top 5 detected keywords (up to 10 triples)

---

## 8. FAQ / Common Questions

### Q: Why does the first Predict click take so long?

The SHAP library (~165 seconds to import) is **lazy-loaded** — it only loads on the first prediction. After that, it's cached and subsequent predictions are instant (< 1 second). This is a one-time cost per server session.

### Q: The SHAP table says "No significant SHAP features" — is the prediction wrong?

**No.** This is completely normal, especially for EOS predictions. The model's prior for EOS is 83.4%, meaning it "expects" EOS by default. If your input doesn't contain strongly discriminative words, the model predicts EOS based on this prior alone. The prediction is still valid — there's just no single word that significantly pushed it further.

### Q: What if Count shows `0/922`?

This means the keyword combination **never appeared together** in any of the 922 EOS training samples. Each keyword may individually be common, but they've never co-occurred in the same failure description in the training data. Your input may describe a novel or unusual pattern.

### Q: Why is TOS hard to predict?

TOS has only **41 training samples** (3.7% of the data). The model needs more examples to learn reliable patterns. The F1 score for TOS is 22.2% — this means TOS predictions should be treated with lower confidence.

### Q: Can I copy data from the tables?

**Yes.** All tables support text selection and copy (Ctrl+C). Click and drag to select cells, or use Ctrl+A within a table.

### Q: What does "Macro F1" mean?

Macro F1 is the **unweighted average** of F1 scores across all 3 classes:

$$\text{Macro F1} = \frac{F1_{\text{EOS}} + F1_{\text{MOS}} + F1_{\text{TOS}}}{3} = \frac{94.6 + 69.6 + 22.2}{3} = 62.1\%$$

It treats all classes equally, regardless of how many samples each has.

### Q: How confident should I be in the predictions?

| Probability Range | Confidence Level | Recommendation |
|-------------------|-----------------|----------------|
| > 90% | High | Trust the prediction |
| 70–90% | Moderate | Review the keywords and SHAP features for confirmation |
| 50–70% | Low | Manual review recommended |
| < 50% | Very Low | The model is uncertain — seek expert judgement |

### Q: What is the ☰ menu button (top right) for?

The Streamlit menu provides:
- **Rerun** — Re-execute the app
- **Settings** — Theme (light/dark), wide mode
- **About** — Streamlit version info
- **Clear cache** — Force reload all cached data and models

---

## 9. Technical Reference

### Model Details

| Property | Value |
|----------|-------|
| Algorithm | Multinomial Naïve Bayes (α=0.1) |
| Feature Extraction | TF-IDF (846 features) |
| Training Data | 1,105 samples (884 train / 221 test, stratified) |
| Class Distribution | EOS: 922 (83.4%), MOS: 142 (12.9%), TOS: 41 (3.7%) |
| Model File Size | ~70 KB total |
| Framework | scikit-learn 1.8.0 |
| Explainability | SHAP 0.49.1 (LinearExplainer) |
| UI | Streamlit 1.54.0 |

### Text Preprocessing

The app cleans input text identically to training:
1. Convert to lowercase
2. Replace non-alphanumeric characters (except `-` and `/`) with spaces
3. Collapse multiple spaces into one
4. Strip leading/trailing whitespace

**Regex:** `[^a-z0-9\s\-/]` → space

### File Locations

| File | Purpose |
|------|---------|
| `app/streamlit_app_v3.py` | Main Streamlit application |
| `models/v3/best_model.joblib` | Trained MultinomialNB model (40 KB) |
| `models/v3/tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer (28 KB) |
| `models/v3/label_encoder.joblib` | Label encoder for 3 classes (0.5 KB) |
| `models/v3/metadata.json` | Model metadata and routing rules |
| `keywords/keyword_list.csv` | 461 keywords with per-class rates |
| `data/Customer_EOS_MOS_TOS.csv` | Training data (1,105 rows) |
| `results/v3_shap_summary.csv` | Pre-computed SHAP importance (top 20 per class) |

---

*Generated for FA Failure Classifier v0.1.2 — NLP_FA_FailureClassifier project*
