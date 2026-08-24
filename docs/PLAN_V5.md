# V5 Plan — NLP FA Failure Classifier
**Project**: NLP FA Failure Classifier — v5a / v5b / v5c  
**Date**: March 2026  
**Data source**: `data/raw_v4/TC2_TC3_PL90_cases_with_root_cause_Status_2026-02-25.xlsx` (~13.9K rows)  
**Input filter**: All rows with non-empty `PSI Failure Desc`, ignoring Product/Family  
**Business goal**: Accurately identify EOS cases for specialist-lab routing — applies to all 3 versions  
**Primary metrics**: Macro F1 · EOS Precision · EOS Recall

---

## Status Tracker

| # | Task | Status | Notes |
|---|------|--------|-------|
| **Phase 0** | **Git push v4** | | |
| 0.1 | Commit v4 to `feature/v4-nlp-fa` | ✅ DONE | |
| 0.2 | Push to remote | ✅ DONE | "v4 model comparison — no streamlit designed for this version" |
| **Phase 1** | **Shared data preparation** | | |
| 1.1 | Create `data/prepare_v5_data.py` | ✅ DONE | Single script → 3 CSVs |
| 1.2 | Print full class distribution (non-empty text, all classes) | ✅ DONE | 0 null text rows — all 13,917 rows have text |
| 1.3 | Generate `data/v5a_eos_vs_noneos.csv` (2-class) | ✅ DONE | EOS=3,554 (25.5%) · Non-EOS=10,363 (74.5%) |
| 1.4 | Generate `data/v5b_eos_backend_other.csv` (3-class) | ✅ DONE | EOS=3,554 · Backend=980 · Other=9,383 |
| 1.5 | Generate `data/v5c_all_classes.csv` (~10-class) | ✅ DONE | 13,914 rows, 10 classes, Lot disposition (3) dropped |
| 1.6 | Verify CSVs: row counts, no nulls, distributions correct | ✅ DONE | All null text=0, null label=0 |
| **Phase 2** | **Environment setup** | | |
| 2.1 | Create `requirements_v5.txt` | ✅ DONE | Based on requirements_v4.txt |
| 2.2 | Create/reuse venv, verify imports | ✅ DONE | Reusing `.venv_v3` — has torch 2.10.0+cpu, transformers 5.2.0, sklearn 1.8.0, xgboost 3.2.0, shap 0.49.1, imbalanced-learn 0.14.1 |
| **Phase 3** | **v5a notebook** — EOS vs Non-EOS (2-class) | | |
| 3.1 | Setup & data loading | ✅ DONE | |
| 3.2 | EDA (distribution, text lengths, word clouds, top words) | ✅ DONE | |
| 3.3 | Preprocessing (TF-IDF, label encode) | ✅ DONE | |
| 3.4 | Train/test split (stratified 80/20) | ✅ DONE | |
| 3.5 | Baseline models (LogReg, SVM, RF, XGB, MNB, CNB) ± class_weight | ✅ DONE | |
| 3.6 | + Data augmentation (swap/delete on minority = EOS) | ✅ DONE | |
| 3.7 | + SMOTE on TF-IDF features | ✅ DONE | |
| 3.8 | + Combined (augmentation + SMOTE) | ✅ DONE | |
| 3.9 | Transformer models (SciBERT, DeBERTa) ± augmented + focal loss | ✅ DONE | |
| 3.10 | Ensemble (hard/soft voting, stacking, best-confidence) | ✅ DONE | |
| 3.11 | Threshold analysis (PR curve, F-beta optimal threshold) | ✅ DONE | Key for specialist-lab routing decisions |
| 3.12 | Results table → `results/v5a_all_results.csv` | ✅ DONE | Accuracy, Macro F1, EOS F1, EOS Prec, EOS Rec |
| 3.13 | SHAP explainability → `results/v5a_shap_summary.csv` | ✅ DONE | Top-20 features per class |
| 3.14 | Keyword analysis → `keywords/keyword_list_v5a.csv` | ✅ DONE | Rates per v5a grouping |
| 3.15 | Save best model → `models/v5a/` | ✅ DONE | + metadata.json with threshold |
| **Phase 4** | **v5b notebook** — EOS + Backend + Other (3-class) | | |
| 4.1 | Setup & data loading | ⬜ TODO | |
| 4.2 | EDA | ⬜ TODO | |
| 4.3 | Preprocessing | ⬜ TODO | |
| 4.4 | Train/test split | ⬜ TODO | |
| 4.5 | Baseline models ± class_weight | ⬜ TODO | |
| 4.6 | + Data augmentation (minority = Backend) | ⬜ TODO | |
| 4.7 | + SMOTE | ⬜ TODO | |
| 4.8 | + Combined | ⬜ TODO | |
| 4.9 | Transformer models ± augmented + focal loss | ⬜ TODO | |
| 4.10 | Ensemble | ⬜ TODO | |
| 4.11 | Threshold analysis (EOS vs Backend focus) | ⬜ TODO | |
| 4.12 | Results table → `results/v5b_all_results.csv` | ⬜ TODO | |
| 4.13 | SHAP → `results/v5b_shap_summary.csv` | ⬜ TODO | |
| 4.14 | Keyword analysis → `keywords/keyword_list_v5b.csv` | ⬜ TODO | EOS vs Backend highlighted |
| 4.15 | Save best model → `models/v5b/` | ⬜ TODO | |
| **Phase 5** | **v5c notebook** — All ~10 classes | | |
| 5.1 | Setup & data loading | ⬜ TODO | |
| 5.2 | EDA (full distribution, pairwise confusion) | ⬜ TODO | |
| 5.3 | Preprocessing | ⬜ TODO | |
| 5.4 | Train/test split | ⬜ TODO | |
| 5.5 | Baseline models ± class_weight | ⬜ TODO | |
| 5.6 | + Data augmentation (all minority classes) | ⬜ TODO | |
| 5.7 | + SMOTE | ⬜ TODO | |
| 5.8 | + Combined | ⬜ TODO | |
| 5.9 | Transformer models ± augmented + focal loss | ⬜ TODO | |
| 5.10 | Ensemble | ⬜ TODO | |
| 5.11 | Threshold analysis (per-class) | ⬜ TODO | |
| 5.12 | Results table → `results/v5c_all_results.csv` | ⬜ TODO | |
| 5.13 | SHAP → `results/v5c_shap_summary.csv` | ⬜ TODO | |
| 5.14 | Keyword analysis → `keywords/keyword_list_v5c.csv` | ⬜ TODO | Per-class rates |
| 5.15 | Save best model → `models/v5c/` | ⬜ TODO | |
| **Phase 6** | **Feature Engineering Exploration** | | |
| 6.1 | EDA on all additional Excel columns (value distributions, missing rate, predictive potential) | ⬜ TODO | Columns: `Prod Chip 11`, `Process`, `Prod Material Nr Descr`, `N0 Complainer Name`, `Nn Complainer Region Descr`, `Nn Incoming Location` |
| 6.2 | Encode promising columns (one-hot for low-cardinality categoricals, TF-IDF if text-like) | ⬜ TODO | |
| 6.3 | Build combined feature matrix: TF-IDF(PSI Failure Desc) + encoded additional features | ⬜ TODO | |
| 6.4 | Retrain best classical models (LogReg, RF) on combined features vs text-only baseline | ⬜ TODO | |
| 6.5 | Decision: if ≥1pp Macro F1 improvement → rerun v5a/b/c with combined features | ⬜ TODO | |
| 6.6 | Apply to all 3 versions (v5a/b/c) | ⬜ TODO | |
| **Phase 7** | **Verification** | | |
| 7.1 | Data integrity (row counts, null checks per CSV) | ⬜ TODO | |
| 7.2 | No data leakage (stratified split, no overlap) | ⬜ TODO | |
| 7.3 | Reproducibility (random_state=42, metadata.json) | ⬜ TODO | |
| 7.4 | Cross-version comparison vs v3 (62.1%) and v4 (54.9%) | ⬜ TODO | |
| 7.5 | Keyword quality review | ⬜ TODO | |
| 7.6 | End-to-end notebook run without errors | ⬜ TODO | |
| **Phase 8** | **Streamlit UI** | | |
| 8.1 | Decide: 3-tab app vs separate apps | ✅ DONE | Separate app per version (`streamlit_app_v5a.py`) |
| 8.2 | Fix slow loading from v4 (lazy load, caching) | ✅ DONE | `python -m streamlit`, local venv, @st.cache_resource |
| 8.3 | Add EOS threshold slider (precision/recall trade-off) | ✅ DONE | 3 presets (β=2/default/β=0.5) + continuous slider |
| 8.4 | Implementation | ✅ DONE | 500-line app with probability bars, keyword table, model features |
| 8.5 | Smoke test | ✅ DONE | Verified EOS/Non-EOS predictions, threshold slider, keyword display |

---

## Known Issues & Caveats (v5a)

| # | Issue | Severity | Detail | Status |
|---|-------|----------|--------|--------|
| K1 | TF-IDF mismatch in saved artifacts | **Critical** | Original notebook §15 saved `tfidf_aug` instead of `tfidf_baseline` for RF-SMOTE model. Model was trained on baseline TF-IDF but saved vectorizer had augmented vocabulary. | ✅ FIXED — Model retrained with `dev/retrain_v5a.py`, all artifacts now consistent. Notebooks §11/§13/§15 code corrected via `dev/fix_notebook_bugs.py`. |
| K2 | `maxabs_scaler.joblib` stale artifact | Low | RF-SMOTE doesn't need NB scaler. Old save logic saved it anyway. | ✅ FIXED — Removed. `metadata.json` has `uses_nb_scaler: false`. |
| K3 | Mixed TF-IDFs across 30-model leaderboard | Info | Each strategy (baseline/aug/smote/comb) uses its own TF-IDF vectorizer on the same raw test texts. Comparison is valid within each strategy's full pipeline but not a perfect apples-to-apples feature-level comparison. | Documented — acceptable design choice. |
| K4 | Transformer results not representative | Info | SciBERT (0.691) and DeBERTa (0.657) trained on CPU only — 2–4 epochs, 2K subsample (FULL_TRAIN=False) or 4 epochs on full data. GPU training (10+ epochs, batch=16, mixed precision) could significantly improve results. | Documented — not a bug, constraint. |
| K5 | Threshold values recalculated | Info | Original thresholds were computed with wrong TF-IDF (β=2: 0.036, β=0.5: 0.525). Corrected values after retrain: β=2: ~0.156, β=0.5: ~0.700. Both are more realistic operating points. | ✅ FIXED — New thresholds in metadata.json. |
| K6 | Customer name feature leakage | Medium | "shiroishi", "visteon" appear as top features because specific customers have higher EOS rates. Correlation, not causation. | Open — consider blocking customer-name features in retraining. |

---

## Key Design Decisions

| # | Decision | Resolution |
|---|----------|-----------|
| D1 | Data source | Only new Excel (2026-02-25, ~13.9K rows). NOT merged with old Tabelle1.csv |
| D2 | v5a classes | `Customer EOS` → **EOS**, everything else → **Non-EOS** |
| D3 | v5b classes | `Customer EOS` → **EOS**, `Backend` → **Backend**, everything else → **Other** |
| D4 | v5c classes | All original ROOT CAUSE labels. Drop `Lot disposition` (3 rows only). Keep `unknown` as valid class ("not classified") |
| D5 | Product/Family | Ignored — no filtering by Prod Chip 11 or device family |
| D6 | Primary metrics | **Macro F1** (primary) + **EOS Precision** + **EOS Recall** (all versions) |
| D7 | Threshold tuning | Single model + adjustable decision threshold — no separate models needed. UI slider later |
| D8 | Class weight penalty | `class_weight='balanced'` for classical models; **focal loss** for transformers |
| D9 | Keywords | Rates computed per version's class grouping → separate CSV per version (same raw text, different rates) |
| D10 | Notebook structure | One notebook per version (3 total), following v3/v4 pattern |
| D11 | Skip v0.4.x | Confirmed — jumping from v4 directly to v5a/b/c |
| D12 | Business goal (all versions) | EOS Precision + Recall tracked everywhere — specialist-lab routing is central to all 3 |
| D13 | Streamlit | Planned (Phase 7 placeholder), design after notebooks. Fix slow loading |

---

## Class Distributions (after non-empty `PSI Failure Desc` filter)

> Counts confirmed by `data/prepare_v5_data.py` — 0 null PSI Failure Desc rows, all 13,917 rows usable.

### v5a — 2 classes (EOS vs Non-EOS)
| Class | Count | % | Note |
|-------|-------|---|------|
| **Non-EOS** | 10,363 | 74.5% | Majority — all non-EOS root causes merged |
| **EOS** | 3,554 | 25.5% | **Minority** ⚠️ — flipped from v3/v4 where EOS was 84–88% majority |
| **Total** | **13,917** | | Imbalance ratio = 2.9x |

### v5b — 3 classes (EOS + Backend + Other)
| Class | Count | % | Note |
|-------|-------|---|------|
| **Other** | 9,383 | 67.4% | Super-majority |
| **EOS** | 3,554 | 25.5% | |
| **Backend** | 980 | 7.0% | Minority — needs augmentation |
| **Total** | **13,917** | | Imbalance ratio = 9.6x |

### v5c — ~10 classes (all root causes)
| Class | ~Count | ~% |
|-------|--------|-----|
| NTF | 5,367 | 39% |
| Customer EOS | 3,554 | 26% |
| Customer OTHER | 1,323 | 10% |
| Wafer Fab | 1,099 | 8% |
| Backend | 980 | 7% |
| Test coverage | 601 | 4% |
| Customer MOS | 373 | 3% |
| Design / Spec. | 367 | 3% |
| unknown | 144 | 1% |
| Customer TOS | 106 | 0.8% |
| ~~Lot disposition~~ | ~~3~~ | ~~dropped~~ |

> `unknown` is kept as a valid class — represents cases "not classified to any known root cause".  
> Imbalance ratio NTF:TOS ≈ 50x — requires aggressive SMOTE + augmentation + balanced class weights.

---

## Notebook Section Template (all 3 notebooks follow this)

| § | Content |
|---|---------|
| 1 | Setup & data loading — print shape, class distribution, imbalance ratio |
| 2 | EDA — class bar chart, text length histograms, word clouds, top-N words per class, overlap analysis |
| 3 | Preprocessing — lowercase/strip, TF-IDF (tune max_features: 1500/3000/5000), label encode |
| 4 | Train/test split — stratified 80/20, print distribution |
| 5 | Baseline models — LogReg(L1/L2), SVM, RF, XGBoost, MultinomialNB, ComplementNB — each ± `class_weight='balanced'` |
| 6 | + Data augmentation — random word swap + deletion on minority class(es) (expand 2–4×) |
| 7 | + SMOTE — on TF-IDF features |
| 8 | + Combined — augmentation + SMOTE |
| 9 | Transformer models — SciBERT + DeBERTa fine-tuning ± augmented data, focal loss |
| 10 | Ensemble — hard/soft voting, stacking (meta-learner), best-confidence selection |
| 11 | **Threshold analysis** — PR curve for EOS class, F-beta optimal (β=2 favors recall, β=0.5 favors precision) |
| 12 | Results comparison table — Accuracy, Macro F1, EOS F1, **EOS Precision**, **EOS Recall** → save CSV |
| 13 | SHAP explainability — LinearExplainer (LogReg), TreeExplainer (RF/XGB), top-20 positive features per class |
| 14 | Keyword analysis — per-class rates, positive EOS keywords, save CSV |
| 15 | Save best model — `best_model.joblib` + `tfidf_vectorizer.joblib` + `label_encoder.joblib` + `metadata.json` |

---

## File Structure

```
NEW FILES:
  data/prepare_v5_data.py              — shared data preparation script
  data/v5a_eos_vs_noneos.csv           — 2-class dataset
  data/v5b_eos_backend_other.csv       — 3-class dataset
  data/v5c_all_classes.csv             — ~10-class dataset
  notebooks/v5a_eos_vs_noneos.ipynb    — binary model notebook
  notebooks/v5b_eos_backend_other.ipynb — 3-class model notebook
  notebooks/v5c_all_classes.ipynb      — multi-class model notebook
  models/v5a/                          — best v5a model + metadata
  models/v5b/                          — best v5b model + metadata
  models/v5c/                          — best v5c model + metadata
  results/v5a_all_results.csv
  results/v5b_all_results.csv
  results/v5c_all_results.csv
  results/v5a_shap_summary.csv
  results/v5b_shap_summary.csv
  results/v5c_shap_summary.csv
  keywords/keyword_list_v5a.csv
  keywords/keyword_list_v5b.csv
  keywords/keyword_list_v5c.csv
  requirements_v5.txt

REFERENCE FILES (read-only, reuse patterns from):
  data/raw_v4/TC2_TC3_PL90_cases_with_root_cause_Status_2026-02-25.xlsx
  data/prepare_v3_data.py              — data prep pattern
  notebooks/v3_all_models_comparison.ipynb — notebook structure template
  notebooks/v4_all_models_comparison.ipynb — data loading, augmentation, SciBERT code
  notebooks/eos_keyword_analysis.ipynb — keyword extraction pattern
```

---

## Execution Order

```
Phase 0 (Git push v4)      ✅ DONE
Phase 1 (Data prep)        ✅ DONE
Phase 2 (Env setup)        ✅ DONE  ← .venv_v3 reused
         ↓
Phase 3 (v5a notebook)
         ↓
Phase 4 (v5b notebook)     ← learnings from v5a applied here
         ↓
Phase 5 (v5c notebook)     ← learnings from v5b applied here
         ↓
Phase 6 (Feature Engineering Exploration)  ← NEW: test other Excel columns
         ↓  (if improvement ≥1pp → loop back to Phases 3-5)
Phase 7 (Verification)
         ↓
Phase 8 (Streamlit UI)     ← design decided after notebooks
```

---

## Baseline Performance (for comparison)

| Version | Model | Accuracy | Macro F1 | EOS F1 | MOS/Non-EOS F1 | TOS F1 |
|---------|-------|----------|----------|--------|-----------------|--------|
| **v3** | MultinomialNB α=0.1 | 90.5% | **62.1%** | 94.6% | 69.6% | 22.2% |
| **v4** | MultinomialNB α=0.5 | 88.1% | 54.9% | 93.9% | 47.8% | 23.1% |
| **v5a** | TBD | — | — | — | — | — |
| **v5b** | TBD | — | — | — | — | — |
| **v5c** | TBD | — | — | — | — | — |

---

_Last updated: March 2026 — Phase 0 ✅ · Phase 1 ✅ · Phase 2 ✅ · Phase 3 ✅ · Phase 4 next_
