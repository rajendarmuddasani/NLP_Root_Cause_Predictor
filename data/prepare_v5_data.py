"""
prepare_v5_data.py
==================
Phase 1 — V5 Data Preparation

Loads the raw Excel (2026-02-25, ~13.9K rows), filters to rows with
non-empty PSI Failure Desc, and produces three CSV files:

  data/v5a_eos_vs_noneos.csv          — 2 classes:  EOS | Non-EOS
  data/v5b_eos_backend_other.csv      — 3 classes:  EOS | Backend | Other
  data/v5c_all_classes.csv            — ~10 classes: all ROOT CAUSE labels
                                         (Lot disposition dropped; unknown kept)

Usage:
    python data/prepare_v5_data.py
"""

import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_PATH = os.path.join(BASE_DIR, 'data', 'raw_v4',
                          'TC2_TC3_PL90_cases_with_root_cause_Status_2026-02-25.xlsx')

OUT_V5A = os.path.join(BASE_DIR, 'data', 'v5a_eos_vs_noneos.csv')
OUT_V5B = os.path.join(BASE_DIR, 'data', 'v5b_eos_backend_other.csv')
OUT_V5C = os.path.join(BASE_DIR, 'data', 'v5c_all_classes.csv')

# ── Column names (as they appear after header=1 parse) ───────────────────
TEXT_COL  = 'PSI Failure Desc'
LABEL_COL = 'ROOT CAUSE'
ID_COL    = 'Nn Notif Nr'

# ── Class labels ──────────────────────────────────────────────────────────
DROP_CLASSES  = ['Lot disposition']   # too few rows (3) — unlearnable
KEEP_COLS     = [ID_COL, TEXT_COL, LABEL_COL]

# ── v5b mapping ───────────────────────────────────────────────────────────
def map_v5b(label: str) -> str:
    if label == 'Customer EOS':
        return 'EOS'
    if label == 'Backend':
        return 'Backend'
    return 'Other'


def _print_distribution(df: pd.DataFrame, label_col: str, title: str) -> None:
    counts = df[label_col].value_counts()
    total  = len(df)
    print(f"\n  {title}  (total: {total:,} rows)")
    print(f"  {'Class':<30} {'Count':>6}  {'%':>6}")
    print(f"  {'-'*46}")
    for cls, cnt in counts.items():
        print(f"  {str(cls):<30} {cnt:>6,}  {cnt/total*100:>5.1f}%")
    imb = counts.iloc[0] / counts.iloc[-1]
    print(f"  Imbalance ratio (majority:minority) = {imb:.1f}x")


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("  V5 Data Preparation")
print("=" * 62)

# ── Step 1: Load Excel ─────────────────────────────────────────────────────
print(f"\n📂 Loading Excel: {os.path.basename(EXCEL_PATH)} …")
xl   = pd.ExcelFile(EXCEL_PATH)
print(f"   Sheets: {xl.sheet_names}")
df_raw = xl.parse('CMA Content', header=1)
df_raw.columns = df_raw.columns.str.strip()
print(f"   Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
print(f"   Columns: {list(df_raw.columns)}")

# ── Step 2: Identify actual column names (defensive) ──────────────────────
# Require both 'FAIL' and 'DESC' to avoid matching 'PSI Item Nr' etc.
text_candidates  = [c for c in df_raw.columns if 'FAIL' in c.upper() and 'DESC' in c.upper()]
if not text_candidates:  # fallback: any column with 'FAIL'
    text_candidates = [c for c in df_raw.columns if 'FAIL' in c.upper()]
label_candidates = [c for c in df_raw.columns if 'ROOT' in c.upper() or 'CAUSE' in c.upper()]
id_candidates    = [c for c in df_raw.columns if 'NOTIF' in c.upper() or 'NR' in c.upper()]

found_text  = text_candidates[0]  if text_candidates  else TEXT_COL
found_label = label_candidates[0] if label_candidates else LABEL_COL
found_id    = id_candidates[0]    if id_candidates    else ID_COL

print(f"\n   Resolved columns:")
print(f"     text  → '{found_text}'")
print(f"     label → '{found_label}'")
print(f"     id    → '{found_id}'")

# ── Step 3: Keep needed columns, drop entirely-null label rows ─────────────
available_keep = [c for c in [found_id, found_text, found_label] if c in df_raw.columns]
df = df_raw[available_keep].copy()
df.columns = [ID_COL if c == found_id else
              TEXT_COL if c == found_text else
              LABEL_COL for c in df.columns]
df = df.dropna(subset=[LABEL_COL])
print(f"\n   Rows with ROOT CAUSE label: {len(df):,}")

# ── Step 4: Root-cause distribution BEFORE text filter ─────────────────────
print("\n📊 Full ROOT CAUSE distribution (ALL rows, including empty text):")
raw_counts = df[LABEL_COL].value_counts()
total_all  = len(df)
print(f"\n  {'Class':<30} {'Count':>6}  {'%':>6}")
print(f"  {'-'*46}")
for cls, cnt in raw_counts.items():
    print(f"  {str(cls):<30} {cnt:>6,}  {cnt/total_all*100:>5.1f}%")

# ── Step 5: Filter to non-empty PSI Failure Desc ──────────────────────────
print(f"\n🔍 Filtering to non-empty '{TEXT_COL}' …")
null_text = df[TEXT_COL].isna().sum()
print(f"   Rows with null text: {null_text:,}")
df = df.dropna(subset=[TEXT_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL].str.len() > 0]
df = df.reset_index(drop=True)
print(f"   Rows after filter:   {len(df):,}")

# ── Step 6: Full distribution AFTER text filter ────────────────────────────
print("\n📊 Full ROOT CAUSE distribution (non-empty PSI Failure Desc only):")
_print_distribution(df, LABEL_COL, "All classes")

# ── Step 7: Build v5a — 2 classes ─────────────────────────────────────────
print("\n" + "─" * 62)
print("  Generating v5a: EOS vs Non-EOS")
df_v5a = df.copy()
df_v5a['label'] = df_v5a[LABEL_COL].apply(
    lambda x: 'EOS' if x == 'Customer EOS' else 'Non-EOS'
)
_print_distribution(df_v5a, 'label', 'v5a (2-class)')
df_v5a[[ID_COL, TEXT_COL, 'label']].to_csv(OUT_V5A, index=False, encoding='utf-8')
print(f"   💾 Saved → {OUT_V5A}")

# ── Step 8: Build v5b — 3 classes ─────────────────────────────────────────
print("\n" + "─" * 62)
print("  Generating v5b: EOS | Backend | Other")
df_v5b = df.copy()
df_v5b['label'] = df_v5b[LABEL_COL].apply(map_v5b)
_print_distribution(df_v5b, 'label', 'v5b (3-class)')
df_v5b[[ID_COL, TEXT_COL, 'label']].to_csv(OUT_V5B, index=False, encoding='utf-8')
print(f"   💾 Saved → {OUT_V5B}")

# ── Step 9: Build v5c — ~10 classes ───────────────────────────────────────
print("\n" + "─" * 62)
print(f"  Generating v5c: all classes (dropping: {DROP_CLASSES})")
df_v5c = df[~df[LABEL_COL].isin(DROP_CLASSES)].copy()
df_v5c['label'] = df_v5c[LABEL_COL]
_print_distribution(df_v5c, 'label', 'v5c (~10-class)')
dropped = len(df) - len(df_v5c)
print(f"   Dropped rows from {DROP_CLASSES}: {dropped}")
print(f"   Kept 'unknown' class as valid (= not classified to any root cause)")
df_v5c[[ID_COL, TEXT_COL, 'label']].to_csv(OUT_V5C, index=False, encoding='utf-8')
print(f"   💾 Saved → {OUT_V5C}")

# ── Step 10: Final integrity summary ──────────────────────────────────────
print("\n" + "=" * 62)
print("  SUMMARY")
print("=" * 62)
for path, name in [(OUT_V5A, 'v5a'), (OUT_V5B, 'v5b'), (OUT_V5C, 'v5c')]:
    loaded = pd.read_csv(path)
    null_text_check  = loaded[TEXT_COL].isna().sum()
    null_label_check = loaded['label'].isna().sum()
    classes = loaded['label'].nunique()
    print(f"\n  {name}: {len(loaded):,} rows | {classes} classes | "
          f"null text={null_text_check} | null label={null_label_check}")
    print(f"       classes: {sorted(loaded['label'].unique())}")

print("\n✅ Phase 1 data preparation complete!\n")
