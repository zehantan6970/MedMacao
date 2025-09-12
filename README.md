# MedMacao
## MedMacao: A Four-Language Medical Term Normalization Dataset for Macao

Goal. Normalize patient expressions in Cantonese (colloquial) / Mandarin (zh) / English (en) / Portuguese (pt) to standard clinical terms (either zh or en).

## Dataset Overview

Size: 10,162 internal medicine entries
Languages / fields (per row):
Chinese 
English
Portuguese 
Cantonese
Cantonese_Colloquial
Category
Primary use: Normalize patient-side expressions (esp. Cantonese_Colloquial) to Chinese or English.
### Licensing notice (important)
The dataset is derived from medical terminology resources (e.g., MedDRA 28.0). Before redistribution, please ensure compliance with the original terminology license(s). If full redistribution is restricted in your jurisdiction, provide indices/mappings or a small demonstrative subset instead of full text terms.
This repository is research-only, with no clinical warranty.

## Quickstart
0) Install
pip install pandas scikit-learn sacrebleu openai anthropic requests
Set model keys (use any subset you have):
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
1) Put data
MedMacao.xlsx   and Curing test set：python normalize_eval.py --data_xlsx MedMacao.xlsx --task yuecol2std --dump_test test_yuecol_n100.xlsx --n 100 --seed 42
2) Run the main experiment (Normalization)
python scripts/normalize_eval.py \
  --data_xlsx data/MedMacao.xlsx \
  --task yuecol2std \
  --models gpt4o,claude,deepseek \
  --n 100 --K 16 --M 10 --eta 10 \
  --outdir results/yuecol

task: yuecol2std (main), also supports en2std, pt2std

models: any subset of {gpt4o, claude, deepseek} you configured

n: samples per task (default 100 for quick runs)

K: LLM samples per item (default 16)

M: RAG candidates per item (default 10)

eta: keep top-η% for confidence voting (default 10)

Outputs

results/yuecol/yuecol2std_<model>_detail.jsonl — per-item predictions

..._summary.json — per-model summary (Accuracy, Top-M hit)

Metrics

Accuracy (success if predicted zh==ref zh or en==ref en, after normalization)

Top-M hit (upper bound, whether the gold term is in retrieved candidates)
