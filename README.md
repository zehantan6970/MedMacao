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

## Run experiments
A) Retrieval‐only baseline (Top-1, no LLM)


python normalize_eval.py --data_xlsx data\MedMacao.xlsx --test_xlsx data\test_yuecol_n100.xlsx ^
  --task yuecol2std --models deepseek --mode retrieval --metric code ^
  --outdir results_noLLM\top1

B) RAG + Majority voting

set DEEPSEEK_API_KEY=sk-...
python normalize_eval.py --data_xlsx data\MedMacao.xlsx --test_xlsx data\test_yuecol_n100.xlsx ^
  --task yuecol2std --models deepseek --mode majority --metric code ^
  --K 16 --M 10 --outdir results_RAGmajority\baseline

C) RAG + DeepConf

set DEEPSEEK_API_KEY=sk-...
python normalize_eval.py --data_xlsx data\MedMacao.xlsx --test_xlsx data\test_yuecol_n100.xlsx ^
  --task yuecol2std --models deepseek --mode deepconf --metric code ^
  --K 16 --M 10 --eta 10 --temp 0.2 --w_sim 0.4 --w_self 0.6 --fewshot 1 ^
  --outdir results\deepseek_dc_k16_m10_eta10_t02

set OPENAI_API_KEY=sk-...
set DEEPSEEK_API_KEY=
python normalize_eval.py --data_xlsx data\MedMacao.xlsx --test_xlsx data\test_yuecol_n100.xlsx ^
  --task yuecol2std --models gpt4o --mode deepconf --metric code ^
  --K 16 --M 10 --eta 10 --temp 0.2 --w_sim 0.4 --w_self 0.6 --fewshot 1 ^
  --outdir results\gpt4o_dc_k16_m10_eta10_t02

Ablations

:: K = 8/16/32
for %k in (8 16 32) do python normalize_eval.py --data_xlsx data\MedMacao.xlsx ^
  --test_xlsx data\test_yuecol_n100.xlsx --task yuecol2std --models deepseek ^
  --mode deepconf --metric code --K %k --M 10 --eta 10 --temp 0.2 ^
  --w_sim 0.4 --w_self 0.6 --outdir results_k8k16k32\ablation

:: M = 5/10/20
for %m in (5 10 20) do python normalize_eval.py --data_xlsx data\MedMacao.xlsx ^
  --test_xlsx data\test_yuecol_n100.xlsx --task yuecol2std --models deepseek ^
  --mode deepconf --metric code --K 16 --M %m --eta 10 --temp 0.2 ^
  --w_sim 0.4 --w_self 0.6 --outdir results_m5m10m20\ablation

:: η = 10 vs 90 (retention %)
python normalize_eval.py --data_xlsx data\MedMacao.xlsx --test_xlsx data\test_yuecol_n100.xlsx ^
  --task yuecol2std --models gpt4o --mode deepconf --metric code ^
  --K 16 --M 10 --eta 90 --temp 0.2 --w_sim 0.4 --w_self 0.6 ^
  --outdir results\ablation_eta90

  Command-line options

--data_xlsx PATH            # dataset (xlsx)
--task {yuecol2std,en2std,pt2std}
--models {gpt4o,deepseek}   # choose one per run
--mode {deepconf,majority,retrieval}
--metric {code,text}        # usually `code` (strict text match)
--n N                       # sample size when dumping test slice
--seed SEED                 # sampling seed
--test_xlsx PATH            # fixed test slice (if provided, no sampling)
--dump_test PATH            # export the sampled slice to xlsx
--K 8|16|32                 # LLM samples per query
--M 5|10|20                 # retrieval candidate count
--eta 10|90                 # retention % for DeepConf
--temp 0.2|0.7              # LLM temperature
--w_sim 0.4                 # weight for retrieval similarity
--w_self 0.6                # weight for self confidence (w_sim + w_self = 1)
--fewshot 0|1|2             # prepend few-shot demonstrations if >0
--outdir DIR                # where to save summaries & details





