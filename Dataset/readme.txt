Dataset Overview

Size: 10,162 internal medicine entries

Languages / fields (per row):

Chinese — 标准中文医学术语（目标之一）

English — Standard English medical term（另一目标）

Portuguese — Standard Portuguese term

Cantonese — 粤语正式/书面术语（如有）

Cantonese_Colloquial — 粤语口语患者描述（新增维度）

Category — 内科类别（见统计）

Primary use: Normalize patient-side expressions (esp. Cantonese_Colloquial) to Chinese or English.

Licensing notice (important)
The dataset is derived from medical terminology resources (e.g., MedDRA 28.0). Before redistribution, please ensure compliance with the original terminology license(s). If full redistribution is restricted in your jurisdiction, provide indices/mappings or a small demonstrative subset instead of full text terms.
This repository is research-only, with no clinical warranty.

Category Statistics (10,162 entries)
Category (zh)	Count	Percent
胃肠系统疾病	4,769	47.0%
感染及侵染类疾病	1,888	18.6%
呼吸系统疾病	993	9.8%
肝胆系统疾病	696	6.8%
血液系统疾病	497	4.9%
肾脏及尿路疾病	497	4.9%
内分泌及代谢疾病	497	4.9%
神经系统疾病	227	2.2%
其他	98	1.0%

generate test_yuecol_n100.xlsx：
python normalize_eval.py --data_xlsx MedMacao.xlsx  --task yuecol2std  --models deepseek  --dump_test test_yuecol_n100.xlsx --n 100 --seed 42
