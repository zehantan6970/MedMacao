# -*- coding: utf-8 -*-
# normalize_eval.py  （2025-09 版）
# - 固定测试集：--dump_test / --test_xlsx
# - 评测口径：--metric code（按“编码”）或 text（按“术语字符串”）
# - few-shot 开关：--fewshot 1/0
# - 合法 ID 约束已加入 Prompt
# - DeepConf 聚合权重：--w_sim / --w_self

import os, re, json, argparse, random, string
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def set_seed(seed=42):
    random.seed(seed)

_PUNC = set(" \t\r\n" + string.punctuation + "，。、《》！？；：（）【】“”‘’·—-")
def norm_text(s: str) -> str:
    if s is None: return ""
    s = s.strip().lower()
    s = "".join(ch for ch in s if ch not in _PUNC)
    return s

def success_text(pred_zh: str, pred_en: str, ref_zh: str, ref_en: str) -> int:
    nz, ne = norm_text(pred_zh), norm_text(pred_en)
    rz, re = norm_text(ref_zh), norm_text(ref_en)
    return int((nz and nz == rz) or (ne and ne == re))

# ------------- 数据加载与抽样 -------------
REQ_COLS = ["编码","Chinese","English","Portuguese","Cantonese","Cantonese_Colloquial"]
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Category" not in df.columns and "SOC分类" in df.columns:
        df = df.rename(columns={"SOC分类": "Category"})
    return df

def ensure_columns(df: pd.DataFrame):
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}. Found: {list(df.columns)}")

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if "Category" in df.columns:
        groups = df.groupby("Category")
        sizes = (groups.size() / len(df) * n).round().astype(int).to_dict()
        picks = []
        for cat, g in groups:
            k = max(0, min(int(sizes.get(cat, 0)), len(g)))
            if k > 0:
                picks.append(g.sample(k, random_state=seed))
        sdf = pd.concat(picks) if picks else df.sample(n, random_state=seed)
        # 若四舍五入产生少量偏差，补齐/裁剪
        if len(sdf) < n:
            rest = n - len(sdf)
            extra = df.drop(sdf.index).sample(min(rest, len(df)-len(sdf)), random_state=seed)
            sdf = pd.concat([sdf, extra]).head(n)
        elif len(sdf) > n:
            sdf = sdf.sample(n, random_state=seed)
    else:
        sdf = df.sample(n, random_state=seed)
    return sdf

def df_to_records(task: str, sdf: pd.DataFrame) -> List[Dict[str,Any]]:
    if task == "yuecol2std":
        src_col = "Cantonese_Colloquial"
    elif task == "en2std":
        src_col = "English"
    elif task == "pt2std":
        src_col = "Portuguese"
    else:
        raise ValueError("task must be one of: yuecol2std, en2std, pt2std")

    recs = []
    for i, r in sdf.reset_index(drop=True).iterrows():
        recs.append({
            "row_idx": int(i),
            "code": int(r.get("编码")) if pd.notna(r.get("编码")) else int(i),
            "src": str(r.get(src_col, "")),
            "ref_zh": str(r.get("Chinese","")),
            "ref_en": str(r.get("English","")),
            "cat": str(r.get("Category",""))
        })
    return recs

def load_records(data_xlsx: str, task: str, n: int, seed: int,
                 test_xlsx: str=None, dump_test: str=None) -> Tuple[List[Dict[str,Any]], pd.DataFrame]:
    df = pd.read_excel(data_xlsx)
    df = normalize_columns(df)
    ensure_columns(df)

    if dump_test:
        sdf = stratified_sample(df, n, seed)
        sdf.to_excel(dump_test, index=False)
        print(f"[OK] dumped fixed test set: {dump_test}  (n={len(sdf)}, seed={seed})")
        return [], df  # 提前返回，主流程会检测到空 records 并退出

    if test_xlsx:
        sdf = pd.read_excel(test_xlsx)
        sdf = normalize_columns(sdf)
        ensure_columns(sdf)
        print(f"[OK] loaded fixed test set: {test_xlsx}  (n={len(sdf)})")
    else:
        sdf = stratified_sample(df, n, seed)
        print(f"[OK] sampled test set from {os.path.basename(data_xlsx)}  (n={len(sdf)}, seed={seed})")

    recs = df_to_records(task, sdf)
    return recs, df

# ------------- 构建术语库索引（R） -------------
def build_index(df: pd.DataFrame):
    zh = df.get("Chinese", pd.Series([""]*len(df))).astype(str).tolist()
    en = df.get("English", pd.Series([""]*len(df))).astype(str).tolist()
    yue = df.get("Cantonese", pd.Series([""]*len(df))).astype(str).tolist()
    codes = df.get("编码", pd.Series(list(range(len(df))))).astype(int).tolist()

    texts = [f"{a} | {b} | {c}".strip() for a,b,c in zip(zh, en, yue)]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), min_df=1)
    X = vec.fit_transform(texts)

    idx2code = {i: codes[i] for i in range(len(codes))}
    code2idx = {codes[i]: i for i in range(len(codes))}
    return vec, X, zh, en, yue, idx2code, code2idx

def retrieve_topM(vec, X, query: str, M: int = 10) -> List[Tuple[int,float]]:
    qv = vec.transform([str(query)])
    sim = cosine_similarity(qv, X)[0]
    idxs = sim.argsort()[::-1][:M]
    return [(int(i), float(sim[i])) for i in idxs]

# ------------- Prompt（A） -------------
def make_allowed_ids_line(cands_idx_sim, idx2code):
    ids = [str(idx2code[idx]) for idx,_ in cands_idx_sim]
    return "允许的 choice_id 列表：[" + ",".join(ids) + "]（你的 choice_id 必须从此列表中选择）"

def build_cand_block(cands_idx_sim, idx2code, zh_terms, en_terms):
    lines, sub_code2idx = [], {}
    for idx, sim in cands_idx_sim:
        code = idx2code[idx]
        zh   = zh_terms[idx]
        en   = en_terms[idx]
        lines.append(f"[{code}] zh: {zh}  |  en: {en}")
        sub_code2idx[code] = idx
    return "\n".join(lines), sub_code2idx

def build_prompt(src, cand_block, allowed_line, fewshot=True):
    FEWSHOT = """示例（仅演示格式，非本题候选）：
患者表述：
肚痛
候选标准术语（仅能从上列候选中选择一项）：
[101] zh: 腹痛  |  en: Abdominal pain
[102] zh: 腹部不适  |  en: Abdominal discomfort
正确输出：
{"choice_id": 101, "term_zh": "腹痛", "term_en": "Abdominal pain", "confidence": 0.95}

现在请处理下面的样本：
"""
    BASE = f"""你是一名医学术语标准化助手。请仅从下面候选中选择一项最匹配的“标准术语”（可输出中文或英文两者之一或两者皆可），严格输出JSON，不要解释。

患者表述：
{src}

候选标准术语（仅能从上列候选中选择一项）：
{cand_block}

{allowed_line}

请输出JSON（务必只选一个已有候选的id，不要编造新术语；若无法判断，选最接近的一项）：
{{
  "choice_id": <从方括号里的编号里选一个整数>,
  "term_zh": "<填写该候选对应中文术语或留空>",
  "term_en": "<填写该候选对应英文术语或留空>",
  "confidence": <0到1之间的小数，表示你对此选择的把握>
}}
"""
    return (FEWSHOT + BASE) if fewshot else BASE

_JSON_RE = re.compile(r"\{[\s\S]*\}")
def parse_json(s: str) -> Dict[str,Any]:
    m = _JSON_RE.search(s or "")
    if not m:
        return {"choice_id": -1, "term_zh":"", "term_en":"", "confidence":0}
    try:
        obj = json.loads(m.group(0))
        cid = int(obj.get("choice_id", -1))
        cz = str(obj.get("term_zh","")).strip()
        ce = str(obj.get("term_en","")).strip()
        conf = float(obj.get("confidence", 0))
        conf = max(0.0, min(1.0, conf))
        return {"choice_id": cid, "term_zh": cz, "term_en": ce, "confidence": conf}
    except:
        return {"choice_id": -1, "term_zh":"", "term_en":"", "confidence":0}

# ------------- LLM 接口 -------------
def call_gpt4o(prompt: str, temperature: float = 0.2) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=temperature, top_p=0.95, n=1, max_tokens=256
        )
        return resp.choices[0].message.content
    except Exception:
        return '{"choice_id": -1, "term_zh":"", "term_en":"", "confidence":0}'

def call_deepseek(prompt: str, temperature: float = 0.2) -> str:
    try:
        import requests
        url = "https://api.deepseek.com/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
                   "Content-Type":"application/json"}
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role":"user","content":prompt}],
            "temperature": temperature, "top_p": 0.95, "n": 1, "max_tokens": 256
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        j = r.json()
        return j["choices"][0]["message"]["content"]
    except Exception:
        return '{"choice_id": -1, "term_zh":"", "term_en":"", "confidence":0}'

def call_claude(prompt: str, temperature: float = 0.2) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=256, temperature=temperature,
            messages=[{"role":"user","content":prompt}]
        )
        return "".join([b.text for b in msg.content if getattr(b, "type", "")=="text"])
    except Exception:
        return '{"choice_id": -1, "term_zh":"", "term_en":"", "confidence":0}'

def call_model(name: str, prompt: str, temperature=0.2) -> str:
    if name == "gpt4o":   return call_gpt4o(prompt, temperature)
    if name == "deepseek":return call_deepseek(prompt, temperature)
    if name == "claude":  return call_claude(prompt, temperature)
    return '{"choice_id": -1, "term_zh":"", "term_en":"", "confidence":0}'

# ------------- DeepConf（离线简化版聚合） -------------
def aggregate_deepconf(samples: List[Dict[str,Any]], eta: int,
                       norm_sims_by_code: Dict[int,float],
                       w_sim=0.7, w_self=0.3) -> Tuple[int,Dict]:
    scored = []
    for s in samples:
        code = int(s.get("choice_id",-1))
        if code < 0: continue
        selfc = float(s.get("confidence",0))
        sim = float(norm_sims_by_code.get(code, 0.0))
        score = w_sim*sim + w_self*selfc
        scored.append((code, score))
    if not scored:
        return -1, {"kept":0, "classes":0}
    scored.sort(key=lambda x: x[1], reverse=True)
    keep_n = max(1, int(len(scored) * (eta/100.0)))
    kept = scored[:keep_n]
    bucket = defaultdict(float)
    for code, score in kept:
        bucket[code] += score
    best_code, _ = max(bucket.items(), key=lambda kv: kv[1])
    return best_code, {"kept": keep_n, "classes": len(bucket)}

# ------------- 三种运行模式 -------------
def run_retrieval_only(rec, vec, X, zh_terms, en_terms, idx2code):
    tops = retrieve_topM(vec, X, rec["src"], M=1)
    if not tops:
        return "", "", 0, 0, {"mode":"retrieval"}
    idx = tops[0][0]
    return zh_terms[idx], en_terms[idx], 1, 1, {"mode":"retrieval", "best_code": idx2code[idx]}

def run_rag_majority(model, rec, vec, X, zh_terms, en_terms, idx2code, K=16, M=10, temperature=0.2, fewshot=True):
    cands = retrieve_topM(vec, X, rec["src"], M=M)
    if not cands:
        return "", "", 0, 0, {"mode":"majority"}
    sims = [sim for _, sim in cands]
    smin, smax = (min(sims), max(sims)) if sims else (0.0, 1.0)
    denom = (smax - smin + 1e-6)
    norm_by_code = {idx2code[idx]: (sim - smin)/denom for idx, sim in cands}
    cand_block, sub_map = build_cand_block(cands, idx2code, zh_terms, en_terms)
    allowed = make_allowed_ids_line(cands, idx2code)
    prompt = build_prompt(rec["src"], cand_block, allowed, fewshot=fewshot)

    votes = defaultdict(int)
    for _ in range(K):
        out = call_model(model, prompt, temperature=temperature if K>1 else 0.0)
        js = parse_json(out)
        cid = int(js.get("choice_id",-1))
        if cid in sub_map:
            votes[cid] += 1
    if not votes:
        idx = cands[0][0]
    else:
        best_code, _ = max(votes.items(), key=lambda kv: kv[1])
        idx = sub_map.get(best_code, cands[0][0])

    topM_hit = 1
    return zh_terms[idx], en_terms[idx], 1, topM_hit, {"mode":"majority","best_code": idx2code[idx]}

def run_rag_deepconf(model, rec, vec, X, zh_terms, en_terms, idx2code,
                     K=16, M=10, eta=10, temperature=0.2, w_sim=0.7, w_self=0.3, fewshot=True):
    cands = retrieve_topM(vec, X, rec["src"], M=M)
    if not cands:
        return "", "", 0, 0, {"mode":"deepconf"}
    sims = [sim for _, sim in cands]
    smin, smax = (min(sims), max(sims)) if sims else (0.0, 1.0)
    denom = (smax - smin + 1e-6)
    norm_by_code = {idx2code[idx]: (sim - smin)/denom for idx, sim in cands}

    cand_block, sub_map = build_cand_block(cands, idx2code, zh_terms, en_terms)
    allowed = make_allowed_ids_line(cands, idx2code)
    prompt = build_prompt(rec["src"], cand_block, allowed, fewshot=fewshot)

    samples = []
    for _ in range(K):
        out = call_model(model, prompt, temperature=temperature if K>1 else 0.0)
        js = parse_json(out)
        if int(js.get("choice_id",-1)) in sub_map:
            samples.append(js)

    if not samples:
        idx = cands[0][0]
        best_code = idx2code[idx]
    else:
        best_code, meta = aggregate_deepconf(samples, eta=eta, norm_sims_by_code=norm_by_code,
                                             w_sim=w_sim, w_self=w_self)
        idx = sub_map.get(best_code, cands[0][0])

    topM_hit = 1
    return zh_terms[idx], en_terms[idx], 1, topM_hit, {"mode":"deepconf","best_code": idx2code[idx]}

# ------------- 主流程 -------------
def main(args):
    set_seed(args.seed)
    records, full_df = load_records(args.data_xlsx, args.task, args.n, args.seed,
                                    test_xlsx=args.test_xlsx, dump_test=args.dump_test)
    if args.dump_test:
        return  # 仅导出测试集

    vec, X, zh_terms, en_terms, yue_terms, idx2code, code2idx = build_index(full_df)
    os.makedirs(args.outdir, exist_ok=True)

    # 方便论文配图：写两份示例
    ex1 = """你是一名医学术语标准化助手。请仅从下面候选中选择一项最匹配的“标准术语”（可输出中文或英文两者之一或两者皆可），严格输出JSON，不要解释。

患者表述：
肚痛

候选标准术语（仅能从上列候选中选择一项）：
[10000039] zh: 腹痛  |  en: Abd. pain
[10000040] zh: 腹痛  |  en: Abdo pain
[10000042] zh: 腹部不适  |  en: Abdo. discomfort
允许的 choice_id 列表：[10000039,10000040,10000042]（你的 choice_id 必须从此列表中选择）

正确输出（示例）：
{"choice_id": 10000039, "term_zh": "腹痛", "term_en": "Abd. pain", "confidence": 0.95}
"""
    with open(os.path.join(args.outdir,"PROMPT_EXAMPLE_1.txt"),"w",encoding="utf-8") as f: f.write(ex1)

    ex2 = """你是一名医学术语标准化助手。请仅从下面候选中选择一项最匹配的“标准术语”，严格输出JSON。
患者表述：
突然間透唔到氣好辛苦
候选标准术语（仅能从上列候选中选择一项）：
[10001052] zh: 急性呼吸窘迫綜合症  |  en: Acute respiratory distress syndrome
[10001104] zh: 部位未明急性上呼吸道感染  |  en: Acute upper respiratory infections of unspecified site
允许的 choice_id 列表：[10001052,10001104]（你的 choice_id 必须从此列表中选择）
"""
    with open(os.path.join(args.outdir,"PROMPT_EXAMPLE_2.txt"),"w",encoding="utf-8") as f: f.write(ex2)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        rows = []
        succ, hit = 0, 0
        for i, rec in enumerate(records):
            if args.mode == "retrieval":
                pred_zh, pred_en, ok, top_hit, meta = run_retrieval_only(rec, vec, X, zh_terms, en_terms, idx2code)
            elif args.mode == "majority":
                pred_zh, pred_en, ok, top_hit, meta = run_rag_majority(m, rec, vec, X, zh_terms, en_terms, idx2code,
                                                                       K=args.K, M=args.M, temperature=args.temp,
                                                                       fewshot=bool(args.fewshot))
            else:
                pred_zh, pred_en, ok, top_hit, meta = run_rag_deepconf(m, rec, vec, X, zh_terms, en_terms, idx2code,
                                                                       K=args.K, M=args.M, eta=args.eta, temperature=args.temp,
                                                                       w_sim=args.w_sim, w_self=args.w_self,
                                                                       fewshot=bool(args.fewshot))
            # 评测口径
            if args.metric == "code":
                pred_code = int(meta.get("best_code",-1))
                success = int(pred_code == rec["code"])
            else:  # text
                success = success_text(pred_zh, pred_en, rec["ref_zh"], rec["ref_en"])

            succ += success; hit += top_hit
            rows.append({
                "idx": i, "code": rec["code"], "src": rec["src"], "cat": rec["cat"],
                "ref_zh": rec["ref_zh"], "ref_en": rec["ref_en"],
                "pred_zh": pred_zh, "pred_en": pred_en,
                "pred_code": meta.get("best_code",-1),
                "success": success, "topM_hit": top_hit, "meta": meta
            })
            if (i+1) % 10 == 0:
                print(f"[{m}/{args.mode}] {i+1}/{len(records)}")

        acc = succ / max(1,len(records))
        topM = hit / max(1,len(records))
        print(f"[{args.task}] {m} ({args.mode})  Accuracy={acc:.3f}  TopMHit={topM:.3f}  n={len(records)}  metric={args.metric}")

        # 输出文件
        detail = os.path.join(args.outdir, f"{args.task}_{m}_{args.mode}_detail.jsonl")
        with open(detail, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False)+"\n")
        summary = {
            "task": args.task, "model": m, "mode": args.mode, "n": len(records),
            "accuracy": acc, "topM_hit": topM,
            "K": args.K, "M": args.M, "eta": args.eta, "temp": args.temp,
            "metric": args.metric, "fewshot": int(args.fewshot),
            "w_sim": args.w_sim, "w_self": args.w_self
        }
        with open(os.path.join(args.outdir, f"{args.task}_{m}_{args.mode}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_xlsx", type=str, required=True)
    ap.add_argument("--task", type=str, choices=["yuecol2std","en2std","pt2std"], required=True)
    ap.add_argument("--models", type=str, required=True, help="e.g., gpt4o,deepseek,claude")
    ap.add_argument("--mode", type=str, choices=["deepconf","majority","retrieval"], default="deepconf")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--eta", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--w_sim", type=float, default=0.7)
    ap.add_argument("--w_self", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="results_norm")
    ap.add_argument("--fewshot", type=int, default=1)
    # 新增：
    ap.add_argument("--test_xlsx", type=str, default=None)
    ap.add_argument("--dump_test", type=str, default=None)
    ap.add_argument("--metric", type=str, choices=["code","text"], default="code")
    args = ap.parse_args()
    main(args)
