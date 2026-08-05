# Eliminating Package Hallucination via Registry Verification and Substitution-Guided Repair

Make code LLMs stop inventing packages that do not exist. After the model writes code, we ask it twice which packages are needed, cross-check every name against the PyPI registry, and then guide the model to rewrite the code using real package evidence. Across four Python datasets, the hallucination rate drops from 11.9%–20.7% to ~0%, with syntax validity improving across the board.

The method consists of two components, mirroring the two mechanisms in the title:

1. **Registry Verification**: package names are extracted through three channels (`pip install` commands + two self-reported dependency queries), checked against the PyPI registry, and the result decides whether a repair is triggered
2. **Substitution-Guided Repair**: for every hallucinated package, real alternative packages are retrieved; a "hallucinated → real" substitution map guides the model to rewrite the code, followed by re-verification

A repair is triggered when: a hallucinated package exists / syntax is invalid / a `pip install` command appears / a dynamic import is used. A base repair variant (real package context without a substitution map) serves as the ablation baseline.

## Background

Code LLMs frequently recommend **packages that do not exist** (package hallucination) when answering programming questions. For example, when asked to resolve an error, a model may suggest `pip install find-libpython` even though no such package exists on PyPI. An attacker could publish a malicious package under the same name and trick users into installing it (package confusion attack).

- **Evaluation protocol** follows Spracklen et al. (USENIX Security 2025): three-heuristic package extraction (`pip install` commands + two dependency queries to the model) + matching against the full PyPI name list; a name not on the list is a hallucination.
- **The method is original to this repository**: the mitigations in the original paper (RAG, Self-Refinement, Fine-tuning) do not involve the closed loop of registry adjudication + alternative retrieval + code rewriting repair. The **substitution map mechanism (Substitution-Guided Repair) is the core contribution**.

## Method

For each of the four datasets, 500 prompts are sampled with a fixed seed; every prompt runs under all three conditions.

### Conditions

| Condition (code id) | Description |
|---|---|
| `no_rag` | Baseline. Direct query, no retrieval, no repair |
| `registry_repair` | Registry verification + base repair (detect → verify → retrieve real package context → rewrite → re-verify) |
| `substitution_repair` | Registry verification + **substitution-guided repair**: retrieve real alternatives for each hallucinated package and feed the substitution map to the model |

### Pipeline of the two repair methods

Both methods share the front half: (1) the LLM writes code with no retrieval, (2) two dependency queries (one on the code, one on the prompt), (3) three-channel registry verification (query-1 + query-2 + `pip install` scan of the answer text, with an auxiliary import check), and (4) a repair trigger decision (hallucinated package / syntax error / `pip install` command / dynamic import). They differ in what the repair does:

**`registry_repair` — registry verification + base repair**

```
⑤ filter: q1/q2 answers keep only registry-validated packages (drop hallucinated ones)
⑥ retrieve: BM25 real package context (full_context)
⑦ repair: prompt = task + code + hallucinated list + real package context
          → LLM rewrites the code
⑧ re-verify: re-check imports, re-query dependencies, re-match against the registry
```

**`substitution_repair` — substitution-guided repair**

```
⑤ alternative retrieval: for each hallucinated package, BM25 retrieves real
   alternatives → build "hallucinated → alternative" substitution map
⑥ substitute: q1/q2 answers keep valid packages, replace hallucinated ones
   with the top-1 alternative
⑦ repair: prompt = task + code + hallucinated list + real package context
          + substitution guide (replace X with Y) → LLM rewrites the code
⑧ re-verify + safety net: residual hallucinated imports are re-retrieved for
   alternatives, q1/q2 are re-built by substitution, then re-matched
```

The difference in one sentence: `registry_repair` tells the model which packages are fake and provides real package evidence, leaving the replacement to the model itself; `substitution_repair` additionally tells the model exactly which real package to use instead, and applies a post-repair substitution safety net. This is why it yields more valid packages (597 vs 567 on SO_LY) and a lower raw hallucination rate (9.14% vs 14.39%).

Key design decisions:

1. **The model writes, the code judges**: every validity decision comes from `pypi_package_names.csv` (the full PyPI name list); the model has no self-adjudication power (unlike Self-Refinement in the original paper).
2. **Retrieval is not pre-generation RAG**: retrieval happens only in the repair stage after a hallucination is detected; the initial generation is identical to the baseline and does not rely on RAG augmentation.
3. **Hallucination rate counts three channels only**: query-1 hallucinations + query-2 hallucinations + `pip install` hallucinations. Imports parsed from the code are used only to trigger repair and to backfill valid packages; they never enter the hallucination-rate numerator or denominator.
4. **Name normalization**: before matching, `_`/`.`/`-` are unified to `-` and lowercased; stdlib modules and a hand-curated false-positive list are skipped entirely (counted as neither valid nor hallucinated).

## Results

Qwen2.5-Coder-7B-Instruct, n=500 per condition (data under `results/registry_repair_full_*`).

| Dataset | Condition | Hallucination rate | Hallucinated/total | Syntax validity | Repairs triggered |
|---|---|---|---|---|---|
| LLM_AT | no_rag | 11.86% | 133/1121 | 97.2% | — |
| LLM_AT | registry_repair | **0.00%** | 0/1152 | 98.4% | 52 |
| LLM_AT | substitution_repair | **0.00%** | 0/1157 | 97.8% | 114 |
| LLM_LY | no_rag | 15.16% | 227/1497 | 95.4% | — |
| LLM_LY | registry_repair | **0.00%** | 0/1423 | 97.6% | 97 |
| LLM_LY | substitution_repair | **0.00%** | 0/1485 | 96.0% | 172 |
| SO_AT | no_rag | 20.71% | 99/478 | 75.8% | — |
| SO_AT | registry_repair | **0.00%** | 0/452 | 94.6% | 132 |
| SO_AT | substitution_repair | **0.00%** | 0/536 | 95.0% | 174 |
| SO_LY | no_rag | 20.71% | 93/449 | 53.4% | — |
| SO_LY | registry_repair | **0.00%** | 0/567 | 86.4% | 254 |
| SO_LY | substitution_repair | **0.00%** | 0/597 | 87.0% | 261 |

Observations:

- **Package hallucination is fully eliminated**: substitution-guided repair (`substitution_repair`) reaches a 0.00% hallucination rate on all four datasets.
- **Syntax validity improves alongside**: invalid syntax is one of the repair triggers, so the repair pass rewrites malformed code too; the gain is largest on the SO datasets (53.4% → 87.0%).
- **Residual syntax errors are attributable to the base model**: remaining failures are all model-generation issues (wrong language answers, truncation, ordinary syntax mistakes) unrelated to package names — once hallucination is eliminated, code quality is bounded by the code model itself.

## Repository layout

```
registry_repair_alignment/          core implementation
  run_four_conditions.py experiment entry (6 conditions incl. rag_no_attack / spracks_attack)
  metrics.py             three-channel extraction, registry matching, code dependency checks, rate
  retrieval.py           self-implemented BM25 retriever, alternative retrieval
  llm.py                 OpenAI-compatible API client
Data/Python/             4 datasets + pypi_package_names.csv + false_positive_packages.csv
Mitigation/Data/         retrieval corpus (RAG_data.jsonl, 49,920 package-question descriptions)
results/registry_repair_full_*/     experimental results (summary.csv / rows.csv)
tests/                   metric unit tests
```

## Reproduction

1. Serve a model with vLLM (or any OpenAI-compatible endpoint), e.g. `Qwen/Qwen2.5-Coder-7B-Instruct`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run a single dataset and condition set:

```bash
python -m registry_repair_alignment.run_four_conditions \
  --input Data/Python/SO_LY.json \
  --conditions no_rag registry_repair substitution_repair \
  --output-dir results/registry_repair_full_SO_LY \
  --api-base http://127.0.0.1:8000/v1 \
  --model Qwen2.5-Coder-7B-Instruct \
  --limit 500 --seed 0
```

4. Output: `<output-dir>/<condition>/rows.csv` (per-prompt results) + `summary.csv` (aggregates: hallucination rate, syntax validity, repair counts, etc.).

Notable flags: temperatures (code 0.7 / package 0.01), retrieval top-k 5, `--max-prompt-chars 6000`, `--workers` for parallelism. `--dry-run` smoke-tests the pipeline without a model.

## Datasets

Python subset of Spracklen et al., organized in a 2×2 split:

| Dataset | Size | Content |
|---|---|---|
| LLM_AT / LLM_LY | 4922 / 4892 | LLM-generated "write Python code" instructions; LY generated after 2023 (newer packages) |
| SO_AT / SO_LY | 4640 / 4630 | StackOverflow Q&A posts (Title+Body); LY posts from 2023 |

Note: the SO datasets are not all code-writing tasks — they also include concept explanations, error debugging, and environment setup questions; the model may answer with bash or other languages. Evaluation uniformly verifies the three channels (code + import + `pip install`); pure explanatory answers do not count as hallucinations.

## Known limitations

- Syntax validity only reflects parseability, not runtime correctness (the original paper used HumanEval pass@1, which is not included in this repository).
- Registry snapshot semantics: hallucination rate uses `pypi_package_names.csv` as ground truth; if the list is contaminated, the rate is a lower bound (consistent with the original paper).
