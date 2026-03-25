# KV Cache Hierarchical Memory Manager

Transformer 모델의 KV Cache를 중요도 기반으로 **SRAM(고속) / HBM(대용량)** 2계층으로 관리하는 연구 프로토타입.

중요도가 높은 토큰은 SRAM에 상주시키고, 낮아지면 HBM으로 내리며, 재참조되거나 의미적으로 유사한 토큰이 들어오면 다시 SRAM으로 승격한다.

## 동작 원리

```
새 토큰 입력
    │
    ▼
코사인 유사도 계산 (새 토큰 vs 캐시 전체)
    │
    ├─ 유사도 > threshold → 중요도 boost
    └─ 그 외              → 중요도 decay
    │
    ▼
SRAM 배치 결정
    ├─ 여유 있음 → SRAM에 추가
    └─ 포화     → 최저 중요도 엔트리와 비교 → evict/swap
    │
    ▼
승격 패스: HBM 중 고중요도 엔트리 → SRAM 최저와 swap
```

## 설치

```bash
pip install -r requirements.txt
```

**요구사항:** Python 3.10+, PyTorch 2.x (CUDA 또는 ROCm)

AMD ROCm 환경:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

## 빠른 시작

### 대화형 데모

```bash
python demo.py                    # GPT-2, SRAM 128
python demo.py --sram 64          # SRAM 용량 조절
python demo.py --model gpt2-medium
```

데모 명령어:
- 텍스트 입력 → 관리된 KV 캐시로 생성
- `:stats` — 캐시 통계 출력
- `:tiers` — SRAM/HBM 엔트리 상세 보기
- `:ppl <텍스트>` — perplexity 비교 (managed vs baseline)
- `:reset` — 캐시 초기화

### Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cache_manager import CacheManager, ImportanceScorer, ManagedGenerator

model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

cache = CacheManager(sram_capacity=128, dim=768, device="cuda")
gen = ManagedGenerator(model, tokenizer, cache)

# 텍스트 생성
output = gen.generate("The history of AI began", max_new_tokens=64)

# Perplexity 비교
result = gen.evaluate_perplexity("Long text here...", max_length=512)
print(f"Baseline: {result['baseline_ppl']:.2f}")
print(f"Managed:  {result['managed_ppl']:.2f}")
print(f"Ratio:    {result['ppl_ratio']:.4f}")
```

### 벤치마크

```bash
# 내장 샘플로 빠른 테스트
python -m kv_cache_manager.benchmark --sram 128 --len 256

# WikiText-103으로 본격 평가
python -m kv_cache_manager.benchmark --dataset wikitext --samples 10 --output results.json

# 하이퍼파라미터 조정
python -m kv_cache_manager.benchmark --sim-thresh 0.25 --boost 0.40 --decay 0.03
```

### Colab 노트북

`demo.ipynb`을 Google Colab에 업로드하면 A100 환경에서 바로 실행 가능.

## 프로젝트 구조

```
kv_cache_manager/
├── __init__.py            # 공개 API
├── importance_scorer.py   # 코사인 유사도 기반 중요도 계산
├── cache_manager.py       # SRAM/HBM 2-tier 캐시 관리 핵심 로직
├── attention_hook.py      # GPT-2 연동 (AttentionHook, ManagedGenerator)
└── benchmark.py           # perplexity 비교 벤치마크 CLI
demo.py                    # 대화형 CLI 데모
demo.ipynb                 # Colab 데모 노트북
requirements.txt
```

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `sram_capacity` | 128 | SRAM 최대 토큰 수 |
| `sim_threshold` | 0.30 | 유사도 판단 기준 |
| `boost` | 0.35 | 유사 시 중요도 상승 계수 |
| `decay` | 0.04 | 매 스텝 중요도 감쇠량 |
| `promote_thresh` | 0.68 | HBM → SRAM 승격 기준 |

## 검증 목표

- Perplexity 차이: baseline 대비 **+5% 이내** (`ppl_ratio ≤ 1.05`)
- SRAM 히트율 추적
- 메모리 사용량 및 처리 속도 비교

## 로드맵

- [x] **1단계 (MVP):** CacheManager 기본 동작, GPT-2 hook 연결, perplexity 비교, CLI 데모
- [ ] **2단계:** HBM 토큰 저차원 압축 (SVD / linear projection), attention weight 기반 스코어러, LLaMA-3 8B 적용

## 제한 사항

- 실제 GPU L2 캐시에 텐서를 고정하는 것은 OS/드라이버 레벨에서 불가능 — SRAM/HBM은 **논리적 구분**으로 정책을 시뮬레이션
- Representative vector는 첫 번째 레이어 key를 헤드 축으로 flatten한 768차원 벡터 사용 (sentence-transformers 기반은 2단계 예정)
- 중요도 업데이트가 O(n)이므로 매우 긴 시퀀스에서는 오버헤드 발생 가능

## 라이선스

MIT
