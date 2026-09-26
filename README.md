# 02 Brand Personality Analyzer v1.9

## 목적
v1.8의 single-facet anchor baseline bias를 점검하기 위한 **controlled diagnostic version**입니다.

이번 버전은 입력 corpus, SentenceTransformer 모델, cosine similarity 계산을 그대로 유지하고,
**semantic anchor만 Aaker (1997)의 최종 42 traits ensemble로 변경**합니다.

## Aaker hierarchy
- 42 traits
- 15 facets
- 5 dimensions

계산:

```text
sentence × 42 trait prompts cosine similarity
        ↓
trait mean within each original Aaker facet
        ↓
15 facet scores
        ↓
equal facet mean within each dimension
        ↓
5 dimension scores
```

## 고정된 요소
- 모델: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- normalized embeddings
- cosine similarity
- prompt template: `A brand that is {trait}.`
- Full Sample / Balanced Sample UI
- Mean ± SD / Radar

## 이번 버전에서 아직 하지 않는 것
- facet z-score / baseline correction
- multiple prompt-template ensemble
- antonym / negative prompts
- dimension centering

이것들을 동시에 바꾸지 않는 이유는 **42-trait ensemble 자체의 효과를 v1.8과 분리해서 검증하기 위해서**입니다.

## 입력
권장 입력:

`01_sentence_proposition_02_ready.csv`

필수 열:
- Brand
- Unit_ID
- Source_URL
- Text
- Researcher_Final

## 주요 출력
- `02_brand_personality_5D_profiles_v1_9.csv`
- `02_brand_personality_15facet_profiles_v1_9.csv`
- `02_brand_personality_42trait_profiles_v1_9.csv`
- `02_brand_personality_unit_scores_v1_9.csv`
- `02_anchor_definition_v1_9.csv`
- `02_diagnostic_metrics_v1_9.csv`
- `02_sample_size_information_v1_9.csv`
- `02_content_units_researcher_approval_v1_9.csv`

## 핵심 진단값
앱은 다음 값을 자동 저장합니다.

- Mean 5D pairwise correlation
- Mean 15-facet pairwise correlation
- PC1 explained variance of standardized 15-facet brand profiles

v1.8 sentence/proposition baseline 참고값:
- 5D mean pairwise correlation ≈ 0.930
- 15-facet mean pairwise correlation ≈ 0.850
- PC1 ≈ 86.1%

v1.9에서 이 값들이 감소하는지 확인합니다.

## 주의
`Primary_Dimension`과 `Relative %`는 v1.8과의 연속성을 위해 남겨두지만,
본 논문의 최종 브랜드 분류값이나 실제 퍼센트로 해석하지 않습니다.
