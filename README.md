# 02 Brand Personality Analyzer v2.0

## 목적
v1.9의 Aaker 42-trait anchor ensemble을 유지한 상태에서,
sentence embedding의 공통성분(common component / anisotropy)이
15 facets와 5 dimensions를 함께 움직이게 하는지 확인하는 통제 진단 버전입니다.

## 유지
- 240 sentence/proposition units
- `paraphrase-multilingual-MiniLM-L12-v2`
- Aaker 42 traits → 15 facets → 5 dimensions
- fixed prompt: `A brand that is {trait}.`
- normalized cosine similarity
- trait→facet→dimension 동일가중 계층집계

## v2.0에서 추가되는 유일한 핵심 변화
1. 분석 sentence embeddings의 평균벡터를 계산
2. sentence embeddings에서 평균을 제거
3. centered sentence embeddings의 top principal component 1개를 추정
4. **같은 평균벡터와 PC1 변환을 42 trait anchor embeddings에도 적용**
5. sentence와 anchor 모두 PC1 투영 제거
6. L2 재정규화
7. corrected cosine similarity 계산

중요: common component는 **sentence corpus만으로 추정**합니다.
Aaker anchors는 common direction 추정에 사용하지 않습니다.

## Raw와 Corrected를 동시에 저장
v2.0은 동일 실행에서:
- Raw v1.9-equivalent
- Mean-center + Top-1 corrected

결과를 모두 저장하여 직접 비교합니다.

## 핵심 진단
- Mean 5D pairwise correlation
- Mean 15-facet pairwise correlation
- PC1 explained variance of standardized brand-level 15-facet profiles
- facet baseline mean range

## 해석
진단지표가 낮아져도 그것만으로 최종 방법을 선택하지 않습니다.
공통요인 제거는 의미정보도 제거할 수 있으므로,
최종 채택은 향후 인간코딩/인간평가 대응 결과와 함께 판단합니다.

## 이론적 근거
- Mu, Bhat, & Viswanath (2017), All-but-the-Top:
  common mean과 상위 지배방향 제거를 통한 embedding postprocessing.
- Su et al. (2021), Whitening Sentence Representations:
  BERT 계열 sentence representation의 anisotropy 문제와 후처리 개선 가능성.
