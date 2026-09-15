# 02 Brand Personality Analyzer v0.2

## 변경점
- 브랜드 전체 원문을 1개 unit으로 처리하던 v0.1 수정
- 크롤링 원문의 줄바꿈/콘텐츠 블록 경계를 우선 보존
- 긴 블록만 문장 경계로 보조 분할
- 브랜드별 Unit_ID 부여
- 분석 전에 content unit 검토 CSV 다운로드 가능
- unit별 Aaker 5차원 cosine similarity 계산 후 브랜드별 평균 집계
- N_Units 기록

## 연구 근거
- Aaker (1997): 5차원 브랜드 개성.
- Pamuksuz, Yun & Humphreys (2021): 브랜드 생성 텍스트의 계산적 브랜드 개성 점수화 및 브랜드 단위 집계 접근.
- Vinyals-Mirabent, Kavaratzis & Fernández-Cavia (2019): 공식 웹사이트의 content unit 기반 정량적 내용분석. Content unit은 독립된 아이디어/주제를 나타내는 단위로 다룸.

## 주의
본 앱은 Pamuksuz et al.의 LDA2Vec/Doc2Vec/RoBERTa 모델을 복제하지 않습니다. 다국어 sentence embedding과 cosine similarity는 본 연구 파일럿의 수정 적용입니다. Relative profile도 확립된 기존 척도가 아니라 탐색적 조작화입니다.
