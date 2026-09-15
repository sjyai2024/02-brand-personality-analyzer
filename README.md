# Module 02 — Brand Personality Analyzer
승인 CSV(Include=1)만 분석합니다.

근거: Aaker (1997) 5차원; Pamuksuz, Yun & Humphreys (2021)의 계산적 브랜드 개성 측정 및 브랜드 단위 집계 접근.

주의: 본 앱은 Pamuksuz et al.의 LDA2Vec/Doc2Vec/RoBERTa 모델을 복제하지 않습니다. 석사논문 파일럿 범위에서 multilingual sentence embedding과 cosine similarity를 적용한 탐색적 조작화입니다.

입력 필수 열: Brand, Original_Text, Include
출력: unit별 원 cosine similarity, 상대 프로파일, 브랜드별 평균 프로파일.
