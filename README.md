# 02 Brand Personality Analyzer v1.0

## 최종 구조
01 승인 CSV → content unit → 연구자 승인 → Aaker (1997) 15 facets cosine similarity → 동일 차원 facet 평균 → 5-dimensional brand personality profile.

### Aaker 15 facets
Sincerity: Down-to-earth, Honest, Wholesome, Cheerful
Excitement: Daring, Spirited, Imaginative, Up-to-date
Competence: Reliable, Intelligent, Successful
Sophistication: Upper-class, Charming
Ruggedness: Outdoorsy, Tough

## 방법론 구분
- Aaker (1997): 5 dimensions / 15 facets의 이론적 구조.
- Vinyals-Mirabent et al. (2019): 공식 웹사이트의 content-unit 기반 분석 근거.
- Pamuksuz, Yun & Humphreys (2021): 브랜드 생성 텍스트를 계산적으로 브랜드 개성 차원에 연결하고 브랜드 단위로 집계하는 접근의 근거.
- Eisend & Stokburger-Sauer (2013): trait/facet 수준 등 브랜드 개성 측정 특성에 관한 메타분석 참고.
- Sung & Tinkham (2005): 한국 문화권의 브랜드 개성 구조 차이에 대한 이론적/파일럿 검토 근거.

## 본 연구의 탐색적 조작화
Sentence Transformer embedding, facet-anchor cosine similarity, 같은 dimension facet 점수의 arithmetic mean, Relative Profile은 위 선행연구의 기존 측정식을 그대로 복제한 것이 아니다. 본 연구가 Aaker 구조를 계산적 텍스트 분석에 적용하기 위해 설정한 탐색적 조작화이다.
