# 02 Brand Personality Analyzer v1.5

파일럿 안정화 버전.

## 분석 모드
- Full Sample — Primary Analysis: 연구자가 승인한 모든 브랜드 관련 Content Unit 사용.
- Balanced Sample — Robustness Check: 브랜드별 동일 N을 random seed=42로 표집하여 표본량 영향 확인.
- 공통 N < 3이면 Balanced 분석을 권장하지 않는 경고 표시.

## 유지 기능
- Aaker (1997) 15 facets → 5 dimensions
- Content Unit 자동 분할 + 연구자 최종 승인
- 브랜드별 N / Word Count / Character Count
- Mean / SD / Relative 보조값
- compact Radar: unit 개별선, 브랜드 평균, Mean ± SD
- 브랜드 간 Radar 비교
- 15 Facet bar chart
- 결과 session 유지
- CSV 개별 다운로드 + ZIP 일괄 다운로드

## 해석
N=1은 표준편차와 안정적 대표성 평가가 불가능합니다. 부족한 브랜드는 공식 브랜드 텍스트 추가 확보가 우선입니다.
