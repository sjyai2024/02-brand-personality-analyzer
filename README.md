# 02 Brand Personality Analyzer v1.7

## NameError 수정
01 v2.x 최종 승인 CSV를 사용할 때 `edited` 변수가 존재하지 않는데
분석 결과 session_state 저장 단계에서 `edited`를 참조하던 오류를 수정했습니다.

이제 session_state에는 실제 01 최종 승인 데이터(`final`)를 저장합니다.

지원:
- 01_all_brands_approved_units.csv 직접 입력
- Full Sample — Primary Analysis
- Balanced Sample — Robustness Check
- Aaker 15 facets → 5 dimensions
- Mean ± SD / Radar / 브랜드 비교 / 15 Facet
- CSV 및 ZIP 다운로드
