# 02 Brand Personality Analyzer v1.8

## 한글 그래프 깨짐 대응
Matplotlib에서 서버에 설치된 한글 폰트를 자동 탐색합니다.

우선순위:
1. Noto Sans CJK KR
2. Noto Sans KR
3. NanumGothic
4. Malgun Gothic
5. AppleGothic

사용 가능한 한글 폰트가 있으면 Radar/legend의 한국어 브랜드명이 정상 출력됩니다.
서버에 한글 폰트가 전혀 없으면 그래프 범례에 안전한 fallback을 사용하고 데이터 표의 원 브랜드명은 그대로 보존합니다.

주의: 폰트 파일을 저장소에 포함하지 않습니다.
