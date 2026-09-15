import streamlit as st
import pandas as pd, numpy as np, re
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Brand Personality Analyzer",layout="wide")
D={
"Sincerity":["sincere","honest","genuine","friendly","cheerful","down-to-earth","wholesome","warm","authentic"],
"Excitement":["daring","spirited","imaginative","up-to-date","trendy","young","unique","independent","exciting","innovative"],
"Competence":["reliable","intelligent","successful","secure","confident","responsible","professional","competent"],
"Sophistication":["upper-class","glamorous","charming","good-looking","refined","elegant","sophisticated","premium"],
"Ruggedness":["outdoorsy","tough","strong","rugged","masculine","robust","adventurous"]}

@st.cache_resource
def model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def split_semantic_units(text):
    """
    연구용 보수적 분할:
    1) 크롤러가 보존한 줄바꿈/블록 경계를 우선 사용
    2) 긴 블록만 문장 종결부호로 분할
    3) 너무 짧은 UI성 조각 제거
    원문은 수정/요약하지 않음
    """
    text=str(text or "").replace("\r\n","\n").replace("\r","\n").strip()
    blocks=[re.sub(r"[ \t]+"," ",b).strip() for b in re.split(r"\n+",text) if b.strip()]
    out=[]
    for b in blocks:
        # 제목+짧은 설명은 블록 자체를 하나의 unit으로 유지
        if len(b)<=280:
            parts=[b]
        else:
            parts=re.split(r"(?<=[.!?。！？])\s+|(?<=다\.)\s+",b)
        for p in parts:
            p=re.sub(r"\s+"," ",p).strip()
            if len(p)>=15:
                out.append(p)
    # 완전 중복만 제거, 순서 유지
    seen=set(); clean=[]
    for x in out:
        k=x.casefold()
        if k not in seen:
            seen.add(k); clean.append(x)
    return clean

def analyze(texts):
    m=model(); dims=list(D)
    ue=m.encode(texts,normalize_embeddings=True,show_progress_bar=False)
    centroids=[]
    for d in dims:
        pe=m.encode([f"A brand that is {t}." for t in D[d]],
                    normalize_embeddings=True,show_progress_bar=False)
        c=pe.mean(0); c=c/max(np.linalg.norm(c),1e-12)
        centroids.append(c)
    sims=ue@np.vstack(centroids).T
    raw=pd.DataFrame(sims,columns=dims)
    # 상대강도 표시용. 확립 척도가 아닌 본 연구 파일럿 조작화.
    e=np.exp((sims-sims.max(axis=1,keepdims=True))/.10)
    rel=e/e.sum(axis=1,keepdims=True)
    rel=pd.DataFrame(rel,columns=[d+"_Relative" for d in dims])
    return raw,rel

st.title("Brand Personality Analyzer · Module 02 v0.2")
st.caption("연구자 승인 브랜드 텍스트 → content unit → Aaker 5차원 프로파일")

with st.expander("분석 규칙",expanded=True):
    st.markdown("""
1. `Include=1`인 연구자 승인 자료만 사용합니다.
2. 크롤링 원문의 **줄바꿈/콘텐츠 블록을 우선적인 content unit 경계**로 사용합니다.
3. 긴 블록만 문장 종결부호를 기준으로 보조 분할합니다.
4. 원문을 요약하거나 다시 쓰지 않습니다.
5. 각 unit에 대해 Aaker 5차원 cosine similarity를 계산한 뒤 브랜드별 평균을 산출합니다.
6. `Relative` 값은 5차원 상대강도를 보기 위한 **본 연구의 파일럿용 탐색적 조작화**이며 기존 확립 척도가 아닙니다.
""")

f=st.file_uploader("01 단계 연구자 승인 CSV",type="csv")
if f:
    df=pd.read_csv(f)
    req={"Brand","Original_Text","Include"}
    if not req.issubset(df.columns):
        st.error("필수 열: Brand, Original_Text, Include"); st.stop()
    inc=pd.to_numeric(df["Include"],errors="coerce").fillna(0).astype(int)
    ok=df[inc==1].copy()
    rows=[]
    counters={}
    for _,r in ok.iterrows():
        brand=str(r["Brand"])
        counters.setdefault(brand,0)
        for txt in split_semantic_units(r["Original_Text"]):
            counters[brand]+=1
            rows.append({
                "Unit_ID":f"{brand}_U{counters[brand]:03d}",
                "Brand":brand,
                "Page_Type":r.get("Page_Type",""),
                "Page_Title":r.get("Page_Title",""),
                "Source_URL":r.get("Source_URL",""),
                "Text":txt
            })
    units=pd.DataFrame(rows)
    st.write(f"승인 페이지 **{len(ok)}개** · 분석 content unit **{len(units)}개**")
    if len(units):
        st.subheader("분석 전 content unit 확인")
        st.dataframe(units,use_container_width=True,height=380)
        st.download_button("Content unit 검토 CSV",
            units.to_csv(index=False).encode("utf-8-sig"),
            "02_content_units_for_review.csv","text/csv")
        st.caption("논문 본분석에서는 이 표를 먼저 검토하여 unit 분할이 적절한지 확인하는 것을 권장합니다.")

    if st.button("5차원 분석 실행",type="primary") and len(units):
        with st.spinner("분석 중입니다. 최초 실행은 모델 다운로드로 시간이 걸릴 수 있습니다."):
            raw,rel=analyze(units["Text"].tolist())
        detail=pd.concat([units.reset_index(drop=True),raw,rel],axis=1)
        dims=list(D); rc=[d+"_Relative" for d in dims]

        raw_summary=detail.groupby("Brand")[dims].agg(["mean","std","count"])
        mean_raw=detail.groupby("Brand")[dims].mean().reset_index()
        mean_rel=detail.groupby("Brand")[rc].mean().reset_index()
        for c in rc: mean_rel[c]*=100
        summary=mean_raw.merge(mean_rel,on="Brand")
        summary["N_Units"]=detail.groupby("Brand").size().reindex(summary.Brand).values
        summary["Primary_Dimension"]=summary[rc].idxmax(axis=1).str.replace("_Relative","",regex=False)

        st.subheader("브랜드별 평균 5차원 프로파일")
        st.dataframe(summary,use_container_width=True)
        st.subheader("Unit별 근거 데이터")
        st.dataframe(detail,use_container_width=True,height=450)

        st.download_button("브랜드 프로파일 CSV",
            summary.to_csv(index=False).encode("utf-8-sig"),
            "02_brand_personality_profiles_v0_2.csv","text/csv")
        st.download_button("Unit별 점수 CSV",
            detail.to_csv(index=False).encode("utf-8-sig"),
            "02_brand_personality_unit_scores_v0_2.csv","text/csv")
        st.warning("Primary Dimension 하나만 결론으로 사용하지 말고, 원 cosine similarity·5차원 평균·unit 수를 함께 검토하십시오.")
