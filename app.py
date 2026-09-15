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
def model(): return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
def units(t):
    b=[re.sub(r"\\s+"," ",x).strip() for x in re.split(r"\\n+",str(t)) if x.strip()]
    return [x for x in b if len(x)>=15]
def analyze(texts):
    m=model(); dims=list(D)
    ue=m.encode(texts,normalize_embeddings=True)
    cents=[]
    for d in dims:
        e=m.encode(["A brand that is "+x+"." for x in D[d]],normalize_embeddings=True)
        c=e.mean(0); cents.append(c/np.linalg.norm(c))
    s=ue@np.vstack(cents).T
    raw=pd.DataFrame(s,columns=dims)
    e=np.exp((s-s.max(1,keepdims=True))/.10); r=e/e.sum(1,keepdims=True)
    rel=pd.DataFrame(r,columns=[x+"_Relative" for x in dims])
    return raw,rel

st.title("Brand Personality Analyzer · Module 02")
st.caption("승인 텍스트 → Aaker 5차원 계산적 브랜드 개성 프로파일")
st.info("Aaker(1997)의 5차원을 사용하고, Pamuksuz et al.(2021)의 계산적 브랜드 개성 측정·집계 논리를 참고합니다. LDA2Vec/RoBERTa를 재학습하지 않고 다국어 문장 임베딩 cosine similarity를 사용하는 것은 본 연구의 파일럿용 수정 적용입니다.")
f=st.file_uploader("연구자 승인 CSV",type="csv")
if f:
    df=pd.read_csv(f)
    need={"Brand","Original_Text","Include"}
    if not need.issubset(df.columns):
        st.error("필수 열: Brand, Original_Text, Include"); st.stop()
    ok=df[pd.to_numeric(df.Include,errors="coerce").fillna(0).astype(int)==1].copy()
    rows=[]; n=1
    for _,r in ok.iterrows():
        for u in units(r.Original_Text):
            rows.append({"Unit_ID":f"U{n:04d}","Brand":r.Brand,"Source_URL":r.get("Source_URL",""),"Text":u}); n+=1
    ud=pd.DataFrame(rows)
    st.write(f"승인 페이지 {len(ok)}개 · content unit {len(ud)}개")
    if st.button("5차원 분석 실행",type="primary") and len(ud):
        with st.spinner("최초 실행 시 모델 다운로드로 시간이 걸릴 수 있습니다."):
            raw,rel=analyze(ud.Text.tolist())
        detail=pd.concat([ud.reset_index(drop=True),raw,rel],axis=1)
        dims=list(D); rc=[x+"_Relative" for x in dims]
        a=detail.groupby("Brand")[dims].mean().reset_index()
        b=detail.groupby("Brand")[rc].mean().reset_index()
        for c in rc:b[c]*=100
        summary=a.merge(b,on="Brand")
        summary["Primary_Dimension"]=summary[rc].idxmax(axis=1).str.replace("_Relative","",regex=False)
        st.subheader("브랜드별 프로파일"); st.dataframe(summary,use_container_width=True)
        st.subheader("근거 단위별 점수"); st.dataframe(detail,use_container_width=True,height=420)
        st.download_button("브랜드 프로파일 CSV",summary.to_csv(index=False).encode("utf-8-sig"),"brand_personality_profiles.csv")
        st.download_button("단위별 분석 CSV",detail.to_csv(index=False).encode("utf-8-sig"),"brand_personality_unit_scores.csv")
        st.warning("Relative Profile은 확립된 기존 척도가 아니라 본 연구의 탐색적 조작화입니다. 원 cosine similarity도 함께 보존·보고하십시오.")
