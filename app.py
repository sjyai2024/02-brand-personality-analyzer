import streamlit as st
import pandas as pd, numpy as np, re
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="02 Brand Personality Analyzer",layout="wide")

# Aaker (1997): 15 facets nested within 5 dimensions
FACETS={
"Sincerity":["Down-to-earth","Honest","Wholesome","Cheerful"],
"Excitement":["Daring","Spirited","Imaginative","Up-to-date"],
"Competence":["Reliable","Intelligent","Successful"],
"Sophistication":["Upper-class","Charming"],
"Ruggedness":["Outdoorsy","Tough"]
}

PRODUCT_TERMS=[
"베스트셀러","세라마이딘","시카페어","바이탈 하이드라","포어레미디","에브리 선 데이",
"워터뱅크","크림 스킨","바운시 앤 펌","래디언-c","퍼펙트 리뉴",
"sleeping beauty technology","core product pillars","hybrid skincare",
"제품","성분","효능","사용법","ingredient","ingredients","formula","formulation",
"clinical","dermatologist","피부과","전문의","스킨케어","토너","세럼","앰플","선크림","spf"]
UI_TERMS=["visit and follow us","brand core value","learn more","shop now","view more","discover more"]

@st.cache_resource
def model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def split_units(text):
    text=str(text or "").replace("\r\n","\n").replace("\r","\n").strip()
    blocks=[re.sub(r"[ \t]+"," ",x).strip() for x in re.split(r"\n+",text) if x.strip()]
    out=[]
    for b in blocks:
        parts=[b] if len(b)<=280 else re.split(r"(?<=[.!?。！？])\s+|(?<=다\.)\s+",b)
        for p in parts:
            p=re.sub(r"\s+"," ",p).strip()
            if len(p)>=15: out.append(p)
    seen=set(); ans=[]
    for x in out:
        k=x.casefold()
        if k not in seen: seen.add(k); ans.append(x)
    return ans

def classify_units(df):
    norm=df.Text.fillna("").astype(str).str.lower().str.replace(r"\s+"," ",regex=True).str.strip()
    dup=df.assign(_n=norm).duplicated(["Brand","_n"],keep="first")
    cats=[]; inc=[]; reasons=[]
    for pos,(_,r) in enumerate(df.iterrows()):
        t=str(r.Text).strip(); tl=t.lower()
        if dup.iloc[pos]:
            cats.append("Duplicate"); inc.append(0); reasons.append("동일 브랜드 내 완전 중복")
        elif len(t)<35 or any(x in tl for x in UI_TERMS):
            cats.append("UI/Heading"); inc.append(0); reasons.append("섹션 제목·CTA 등 UI성 텍스트")
        elif any(x in tl for x in PRODUCT_TERMS):
            cats.append("Product/Functional"); inc.append(0); reasons.append("제품·성분·효능·기술 등 기능적 내용")
        else:
            cats.append("Brand"); inc.append(1); reasons.append("브랜드 정체성·철학·가치·방향성 후보")
    x=df.copy(); x["Auto_Category"]=cats; x["Auto_Include"]=inc
    x["Review_Reason"]=reasons; x["Researcher_Final"]=inc; x["Researcher_Note"]=""
    return x

def analyze(texts):
    m=model()
    # Unit × 15 facet cosine similarities
    facet_names=[]; facet_dim=[]; anchors=[]
    for dim,fs in FACETS.items():
        for f in fs:
            facet_names.append(f); facet_dim.append(dim)
            anchors.append(f"A brand that is {f.lower()}.")
    ue=m.encode(texts,normalize_embeddings=True,show_progress_bar=False)
    ae=m.encode(anchors,normalize_embeddings=True,show_progress_bar=False)
    fsims=ue@ae.T
    facet_df=pd.DataFrame(fsims,columns=[f"Facet_{f}" for f in facet_names])

    # 본 연구 조작화: 같은 dimension 소속 facet의 arithmetic mean
    dim_scores={}
    for dim,fs in FACETS.items():
        idx=[facet_names.index(f) for f in fs]
        dim_scores[dim]=fsims[:,idx].mean(axis=1)
    dim_df=pd.DataFrame(dim_scores)

    # 시각화용 상대 프로파일 (보조지표)
    ds=dim_df.to_numpy()
    ex=np.exp((ds-ds.max(axis=1,keepdims=True))/.10)
    rel=ex/ex.sum(axis=1,keepdims=True)
    rel_df=pd.DataFrame(rel,columns=[d+"_Relative" for d in FACETS])
    return facet_df,dim_df,rel_df

st.title("02 Brand Personality Analyzer · v1.0")
st.caption("Aaker (1997) 15 facets → 5 dimensions · 연구자 승인 기반")

with st.expander("최종 측정체계",expanded=True):
    st.markdown("""
**Aaker (1997)의 5차원·15 facet 구조를 공통 분석틀로 사용합니다.**

- Sincerity → Down-to-earth, Honest, Wholesome, Cheerful
- Excitement → Daring, Spirited, Imaginative, Up-to-date
- Competence → Reliable, Intelligent, Successful
- Sophistication → Upper-class, Charming
- Ruggedness → Outdoorsy, Tough

각 승인 content unit과 15 facet anchor의 cosine similarity를 계산하고,
같은 차원에 속하는 facet 점수의 **산술평균**으로 5차원 점수를 산출합니다.

**주의:** 임베딩 cosine similarity와 facet 평균 계산은 Aaker(1997)의 원래 설문척도 자체가 아니라,
Aaker의 구조를 계산적 텍스트 분석에 적용하기 위한 **본 연구의 탐색적 조작화**입니다.
""")

f=st.file_uploader("01 연구자 승인 CSV",type="csv")
if f:
    src=pd.read_csv(f)
    if not {"Brand","Original_Text","Include"}.issubset(src.columns):
        st.error("필수 열: Brand, Original_Text, Include"); st.stop()
    ok=src[pd.to_numeric(src.Include,errors="coerce").fillna(0).astype(int)==1].copy()

    rows=[]; cnt={}
    for _,r in ok.iterrows():
        b=str(r.Brand); cnt.setdefault(b,0)
        for t in split_units(r.Original_Text):
            cnt[b]+=1
            rows.append({"Unit_ID":f"{b}_U{cnt[b]:03d}","Brand":b,
                         "Page_Type":r.get("Page_Type",""),"Page_Title":r.get("Page_Title",""),
                         "Source_URL":r.get("Source_URL",""),"Text":t})
    review=classify_units(pd.DataFrame(rows))
    st.write(f"승인 페이지 **{len(ok)}개** · content unit **{len(review)}개**")

    st.subheader("1. Content Unit 연구자 승인")
    edited=st.data_editor(
        review,
        disabled=[c for c in review.columns if c not in ["Researcher_Final","Researcher_Note"]],
        column_config={
            "Researcher_Final":st.column_config.SelectboxColumn("Researcher_Final",options=[1,0],required=True),
            "Researcher_Note":st.column_config.TextColumn("Researcher_Note")
        },
        use_container_width=True,height=480,key="editor"
    )
    st.download_button("연구자 승인 Unit CSV",
        edited.to_csv(index=False).encode("utf-8-sig"),
        "02_content_units_researcher_approval_v1_0.csv","text/csv")

    final=edited[pd.to_numeric(edited.Researcher_Final,errors="coerce").fillna(0).astype(int)==1].copy()
    st.write(f"최종 분석 대상: **{len(final)}개 unit**")

    if st.button("2. 최종 승인 Unit 15 Facet → 5차원 분석",type="primary") and len(final):
        with st.spinner("15 facet 의미 유사도 분석 중..."):
            facet,dim,rel=analyze(final.Text.tolist())
        detail=pd.concat([final.reset_index(drop=True),facet,dim,rel],axis=1)

        dims=list(FACETS); rc=[d+"_Relative" for d in dims]
        mean_dim=detail.groupby("Brand")[dims].mean().reset_index()
        mean_rel=detail.groupby("Brand")[rc].mean().reset_index()
        for c in rc: mean_rel[c]*=100
        summary=mean_dim.merge(mean_rel,on="Brand")
        summary["N_Units"]=detail.groupby("Brand").size().reindex(summary.Brand).values
        summary["Primary_Dimension"]=summary[rc].idxmax(axis=1).str.replace("_Relative","",regex=False)

        # 브랜드별 facet 평균도 별도 산출
        fcols=[c for c in detail.columns if c.startswith("Facet_")]
        facet_summary=detail.groupby("Brand")[fcols].mean().reset_index()

        st.subheader("브랜드별 Aaker 5차원 프로파일")
        st.dataframe(summary,use_container_width=True)
        st.subheader("브랜드별 15 Facet 평균")
        st.dataframe(facet_summary,use_container_width=True)
        st.subheader("Unit별 분석 근거")
        st.dataframe(detail,use_container_width=True,height=430)

        st.download_button("5차원 브랜드 프로파일 CSV",
            summary.to_csv(index=False).encode("utf-8-sig"),
            "02_brand_personality_5D_profiles_v1_0.csv","text/csv")
        st.download_button("15 Facet 브랜드 프로파일 CSV",
            facet_summary.to_csv(index=False).encode("utf-8-sig"),
            "02_brand_personality_15facet_profiles_v1_0.csv","text/csv")
        st.download_button("Unit별 전체 점수 CSV",
            detail.to_csv(index=False).encode("utf-8-sig"),
            "02_brand_personality_unit_scores_v1_0.csv","text/csv")

st.divider()
st.caption("v1.0: 임의 확장어를 제거하고 Aaker(1997)의 15 facet 구조를 semantic anchor로 사용합니다. 한국 문화권의 척도 차이는 후속 인간평가 파일럿에서 검토합니다.")
