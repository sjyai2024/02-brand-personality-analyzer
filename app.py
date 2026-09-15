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
    m=model(); dims=list(D)
    ue=m.encode(texts,normalize_embeddings=True,show_progress_bar=False)
    cs=[]
    for d in dims:
        e=m.encode([f"A brand that is {t}." for t in D[d]],normalize_embeddings=True,show_progress_bar=False)
        c=e.mean(0); cs.append(c/max(np.linalg.norm(c),1e-12))
    sims=ue@np.vstack(cs).T
    raw=pd.DataFrame(sims,columns=dims)
    ex=np.exp((sims-sims.max(1,keepdims=True))/.10); rel=ex/ex.sum(1,keepdims=True)
    return raw,pd.DataFrame(rel,columns=[d+"_Relative" for d in dims])

st.title("Brand Personality Analyzer · Module 02 v0.3")
st.caption("01 승인 페이지 → content unit → 연구자 승인 → Aaker 5차원 분석")

st.markdown("""
**연구 흐름**
1. 01 단계 승인 CSV 업로드  
2. Content unit 자동 분할  
3. 제품/기능·UI·중복 후보 자동 표시  
4. **연구자가 표에서 `Researcher_Final`을 1/0으로 최종 승인**  
5. 승인 unit만 Aaker 5차원 분석  
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

    st.subheader("연구자 승인")
    st.caption("Researcher_Final: 1=분석 포함, 0=제외. 자동판정은 보조수단이며 연구자가 수정할 수 있습니다.")
    edited=st.data_editor(
        review,
        disabled=[c for c in review.columns if c not in ["Researcher_Final","Researcher_Note"]],
        column_config={
            "Researcher_Final":st.column_config.SelectboxColumn("Researcher_Final",options=[1,0],required=True),
            "Researcher_Note":st.column_config.TextColumn("Researcher_Note")
        },
        use_container_width=True,height=500,key="editor"
    )
    st.download_button("연구자 승인 결과 CSV",
        edited.to_csv(index=False).encode("utf-8-sig"),
        "02_content_units_researcher_approval.csv","text/csv")

    final=edited[pd.to_numeric(edited.Researcher_Final,errors="coerce").fillna(0).astype(int)==1].copy()
    st.write(f"최종 분석 대상: **{len(final)}개 unit**")

    if st.button("최종 승인 Unit만 5차원 분석",type="primary") and len(final):
        with st.spinner("분석 중..."):
            raw,rel=analyze(final.Text.tolist())
        detail=pd.concat([final.reset_index(drop=True),raw,rel],axis=1)
        dims=list(D); rc=[d+"_Relative" for d in dims]
        a=detail.groupby("Brand")[dims].mean().reset_index()
        b=detail.groupby("Brand")[rc].mean().reset_index()
        for c in rc:b[c]*=100
        summary=a.merge(b,on="Brand")
        summary["N_Units"]=detail.groupby("Brand").size().reindex(summary.Brand).values
        summary["Primary_Dimension"]=summary[rc].idxmax(axis=1).str.replace("_Relative","",regex=False)

        st.subheader("브랜드별 5차원 프로파일")
        st.dataframe(summary,use_container_width=True)
        st.subheader("Unit별 분석 근거")
        st.dataframe(detail,use_container_width=True,height=430)

        st.download_button("02 브랜드 프로파일 CSV",
            summary.to_csv(index=False).encode("utf-8-sig"),
            "02_brand_personality_profiles_v0_3.csv","text/csv")
        st.download_button("02 Unit별 점수 CSV",
            detail.to_csv(index=False).encode("utf-8-sig"),
            "02_brand_personality_unit_scores_v0_3.csv","text/csv")

st.divider()
st.caption("자동분류와 Relative Profile은 본 연구의 파일럿용 탐색적 조작화입니다. 자동판정은 연구자의 최종 코딩을 대체하지 않습니다.")
