import streamlit as st
import pandas as pd, numpy as np, re, io, zipfile
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sentence_transformers import SentenceTransformer

def configure_korean_font():
    """Streamlit Cloud/Linux와 macOS에서 사용 가능한 한글 폰트를 자동 선택."""
    preferred=["Noto Sans CJK KR","Noto Sans KR","NanumGothic","Malgun Gothic","AppleGothic"]
    available={f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"]=name
            plt.rcParams["axes.unicode_minus"]=False
            return name
    # 설치 폰트가 없으면 DejaVu Sans를 사용하되 앱에서 영문 표시명 fallback을 적용
    plt.rcParams["axes.unicode_minus"]=False
    return None

KOREAN_FONT=configure_korean_font()

st.set_page_config(page_title="02 Brand Personality Analyzer",layout="wide")
FACETS={"Sincerity":["Down-to-earth","Honest","Wholesome","Cheerful"],
"Excitement":["Daring","Spirited","Imaginative","Up-to-date"],
"Competence":["Reliable","Intelligent","Successful"],
"Sophistication":["Upper-class","Charming"],
"Ruggedness":["Outdoorsy","Tough"]}
PRODUCT_TERMS=["베스트셀러","세라마이딘","시카페어","워터뱅크","크림 스킨","바운시 앤 펌",
"sleeping beauty technology","core product pillars","hybrid skincare","제품","성분","효능","사용법",
"ingredient","ingredients","formula","formulation","clinical","dermatologist","피부과","전문의",
"스킨케어","토너","세럼","앰플","선크림","spf"]
UI_TERMS=["visit and follow us","brand core value","learn more","shop now","view more","discover more"]

@st.cache_resource
def model(): return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def split_units(text):
    text=str(text or "").replace("\r\n","\n").replace("\r","\n").strip()
    blocks=[re.sub(r"[ \t]+"," ",x).strip() for x in re.split(r"\n+",text) if x.strip()]
    out=[]
    for b in blocks:
        parts=[b] if len(b)<=280 else re.split(r"(?<=[.!?。！？])\s+|(?<=다\.)\s+",b)
        for p in parts:
            p=re.sub(r"\s+"," ",p).strip()
            if len(p)>=15: out.append(p)
    return list(dict.fromkeys(out))

def classify(df):
    norm=df.Text.fillna("").astype(str).str.lower().str.replace(r"\s+"," ",regex=True).str.strip()
    dup=df.assign(_n=norm).duplicated(["Brand","_n"],keep="first")
    rows=[]
    for pos,(_,r) in enumerate(df.iterrows()):
        t=str(r.Text).strip();tl=t.lower()
        if dup.iloc[pos]: c,i,z="Duplicate",0,"동일 브랜드 내 완전 중복"
        elif len(t)<35 or any(x in tl for x in UI_TERMS): c,i,z="UI/Heading",0,"섹션 제목·CTA 등 UI성 텍스트"
        elif any(x in tl for x in PRODUCT_TERMS): c,i,z="Product/Functional",0,"제품·성분·효능·기술 등 기능적 내용"
        else:c,i,z="Brand",1,"브랜드 정체성·철학·가치·방향성 후보"
        q=r.to_dict();q.update(Auto_Category=c,Auto_Include=i,Review_Reason=z,Researcher_Final=i,Researcher_Note="")
        rows.append(q)
    return pd.DataFrame(rows)

def analyze(texts):
    m=model();names=[];anchors=[]
    for d,fs in FACETS.items():
        for f in fs:names.append(f);anchors.append(f"A brand that is {f.lower()}.")
    ue=m.encode(texts,normalize_embeddings=True,show_progress_bar=False)
    ae=m.encode(anchors,normalize_embeddings=True,show_progress_bar=False)
    fs=ue@ae.T
    fdf=pd.DataFrame(fs,columns=["Facet_"+x for x in names])
    dd={}
    for d,fl in FACETS.items():
        idx=[names.index(x) for x in fl];dd[d]=fs[:,idx].mean(1)
    ddf=pd.DataFrame(dd)
    a=ddf.to_numpy();ex=np.exp((a-a.max(1,keepdims=True))/.10);rel=ex/ex.sum(1,keepdims=True)
    return fdf,ddf,pd.DataFrame(rel,columns=[d+"_Relative" for d in FACETS])

def safe_brand_label(x):
    x=str(x)
    if KOREAN_FONT:
        return x
    # 한글 폰트가 없는 서버에서는 도메인/원문 브랜드명 대신 ASCII fallback이 있으면 사용
    ascii_only="".join(ch for ch in x if ord(ch)<128).strip()
    return ascii_only if ascii_only else "Brand"

def radar(ax, labels, mean, sd=None, unit_rows=None, title=""):
    n=len(labels);angles=np.linspace(0,2*np.pi,n,endpoint=False).tolist();angles+=angles[:1]
    if unit_rows is not None:
        for row in unit_rows:
            v=list(row)+[row[0]]
            ax.plot(angles,v,linewidth=.7,alpha=.18)
    mv=list(mean)+[mean[0]]
    ax.plot(angles,mv,linewidth=2.4,label="Mean")
    ax.fill(angles,mv,alpha=.08)
    if sd is not None:
        lo=np.maximum(np.array(mean)-np.array(sd),0);hi=np.array(mean)+np.array(sd)
        lo=list(lo)+[lo[0]];hi=list(hi)+[hi[0]]
        ax.plot(angles,lo,linestyle="--",linewidth=1,label="Mean - SD")
        ax.plot(angles,hi,linestyle="--",linewidth=1,label="Mean + SD")
    ax.set_xticks(angles[:-1]);ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title,pad=14,fontsize=11)
    ax.legend(loc="upper right",bbox_to_anchor=(1.18,1.12),fontsize=6.5,frameon=False)

def zip_csv(files):
    b=io.BytesIO()
    with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:
        for name,df in files.items():z.writestr(name,df.to_csv(index=False,encoding="utf-8-sig"))
    return b.getvalue()

st.title("02 Brand Personality Analyzer · v1.8")
st.caption("Aaker 15 facets → 5 dimensions · Mean ± SD · Radar visualization")
st.info("시각화는 Yoo & Lee (2025)의 다차원 정량값 방사형 차트, 평균·표준편차, 하위 사례와 대표값을 함께 제시하는 방식을 참고하여 브랜드 개성 분석에 적용했습니다.")

f=st.file_uploader("01 최종 승인 통합 CSV",type="csv",help="권장: 01_all_brands_approved_units.csv")
if f:
    src=pd.read_csv(f)
    new01={"Brand","Unit_ID","Source_URL","Text","Researcher_Final"}
    legacy={"Brand","Original_Text","Include"}

    if new01.issubset(src.columns):
        final=src[pd.to_numeric(src["Researcher_Final"],errors="coerce").fillna(0).astype(int)==1].copy()
        st.success(f"01 최종 승인 CSV 인식 완료: {final['Brand'].nunique()}개 브랜드 · {len(final)}개 승인 Unit")
        st.subheader("1. 01 최종 승인 데이터 확인")
        st.caption("01에서 Content Unit 분리와 연구자 승인이 끝난 자료입니다. 02에서는 재분할·재승인하지 않습니다.")
        st.dataframe(final,use_container_width=True,height=340,hide_index=True)

    elif legacy.issubset(src.columns):
        ok=src[pd.to_numeric(src.Include,errors="coerce").fillna(0).astype(int)==1]
        rows=[];cnt={}
        for _,r in ok.iterrows():
            b=str(r.Brand);cnt.setdefault(b,0)
            for t in split_units(r.Original_Text):
                cnt[b]+=1;rows.append({"Unit_ID":f"{b}_U{cnt[b]:03d}","Brand":b,
                "Page_Type":r.get("Page_Type",""),"Page_Title":r.get("Page_Title",""),
                "Source_URL":r.get("Source_URL",""),"Text":t})
        review=classify(pd.DataFrame(rows))
        st.warning("구형 01 CSV 형식입니다. 호환을 위해 02에서 Unit 승인 단계를 수행합니다.")
        edited=st.data_editor(review,disabled=[c for c in review if c not in ["Researcher_Final","Researcher_Note"]],
          column_config={"Researcher_Final":st.column_config.SelectboxColumn(options=[1,0],required=True)},
          use_container_width=True,height=420,key="editor")
        final=edited[pd.to_numeric(edited.Researcher_Final,errors="coerce").fillna(0).astype(int)==1].copy()
    else:
        st.error("지원하지 않는 CSV 형식입니다. 01에서 받은 01_all_brands_approved_units.csv를 업로드하세요.")
        st.stop()

    final["Word_Count"]=final["Text"].astype(str).str.split().str.len()
    final["Character_Count"]=final["Text"].astype(str).str.len()
    sample_info=final.groupby("Brand").agg(
        N_Units=("Unit_ID","count"),
        Word_Count=("Word_Count","sum"),
        Character_Count=("Character_Count","sum")).reset_index()

    st.subheader("2. 브랜드별 표본 크기 확인")
    st.dataframe(sample_info,use_container_width=True,hide_index=True)
    mode=st.radio("분석 모드",["Full Sample — Primary Analysis","Balanced Sample — Robustness Check"],horizontal=True)
    analysis_df=final.copy()
    common_n=int(sample_info["N_Units"].min()) if len(sample_info) else 0
    if mode=="Full Sample — Primary Analysis":
        st.success("본 분석: 연구자가 승인한 모든 브랜드 관련 Content Unit을 사용합니다.")
    if mode=="Balanced Sample — Robustness Check":
        st.warning(f"현재 모든 브랜드를 포함할 경우 공통 가능한 최대 N은 {common_n}입니다.")
        if common_n < 3:
            st.error("Balanced Sample은 현재 권장하지 않습니다. 공통 N이 3 미만입니다. 표본이 부족한 브랜드의 공식 브랜드 텍스트를 추가 확보한 뒤 강건성 확인에 사용하십시오.")
        target_n=st.number_input("브랜드당 Unit 수 (N)",1,max(1,common_n),max(1,common_n),1)
        seed=st.number_input("Random seed",0,value=42,step=1)
        parts=[g.sample(n=int(target_n),random_state=int(seed)) for _,g in final.groupby("Brand",sort=False) if len(g)>=target_n]
        analysis_df=pd.concat(parts,ignore_index=True) if parts else final.iloc[0:0].copy()
        st.caption("동일 seed에서는 동일 unit이 선택됩니다.")
        st.dataframe(analysis_df[["Unit_ID","Brand","Text"]],use_container_width=True,height=250)

    st.write(f"이번 분석에 사용되는 unit: **{len(analysis_df)}개**")
    if st.button("3. 분석 실행",type="primary") and len(analysis_df):
        with st.spinner("분석 중..."):
            facet,dim,rel=analyze(analysis_df.Text.tolist())
        detail=pd.concat([analysis_df.reset_index(drop=True),facet,dim,rel],axis=1)
        ds=list(FACETS);rc=[d+"_Relative" for d in ds]
        mean=detail.groupby("Brand")[ds].mean()
        sd=detail.groupby("Brand")[ds].std(ddof=1)
        summary=mean.reset_index()
        for d in ds: summary[d+"_SD"]=summary.Brand.map(sd[d])
        rr=detail.groupby("Brand")[rc].mean()*100
        for c in rc: summary[c]=summary.Brand.map(rr[c])
        summary["N_Units"]=summary.Brand.map(detail.groupby("Brand").size())
        summary["Primary_Dimension"]=summary[ds].idxmax(axis=1)
        fcols=[c for c in detail if c.startswith("Facet_")]
        fsum=detail.groupby("Brand")[fcols].mean().reset_index()
        st.session_state["R"]={"approval":final.copy(),"summary":summary,"facets":fsum,
                               "detail":detail,"sample_info":sample_info,"mode":mode}

if "R" in st.session_state:
    R=st.session_state.R;summary=R["summary"];detail=R["detail"];fsum=R["facets"];ds=list(FACETS)
    st.divider();st.header("분석 결과 미리보기")
    st.caption(f"Analysis mode: {R.get('mode','Full Sample')}")
    st.subheader("브랜드별 원자료 정보량")
    st.dataframe(R["sample_info"],use_container_width=True,hide_index=True)
    preview_cols=["Brand","Primary_Dimension","N_Units"]+list(FACETS.keys())
    st.dataframe(summary[preview_cols],use_container_width=True,hide_index=True)
    with st.expander("전체 결과표 보기 (SD · Relative 포함)"):
        st.dataframe(summary,use_container_width=True,hide_index=True)

    brand=st.selectbox("브랜드 상세 보기",summary.Brand.tolist())
    one=summary[summary.Brand==brand].iloc[0]
    mean=[one[d] for d in ds];sd=[one[d+"_SD"] for d in ds]
    units=detail[detail.Brand==brand][ds].to_numpy()
    fig=plt.figure(figsize=(4.8,4.8));ax=fig.add_subplot(111,polar=True)
    radar(ax,ds,mean,sd,units,f"{brand} · 5D profile")
    _, chart_col, _ = st.columns([1, 2.1, 1])
    with chart_col:
        st.pyplot(fig, use_container_width=True)
    st.caption("얇은 선: 승인 content unit · 굵은 선: 브랜드 평균 · 점선: 평균 ± 1 SD")

    st.subheader("브랜드 간 Radar 비교")
    choices=st.multiselect("비교 브랜드 선택 (2~3개 권장)",summary.Brand.tolist(),default=summary.Brand.tolist()[:2])
    if choices:
        fig2=plt.figure(figsize=(4.8,4.8));ax2=fig2.add_subplot(111,polar=True)
        angles=np.linspace(0,2*np.pi,len(ds),endpoint=False).tolist();angles+=angles[:1]
        for b in choices:
            r=summary[summary.Brand==b].iloc[0];v=[r[d] for d in ds];v+=v[:1]
            ax2.plot(angles,v,linewidth=2,label=safe_brand_label(b))
        ax2.set_xticks(angles[:-1]);ax2.set_xticklabels(ds);ax2.set_title("Brand comparison",pad=20);ax2.legend()
        _, compare_col, _ = st.columns([1, 2.1, 1])
        with compare_col:
            st.pyplot(fig2, use_container_width=True)

    st.subheader("5차원 Mean ± SD")
    stat=pd.DataFrame({"Dimension":ds,"Mean":[one[d] for d in ds],"SD":[one[d+"_SD"] for d in ds],
                       "Relative %":[one[d+"_Relative"] for d in ds]})
    st.dataframe(stat,use_container_width=True,hide_index=True)

    st.subheader("15 Facet 미리보기")
    ff=fsum[fsum.Brand==brand].drop(columns="Brand").T.reset_index()
    ff.columns=["Facet","Mean cosine similarity"]
    fig3,ax3=plt.subplots(figsize=(6.2,4.4));ax3.barh(ff.Facet,ff["Mean cosine similarity"]);ax3.invert_yaxis()
    ax3.set_title(f"{brand} · 15 facets", fontsize=11);ax3.tick_params(labelsize=8);plt.tight_layout()
    _, facet_col, _ = st.columns([0.7, 2.6, 0.7])
    with facet_col:
        st.pyplot(fig3, use_container_width=True)

    with st.expander("Unit별 분석 근거"):st.dataframe(detail[detail.Brand==brand],use_container_width=True,height=420)

    files={"02_sample_size_information_v1_8.csv":R["sample_info"],"02_content_units_researcher_approval_v1_8.csv":R["approval"],
    "02_brand_personality_5D_profiles_v1_8.csv":summary,
    "02_brand_personality_15facet_profiles_v1_8.csv":fsum,
    "02_brand_personality_unit_scores_v1_8.csv":detail}
    st.header("결과 다운로드")
    st.download_button("모든 결과 ZIP 다운로드",zip_csv(files),"02_brand_personality_results_v1_8.zip","application/zip")
    cols=st.columns(4)
    for c,(n,d) in zip(cols,files.items()):c.download_button(n.replace(".csv",""),d.to_csv(index=False).encode("utf-8-sig"),n,"text/csv")

st.caption("Radar/Mean±SD 시각화는 Yoo & Lee (2025)의 정량화 결과 표현 방식을 참고한 시각적 적용이며, 본 연구의 브랜드 개성 지표 자체는 Aaker 5차원 구조에 기반합니다.")
