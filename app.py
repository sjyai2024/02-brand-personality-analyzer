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
FACETS={
"Sincerity":{
    "Down-to-earth":["down-to-earth","family-oriented","small-town"],
    "Honest":["honest","sincere","real"],
    "Wholesome":["wholesome","original"],
    "Cheerful":["cheerful","sentimental","friendly"]
},
"Excitement":{
    "Daring":["daring","trendy","exciting"],
    "Spirited":["spirited","cool","young"],
    "Imaginative":["imaginative","unique"],
    "Up-to-date":["up-to-date","independent","contemporary"]
},
"Competence":{
    "Reliable":["reliable","hard working","secure"],
    "Intelligent":["intelligent","technical","corporate"],
    "Successful":["successful","leader","confident"]
},
"Sophistication":{
    "Upper-class":["upper class","glamorous","good looking"],
    "Charming":["charming","feminine","smooth"]
},
"Ruggedness":{
    "Outdoorsy":["outdoorsy","masculine","Western"],
    "Tough":["tough","rugged"]
}
}
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

def _l2_normalize(x, eps=1e-12):
    n=np.linalg.norm(x,axis=1,keepdims=True)
    return x/np.maximum(n,eps)

def _build_hierarchy(trait_sim, trait_names, facet_trait_indices, facet_order):
    trait_df=pd.DataFrame(
        trait_sim,
        columns=["Trait_"+re.sub(r"[^A-Za-z0-9]+","_",x).strip("_") for x in trait_names]
    )

    facet_scores={}
    for dim,facet in facet_order:
        idx=facet_trait_indices[(dim,facet)]
        facet_scores[facet]=trait_sim[:,idx].mean(1)

    facet_df=pd.DataFrame({
        "Facet_"+re.sub(r"[^A-Za-z0-9]+","_",facet).strip("_"):facet_scores[facet]
        for _,facet in facet_order
    })

    dim_scores={}
    for dim,facet_map in FACETS.items():
        dim_scores[dim]=np.column_stack([facet_scores[f] for f in facet_map]).mean(1)
    dim_df=pd.DataFrame(dim_scores)

    a=dim_df.to_numpy()
    ex=np.exp((a-a.max(1,keepdims=True))/.10)
    rel=ex/ex.sum(1,keepdims=True)
    rel_df=pd.DataFrame(rel,columns=[d+"_Relative" for d in FACETS])
    return trait_df,facet_df,dim_df,rel_df

def analyze(texts):
    """
    v2.0 controlled diagnostic.

    Fixed from v1.9:
    - same multilingual SentenceTransformer
    - same 240 sentence/proposition corpus
    - same Aaker 42 trait anchors
    - same fixed prompt template
    - same trait -> facet -> dimension hierarchy

    New in v2.0:
    - estimate common mean + top principal component from sentence embeddings only
    - apply the SAME affine/projection transform to sentence and anchor embeddings
    - re-normalize corrected vectors
    - compute cosine similarity in corrected space

    This implements an All-but-the-Top-style common-component correction.
    """
    m=model()

    trait_names=[]
    trait_prompts=[]
    facet_trait_indices={}
    facet_order=[]

    for dim,facet_map in FACETS.items():
        for facet,traits in facet_map.items():
            facet_order.append((dim,facet))
            idx=[]
            for trait in traits:
                idx.append(len(trait_names))
                trait_names.append(trait)
                trait_prompts.append(f"A brand that is {trait.lower()}.")
            facet_trait_indices[(dim,facet)]=idx

    # Raw normalized embeddings.
    ue=m.encode(texts,normalize_embeddings=True,show_progress_bar=False)
    ae=m.encode(trait_prompts,normalize_embeddings=True,show_progress_bar=False)

    # Raw baseline for within-run comparison.
    raw_trait_sim=ue@ae.T
    raw_trait_df,raw_facet_df,raw_dim_df,raw_rel_df=_build_hierarchy(
        raw_trait_sim,trait_names,facet_trait_indices,facet_order
    )

    # ---- Common-component correction ----
    # Reference geometry is learned ONLY from the sentence corpus, not from anchors.
    mean_vec=ue.mean(axis=0,keepdims=True)
    uc=ue-mean_vec
    ac=ae-mean_vec

    # Top PC estimated from centered sentence embeddings.
    _,sv,vh=np.linalg.svd(uc,full_matrices=False)
    pc1=vh[0]
    pc1_var=float((sv[0]**2)/(sv**2).sum()) if len(sv) and (sv**2).sum()>0 else np.nan

    # Remove top-1 common direction from both sentence and anchor embeddings.
    uc=uc-(uc@pc1[:,None])*pc1[None,:]
    ac=ac-(ac@pc1[:,None])*pc1[None,:]

    # Cosine requires re-normalization after projection.
    uc=_l2_normalize(uc)
    ac=_l2_normalize(ac)

    corrected_trait_sim=uc@ac.T
    trait_df,facet_df,dim_df,rel_df=_build_hierarchy(
        corrected_trait_sim,trait_names,facet_trait_indices,facet_order
    )

    # Anchor definition.
    anchor_rows=[]
    for dim,facet_map in FACETS.items():
        n_facets=len(facet_map)
        for facet,traits in facet_map.items():
            n_traits=len(traits)
            for trait in traits:
                anchor_rows.append({
                    "Dimension":dim,
                    "Facet":facet,
                    "Trait":trait,
                    "Prompt":f"A brand that is {trait.lower()}.",
                    "Trait_Weight_within_Facet":1/n_traits,
                    "Facet_Weight_within_Dimension":1/n_facets,
                    "Effective_Trait_Weight_within_Dimension":(1/n_traits)*(1/n_facets)
                })
    anchor_df=pd.DataFrame(anchor_rows)

    # Reproducibility vectors in long format.
    vector_rows=[]
    for i,v in enumerate(mean_vec.ravel()):
        vector_rows.append({"Vector_Type":"Corpus_Mean","Embedding_Dimension":i,"Value":float(v)})
    for i,v in enumerate(pc1.ravel()):
        vector_rows.append({"Vector_Type":"Top_PC1","Embedding_Dimension":i,"Value":float(v)})
    vector_df=pd.DataFrame(vector_rows)

    correction_meta=pd.DataFrame([{
        "App_Version":"2.0",
        "Reference_Set":"analysis sentence embeddings only",
        "Reference_N":len(texts),
        "Mean_Centering":"Yes",
        "Top_PCs_Removed":1,
        "Reference_PC1_Explained_Variance":pc1_var,
        "Anchor_Transform":"same sentence-derived mean and PC1 applied to anchors",
        "Post_Projection_Normalization":"L2",
        "Similarity":"cosine via normalized dot product"
    }])

    return {
        "trait":trait_df,
        "facet":facet_df,
        "dim":dim_df,
        "rel":rel_df,
        "raw_trait":raw_trait_df,
        "raw_facet":raw_facet_df,
        "raw_dim":raw_dim_df,
        "raw_rel":raw_rel_df,
        "anchor_def":anchor_df,
        "vector_df":vector_df,
        "correction_meta":correction_meta
    }

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

st.title("02 Brand Personality Analyzer · v2.0")
st.caption("Aaker 42 traits → 15 facets → 5 dimensions · Mean-centering + Top-1 common-component removal")
st.info("v1.9는 v1.9과 동일한 모델·입력·cosine 계산을 유지하고, anchor만 Aaker(1997)의 최종 42 traits ensemble로 변경한 통제 진단 버전입니다. Baseline correction과 prompt template ensemble은 아직 적용하지 않습니다.")

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
            A=analyze(analysis_df.Text.tolist())
        trait=A["trait"];facet=A["facet"];dim=A["dim"];rel=A["rel"]
        raw_trait=A["raw_trait"];raw_facet=A["raw_facet"];raw_dim=A["raw_dim"];raw_rel=A["raw_rel"]
        anchor_def=A["anchor_def"]

        # Corrected values use the standard column names.
        # Raw diagnostic values are kept with Raw_ prefixes.
        raw_trait=raw_trait.add_prefix("Raw_")
        raw_facet=raw_facet.add_prefix("Raw_")
        raw_dim=raw_dim.add_prefix("Raw_")
        raw_rel=raw_rel.add_prefix("Raw_")
        detail=pd.concat([
            analysis_df.reset_index(drop=True),
            raw_trait,raw_facet,raw_dim,raw_rel,
            trait,facet,dim,rel
        ],axis=1)
        ds=list(FACETS);rc=[d+"_Relative" for d in ds]
        # Corrected primary summaries
        mean=detail.groupby("Brand")[ds].mean()
        sd=detail.groupby("Brand")[ds].std(ddof=1)
        summary=mean.reset_index()
        for d in ds:
            summary[d+"_SD"]=summary.Brand.map(sd[d])
        rr=detail.groupby("Brand")[rc].mean()*100
        for c in rc:
            summary[c]=summary.Brand.map(rr[c])
        summary["N_Units"]=summary.Brand.map(detail.groupby("Brand").size())
        summary["Primary_Dimension"]=summary[ds].idxmax(axis=1)

        # Raw within-run baseline summaries
        raw_ds=["Raw_"+d for d in ds]
        raw_rc=["Raw_"+d+"_Relative" for d in ds]
        raw_mean=detail.groupby("Brand")[raw_ds].mean().reset_index()
        raw_mean=raw_mean.rename(columns={f"Raw_{d}":d for d in ds})
        raw_mean["N_Units"]=raw_mean["Brand"].map(detail.groupby("Brand").size())
        raw_mean["Primary_Dimension"]=raw_mean[ds].idxmax(axis=1)

        fcols=[c for c in detail if c.startswith("Facet_")]
        tcols=[c for c in detail if c.startswith("Trait_")]
        raw_fcols=[c for c in detail if c.startswith("Raw_Facet_")]
        raw_tcols=[c for c in detail if c.startswith("Raw_Trait_")]

        fsum=detail.groupby("Brand")[fcols].mean().reset_index()
        tsum=detail.groupby("Brand")[tcols].mean().reset_index()
        raw_fsum=detail.groupby("Brand")[raw_fcols].mean().reset_index()
        raw_tsum=detail.groupby("Brand")[raw_tcols].mean().reset_index()

        # -------- Diagnostic helpers --------
        def common_factor_metrics(dim_df,facet_df,dim_cols,facet_cols,label):
            dim_corr=dim_df[dim_cols].corr()
            dim_off=dim_corr.to_numpy()[np.triu_indices(len(dim_cols),1)]
            facet_corr=facet_df[facet_cols].corr()
            facet_off=facet_corr.to_numpy()[np.triu_indices(len(facet_cols),1)]

            X=facet_df[facet_cols].copy()
            X=(X-X.mean())/X.std(ddof=0).replace(0,np.nan)
            X=X.fillna(0).to_numpy()
            _,sv,_=np.linalg.svd(X,full_matrices=False)
            pc1=float((sv[0]**2)/(sv**2).sum()) if len(sv) and (sv**2).sum()>0 else np.nan

            facet_means=facet_df[facet_cols].mean()
            return {
                "Representation":label,
                "Mean_5D_Pairwise_Correlation":float(dim_off.mean()) if len(dim_off) else np.nan,
                "Min_5D_Pairwise_Correlation":float(dim_off.min()) if len(dim_off) else np.nan,
                "Max_5D_Pairwise_Correlation":float(dim_off.max()) if len(dim_off) else np.nan,
                "Mean_15Facet_Pairwise_Correlation":float(facet_off.mean()) if len(facet_off) else np.nan,
                "PC1_Explained_Variance_15Facet_Standardized":pc1,
                "Facet_Baseline_Mean_Range":float(facet_means.max()-facet_means.min())
            }

        raw_facet_for_diag=raw_fsum.rename(columns={c:c.replace("Raw_","",1) for c in raw_fcols})
        corrected_metrics=common_factor_metrics(summary,fsum,ds,fcols,"Corrected_MeanCenter_Top1")
        raw_metrics=common_factor_metrics(raw_mean,raw_facet_for_diag,ds,fcols,"Raw_v1.9_Equivalent")

        diagnostic=pd.DataFrame([raw_metrics,corrected_metrics])
        diagnostic["App_Version"]="2.0"
        diagnostic["N_Brands"]=summary["Brand"].nunique()
        diagnostic["N_Units"]=len(detail)
        diagnostic["Anchor_Method"]="Aaker 42 traits -> 15 facets -> 5 dimensions"
        diagnostic["Correction_Method"]=[
            "None",
            "sentence-reference mean-centering + top-1 PC removal + L2 renormalization"
        ]
        st.session_state["R"]={
            "approval":final.copy(),
            "summary":summary,
            "raw_summary":raw_mean,
            "facets":fsum,
            "raw_facets":raw_fsum,
            "traits":tsum,
            "raw_traits":raw_tsum,
            "detail":detail,
            "sample_info":sample_info,
            "mode":mode,
            "anchor_def":anchor_def,
            "diagnostic":diagnostic,
            "correction_meta":A["correction_meta"],
            "correction_vectors":A["vector_df"]
        }

if "R" in st.session_state:
    R=st.session_state.R;summary=R["summary"];detail=R["detail"];fsum=R["facets"];tsum=R["traits"];ds=list(FACETS)
    st.divider();st.header("분석 결과 미리보기")
    st.caption(f"Analysis mode: {R.get('mode','Full Sample')}")
    st.subheader("브랜드별 원자료 정보량")
    st.dataframe(R["sample_info"],use_container_width=True,hide_index=True)
    preview_cols=["Brand","Primary_Dimension","N_Units"]+list(FACETS.keys())
    st.dataframe(summary[preview_cols],use_container_width=True,hide_index=True)
    with st.expander("전체 결과표 보기 (SD · Relative 포함)"):
        st.dataframe(summary,use_container_width=True,hide_index=True)

    st.subheader("v2.0 공통요인 보정 진단")
    st.dataframe(R["diagnostic"],use_container_width=True,hide_index=True)
    st.caption("비교 기준: v1.9 sentence/proposition baseline의 5D 평균상관≈0.930, 15-facet 평균상관≈0.850, PC1≈86.1%. 값이 낮아지는지 확인합니다.")

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
    ax3.set_title(f"{brand} · corrected 15 facets", fontsize=11);ax3.tick_params(labelsize=8);plt.tight_layout()
    _, facet_col, _ = st.columns([0.7, 2.6, 0.7])
    with facet_col:
        st.pyplot(fig3, use_container_width=True)

    st.subheader("42 Trait 미리보기 · corrected space")
    tt=tsum[tsum.Brand==brand].drop(columns="Brand").T.reset_index()
    tt.columns=["Trait","Mean cosine similarity"]
    st.dataframe(tt,use_container_width=True,height=360,hide_index=True)

    with st.expander("Unit별 분석 근거"):st.dataframe(detail[detail.Brand==brand],use_container_width=True,height=420)

    files={
    "02_sample_size_information_v2_0.csv":R["sample_info"],
    "02_content_units_researcher_approval_v2_0.csv":R["approval"],
    "02_brand_personality_5D_profiles_corrected_v2_0.csv":summary,
    "02_brand_personality_5D_profiles_raw_v2_0.csv":R["raw_summary"],
    "02_brand_personality_15facet_profiles_corrected_v2_0.csv":fsum,
    "02_brand_personality_15facet_profiles_raw_v2_0.csv":R["raw_facets"],
    "02_brand_personality_42trait_profiles_corrected_v2_0.csv":tsum,
    "02_brand_personality_42trait_profiles_raw_v2_0.csv":R["raw_traits"],
    "02_brand_personality_unit_scores_v2_0.csv":detail,
    "02_anchor_definition_v2_0.csv":R["anchor_def"],
    "02_diagnostic_metrics_v2_0.csv":R["diagnostic"],
    "02_common_component_metadata_v2_0.csv":R["correction_meta"],
    "02_common_component_vectors_v2_0.csv":R["correction_vectors"]
    }
    st.header("결과 다운로드")
    st.download_button("모든 결과 ZIP 다운로드",zip_csv(files),"02_brand_personality_results_v2_0.zip","application/zip")
    cols=st.columns(4)
    for c,(n,d) in zip(cols,files.items()):c.download_button(n.replace(".csv",""),d.to_csv(index=False).encode("utf-8-sig"),n,"text/csv")

st.caption("v1.9는 Aaker(1997)의 최종 42 traits–15 facets–5 dimensions 구조를 sentence embedding에 계층적으로 적용한 통제 진단 버전입니다. 이는 Aaker의 원 설문 계산식을 재현하는 것이 아니라 본 연구의 computational operationalization입니다.")
