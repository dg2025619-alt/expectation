import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
CUTOFF_YEAR = 2025          # 수업 기준 기간: 이 해까지의 자료만 사용
MIN_OBS_DAYS = 300          # 이 관측일수 미만인 해는 제외


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data
def yearly_average(df):
    grouped = df.groupby("연도").agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count"),
    ).reset_index()

    # 기준 기간(2025년까지) 이후 자료 제외
    grouped = grouped[grouped["연도"] <= CUTOFF_YEAR]
    # 관측일이 300일 미만인 해 제외
    grouped = grouped[grouped["관측일수"] >= MIN_OBS_DAYS]

    grouped = grouped.sort_values("연도").reset_index(drop=True)
    return grouped


st.title("🌡️ 서울 기온 예측기")
st.caption("서울 연평균기온 데이터를 이용해 회귀 직선을 구하고, 특정 연도의 예상 기온을 보여줍니다.")

with st.spinner("데이터를 불러오는 중..."):
    raw_df = load_data()
    yearly_df = yearly_average(raw_df)

if yearly_df.empty:
    st.error("조건을 만족하는 연도 데이터가 없습니다.")
    st.stop()

# 회귀 직선 계산 (연평균기온에 대한 선형회귀)
x = yearly_df["연도"].values.astype(float)
y = yearly_df["연평균기온"].values.astype(float)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

n_years = len(yearly_df)
start_year = int(yearly_df["연도"].min())
end_year = int(yearly_df["연도"].max())

# ---------------- 산점도 + 회귀 직선 ----------------
st.subheader("연도별 평균기온 산점도와 회귀 직선")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=yearly_df["연도"],
    y=yearly_df["연평균기온"],
    mode="markers",
    name="연평균기온",
    marker=dict(color="royalblue", size=8),
))

line_x = np.array([start_year, end_year])
line_y = slope * line_x + intercept
fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode="lines",
    name="회귀 직선",
    line=dict(color="firebrick", width=3),
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("상관계수 (r)", f"{r_value:.4f}")
with col2:
    st.metric("결정계수 (R²)", f"{r_squared:.4f}")

st.info(
    f"회귀 직선은 **{n_years}개 연도**의 데이터로 만들어졌습니다.  \n"
    f"시작 연도: **{start_year}년**, 끝 연도: **{end_year}년**  \n"
    f"(기준: {CUTOFF_YEAR}년까지 자료, 연간 관측일수 {MIN_OBS_DAYS}일 이상인 해만 사용)"
)

# ---------------- 슬라이더로 예상 기온 확인 ----------------
st.subheader("연도별 예상 기온 확인")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=end_year,
    step=1,
)

predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding: 30px 0;">
        <div style="font-size:22px; color:gray;">{selected_year}년 예상 평균기온</div>
        <div style="font-size:72px; font-weight:bold; color:#d62728;">
            {predicted_temp:.2f} °C
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_year < start_year or selected_year > end_year:
    st.warning("선택한 연도가 회귀 직선을 만든 실제 데이터 범위를 벗어났습니다. 예측값은 참고용입니다.")

with st.expander("사용한 연평균기온 데이터 보기"):
    st.dataframe(yearly_df, use_container_width=True)
