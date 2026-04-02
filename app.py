import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from matplotlib import font_manager, rc
import joblib  # 저장된 모델(.pkl)을 불러오기 위한 라이브러리
import os

# --- 1. 한글 폰트 설정 (Windows 기준) ---
@st.cache_resource
def set_korean_font():
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# --- 2. 데이터 및 저장된 모델 로드 ---
@st.cache_resource
def load_assets():
    # 데이터 로드
    df_spotify = pd.read_csv('dataset/spotify-2023.csv', encoding='latin-1')
    df_weather = pd.read_csv('dataset/weather_2019_2023_total.csv')
    
    # 날짜 병합 및 로그 변환 전처리
    df_spotify['release_date'] = pd.to_datetime(
        df_spotify['released_year'].astype(str) + '-' + 
        df_spotify['released_month'].astype(str).str.zfill(2) + '-' + 
        df_spotify['released_day'].astype(str).str.zfill(2), errors='coerce'
    )
    df_weather['일시'] = pd.to_datetime(df_weather['일시'])
    df_merged = pd.merge(df_spotify, df_weather, left_on='release_date', right_on='일시', how='inner')
    
    df_merged['streams'] = pd.to_numeric(df_merged['streams'], errors='coerce')
    df_merged = df_merged.dropna(subset=['streams', 'release_date'])
    df_merged['streams_log'] = np.log1p(df_merged['streams'])
    
    # [중요] 미리 학습된 모델 파일(.pkl) 불러오기
    try:
        loaded_model = joblib.load('model/spotify_weather_model_final.pkl')
    except Exception as e:
        st.error(f"모델 파일(spotify_weather_model_final.pkl)을 찾을 수 없습니다: {e}")
        loaded_model = None

    column_mapping = {
        'streams': '스트리밍', 'bpm': '박자(BPM)', 'danceability_%': '댄스성',
        'valence_%': '곡의밝기', 'energy_%': '에너지', 'acousticness_%': '어쿠스틱',
        'instrumentalness_%': '악기비중', 'liveness_%': '현장감', 'speechiness_%': '가사비중',
        '평균기온(°C)': '기온', '일강수량(mm)': '강수량', '평균 상대습도(%)': '습도', '합계 일조시간(hr)': '일조량'
    }
    
    feature_cols = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                    'instrumentalness_%', 'liveness_%', 'speechiness_%',
                    '평균기온(°C)', '일강수량(mm)', '평균 상대습도(%)', '합계 일조시간(hr)']
    
    return df_merged, loaded_model, column_mapping, feature_cols

df_merged, final_model, col_map, feature_list = load_assets()

# --- 3. 성능 평가 데이터 준비 (R2 점수 출력용) ---
X = df_merged[feature_list].fillna(0)
y = df_merged['streams_log']
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if final_model:
    y_pred = final_model.predict(X_test)
    r2_val = r2_score(y_test, y_pred)
else:
    y_pred, r2_val = None, 0

# --- 4. 메인 UI 구성 ---
st.title("🎵 날씨 기반 음원 흥행 예측 시스템")
tab1, tab2, tab3 = st.tabs(["🤖 실시간 예측 서비스", "📊 모델 분석 결과", "📋 데이터 통계 및 분포"])

# --- Tab 1: 예측 서비스 ---
with tab1:
    st.subheader("실시간 흥행도 예측 서비스")
    with st.form("predict_form"):
        col_w, col_m = st.columns(2)
        with col_w:
            st.markdown("##### 🌦️ 기상 조건 설정")
            temp = st.slider("평균 기온 (°C)", -10.0, 35.0, 20.0)
            rain = st.slider("일 강수량 (mm)", 0.0, 100.0, 0.0)
            humi = st.slider("평균 습도 (%)", 0, 100, 50)
            sun = st.slider("일조 시간 (hr)", 0.0, 15.0, 8.0)
        with col_m:
            st.markdown("##### 🎼 음원 속성 설정")
            bpm_val = st.number_input("박자 (BPM)", 60, 200, 120)
            dance_val = st.slider("댄스성 (%)", 0, 100, 70)
            valence_val = st.slider("곡의 밝기 (Valence %)", 0, 100, 50)
            energy_val = st.slider("에너지 (%)", 0, 100, 60)
            acoustic_val = st.slider("어쿠스틱 (%)", 0, 100, 20)
        
        submitted = st.form_submit_button("🚀 흥행도 예측하기")
        
    if submitted and final_model:
        input_data = [[bpm_val, dance_val, valence_val, energy_val, acoustic_val, 0, 15, 5, temp, rain, humi, sun]]
        log_pred = final_model.predict(input_data)[0]
        real_pred = np.expm1(log_pred)
        st.success(f"### 📈 예상 스트리밍 횟수: 약 {int(real_pred):,} 회")

# --- Tab 2: 모델 분석 결과 ---
with tab2:
    st.subheader("로드된 모델의 예측 성능 및 상관관계")
    col_r1, col_r2 = st.columns([1, 1])
    
    with col_r1:
        st.write("#### 📈 최종 모델 결정계수 (R2 Score)")
        st.metric("랜덤 포레스트 성능", f"{r2_val:.4f}")
        st.caption("0보다 클 경우 평균보다 우수한 예측력을 의미합니다.")
        
    with col_r2:
        st.write("#### 🎯 실제값 vs 예측값 분포")
        if final_model:
            fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
            ax_scatter.scatter(y_test, y_pred, alpha=0.5, color='#4682b4')
            ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            st.pyplot(fig_scatter)

    st.divider()
    st.write("#### 📊 변수 간 상관관계 히트맵")
    corr_data = df_merged[list(col_map.keys())].rename(columns=col_map).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# --- Tab 3: 통계 및 분포/중요도 ---
with tab3:
    st.subheader("데이터 통계 및 변수 영향력 분석")
    c1, c2, c3 = st.columns(3)
    c1.metric("데이터 규모", f"{len(df_merged)}건")
    c2.metric("분석 변수", f"{len(feature_list)}개")
    c3.metric("모델 상태", "최적화 완료" if final_model else "로드 실패")
    
    st.write("#### 📉 변수별 데이터 분포")
    dist_feature = st.selectbox("확인할 변수 선택", list(col_map.values())[1:])
    eng_feature = [k for k, v in col_map.items() if v == dist_feature][0]
    
    # 변수별 맞춤 색상 적용
    color_map = {'기온': 'tomato', '강수량': 'lightskyblue', '습도': 'cadetblue', '일조량': 'gold', 'BPM': 'royalblue'}
    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    sns.histplot(df_merged[eng_feature], kde=True, color=color_map.get(dist_feature, 'teal'), ax=ax_dist)
    st.pyplot(fig_dist)
    
    st.divider()
    st.write("#### 🏆 핵심 요인 분석 (Feature Importance)")
    if final_model:
        importances = pd.Series(final_model.feature_importances_, 
                                index=[col_map.get(f, f) for f in feature_list]).sort_values()
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        importances.plot(kind='barh', color='skyblue', edgecolor='black', ax=ax_imp)
        st.pyplot(fig_imp)