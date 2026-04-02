import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import os

# --- 1. 그래프 기본 설정 (한글 폰트 미사용) ---
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid") # 배포 시 깔끔한 디자인 적용

# --- 2. 데이터 및 저장된 모델 로드 ---
@st.cache_resource
def load_assets():
    # 데이터 로드
    df_spotify = pd.read_csv('dataset/spotify-2023.csv', encoding='latin-1')
    df_weather = pd.read_csv('dataset/weather_2019_2023_total.csv')
    
    # 데이터 전처리 (날짜 병합 및 로그 변환)
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
    
    # [수정] 그래프에 표시될 라벨만 영어로 매핑 (깨짐 방지)
    column_mapping = {
        'streams': 'Streams', 'bpm': 'BPM', 'danceability_%': 'Danceability',
        'valence_%': 'Valence', 'energy_%': 'Energy', 'acousticness_%': 'Acousticness',
        'instrumentalness_%': 'Instrumentalness', 'liveness_%': 'Liveness', 'speechiness_%': 'Speechiness',
        '평균기온(°C)': 'Temp', '일강수량(mm)': 'Rain', '평균 상대습도(%)': 'Humidity', '합계 일조시간(hr)': 'Sunshine'
    }
    
    feature_cols = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 'acousticness_%', 
                    'instrumentalness_%', 'liveness_%', 'speechiness_%',
                    '평균기온(°C)', '일강수량(mm)', '평균 상대습도(%)', '합계 일조시간(hr)']
    
    # 모델 불러오기
    try:
        loaded_model = joblib.load('model/spotify_weather_model_final.pkl')
    except:
        loaded_model = None

    return df_merged, loaded_model, column_mapping, feature_cols

df_merged, final_model, col_map, feature_list = load_assets()

# --- 3. 성능 평가 데이터 준비 ---
X = df_merged[feature_list].fillna(0)
y = df_merged['streams_log']
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if final_model:
    y_pred = final_model.predict(X_test)
    r2_val = r2_score(y_test, y_pred)
else:
    y_pred, r2_val = None, 0

# --- 4. 메인 UI (설명은 한글로 유지) ---
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
    st.subheader("모델 예측 성능 및 상관관계")
    col_r1, col_r2 = st.columns([1, 1])
    
    with col_r1:
        st.write("#### 📈 최종 모델 결정계수 (R2 Score)")
        st.metric("랜덤 포레스트 성능", f"{r2_val:.4f}")
        
    with col_r2:
        st.write("#### 🎯 Actual vs Predicted (Log)")
        if final_model:
            fig_scatter, ax_scatter = plt.subplots(figsize=(5, 4))
            ax_scatter.scatter(y_test, y_pred, alpha=0.5, color='#4682b4')
            ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            st.pyplot(fig_scatter)

    st.divider()
    st.write("#### 📊 상관관계 히트맵 (Feature Correlation)")
    # 그래프에 들어가는 컬럼명을 영어로 치환하여 깨짐 방지
    corr_data = df_merged[list(col_map.keys())].rename(columns=col_map).corr()
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# --- Tab 3: 통계 및 분포/중요도 ---
with tab3:
    st.subheader("데이터 통계 및 변수 영향력")
    c1, c2, c3 = st.columns(3)
    c1.metric("데이터 규모", f"{len(df_merged)}건")
    c2.metric("분석 변수", f"{len(feature_list)}개")
    c3.metric("모델 상태", "최적화 완료" if final_model else "로드 실패")
    
    st.write("#### 📉 변수별 데이터 분포 (Distribution)")
    # 선택 메뉴는 한글로 보여주되, 실제 데이터 접근은 영어 라벨을 사용
    dist_feature_kor = st.selectbox("확인할 변수 선택", list(col_map.values())[1:])
    # 선택한 한글명에 해당하는 영어 컬럼명 찾기
    eng_feature = [k for k, v in col_map.items() if v == dist_feature_kor][0]
    
    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    sns.histplot(df_merged[eng_feature], kde=True, color='teal', ax=ax_dist)
    ax_dist.set_xlabel(dist_feature_kor.encode('ascii', 'ignore').decode('ascii')) # 그래프 축 영어로 강제
    ax_dist.set_xlabel(col_map[eng_feature]) # 미리 정의된 영어 라벨 사용
    st.pyplot(fig_dist)
    
    st.divider()
    st.write("#### 🏆 핵심 요인 분석 (Feature Importance)")
    if final_model:
        # 인덱스를 영어(col_map의 값)로 설정하여 그래프 출력
        importances = pd.Series(final_model.feature_importances_, 
                                index=[col_map.get(f, f) for f in feature_list]).sort_values()
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        importances.plot(kind='barh', color='skyblue', edgecolor='black', ax=ax_imp)
        ax_imp.set_title("Top Success Drivers")
        st.pyplot(fig_imp)