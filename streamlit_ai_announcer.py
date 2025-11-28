"""
LoL AI 해설위원 웹서비스
Streamlit 기반 실시간 AI 해설위원 시스템

목적: 사용자가 경기를 선택하고 AI 해설위원 해설을 들을 수 있는 웹 인터페이스
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
import base64
import asyncio
import threading

# 모델 관련 imports
try:
    from tensorflow.keras.models import load_model
    import openai
    import edge_tts
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    st.warning("⚠️ 일부 라이브러리가 설치되지 않았습니다. 모델 기능이 제한될 수 있습니다.")

# 페이지 설정
st.set_page_config(
    page_title="Esport AI 해설위원",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .match-card {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: white;
        transition: all 0.3s ease;
        color: inherit;
    }
    
    .match-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
    }
    
    /* 다크모드 지원 */
    @media (prefers-color-scheme: dark) {
        .match-card {
            background: #2d3748 !important;
            border-color: #4a5568 !important;
            color: #ffffff !important;
        }
    }
    
    /* Streamlit 다크모드 감지 */
    .stApp[data-theme="dark"] .match-card {
        background: #2d3748 !important;
        border-color: #4a5568 !important;
        color: #ffffff !important;
    }
    
    .timeline-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 20px;
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997);
        transition: width 0.5s ease;
    }
    
    .announcer-box {
        border-left: 4px solid #667eea;
        background: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        word-wrap: break-word;
        white-space: pre-wrap;
        line-height: 1.6;
        color: inherit;
    }
    
    .announcer-text {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        line-height: 1.8;
        font-size: 14px;
        border: 1px solid #e1e5e9;
        word-break: keep-all;
        color: inherit;
    }
    
    /* 다크모드에서 해설위원 박스 및 텍스트 */
    @media (prefers-color-scheme: dark) {
        .announcer-box {
            background: #2d3748 !important;
            border-left-color: #63b3ed !important;
            color: #ffffff !important;
        }
        
        .announcer-text {
            background: #1a202c !important;
            border-color: #4a5568 !important;
            color: #e2e8f0 !important;
        }
    }
    
    /* Streamlit 다크모드 */
    .stApp[data-theme="dark"] .announcer-box {
        background: #2d3748 !important;
        border-left-color: #63b3ed !important;
        color: #ffffff !important;
    }
    
    .stApp[data-theme="dark"] .announcer-text {
        background: #1a202c !important;
        border-color: #4a5568 !important;
        color: #e2e8f0 !important;
        overflow-wrap: break-word;
    }
    
    .time-marker {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        margin-right: 1rem;
    }
    
    /* 다크모드에서 시간 마커 */
    @media (prefers-color-scheme: dark) {
        .time-marker {
            background: #63b3ed !important;
            color: #1a202c !important;
        }
    }
    
    .stApp[data-theme="dark"] .time-marker {
        background: #63b3ed !important;
        color: #1a202c !important;
    }
    
    /* 다크모드에서 텍스트 가독성 향상 */
    @media (prefers-color-scheme: dark) {
        .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #e2e8f0 !important;
        }
    }
    
    .stApp[data-theme="dark"] .stMarkdown,
    .stApp[data-theme="dark"] .stMarkdown p,
    .stApp[data-theme="dark"] .stMarkdown h1,
    .stApp[data-theme="dark"] .stMarkdown h2,
    .stApp[data-theme="dark"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }
    
    /* 다크모드에서 해설 전문 영역 */
    @media (prefers-color-scheme: dark) {
        .stExpander > div > div {
            background-color: #2d3748 !important;
            color: #e2e8f0 !important;
        }
    }
    
    .stApp[data-theme="dark"] .stExpander > div > div {
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
    }
    
    .stApp[data-theme="dark"] .stExpander div[style*="background-color: #f0f2f6"] {
        background-color: #2d3748 !important;
        border-left: 4px solid #63b3ed !important;
        color: #e2e8f0 !important;
    }
    
    /* 다크모드에서 해설 전문 컨텐츠 */
    @media (prefers-color-scheme: dark) {
        .commentary-content {
            background-color: #2d3748 !important;
            border-left: 4px solid #63b3ed !important;
            color: #e2e8f0 !important;
        }
    }
    
    .stApp[data-theme="dark"] .commentary-content {
        background-color: #2d3748 !important;
        border-left: 4px solid #63b3ed !important;
        color: #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'selected_match' not in st.session_state:
    st.session_state.selected_match = None
if 'announcements' not in st.session_state:
    st.session_state.announcements = {}
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = {}
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'custom_input' not in st.session_state:
    st.session_state.custom_input = None

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🎮LOL AI 해설위원</h1>
    <p>e스포츠 경기 해설 시스템</p>
</div>
""", unsafe_allow_html=True)

# 데이터 로딩 함수 (캐싱 적용)
@st.cache_data(show_spinner="📊 경기 데이터를 로딩 중...", ttl=600)
def load_match_data():
    """경기 데이터 로딩 (10분 캐싱)"""
    try:
        df = pd.read_csv('lol_for_user.csv')
        
        # 롤드컵 경기만 필터링 (Worlds 관련 키워드로 검색)
        worlds_keywords = ['worlds', 'world', 'championship', 'wcs']
        worlds_df = df[df['league'].str.lower().str.contains('|'.join(worlds_keywords), na=False)]
        
        # 롤드컵 경기가 있으면 우선 선택, 없으면 전체에서 선택
        if len(worlds_df) > 0:
            selected_df = worlds_df
            filter_info = f"롤드컵 경기 {len(worlds_df)}개 발견"
        else:
            selected_df = df
            filter_info = "롤드컵 경기를 찾을 수 없어 전체 경기에서 선택"
        
        # 최대 10경기로 제한
        if len(selected_df) > 10:
            selected_df = selected_df.sample(n=10).reset_index(drop=True)  # random_state 제거!
            
        st.success(f"✅ {len(selected_df):,}개 경기 데이터 로딩 완료! ({filter_info})")
        return selected_df
    except FileNotFoundError:
        st.error("❌ lol_for_user.csv 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        st.error(f"❌ 데이터 로딩 중 오류 발생: {e}")
        return None

@st.cache_resource(show_spinner="🤖 LSTM 모델을 로딩 중...")
def load_lstm_model():
    """LSTM 모델 로딩 (세션 동안 캐싱)"""
    if not MODELS_AVAILABLE:
        st.warning("⚠️ TensorFlow가 설치되지 않아 LSTM 모델을 사용할 수 없습니다.")
        return None
    try:
        model = load_model('best_lol_lstm_model.h5')
        
        # 모델 구조 정보 출력
        st.success("✅ Progressive LSTM 모델 로딩 완료!")
        
        # 모델 정보 표시 (디버깅용)
        with st.expander("🔍 Progressive LSTM 모델 정보 (디버깅)", expanded=False):
            st.text("모델 구조:")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
            
            st.text(f"입력 형태: {model.input_shape}")
            st.text(f"출력 형태: {model.output_shape}")
        
        return model
    except FileNotFoundError:
        st.error("❌ best_lol_lstm_model.h5 파일을 찾을 수 없습니다.")
        st.info("💡 먼저 lol_lstm_training.py를 실행하여 모델을 생성하세요!")
        return None
    except Exception as e:
        st.error(f"❌ Progressive LSTM 모델 로딩 실패: {e}")
        return None

@st.cache_resource(show_spinner="🔑 OpenAI 클라이언트 설정 중...")
def setup_openai_client():
    """OpenAI 클라이언트 설정 (세션 동안 캐싱)"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            # API 연결 테스트
            client.models.list()
            st.success("✅ OpenAI API 연결 성공!")
            return client
        except Exception as e:
            st.error(f"❌ OpenAI API 연결 실패: {e}")
            return None
    else:
        st.warning("⚠️ OpenAI API 키가 설정되지 않았습니다.")
        return None

@st.cache_data(show_spinner="🔍 경기 데이터 필터링 중...", ttl=60)
def filter_match_data(df, selected_league, selected_team):
    """경기 데이터 필터링 (1분 캐싱)"""
    filtered_df = df.copy()
    
    if selected_league != '전체':
        filtered_df = filtered_df[filtered_df['league'] == selected_league]
    
    if selected_team != '전체':
        filtered_df = filtered_df[(filtered_df['team1'] == selected_team) | 
                                  (filtered_df['team2'] == selected_team)]
    
    return filtered_df

@st.cache_data(show_spinner="📈 게임 통계 계산 중...", ttl=300)
def get_data_statistics(df):
    """데이터 통계 정보 계산 (5분 캐싱)"""
    stats = {
        'total_matches': len(df),
        'unique_leagues': len(df['league'].unique()),
        'unique_teams': len(set(df['team1'].unique()) | set(df['team2'].unique())),
        'date_range': {
            'start': df['formatted_date'].iloc[0] if 'formatted_date' in df.columns else '데이터 확인 필요',
            'end': df['formatted_date'].iloc[-1] if 'formatted_date' in df.columns else '데이터 확인 필요'
        },
        'league_distribution': df['league'].value_counts().to_dict()
    }
    return stats

def generate_custom_announcements_by_time(match_data, time_data):
    """시점별 사용자 입력 특성을 기반으로 해설 생성"""
    
    try:
        # 각 시점별로 해설 생성
        announcements = {}
        win_probabilities = {}
        
        # LSTM 모델로 점진적 승률 예측
        if lstm_model is not None:
            for time_point in [10, 15, 20, 25]:
                if time_point in time_data:
                    # 해당 시점까지의 데이터로 LSTM 입력 준비
                    lstm_input = []
                    
                    for t in [10, 15, 20, 25]:
                        if t <= time_point and t in time_data:
                            features = time_data[t]
                            lstm_input.append([
                                features['golddiff'],
                                features['xpdiff'], 
                                features['csdiff'],
                                features['kills'],
                                features['deaths'],
                                features['assists']
                            ])
                        else:
                            lstm_input.append([0, 0, 0, 0, 0, 0])  # 패딩
                    
                    # LSTM 예측
                    lstm_array = np.array([lstm_input], dtype=np.float32)  # (1, 4, 6)
                    
                    # 디버깅: 입력 데이터 확인
                    st.write(f"🔍 {time_point}분 LSTM 입력 데이터:")
                    st.write(lstm_array[0])
                    
                    raw_pred = lstm_model.predict(lstm_array, verbose=0)[0][0]
                    win_prob = float(raw_pred)
                    
                    # 디버깅: 원시 예측값 확인
                    st.write(f"🔍 {time_point}분 원시 예측값: {raw_pred}")
                    
                    win_prob = max(0.1, min(0.9, win_prob))  # 10%~90% 범위로 제한
                    
                    # 디버깅: 클리핑 후 값 확인
                    st.write(f"🔍 {time_point}분 클리핑 후: {win_prob}")
                    
                    win_probabilities[time_point] = win_prob
                    
                    # LLM 해설 생성
                    announcements[time_point] = generate_llm_commentary(
                        match_data, time_point, win_prob
                    )
        else:
            # LSTM 모델이 없는 경우 기본값
            for time_point in [10, 15, 20, 25]:
                if time_point in time_data:
                    win_probabilities[time_point] = 0.5
                    announcements[time_point] = [
                        f"{time_point}분 기본 해설 1",
                        f"{time_point}분 기본 해설 2", 
                        f"{time_point}분 기본 해설 3"
                    ]
        
        st.session_state.announcements = announcements
        st.session_state.win_probabilities = win_probabilities  # 승률 정보 저장
        
        # 음성 파일 생성
        generate_audio_files(announcements)
        
        st.success("✅ 시점별 커스텀 상황 분석 완료!")
        
        # 결과 미리보기 - 시점별 승률 표시
        prob_summary = " | ".join([f"{t}분: {p*100:.1f}%" for t, p in win_probabilities.items()])
        st.info(f"📊 시점별 승률: {prob_summary}")
        
    except Exception as e:
        st.error(f"❌ 시점별 해설 생성 중 오류 발생: {e}")

# 데이터 로딩
df = load_match_data()
lstm_model = load_lstm_model()
openai_client = setup_openai_client()

if df is None:
    st.stop()

# 헬퍼 함수들 (미리 정의)
def generate_announcements(match_data):
    """AI를 활용하여 해설 생성"""
    
    if not MODELS_AVAILABLE:
        # 모델이 없는 경우 샘플 데이터
        st.session_state.announcements = {
            10: [
                f"10분 시점, {match_data['team1']}이 골드 리드를 가져가고 있습니다. 현재 상황에서는 안정적인 운영이 필요해 보입니다.",
                f"와! {match_data['team1']}의 플레이가 정말 인상적이네요! 이대로 가면 스노우볼링이 시작될 것 같습니다!",
                f"골드 차이를 분석해보면, {match_data['team1']}이 약간의 우위를 점하고 있으나, {match_data['team2']}도 충분히 역전 가능한 상황입니다."
            ],
            15: [
                f"15분, 미드 게임에 접어들면서 {match_data['team2']}의 반격이 시작됩니다.",
                f"오! 이제 진짜 경기가 시작이네요! {match_data['team2']}가 멋진 플레이를 보여주고 있습니다!",
                f"오브젝트 컨트롤 측면에서 보면, 두 팀 모두 신중한 접근을 하고 있어 승부의 향방을 예측하기 어려운 상황입니다."
            ],
            20: [
                f"20분, 후반 페이즈로 접어들며 팀파이트의 중요성이 부각됩니다.",
                f"긴장하세요! 이제부터가 진짜 승부처입니다! 어느 팀이 먼저 이니시를 걸까요?",
                f"현재 아이템 빌드와 레벨 격차를 고려할 때, 다음 대규모 교전이 경기의 판도를 좌우할 것으로 보입니다."
            ],
            25: [
                f"25분, 게임이 막바지에 접어들었습니다. {match_data['team1'] if match_data['result'] == 1 else match_data['team2']}가 승기를 잡고 있습니다!",
                f"와우! 이제 정말 클라이맥스네요! 어느 팀이 마지막 한 방을 날릴까요? 정말 흥미진진합니다!",
                f"게임 후반부 아이템 완성도와 포지셔닝이 승부의 핵심이 되겠습니다. 한 번의 실수가 게임을 좌우할 수 있는 상황입니다."
            ]
        }
        st.success("✅ 샘플 해설이 생성되었습니다!")
        return
    
    try:
        # LSTM 예측
        lstm_predictions = predict_with_lstm(match_data)
        
        # LLM 해설 생성
        announcements = {}
        time_points = [10, 15, 20, 25]  # 25분 추가
        
        for time_point in time_points:
            win_prob = lstm_predictions.get(time_point, 0.5)
            
            # 고급 특징 추출
            advanced_features = {}
            if time_point > 10:
                hg_col = f'HiddenGold_{time_point-5}_{time_point}'
                if hg_col in match_data and pd.notna(match_data[hg_col]):
                    advanced_features['hidden_gold'] = match_data[hg_col]
            
            if time_point > 15:
                acc_col = f'Acc_golddiff_{time_point-10}_{time_point}'
                if acc_col in match_data and pd.notna(match_data[acc_col]):
                    advanced_features['acceleration'] = match_data[acc_col]

            announcements[time_point] = generate_llm_commentary(
                match_data, time_point, win_prob, advanced_features
            )
        
        st.session_state.announcements = announcements
        
        # 음성 파일 생성
        generate_audio_files(announcements)
        
        st.success("✅ AI 해설이 성공적으로 생성되었습니다!")
        
    except Exception as e:
        st.error(f"❌ 해설 생성 중 오류 발생: {e}")

def predict_with_lstm(match_data):
    """LSTM 모델로 점진적 승률 예측 - 각 시점별로 그 때까지의 데이터만 사용"""
    if lstm_model is None:
        st.warning("⚠️ LSTM 모델이 로딩되지 않아 기본값을 사용합니다.")
        return {10: 0.6, 15: 0.7, 20: 0.8, 25: 0.9}
    
    try:
        st.info("🤖 점진적 LSTM 예측 시작...")
        results = {}
        
        # 각 시점별 데이터 추출
        time_features = {}
        for time_point in [10, 15, 20, 25]:
            time_features[time_point] = [
                float(match_data.get(f'golddiffat{time_point}', 0)),
                float(match_data.get(f'xpdiffat{time_point}', 0)),
                float(match_data.get(f'killsat{time_point}', 0)),
                float(match_data.get(f'deathsat{time_point}', 0)),
                float(match_data.get(f'assistsat{time_point}', 0)),
                float(match_data.get(f'csdiffat{time_point}', 0))
            ]
        
        # 1. 10분 시점 예측 (10분 데이터 + 패딩)
        data_10 = [
            time_features[10],
            [0, 0, 0, 0, 0, 0],  # 패딩
            [0, 0, 0, 0, 0, 0],  # 패딩
            [0, 0, 0, 0, 0, 0]   # 패딩
        ]
        lstm_input_10 = np.array(data_10).reshape(1, 4, 6)
        pred_10 = float(lstm_model.predict(lstm_input_10, verbose=0)[0][0])
        results[10] = max(0.1, min(0.9, pred_10))
        
        # 2. 15분 시점 예측 (10분+15분 데이터 + 패딩)
        data_15 = [
            time_features[10],
            time_features[15],
            [0, 0, 0, 0, 0, 0],  # 패딩
            [0, 0, 0, 0, 0, 0]   # 패딩
        ]
        lstm_input_15 = np.array(data_15).reshape(1, 4, 6)
        pred_15 = float(lstm_model.predict(lstm_input_15, verbose=0)[0][0])
        results[15] = max(0.1, min(0.9, pred_15))
        
        # 3. 20분 시점 예측 (10분+15분+20분 데이터 + 패딩)
        data_20 = [
            time_features[10],
            time_features[15],
            time_features[20],
            [0, 0, 0, 0, 0, 0]   # 패딩
        ]
        lstm_input_20 = np.array(data_20).reshape(1, 4, 6)
        pred_20 = float(lstm_model.predict(lstm_input_20, verbose=0)[0][0])
        results[20] = max(0.1, min(0.9, pred_20))
        
        # 4. 25분 시점 예측 (전체 데이터 사용)
        data_25 = [
            time_features[10],
            time_features[15],
            time_features[20],
            time_features[25]
        ]
        lstm_input_25 = np.array(data_25).reshape(1, 4, 6)
        pred_25 = float(lstm_model.predict(lstm_input_25, verbose=0)[0][0])
        results[25] = max(0.1, min(0.9, pred_25))
        
        # 결과 로깅
        prob_summary = " | ".join([f"{t}분: {p*100:.1f}%" for t, p in results.items()])
        st.success(f"✅ 점진적 LSTM 예측 완료! {prob_summary}")
        
        return results
        
    except Exception as e:
        st.error(f"❌ LSTM 예측 오류: {e}")
        st.info("🔄 기본 예측값을 사용합니다.")
        return {10: 0.5, 15: 0.5, 20: 0.5, 25: 0.5}

# extract_features_for_lstm 함수는 새로운 점진적 예측 방식에서 더 이상 사용되지 않음
# 각 시점별 예측이 predict_with_lstm 함수 내에서 직접 처리됨

def generate_llm_commentary(match_data, time_point, win_prob, advanced_features=None):
    """LLM으로 해설 생성"""
    if openai_client is None:
        return [
            f"{time_point}분 기본 해설 1",
            f"{time_point}분 기본 해설 2", 
            f"{time_point}분 기본 해설 3"
        ]
    
    try:
        styles = ["프로페셔널", "열정적", "분석적"]
        commentaries = []
        
        # 고급 특징 문자열 생성 (기본값 처리)
        advanced_prompt = ""
        if advanced_features:
            feature_texts = []
            if 'hidden_gold' in advanced_features and advanced_features['hidden_gold'] != 0:
                feature_texts.append(f"- 숨겨진 골드 성장: {advanced_features['hidden_gold']:.0f}")
            if 'acceleration' in advanced_features and advanced_features['acceleration'] != 0:
                feature_texts.append(f"- 성장 가속도: {advanced_features['acceleration']:.0f}")
            
            if feature_texts:
                advanced_prompt = "\n추가 분석 정보:\n" + "\n".join(feature_texts)

        for style in styles:
            prompt = f"""
            {time_point}분 시점의 LoL 경기 상황에 대한 해설을 생성해주세요.
            
            경기 정보:
            - 팀1: {match_data['team1']}
            - 팀2: {match_data['team2']}
            - 현재 승률: {win_prob*100:.1f}%
            - 해설 스타일: {style}
            {advanced_prompt}
            
            조건:
            1. 30초~45초 분량의 자연스러운 해설 (150-200자)
            2. {style} 스타일에 맞는 톤과 어조 사용
            3. LoL 전문 용어와 상황 분석 포함 (특히 추가 분석 정보를 적극 활용)
            4. 흥미진진하고 몰입감 있는 표현
            5. 반드시 완전한 문장으로 끝나도록 작성
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # 파인튜닝 모델 ID로 변경 필요
                messages=[
                    {"role": "system", "content": "당신은 LoL 전문 해설위원입니다. 항상 완전한 문장으로 해설을 마무리해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # 토큰 수를 늘려서 텍스트 잘림 방지
                temperature=0.8,  # 창의성 증가
                presence_penalty=0.1,  # 반복 방지
                frequency_penalty=0.1  # 다양성 증가
            )
            
            # 응답 텍스트 정리 및 검증
            commentary = response.choices[0].message.content.strip()
            
            # 텍스트가 중간에 끊어졌는지 확인하고 보정
            if commentary and not commentary.endswith(('.', '!', '?', '다', '요', '네', '죠', '까', '군요', '습니다')):
                # 마지막 완전한 문장까지만 사용
                sentences = commentary.split('.')
                if len(sentences) > 1:
                    commentary = '.'.join(sentences[:-1]) + '.'
                else:
                    commentary += "... 경기가 더욱 흥미진진해지고 있습니다!"
            
            commentaries.append(commentary)
        
        return commentaries
        
    except Exception as e:
        st.error(f"LLM 해설 생성 오류: {e}")
        return [f"{time_point}분 해설 {i+1}" for i in range(3)]

def generate_audio_files(announcements):
    """Edge TTS로 음성 파일 생성"""
    try:
        import asyncio
        import edge_tts
        import io
        
        voices = [
            "ko-KR-InJoonNeural",    # 남성 해설위원 1
            "ko-KR-SunHiNeural",     # 여성 해설위원
            "ko-KR-BongJinNeural"    # 남성 해설위원 2
        ]
        
        async def create_speech(text, voice, rate="+0%", pitch="+0Hz"):
            """비동기로 음성 생성 (개선된 버전)"""
            try:
                # SSML 형식으로 음성 품질 향상
                ssml_text = f"""
                <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ko-KR">
                    <voice name="{voice}">
                        <prosody rate="{rate}" pitch="{pitch}">
                            {text}
                        </prosody>
                    </voice>
                </speak>
                """
                
                communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                return audio_bytes
            except Exception as e:
                st.warning(f"⚠️ 음성 생성 중 오류: {e}")
                return None
        
        # 각 시점별, 해설위원별 음성 파일 생성
        voice_settings = [
            ("ko-KR-InJoonNeural", "+10%", "+0Hz"),    # 남성 해설위원 1 - 빠르게
            ("ko-KR-SunHiNeural", "+0%", "+50Hz"),     # 여성 해설위원 - 높은 톤
            ("ko-KR-BongJinNeural", "-10%", "-20Hz")   # 남성 해설위원 2 - 느리고 낮은 톤
        ]
        
        for time_point, texts in announcements.items():
            for i, text in enumerate(texts):
                if i >= len(voice_settings):
                    continue
                    
                try:
                    voice, rate, pitch = voice_settings[i]
                    
                    # 텍스트 정리 (특수문자 제거, 줄바꿈 정리)
                    clean_text = text.strip()
                    clean_text = clean_text.replace('\n', ' ').replace('\r', ' ')
                    clean_text = ' '.join(clean_text.split())  # 여러 공백을 하나로
                    
                    # 텍스트 길이 확인 및 경고
                    if len(clean_text) > 500:
                        st.warning(f"⚠️ {time_point}분 해설위원 {i+1} 텍스트가 길어 음성 생성 시간이 오래 걸릴 수 있습니다.")
                    
                    # 비동기 음성 생성 (전체 텍스트 사용)
                    audio_bytes = asyncio.run(create_speech(clean_text, voice, rate, pitch))
                    
                    if audio_bytes and len(audio_bytes) > 0:
                        audio_key = f"{time_point}_{i}"
                        st.session_state.audio_files[audio_key] = audio_bytes
                        st.success(f"🔊 {time_point}분 해설위원 {i+1} 음성 생성 완료 ({len(clean_text)}자, {len(audio_bytes)} bytes)")
                    else:
                        st.error(f"❌ {time_point}분 해설위원 {i+1} 음성 데이터가 비어있습니다.")
                    
                except Exception as audio_error:
                    st.error(f"❌ {time_point}분 해설위원 {i+1} 음성 생성 실패: {audio_error}")
                    # 재시도 로직 추가
                    try:
                        st.info(f"🔄 {time_point}분 해설위원 {i+1} 음성 재생성 시도...")
                        simple_audio = asyncio.run(create_speech(clean_text, voice))
                        if simple_audio:
                            audio_key = f"{time_point}_{i}"
                            st.session_state.audio_files[audio_key] = simple_audio
                            st.success(f"✅ 재시도 성공!")
                    except:
                        st.warning(f"⚠️ 재시도도 실패했습니다.")
                    continue
        
        st.success("✅ 음성 파일 생성 완료!")
                
    except ImportError:
        st.warning("⚠️ edge-tts 라이브러리가 설치되지 않았습니다. 음성 기능이 비활성화됩니다.")
        st.info("설치 명령: pip install edge-tts")
    except Exception as e:
        st.error(f"❌ 음성 생성 오류: {e}")

def predict_with_custom_features(features_dict):
    """사용자 입력 특성으로 LSTM 예측 - 휴리스틱 기반 승률 계산"""
    try:
        # 각 시점별 우위도 기반 승률 계산 (더 현실적인 방식)
        def calculate_win_prob_from_stats(gold_diff, xp_diff, kill_diff, time_point):
            """시점별 스탯을 기준으로 승률 계산"""
            
            # 시간에 따른 스탯의 중요도 변화 반영
            time_multiplier = {10: 0.7, 15: 0.85, 20: 1.0, 25: 1.2}
            multiplier = time_multiplier.get(time_point, 1.0)
            
            # 가중 우위도 계산
            advantage = (
                (gold_diff / 1000) * 0.4 * multiplier +      # 골드차 (시간에 따라 중요도 증가)
                (xp_diff / 800) * 0.3 * multiplier +         # XP차
                kill_diff * 0.2 +                            # KD차 (시간 무관하게 중요)
                np.random.normal(0, 0.05)                     # 약간의 노이즈 (현실성)
            )
            
            # 시그모이드 함수로 0~1 승률 변환
            win_prob = 1 / (1 + np.exp(-advantage * 1.5))  # 1.5는 민감도 조절
            return max(0.1, min(0.9, win_prob))
        
        # 시간별 스탯 계산 (점진적 성장 시뮬레이션)
        results = {}
        base_gold = features_dict['golddiff']
        base_xp = features_dict['xpdiff']
        base_kills = features_dict['kills']
        base_deaths = features_dict['deaths']
        
        for time_point in [10, 15, 20, 25]:
            # 시간에 따른 성장률 (25분 기준 100%)
            time_factor = time_point / 25.0
            
            # 점진적 스탯 계산
            current_gold = base_gold * time_factor
            current_xp = base_xp * time_factor
            current_kill_diff = (base_kills - base_deaths) * time_factor
            
            # 승률 계산
            win_prob = calculate_win_prob_from_stats(
                current_gold, current_xp, current_kill_diff, time_point
            )
            
            results[time_point] = win_prob
        
        # 결과 로깅
        prob_summary = " | ".join([f"{t}분: {p*100:.1f}%" for t, p in results.items()])
        st.info(f"📊 휴리스틱 승률 계산: {prob_summary}")
        
        return results
            
    except Exception as e:
        st.error(f"❌ 커스텀 예측 오류: {e}")
        return {10: 0.5, 15: 0.5, 20: 0.5, 25: 0.5}

def generate_custom_announcements(match_data, custom_features):
    """사용자 입력 특성을 기반으로 해설 생성"""
    
    try:
        # LSTM 예측
        lstm_predictions = predict_with_custom_features(custom_features)
        
        # LLM 해설 생성
        announcements = {}
        time_points = [10, 15, 20, 25]
        
        for time_point in time_points:
            win_prob = lstm_predictions.get(time_point, 0.5)
            
            # 고급 특징은 커스텀 입력에서는 시뮬레이션하기 어려우므로 전달하지 않음
            announcements[time_point] = generate_llm_commentary(
                match_data, time_point, win_prob
            )
        
        st.session_state.announcements = announcements
        
        # 음성 파일 생성
        generate_audio_files(announcements)
        
        st.success("✅ 커스텀 상황 분석 완료!")
        
        # 결과 미리보기
        st.info(f"📊 예측 승률: 20분 시점 {lstm_predictions[20]*100:.1f}%")
        
    except Exception as e:
        st.error(f"❌ 커스텀 해설 생성 중 오류 발생: {e}")

# 사이드바 - 경기 선택
st.sidebar.markdown("## 🎯 경기 선택 방법")

# 탭으로 두 가지 방법 구분
tab1, tab2 = st.sidebar.tabs(["📋 경기 목록", "⌨️ 직접 입력"])

with tab1:
    st.markdown("### 실제 경기에서 선택")
    
    # 데이터 필터링 없이 전체 경기 사용
    filtered_df = df
    
    # 경기 목록 표시 (페이지네이션)
    st.markdown(f"### 📋 경기 목록 ({len(filtered_df)}경기)")
    
    # 페이지네이션 설정
    matches_per_page = 5  # 사이드바에서는 5경기씩
    total_matches = len(filtered_df)
    total_pages = (total_matches - 1) // matches_per_page + 1 if total_matches > 0 else 1
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("◀", key="prev_page") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
        
        with col2:
            st.write(f"페이지 {st.session_state.current_page}/{total_pages}")
        
        with col3:
            if st.button("▶", key="next_page") and st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
    
    # 현재 페이지의 경기들 표시
    start_idx = (st.session_state.current_page - 1) * matches_per_page
    end_idx = start_idx + matches_per_page
    current_matches = filtered_df.iloc[start_idx:end_idx]
    
    selected_index = None
    for idx, row in current_matches.iterrows():
        match_display = f"{row['team1']} vs {row['team2']}"
        date_display = row['formatted_date'] if 'formatted_date' in row else "날짜 미상"
        
        if st.button(
            f"🏆 {match_display}\n📅 {date_display}",
            key=f"match_{idx}",
            use_container_width=True
        ):
            selected_index = idx
            st.session_state.selected_match = row
            st.session_state.custom_input = None  # 커스텀 입력 모드 해제
            st.session_state.announcements = {}
            st.session_state.audio_files = {}

with tab2:
    st.markdown("### 직접 게임 상황 입력")
    st.caption("각 시점별로 게임 상황을 입력해서 AI 해설위원 반응을 확인하세요!")
    
    # 팀 이름 입력 (폼 밖에서)
    col_team1, col_team2 = st.columns(2)
    with col_team1:
        team1_name = st.text_input("팀1 이름", value="블루팀", key="team1_input")
    with col_team2:
        team2_name = st.text_input("팀2 이름", value="레드팀", key="team2_input")
    
    # 빠른 프리셋 버튼들 (폼 밖에서)
    st.markdown("**🚀 빠른 시나리오 선택:**")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("📈 우리팀 우세", use_container_width=True):
            st.session_state.preset_scenario = "winning"
            st.rerun()
    with preset_col2:
        if st.button("⚖️ 균등한 경기", use_container_width=True):
            st.session_state.preset_scenario = "balanced"
            st.rerun()
    with preset_col3:
        if st.button("📉 상대팀 우세", use_container_width=True):
            st.session_state.preset_scenario = "losing"
            st.rerun()
    
    # 선택된 시나리오에 따른 기본값 설정
    if 'preset_scenario' not in st.session_state:
        st.session_state.preset_scenario = "balanced"
    
    # 현재 선택된 시나리오 표시
    scenario_names = {"winning": "📈 우리팀 우세", "balanced": "⚖️ 균등한 경기", "losing": "📉 상대팀 우세"}
    st.info(f"현재 시나리오: {scenario_names[st.session_state.preset_scenario]}")
    
    with st.form("custom_features"):
        st.markdown("**📊 시점별 게임 상황 입력**")
        
        # 4개 시점별 탭으로 구분
        tab_10, tab_15, tab_20, tab_25 = st.tabs(["⏰ 10분", "⏰ 15분", "⏰ 20분", "⏰ 25분"])
        
        time_data = {}
        
        for tab, time_point in [(tab_10, 10), (tab_15, 15), (tab_20, 20), (tab_25, 25)]:
            with tab:
                st.markdown(f"**{time_point}분 시점**")
                
                # 시나리오별 기본값 설정
                if st.session_state.preset_scenario == "winning":
                    # 우리팀 우세 시나리오
                    if time_point == 10:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = 800, 500, 3, 1, 4, 15
                    elif time_point == 15:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = 2000, 1200, 7, 2, 12, 35
                    elif time_point == 20:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = 3500, 2000, 12, 4, 20, 55
                    else:  # 25분
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = 5000, 2800, 18, 6, 30, 75
                    scenario_desc = "우리팀 우세"
                elif st.session_state.preset_scenario == "losing":
                    # 상대팀 우세 시나리오
                    if time_point == 10:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = -600, -400, 1, 3, 2, -12
                    elif time_point == 15:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = -1500, -900, 3, 7, 5, -28
                    elif time_point == 20:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = -2800, -1600, 5, 12, 8, -45
                    else:  # 25분
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = -4200, -2400, 7, 18, 12, -65
                    scenario_desc = "상대팀 우세"
                else:
                    # 균등한 경기 시나리오
                    if time_point == 10:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = 200, 150, 2, 2, 3, 5
                    elif time_point == 15:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = 400, 300, 5, 5, 8, 10
                    elif time_point == 20:
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = 600, 500, 8, 8, 15, 15
                    else:  # 25분
                        default_gold, default_xp, default_kills, default_deaths, default_assists, default_cs = 800, 700, 12, 12, 22, 20
                    scenario_desc = "균등한 경기"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    golddiff = st.number_input(f"골드 차이 ({time_point}분)", 
                                             value=default_gold,
                                             min_value=-30000, max_value=30000, step=100,
                                             help="양수면 우리팀이 앞섬, 음수면 상대팀이 앞섬",
                                             key=f"gold_{time_point}_{st.session_state.preset_scenario}")
                    xpdiff = st.number_input(f"경험치 차이 ({time_point}분)", 
                                           value=default_xp,
                                           min_value=-15000, max_value=15000, step=50,
                                           key=f"xp_{time_point}_{st.session_state.preset_scenario}")
                    kills = st.number_input(f"킬 수 ({time_point}분)", 
                                          value=default_kills,
                                          min_value=0, max_value=50, step=1,
                                          key=f"kills_{time_point}_{st.session_state.preset_scenario}")
                
                with col2:
                    deaths = st.number_input(f"데스 수 ({time_point}분)", 
                                           value=default_deaths,
                                           min_value=0, max_value=50, step=1,
                                           key=f"deaths_{time_point}_{st.session_state.preset_scenario}")
                    assists = st.number_input(f"어시스트 수 ({time_point}분)", 
                                            value=default_assists,
                                            min_value=0, max_value=100, step=1,
                                            key=f"assists_{time_point}_{st.session_state.preset_scenario}")
                    csdiff = st.number_input(f"CS 차이 ({time_point}분)", 
                                           value=default_cs,
                                           min_value=-300, max_value=300, step=5,
                                           help="미니언/몬스터 처치 수 차이",
                                           key=f"cs_{time_point}_{st.session_state.preset_scenario}")
                
                # 기본값 정보 표시
                st.caption(f"💡 현재 시나리오: {scenario_desc} | {time_point}분 기본값 적용됨")
                if default_gold > 0:
                    st.success(f"📈 우리팀 +{default_gold:,} 골드, {default_kills}:{default_deaths} KDA")
                elif default_gold < 0:
                    st.error(f"📉 상대팀 +{abs(default_gold):,} 골드, {default_deaths}:{default_kills} KDA")
                else:
                    st.info(f"⚖️ 균등한 상황, {default_kills}:{default_deaths} KDA")
                
                time_data[time_point] = {
                    'golddiff': golddiff,
                    'xpdiff': xpdiff,
                    'kills': kills,
                    'deaths': deaths,
                    'assists': assists,
                    'csdiff': csdiff
                }
        
        if st.form_submit_button("🚀 AI 분석 시작", use_container_width=True, type="primary"):
            # 시점별 커스텀 입력 데이터 저장
            team1 = st.session_state.get("team1_input", "블루팀")
            team2 = st.session_state.get("team2_input", "레드팀")
            
            custom_match = {
                'team1': team1,
                'team2': team2,
                'league': '사용자 입력',
                'result': 1 if time_data[25]['golddiff'] > 0 else 0,  # 25분 기준으로 결과 결정
                'formatted_date': '직접 입력',
                'patch': 'Custom'
            }
            
            st.session_state.custom_input = time_data
            st.session_state.selected_match = custom_match
            st.session_state.announcements = {}
            st.session_state.audio_files = {}
            
            # 즉시 분석 실행
            with st.spinner("🤖 AI가 시점별 입력 상황을 분석 중..."):
                generate_custom_announcements_by_time(custom_match, time_data)
                # 분석 완료 후 페이지 새로고침으로 결과 표시
                st.rerun()

# 메인 컨텐츠
if st.session_state.selected_match is not None:
    match = st.session_state.selected_match
    
    # 커스텀 입력 모드인지 확인
    is_custom = hasattr(st.session_state, 'custom_input') and st.session_state.custom_input is not None
    
    # 선택된 경기 정보
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_custom:
            # 커스텀 입력 데이터 구조 확인 및 안전한 접근
            if isinstance(st.session_state.custom_input, dict):
                # 시점별 데이터인 경우 25분 기준으로 표시
                if 25 in st.session_state.custom_input:
                    sample_data = st.session_state.custom_input[25]
                    
                    # 승부 결과 계산 (25분 기준)
                    gold_diff = sample_data.get('golddiff', 0)
                    kill_diff = sample_data.get('kills', 0) - sample_data.get('deaths', 0)
                    
                    # 간단한 승부 예측 로직
                    if gold_diff > 1000 and kill_diff > 2:
                        winner = match['team1']
                        win_status = "크게 앞섬"
                        win_color = "#4CAF50"
                    elif gold_diff > 0 and kill_diff >= 0:
                        winner = match['team1']
                        win_status = "약간 앞섬"
                        win_color = "#8BC34A"
                    elif gold_diff < -1000 and kill_diff < -2:
                        winner = match['team2']
                        win_status = "크게 뒤짐"
                        win_color = "#F44336"
                    elif gold_diff < 0 and kill_diff <= 0:
                        winner = match['team2']
                        win_status = "약간 뒤짐"
                        win_color = "#FF9800"
                    else:
                        winner = "무승부"
                        win_status = "팽팽한 접전"
                        win_color = "#9E9E9E"
                    
                    st.markdown(f"""
                    <div class="match-card">
                        <h2>⌨️ {match['team1']} vs {match['team2']}</h2>
                        <p><strong>모드:</strong> 사용자 직접 입력</p>
                        <p><strong>예상 결과 (25분 기준):</strong> 
                           <span style="color: {win_color}; font-weight: bold;">
                           {winner} {win_status}
                           </span>
                        </p>
                        <p><strong>입력 상황 (25분 기준):</strong></p>
                        <ul>
                        <li>골드 차이: {sample_data.get('golddiff', 0):+,}</li>
                        <li>경험치 차이: {sample_data.get('xpdiff', 0):+,}</li>
                        <li>킬/데스/어시: {sample_data.get('kills', 0)}/{sample_data.get('deaths', 0)}/{sample_data.get('assists', 0)}</li>
                        <li>CS 차이: {sample_data.get('csdiff', 0):+}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 승률 정보가 있을 때만 표시
                    if 'win_probabilities' in st.session_state and st.session_state.win_probabilities:
                        win_probs = st.session_state.win_probabilities
                        st.markdown(f"""
                        **📊 시점별 예상 승률:**
                        - 10분: {win_probs.get(10, 0.5)*100:.1f}%
                        - 15분: {win_probs.get(15, 0.5)*100:.1f}%
                        - 20분: {win_probs.get(20, 0.5)*100:.1f}%
                        - 25분: {win_probs.get(25, 0.5)*100:.1f}%
                        """)
                else:
                    st.markdown(f"""
                    <div class="match-card">
                        <h2>⌨️ {match['team1']} vs {match['team2']}</h2>
                        <p><strong>모드:</strong> 사용자 직접 입력</p>
                        <p><strong>상태:</strong> 시점별 데이터 입력 완료</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="match-card">
                    <h2>⌨️ {match['team1']} vs {match['team2']}</h2>
                    <p><strong>모드:</strong> 사용자 직접 입력</p>
                    <p><strong>상태:</strong> 커스텀 데이터 로딩됨</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="match-card">
                <h2>🏆 {match['team1']} vs {match['team2']}</h2>
                <p><strong>리그:</strong> {match['league']}</p>
                <p><strong>날짜:</strong> {match.get('formatted_date', '날짜 미상')}</p>
                <p><strong>패치:</strong> {match.get('patch', 'Unknown')}</p>
                <p><strong>승리팀:</strong> {'팀1' if match['result'] == 1 else '팀2'} 
                   ({match['team1'] if match['result'] == 1 else match['team2']})</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎮 게임 컨트롤")
        
        # 분석 시작 버튼 (커스텀 모드가 아닐 때만 표시)
        if not is_custom:
            if st.button("🚀 AI 분석 시작", use_container_width=True, type="primary"):
                with st.spinner("🤖 AI가 경기를 분석 중..."):
                    generate_announcements(match)
        else:
            st.info("💡 사용자 입력 모드에서는 자동으로 분석이 완료되었습니다.")
        
    # 해설 생성 완료 후 결과 표시
    if st.session_state.announcements:
        st.markdown("---")
        st.markdown("## 🎤 AI 해설 결과")
        st.success("✅ AI 해설이 성공적으로 생성되었습니다!")
        
        # 모든 해설을 한 번에 표시 (시간 순서대로)
        for time_point in sorted(st.session_state.announcements.keys()):
            announcements = st.session_state.announcements[time_point]
            
            st.markdown(f"""
            <div class="announcer-box">
                <span class="time-marker">{time_point}분</span>
                <strong>경기 상황 분석</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # 탭으로 3명의 해설위원 구분
            tab1, tab2, tab3 = st.tabs([
                f"🎩 프로페셔널 ({time_point}분)", 
                f"🔥 열정적 ({time_point}분)", 
                f"📊 분석적 ({time_point}분)"
            ])
            
            announcer_configs = [
                (tab1, "프로페셔널 해설위원", 0),
                (tab2, "열정적 해설위원", 1), 
                (tab3, "분석적 해설위원", 2)
            ]
            
            for tab, name, i in announcer_configs:
                with tab:
                    if i < len(announcements):
                        # 해설 텍스트를 더 큰 영역에서 표시
                        commentary_text = announcements[i].strip()
                        
                        # 텍스트 길이와 완전성 체크
                        if len(commentary_text) < 50:
                            st.warning(f"⚠️ 해설이 너무 짧습니다: {commentary_text}")
                        elif not commentary_text.endswith(('.', '!', '?', '다', '요', '네', '죠', '까', '군요', '습니다')):
                            st.warning(f"⚠️ 해설이 중간에 잘렸을 수 있습니다: {commentary_text}")
                        
                        # 해설 텍스트를 확장 가능한 영역에 표시
                        with st.expander(f"📖 {name} 해설 전문", expanded=True):
                            st.markdown(f"""
                            <div class="commentary-content" style="
                                background-color: #f0f2f6;
                                padding: 20px;
                                border-radius: 10px;
                                border-left: 4px solid #ff6b6b;
                                font-size: 16px;
                                line-height: 1.6;
                                word-wrap: break-word;
                                white-space: pre-wrap;
                                color: inherit;
                            ">
                                {commentary_text}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 음성 재생 버튼
                        audio_key = f"{time_point}_{i}"
                        if audio_key in st.session_state.audio_files and st.session_state.audio_files[audio_key]:
                            st.markdown("**🔊 음성 해설:**")
                            try:
                                st.audio(st.session_state.audio_files[audio_key], format='audio/wav')
                            except:
                                st.audio(st.session_state.audio_files[audio_key])
                        else:
                            st.info("🔄 음성 파일을 생성 중입니다...")
                    else:
                        st.warning("해설을 준비 중입니다...")
            
            st.divider()  # 시간대별 구분선
        
    # 해설 분석 완료 메시지
    if st.session_state.announcements:
        col_play, col_pause = st.columns(2)
        with col_play:
            if st.button("▶️ 재생", use_container_width=True):
                st.session_state.is_playing = True
        with col_pause:
            if st.button("⏸️ 일시정지", use_container_width=True):
                st.session_state.is_playing = False
        
        # 리셋 버튼
        if st.button("🔄 리셋", use_container_width=True):
            st.session_state.game_progress = 0
            st.session_state.is_playing = False

    # 해설 완료 후 안내 메시지
    if st.session_state.announcements:
        st.success("🎉 모든 시점의 AI 해설이 생성되었습니다!")

else:
    # 경기 선택 안내
    # 캐싱된 통계 함수 사용
    stats = get_data_statistics(df)
    
    st.markdown(f"""
    ## 👈 경기를 선택해주세요
    
    왼쪽 사이드바에서 원하는 경기를 선택하시면:
    
    1. **🤖 AI 분석**: LSTM 모델이 경기 데이터를 분석합니다
    2. **🎤 해설 생성**: 파인튜닝된 LLM이 3가지 스타일의 해설을 생성합니다
    3. **🔊 음성 변환**: Edge TTS가 해설을 음성으로 변환합니다
    4. **▶️ 실시간 재생**: 25분 경기를 2분으로 압축하여 재생합니다
    
    ---
    
    ### 📊 사용 가능한 데이터 (캐싱됨)
    - **총 경기 수**: {stats['total_matches']:,}경기
    - **리그 수**: {stats['unique_leagues']}개 리그
    - **팀 수**: {stats['unique_teams']}개 팀
    - **날짜 범위**: {stats['date_range']['start']} ~ {stats['date_range']['end']}
    
    ### 🏆 리그별 경기 분포
    """)
    
    # 리그별 경기 수를 차트로 표시
    if stats['league_distribution']:
        league_df = pd.DataFrame([
            {'리그': k, '경기수': v} 
            for k, v in list(stats['league_distribution'].items())[:10]  # 상위 10개만 표시
        ])
        st.bar_chart(league_df.set_index('리그'), use_container_width=True)

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🎮 LoL AI 해설위원 시스템 | 데이터사이언스 프로젝트</p>
    <p>Powered by LSTM + LLM + Edge TTS</p>
</div>
""", unsafe_allow_html=True)