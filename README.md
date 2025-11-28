# 🎮 LoL AI 해설위원 시스템

## 📋 프로젝트 개요
League of Legends 경기 데이터를 분석하여 AI가 실시간으로 해설을 생성하는 멀티모달 시스템입니다.

## 🏗️ 시스템 아키텍처
```
LSTM 승률 예측 → LLM 해설 생성 → TTS 음성 변환 → Streamlit 웹 UI
```

## 📁 프로젝트 구조

```
Ai_anchor/
├── 📂 src/                          # 핵심 소스 코드
│   ├── progressive_lstm_preprocessing.py  # 데이터 전처리
│   ├── lol_lstm_training.py              # LSTM 모델 훈련
│   └── streamlit_ai_announcer.py         # 웹 애플리케이션 (메인)
├── 📂 data/                         # 데이터 파일
│   ├── 2025_LoL_esports_match_data_from_OraclesElixir.csv  # 원본 데이터
│   ├── lol_for_user.csv                  # 처리된 사용자 데이터
│   └── lol_lstm_data_v2.npz              # 훈련용 데이터
├── 📂 models/                       # 학습된 모델
│   └── best_lol_lstm_model.h5            # LSTM 승률 예측 모델
├── 📂 temp/                         # 개발/테스트 파일
│   └── openai_finetuning.py              # OpenAI 파인튜닝 (옵션)
└── 📂 .venv/                        # Python 가상환경
```

## 🚀 실행 방법

### 1. 환경 설정
```bash
cd Ai_anchor
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 웹 애플리케이션 실행
```bash
cd src
streamlit run streamlit_ai_announcer.py
```

### 3. OpenAI API 키 설정 (선택사항)
```bash
# PowerShell에서 환경변수 설정
$env:OPENAI_API_KEY="your_openai_api_key_here"

# 또는 CMD에서
set OPENAI_API_KEY=your_openai_api_key_here
```

## 🔧 핵심 기능

### ⚡ LSTM 승률 예측
- **입력**: 10분, 15분, 20분, 25분 시점별 게임 상황
- **출력**: 점진적 승률 변화 예측
- **특징**: Progressive 데이터 증강으로 실시간 예측 최적화

### 🎤 LLM 해설 생성  
- **모델**: GPT-3.5-turbo (파인튜닝 옵션)
- **스타일**: 전문적, 흥미진진, 분석적 해설
- **입력**: 승률 + 게임 상황 데이터

### 🔊 TTS 음성 변환
- **엔진**: Microsoft Edge TTS
- **언어**: 한국어 (다양한 음성 지원)
- **출력**: 실시간 음성 해설

### 🌐 Streamlit 웹 UI
- **기능**: 매치 선택, 커스텀 입력, 오디오 재생
- **인터페이스**: 사용자 친화적 대시보드

## 📊 데이터 처리 파이프라인

1. **원본 데이터**: Oracle's Elixir LoL 2025 데이터 (7,014 게임)
2. **데이터 증강**: Progressive 방식으로 28,056 훈련 샘플 생성
3. **특성 추출**: 골드차, 경험치차, CS차, KDA 등 6개 핵심 특성
4. **모델 훈련**: LSTM (128→64→32 구조) + BatchNorm + Dropout

## 🎯 주요 혁신사항

### Progressive 데이터 증강
- **문제**: 훈련 시 전체 데이터 vs 추론 시 부분 데이터 불일치
- **해결**: 1게임 → 4개 훈련 샘플로 변환 (패딩 패턴 포함)
- **결과**: 실시간 예측 정확도 대폭 향상

## 🛠️ 기술 스택
- **ML/DL**: TensorFlow, NumPy, Pandas
- **LLM**: OpenAI GPT-3.5-turbo
- **TTS**: Microsoft Edge TTS
- **웹**: Streamlit
- **언어**: Python 3.8+

## 📈 성능 지표
- **LSTM 모델**: 실시간 승률 예측 정확도 최적화
- **처리 속도**: 웹 인터페이스에서 실시간 응답
- **사용자 경험**: 직관적인 UI/UX 설계

## 🔍 사용법

### 매치 분석
1. 웹 앱에서 원하는 LoL 매치 선택
2. 시점별(10분, 15분, 20분, 25분) 승률 확인  
3. AI 해설 생성 및 음성 재생

### 커스텀 시나리오
1. "커스텀 시나리오 입력" 탭 선택
2. 각 시점별 게임 상황 수치 입력
3. LSTM 예측 및 해설 생성

## 📝 주의사항
- OpenAI API 키가 필요한 경우 환경변수로 설정
- 모델 파일과 데이터는 `models/`, `data/` 폴더에 위치
- 가상환경 활성화 후 실행 권장
 
**목적**: LoL 경기 분석 및 AI 해설 시스템  
**라이선스**: MIT