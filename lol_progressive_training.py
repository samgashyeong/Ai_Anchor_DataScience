# LoL 점진적 예측을 위한 LSTM 재학습 코드
# 시점별 데이터로 쪼개서 학습하는 올바른 방식

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🔥 점진적 예측을 위한 LSTM 재학습 시작!")
print("="*60)

# =================================================================
# 1. 올바른 데이터 전처리 - 시점별 슬라이싱
# =================================================================

def create_progressive_training_data(df):
    """
    게임 데이터를 시점별로 쪼개서 학습 데이터 생성
    하나의 게임 → 여러 개의 학습 샘플
    """
    
    progressive_data = []
    labels = []
    
    print("📊 시점별 데이터 생성 중...")
    
    for idx, game in df.iterrows():
        game_result = game['result']  # 최종 결과
        
        # 각 시점별로 학습 데이터 생성
        for time_point in [10, 15, 20, 25]:
            try:
                # 해당 시점까지의 누적 데이터 구성
                time_features = []
                
                # 10분까지만 보고 예측
                if time_point >= 10:
                    time_features.append([
                        game.get('golddiffat10', 0),
                        game.get('xpdiffat10', 0),
                        game.get('killsat10', 0),
                        game.get('deathsat10', 0),
                        game.get('assistsat10', 0),
                        game.get('csdiffat10', 0)
                    ])
                
                # 15분까지 보고 예측
                if time_point >= 15:
                    time_features.append([
                        game.get('golddiffat15', 0),
                        game.get('xpdiffat15', 0),
                        game.get('killsat15', 0),
                        game.get('deathsat15', 0),
                        game.get('assistsat15', 0),
                        game.get('csdiffat15', 0)
                    ])
                
                # 20분까지 보고 예측
                if time_point >= 20:
                    time_features.append([
                        game.get('golddiffat20', 0),
                        game.get('xpdiffat20', 0),
                        game.get('killsat20', 0),
                        game.get('deathsat20', 0),
                        game.get('assistsat20', 0),
                        game.get('csdiffat20', 0)
                    ])
                
                # 25분까지 보고 예측
                if time_point >= 25:
                    time_features.append([
                        game.get('golddiffat25', 0),
                        game.get('xpdiffat25', 0),
                        game.get('killsat25', 0),
                        game.get('deathsat25', 0),
                        game.get('assistsat25', 0),
                        game.get('csdiffat25', 0)
                    ])
                
                # 패딩으로 4개 시점 맞추기
                while len(time_features) < 4:
                    time_features.append([0, 0, 0, 0, 0, 0])
                
                progressive_data.append(time_features[:4])  # 4개 시점만
                labels.append(game_result)
                
            except Exception as e:
                print(f"⚠️ 게임 {idx} 처리 중 오류: {e}")
                continue
    
    return np.array(progressive_data), np.array(labels)

# =================================================================
# 2. 데이터 로딩 및 변환
# =================================================================

try:
    # 기존 데이터 로드
    df = pd.read_csv('2025_LoL_esports_match_data_from_OraclesElixir.csv')
    
    # 블루팀만 필터링
    df_blue = df[df['participantid'] == 100].copy()
    print(f"📋 블루팀 경기 데이터: {len(df_blue)}경기")
    
    # 점진적 학습 데이터 생성
    X_progressive, y_progressive = create_progressive_training_data(df_blue)
    
    print(f"✅ 점진적 데이터 생성 완료!")
    print(f"   원본 경기 수: {len(df_blue)}")
    print(f"   학습 샘플 수: {len(X_progressive)} (약 {len(X_progressive)/len(df_blue):.1f}배 증가)")
    print(f"   데이터 형태: {X_progressive.shape}")
    
except Exception as e:
    print(f"❌ 데이터 로딩 실패: {e}")
    exit()

# =================================================================
# 3. 데이터 분할 및 정규화
# =================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_progressive, y_progressive, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_progressive
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,  # 0.2 * 0.25 = 0.05 전체의 5%가 검증셋
    random_state=42,
    stratify=y_train
)

print(f"📊 데이터 분할 완료:")
print(f"   학습셋: {X_train.shape[0]:,}개")
print(f"   검증셋: {X_val.shape[0]:,}개")  
print(f"   테스트셋: {X_test.shape[0]:,}개")

# =================================================================
# 4. 기존 LSTM 모델 구조 사용하여 재학습
# =================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

def create_progressive_lstm_model(input_shape, learning_rate=0.001):
    """점진적 예측을 위한 LSTM 모델 (기존 구조 동일)"""
    
    model = Sequential([
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, 
             kernel_regularizer=l2(0.001), input_shape=input_shape),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        LSTM(32, dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        
        Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        
        # 출력: 승률 (0~1)
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# 모델 생성
model = create_progressive_lstm_model((4, 6))  # (time_steps, features)
print("🤖 점진적 예측 LSTM 모델 생성 완료!")
model.summary()

# =================================================================
# 5. 모델 학습
# =================================================================

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'progressive_lol_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print("\n🚀 점진적 예측 모델 학습 시작!")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

print("✅ 점진적 예측 모델 학습 완료!")

# =================================================================
# 6. 모델 평가
# =================================================================

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 테스트 예측
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# 성능 평가
test_accuracy = np.mean(y_pred == y_test)
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 점진적 예측 모델 성능:")
print(f"   정확도: {test_accuracy:.4f} ({test_accuracy:.1%})")
print(f"   AUC: {test_auc:.4f}")

print("\n📋 분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['패배', '승리']))

# =================================================================
# 7. 테스트: 실제 시점별 예측 검증
# =================================================================

print("\n🧪 시점별 예측 테스트:")

# 샘플 게임 하나로 테스트
sample_game = df_blue.iloc[0]

for time_point in [10, 15, 20, 25]:
    # 해당 시점까지의 데이터만 구성
    test_features = []
    
    if time_point >= 10:
        test_features.append([
            sample_game.get('golddiffat10', 0),
            sample_game.get('xpdiffat10', 0),
            sample_game.get('killsat10', 0),
            sample_game.get('deathsat10', 0),
            sample_game.get('assistsat10', 0),
            sample_game.get('csdiffat10', 0)
        ])
    
    if time_point >= 15:
        test_features.append([
            sample_game.get('golddiffat15', 0),
            sample_game.get('xpdiffat15', 0),
            sample_game.get('killsat15', 0),
            sample_game.get('deathsat15', 0),
            sample_game.get('assistsat15', 0),
            sample_game.get('csdiffat15', 0)
        ])
    
    if time_point >= 20:
        test_features.append([
            sample_game.get('golddiffat20', 0),
            sample_game.get('xpdiffat20', 0),
            sample_game.get('killsat20', 0),
            sample_game.get('deathsat20', 0),
            sample_game.get('assistsat20', 0),
            sample_game.get('csdiffat20', 0)
        ])
    
    if time_point >= 25:
        test_features.append([
            sample_game.get('golddiffat25', 0),
            sample_game.get('xpdiffat25', 0),
            sample_game.get('killsat25', 0),
            sample_game.get('deathsat25', 0),
            sample_game.get('assistsat25', 0),
            sample_game.get('csdiffat25', 0)
        ])
    
    # 패딩
    while len(test_features) < 4:
        test_features.append([0, 0, 0, 0, 0, 0])
    
    # 예측
    test_input = np.array(test_features).reshape(1, 4, 6)
    win_prob = model.predict(test_input, verbose=0)[0][0]
    
    print(f"   {time_point}분 시점 승률: {win_prob*100:.1f}%")

print(f"   실제 결과: {'승리' if sample_game['result'] == 1 else '패배'}")

print("\n🎉 점진적 예측 모델 준비 완료!")
print("이제 Streamlit에서 이 모델(progressive_lol_lstm_model.h5)을 사용하세요!")