#LoL 프로 경기 예측을 위한 LSTM 모델 학습 코드
#전처리된 데이터(lol_lstm_data.npz)를 사용하여 경기 결과 예측

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

print("🤖 LoL 프로 경기 예측 LSTM 모델 학습 시작!")
print("="*60)

# =================================================================
# 1. 데이터 로딩
# =================================================================

print("📂 점진적 LSTM 데이터 로딩 중...")

try:
    # 점진적 전처리된 데이터 로딩
    data = np.load('processed_data/progressive_lstm_data.npz')
    X_train = data['X_train']
    y_train = data['y_train'] 
    X_val = data['X_val']
    y_val = data['y_val']
    feature_names = data['feature_names']
    
    # 점진적 데이터에는 추가 특징 없음 (LSTM 메인 특징만 사용)
    X_additional_train = X_additional_val = None
    additional_feature_names = []
    
    # 테스트 데이터는 검증 데이터로 대체 (점진적 데이터에는 별도 테스트셋 없음)
    X_test = X_val
    y_test = y_val
    
    print("✅ 점진적 LSTM 데이터 로딩 완료!")
    print(f"   📊 X_train: {X_train.shape} (점진적 증강됨)")
    print(f"   📊 X_val:   {X_val.shape}")
    print(f"   📊 X_test:  {X_test.shape}")
    print(f"   🎯 특징들: {list(feature_names)}")
    print(f"   🕐 시간대: [10분, 15분, 20분, 25분]")
    print(f"   💡 점진적 증강: 1게임 → 4샘플 (시점별 패딩 적용)")
    
    # 점진적 데이터 구조 확인
    print(f"\n🔍 점진적 데이터 구조 확인:")
    print(f"   총 원본 게임 수: 약 {len(X_train)//4:,}게임")
    print(f"   증강된 샘플 수: {len(X_train):,}샘플")
    print(f"   승률: {np.mean(y_train)*100:.1f}%")
    
except FileNotFoundError:
    print("❌ 'processed_data/progressive_lstm_data.npz' 파일을 찾을 수 없습니다.")
    print("먼저 progressive_lstm_preprocessing.py를 실행해주세요.")
    exit()

# =================================================================
# 2. 모델 아키텍처 설계
# =================================================================

print("\n" + "="*60)
print("🏗️ 점진적 예측 LSTM 모델 아키텍처 설계")
print("="*60)

def create_progressive_lstm_model(input_shape, learning_rate=0.001):
    """
    점진적 예측을 위한 LSTM 모델 생성
    - 시점별 패딩을 고려한 LSTM 구조
    - 입력: (4 timepoints, 6 features)
    - 출력: 승률 (0~1)
    """
    
    model = Sequential([
        # 첫 번째 LSTM 레이어 - 시퀀스 유지
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, 
             kernel_regularizer=l2(0.001), input_shape=input_shape),
        BatchNormalization(),
        
        # 두 번째 LSTM 레이어 - 시퀀스 유지
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        # 세 번째 LSTM 레이어 - 최종 출력
        LSTM(32, dropout=0.2, recurrent_dropout=0.2,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        
        # Dense 레이어들
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        
        Dense(8, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        
        # 출력 레이어 - 승률 예측
        Dense(1, activation='sigmoid')
    ])
    
    # 컴파일
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# 모델 생성 (단일 LSTM 입력)
input_shape = (X_train.shape[1], X_train.shape[2])  # (4, 6)

print(f"📊 입력 형태: {input_shape}")
print(f"📊 특성: {list(feature_names)}")
print(f"📊 시점별 패딩 적용: 점진적 예측 지원")

model = create_progressive_lstm_model(input_shape)

print("🤖 점진적 예측 LSTM 모델 아키텍처:")
model.summary()

# =================================================================
# 3. 콜백 설정
# =================================================================

print("\n📋 학습 콜백 설정...")

# 조기 종료
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# 학습률 감소
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# 최고 모델 저장
model_checkpoint = ModelCheckpoint(
    '../models/best_lol_lstm_model.h5',  # 점진적 모델명으로 변경
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]
print("✅ 콜백 설정 완료!")

# =================================================================
# 4. 모델 학습
# =================================================================

print("\n" + "="*60)
print("🚀 LSTM 모델 학습 시작")
print("="*60)

# 학습 시작
print("🔥 점진적 LSTM 모델 학습 중... (시간이 오래 걸릴 수 있습니다)")

# 입력 데이터 준비 (점진적 데이터는 단일 LSTM 입력만 사용)
train_inputs = X_train
val_inputs = X_val
print(f"✅ 점진적 입력 모드: LSTM({X_train.shape}) - 시점별 패딩 포함")

history = model.fit(
    train_inputs, y_train,
    validation_data=(val_inputs, y_val),
    epochs=100,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)

print("✅ 모델 학습 완료!")

# =================================================================
# 5. 모델 평가
# =================================================================

print("\n" + "="*60)
print("📊 모델 성능 평가")
print("="*60)

# 최고 모델 로딩
best_model = tf.keras.models.load_model('../models/best_lol_lstm_model.h5')

# 테스트 데이터 준비
test_inputs = X_test

y_pred_proba = best_model.predict(test_inputs)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# 성능 지표 계산
test_accuracy = np.mean(y_pred == y_test)
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"🎯 테스트 정확도: {test_accuracy:.4f} ({test_accuracy:.1%})")
print(f"🎯 테스트 AUC: {test_auc:.4f}")

# 상세한 분류 리포트
print("\n📋 상세 분류 리포트:")
print(classification_report(y_test, y_pred, 
                          target_names=['패배', '승리'],
                          digits=4))

# =================================================================
# 6. 결과 시각화
# =================================================================

print("\n📈 학습 과정 및 결과 시각화 중...")

# 그래프 설정
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Esport LSTM Model Result', fontsize=16, fontweight='bold')

# 1. 손실 함수 그래프
axes[0,0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0,0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0,0].set_title('Loss', fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. 정확도 그래프
axes[0,1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0,1].set_title('Accuracy', fontweight='bold')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Predicted Lose', 'Predicted Win'],
           yticklabels=['Actual Lose', 'Actual Win'],
           ax=axes[1,0])
axes[1,0].set_title('Confusion Matrix', fontweight='bold')

# 4. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1,1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {test_auc:.3f})')
axes[1,1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
axes[1,1].set_title('ROC Curve', fontweight='bold')
axes[1,1].set_xlabel('False Positive Rate')
axes[1,1].set_ylabel('True Positive Rate')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lol_lstm_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 시각화 완료! 'lol_lstm_results.png' 파일로 저장됨")

# =================================================================
# 7. 특징 중요도 분석 (간접적)
# =================================================================

print("\n" + "="*60)
print("🔍 특징 분석")
print("="*60)

# 각 시간대별 예측 성능 비교
print("📊 시간대별 중요도 분석:")

time_steps_minutes = [10, 15, 20, 25]  # 분 단위 표시용

for i, minute in enumerate(time_steps_minutes):
    # 데이터를 복사하여 마스킹 준비
    X_masked = X_test.copy()
    
    # 현재 시점(i) 이후의 데이터는 모두 0으로 만듦 (Padding 효과)
    if i < 3: # 25분(마지막)이 아닐 때만
        X_masked[:, i+1:, :] = 0 
        
    # 예측 수행 (형태는 (N, 4, 6)으로 유지됨)
    y_single_pred = best_model.predict(X_masked, verbose=0)
    single_auc = roc_auc_score(y_test, y_single_pred)
    
    print(f"   🕐 {minute}분 시점 데이터만 사용 시: AUC = {single_auc:.4f}")

# 특징별 평균값 분석
print(f"\n📈 특징별 평균 분석 ({list(feature_names)}):")
for i, feature in enumerate(feature_names):
    # 25분 시점(마지막 타임스텝)의 데이터로 비교
    win_mean = np.mean(X_test[y_test==1, -1, i]) 
    lose_mean = np.mean(X_test[y_test==0, -1, i])
    print(f"   📊 {feature}:")
    print(f"      승리팀 평균: {win_mean:8.2f}")
    print(f"      패배팀 평균: {lose_mean:8.2f}")
    print(f"      차이: {win_mean-lose_mean:13.2f}")

# =================================================================
# 8. 모델 저장 및 결과 요약
# =================================================================

print("\n" + "="*60)
print("💾 최종 결과 저장 및 요약")
print("="*60)

# 학습 히스토리 저장
history_df = pd.DataFrame(history.history)
history_df.to_csv('lol_lstm_training_history.csv', index=False)

# 예측 결과 저장
results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'probability': y_pred_proba.flatten()
})
results_df.to_csv('lol_lstm_predictions.csv', index=False)

# 최종 요약
print(f"""
🎉 LoL LSTM 모델 학습 완료!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 최종 성능:
   • 테스트 정확도: {test_accuracy:.1%}
   • 테스트 AUC:    {test_auc:.4f}
   • 총 에포크:     {len(history.history['loss'])}

🎯 데이터셋 정보:
   • 학습 데이터:   {X_train.shape[0]:,}경기
   • 검증 데이터:   {X_val.shape[0]:,}경기  
   • 테스트 데이터: {X_test.shape[0]:,}경기
   
🏗️ 모델 구조:
   • LSTM 입력:     {lstm_input_shape}
   • 추가 특징:     {additional_input_shape[0]}개
   • LSTM 레이어:   3개 (128→64→32)
   • Dense 레이어:  다중 입력 구조
   • 총 파라미터:   약 {model.count_params():,}개

💾 저장된 파일:
   • best_lol_lstm_model.h5 (최고 성능 모델)
   • lol_lstm_results.png (결과 시각화)
   • lol_lstm_training_history.csv (학습 과정)
   • lol_lstm_predictions.csv (예측 결과)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("🚀 모델이 준비되었습니다! 이제 실시간 경기 예측에 사용할 수 있습니다!")

# 사용 예시 코드 출력
print(f"""
💡 점진적 예측 모델 사용 예시:

# 저장된 모델 로딩
import tensorflow as tf
model = tf.keras.models.load_model('progressive_lol_lstm_model.h5')

# 새로운 경기 데이터로 예측 (점진적 방식)
# 1. LSTM 입력 (shape: (1, 4, 6)) - 시점별 패딩 포함
lstm_data = np.array([[[gold_diff_10, xp_diff_10, cs_diff_10, kills_10, deaths_10, assists_10],
                       [gold_diff_15, xp_diff_15, cs_diff_15, kills_15, deaths_15, assists_15],
                       [0, 0, 0, 0, 0, 0],  # 20분 데이터가 없으면 패딩
                       [0, 0, 0, 0, 0, 0]]])  # 25분 데이터가 없으면 패딩

# 예측 (단일 입력 - 추가 특징 없음)
win_probability = model.predict(lstm_data, verbose=0)[0][0]
    
print(f"블루팀 승리 확률: {{win_probability:.1%}}")
""")

# =================================================================
# 7. 점진적 예측 테스트
# =================================================================

print("\n" + "="*60)
print("🧪 점진적 예측 테스트")
print("="*60)

# 스케일러 로딩 (추론 시 필요)
try:
    import pickle
    with open('processed_data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("🔍 점진적 예측 샘플 테스트:")
    
    # 샘플 게임 데이터 (정규화 전)
    sample_game = {
        10: [1000, 500, 15, 3, 1, 4],    # 10분: 골드차, XP차, CS차, 킬, 데스, 어시
        15: [2000, 800, 25, 5, 2, 8],    # 15분
        20: [3500, 1200, 45, 8, 3, 12],  # 20분
        25: [5000, 1600, 65, 12, 5, 18]  # 25분
    }
    
    for timepoint in [10, 15, 20, 25]:
        # 해당 시점까지의 데이터만 사용하여 추론
        test_sample = []
        
        for t in [10, 15, 20, 25]:
            if t <= timepoint:
                test_sample.append(sample_game[t])
            else:
                test_sample.append([0, 0, 0, 0, 0, 0])  # 패딩
        
        # 정규화 적용
        test_input = np.array([test_sample], dtype=np.float32)
        test_flat = test_input.reshape(-1, 6)
        non_zero_mask = np.any(test_flat != 0, axis=1)
        if np.any(non_zero_mask):
            test_flat[non_zero_mask] = scaler.transform(test_flat[non_zero_mask])
        test_scaled = test_flat.reshape(1, 4, 6)
        
        # 예측
        win_prob = best_model.predict(test_scaled, verbose=0)[0][0]
        print(f"   {timepoint}분 시점 예측 승률: {win_prob*100:.1f}%")
        
except Exception as e:
    print(f"   ⚠️ 점진적 예측 테스트 오류: {e}")

print("\n🚀 점진적 예측 모델이 준비되었습니다!")
print("🎯 이제 streamlit_ai_announcer.py의 predict_with_lstm 함수를 progressive_lol_lstm_model.h5로 교체하세요!")

print(f"""
💡 점진적 예측 모델 사용법:

1. 모델 로딩: tf.keras.models.load_model('progressive_lol_lstm_model.h5')
2. 스케일러 로딩: pickle.load(open('processed_data/scaler.pkl', 'rb'))
3. 시점별 데이터를 패딩하여 (4, 6) 형태로 만들기
4. 정규화 후 예측

✅ 학습/추론 분포 일치: 제로패딩을 포함한 학습으로 올바른 예측 가능!
""")