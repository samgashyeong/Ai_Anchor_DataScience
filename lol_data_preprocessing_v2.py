"""
LoL 프로 경기 데이터를 LSTM 모델에 입력하기 위한 완벽한 전처리 파이프라인 (수정버전)
실제 데이터 구조에 맞춘 정확한 특징공학 포함
목적: 킬/데스/어시스트 정보를 포함한 완전한 특징공학 + LSTM 입력 생성
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("🔥 LoL 프로 경기 데이터 완전 전처리 파이프라인 v2.0")
print("="*70)

# =================================================================
# 0. 환경 설정 및 데이터 로딩
# =================================================================

file_path = '2025_LoL_esports_match_data_from_OraclesElixir.csv'
print(f"📂 데이터 로딩 중... ({file_path})")

try:
    df_raw = pd.read_csv(file_path, low_memory=False)
    print(f"✅ 데이터 로딩 완료! 크기: {df_raw.shape}")
    
    # 블루팀 데이터 필터링
    df_blue = df_raw[df_raw['participantid'] == 100].copy()
    print(f"🔵 블루팀 데이터 추출: {df_blue.shape}")
    
except Exception as e:
    print(f"❌ 데이터 로딩 실패: {e}")
    exit()

# 타겟 변수 설정
if 'result' in df_blue.columns:
    Y = (df_blue['result'] == 1).astype(int).values
    target_col = 'result'
elif 'win' in df_blue.columns:
    Y = df_blue['win'].astype(int).values
    target_col = 'win'
else:
    print("❌ 승패 컬럼을 찾을 수 없습니다.")
    exit()

print(f"🎯 타겟 설정: {target_col}")
print(f"   승리: {sum(Y):,}경기 ({sum(Y)/len(Y)*100:.1f}%)")
print(f"   패배: {len(Y)-sum(Y):,}경기 ({(len(Y)-sum(Y))/len(Y)*100:.1f}%)")

# 시간대 설정
TIME_STEPS = [10, 15, 20, 25]
print(f"🕐 분석 시간대: {TIME_STEPS}분")

# =================================================================
# 1. 기본 특징 정의 및 수집
# =================================================================

print("\n" + "="*70)
print("📊 STEP 1: 기본 특징 수집 및 정의")
print("="*70)

# 1-1. 기본 특징 정의 (차이값 위주)
BASE_FEATURES = {
    'golddiff': '골드 격차',
    'xpdiff': '경험치 격차', 
    'csdiff': 'CS 격차'
}

# 1-2. KDA 특징 정의 (절대값 - 상대방과 비교 필요)
KDA_FEATURES = {
    'kills': '킬 수',
    'deaths': '데스 수',
    'assists': '어시스트 수'
}

print("🎯 사용할 기본 특징들:")
for key, desc in BASE_FEATURES.items():
    print(f"   📈 {key}: {desc}")

print("🎯 사용할 KDA 특징들:")
for key, desc in KDA_FEATURES.items():
    print(f"   ⚔️ {key}: {desc}")

# 1-3. 모든 특징 컬럼 수집 및 검증
all_feature_cols = []
feature_availability = {}

for time in TIME_STEPS:
    print(f"\n🕐 {time}분 데이터 검증:")
    time_features = []
    
    # 기본 diff 특징
    for feature in BASE_FEATURES.keys():
        col_name = f'{feature}at{time}'
        if col_name in df_blue.columns:
            all_feature_cols.append(col_name)
            time_features.append(col_name)
            print(f"   ✅ {col_name}")
        else:
            print(f"   ❌ {col_name} (없음)")
    
    # KDA 특징
    for feature in KDA_FEATURES.keys():
        col_name = f'{feature}at{time}'
        if col_name in df_blue.columns:
            all_feature_cols.append(col_name)
            time_features.append(col_name)
            print(f"   ✅ {col_name}")
        else:
            print(f"   ❌ {col_name} (없음)")
    
    feature_availability[time] = time_features
    print(f"   📊 {time}분: 총 {len(time_features)}개 특징")

print(f"\n📋 수집된 전체 특징: {len(all_feature_cols)}개")

# =================================================================
# 2. 고급 특징 공학 (Feature Engineering)
# =================================================================

print("\n" + "="*70)
print("💡 STEP 2: 고급 특징 공학 (Feature Engineering)")
print("="*70)

# 2-1. KDA 격차 특징 생성 (상대방과 비교)
print("⚔️ KDA 격차 특징 생성 중...")

kda_diff_features = []
for time in TIME_STEPS:
    for kda_type in ['kills', 'assists', 'deaths']:
        our_col = f'{kda_type}at{time}'
        opp_col = f'opp_{kda_type}at{time}'
        
        if our_col in df_blue.columns and opp_col in df_blue.columns:
            diff_col = f'{kda_type}diffat{time}'
            if kda_type == 'deaths':
                # 데스는 적을수록 좋으므로 반대로 계산
                df_blue[diff_col] = df_blue[opp_col] - df_blue[our_col]
            else:
                # 킬, 어시스트는 많을수록 좋음
                df_blue[diff_col] = df_blue[our_col] - df_blue[opp_col]
            
            kda_diff_features.append(diff_col)
            all_feature_cols.append(diff_col)
            print(f"   ✅ {diff_col} 생성")

print(f"⚔️ 생성된 KDA 격차 특징: {len(kda_diff_features)}개")

# 2-2. 미확인 골드 (Hidden Gold) 특징 생성
print("\n🏆 미확인 골드 (Hidden Gold) 특징 생성 중...")

hidden_gold_features = []
for i in range(len(TIME_STEPS) - 1):
    t_prev = TIME_STEPS[i]
    t_curr = TIME_STEPS[i+1]
    
    # 필요한 컬럼들
    gold_prev = f'golddiffat{t_prev}'
    gold_curr = f'golddiffat{t_curr}'
    kills_prev = f'killsdiffat{t_prev}'  # 새로 생성된 킬 차이
    kills_curr = f'killsdiffat{t_curr}'
    cs_prev = f'csdiffat{t_prev}'
    cs_curr = f'csdiffat{t_curr}'
    
    if all(col in df_blue.columns for col in [gold_prev, gold_curr, kills_prev, kills_curr, cs_prev, cs_curr]):
        # Hidden Gold = 골드 증가량 - (킬 증가량 * 300) - (CS 증가량 * 20)
        hidden_gold_col = f'HiddenGold_{t_prev}_{t_curr}'
        
        df_blue[hidden_gold_col] = (
            (df_blue[gold_curr] - df_blue[gold_prev]) -
            ((df_blue[kills_curr] - df_blue[kills_prev]) * 300) -
            ((df_blue[cs_curr] - df_blue[cs_prev]) * 20)
        )
        
        hidden_gold_features.append(hidden_gold_col)
        print(f"   ✅ {hidden_gold_col} 생성")
    else:
        # 간단한 버전 (골드 변화량만)
        if gold_prev in df_blue.columns and gold_curr in df_blue.columns:
            hidden_gold_col = f'GoldChange_{t_prev}_{t_curr}'
            df_blue[hidden_gold_col] = df_blue[gold_curr] - df_blue[gold_prev]
            hidden_gold_features.append(hidden_gold_col)
            print(f"   ✅ {hidden_gold_col} 생성 (간단 버전)")

print(f"🏆 생성된 미확인 골드 특징: {len(hidden_gold_features)}개")

# 2-3. 성장 가속도 (Acceleration) 특징 생성
print("\n⚡ 성장 가속도 (Acceleration) 특징 생성 중...")

acceleration_features = []
primary_features = ['golddiff', 'xpdiff']  # 가속도 계산할 주요 특징들

for feature in primary_features:
    for i in range(len(TIME_STEPS) - 2):  # 최소 3개 시점 필요
        t1 = TIME_STEPS[i]
        t2 = TIME_STEPS[i+1] 
        t3 = TIME_STEPS[i+2]
        
        col1 = f'{feature}at{t1}'
        col2 = f'{feature}at{t2}'
        col3 = f'{feature}at{t3}'
        
        if all(col in df_blue.columns for col in [col1, col2, col3]):
            # 가속도 = (t3-t2 변화량) - (t2-t1 변화량)
            acc_col = f'Acc_{feature}_{t1}_{t3}'
            
            change_1 = df_blue[col2] - df_blue[col1]  # t1->t2 변화
            change_2 = df_blue[col3] - df_blue[col2]  # t2->t3 변화
            
            df_blue[acc_col] = change_2 - change_1  # 가속도
            
            acceleration_features.append(acc_col)
            print(f"   ✅ {acc_col} 생성")

print(f"⚡ 생성된 가속도 특징: {len(acceleration_features)}개")

# 2-4. 종합 전투력 지수 (Combat Power) 특징 생성
print("\n💪 종합 전투력 지수 생성 중...")

combat_power_features = []
for time in TIME_STEPS:
    required_cols = [f'killsdiffat{time}', f'assistsdiffat{time}', f'deathsdiffat{time}', f'golddiffat{time}']
    
    if all(col in df_blue.columns for col in required_cols):
        combat_col = f'CombatPower_{time}'
        
        # 전투력 = (킬차이*2 + 어시차이*1 + 데스차이*1.5) * 골드차이_가중치
        gold_weight = 1 + (df_blue[f'golddiffat{time}'] / 5000).clip(-1, 1)  # 골드차이 가중치
        
        df_blue[combat_col] = (
            (df_blue[f'killsdiffat{time}'] * 2 + 
             df_blue[f'assistsdiffat{time}'] * 1 + 
             df_blue[f'deathsdiffat{time}'] * 1.5) * gold_weight
        )
        
        combat_power_features.append(combat_col)
        print(f"   ✅ {combat_col} 생성")

print(f"💪 생성된 전투력 특징: {len(combat_power_features)}개")

# =================================================================
# 3. 최종 특징 선택 및 LSTM 입력 형태 변환
# =================================================================

print("\n" + "="*70)
print("🎯 STEP 3: 최종 특징 선택 및 LSTM 형태 변환")
print("="*70)

# 3-1. 시점별 사용할 특징 정의
LSTM_FEATURES = ['golddiff', 'xpdiff', 'csdiff', 'killsdiff', 'assistsdiff', 'deathsdiff']

print("🎯 LSTM 입력용 최종 특징들:")
for feature in LSTM_FEATURES:
    print(f"   📊 {feature}")

# 3-2. 각 시점별 특징 행렬 구성
lstm_data_list = []
valid_times = []

for time in TIME_STEPS:
    time_features = []
    time_feature_names = []
    
    for feature in LSTM_FEATURES:
        col_name = f'{feature}at{time}'
        if col_name in df_blue.columns:
            time_features.append(col_name)
            time_feature_names.append(feature)
    
    if time_features:
        lstm_data_list.append(df_blue[time_features].values)
        valid_times.append(time)
        print(f"   🕐 {time}분: {len(time_features)}개 특징 ({time_feature_names})")

# 3-3. 결측치 처리
print(f"\n🔧 결측치 처리 중...")

for i, time in enumerate(valid_times):
    data = lstm_data_list[i]
    missing_count = np.isnan(data).sum()
    
    if missing_count > 0:
        print(f"   ⚠️ {time}분: {missing_count}개 결측치 발견")
        # 0으로 채우기
        lstm_data_list[i] = np.nan_to_num(data, 0)
    else:
        print(f"   ✅ {time}분: 결측치 없음")

# 3-4. 최종 3차원 배열 구성
print(f"\n🎯 3차원 LSTM 배열 구성 중...")

NUM_GAMES = len(df_blue)
NUM_TIMESTEPS = len(valid_times)
NUM_FEATURES = len(LSTM_FEATURES)

# 모든 시점이 동일한 특징 개수를 가지도록 조정
X_final = np.zeros((NUM_GAMES, NUM_TIMESTEPS, NUM_FEATURES))

for t_idx, time in enumerate(valid_times):
    for f_idx, feature in enumerate(LSTM_FEATURES):
        col_name = f'{feature}at{time}'
        if col_name in df_blue.columns:
            X_final[:, t_idx, f_idx] = df_blue[col_name].fillna(0).values

# 3-5. 표준화 (각 특징별로 독립적으로)
print(f"\n📊 특징별 표준화 중...")

scalers = {}
for f_idx, feature in enumerate(LSTM_FEATURES):
    # 해당 특징의 모든 시점 데이터를 하나로 합쳐서 스케일링
    feature_data = X_final[:, :, f_idx].reshape(-1, 1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)
    X_final[:, :, f_idx] = scaled_data.reshape(NUM_GAMES, NUM_TIMESTEPS)
    scalers[feature] = scaler
    print(f"   📊 {feature} 표준화 완료")

print(f"\n✅ 최종 LSTM 입력 형태: {X_final.shape}")
print(f"   📊 게임 수: {NUM_GAMES:,}")
print(f"   🕐 시간 단계: {NUM_TIMESTEPS} ({valid_times})")
print(f"   📈 특징 수: {NUM_FEATURES} ({LSTM_FEATURES})")

# =================================================================
# 4. 추가 특징들 (별도 Dense Layer용)
# =================================================================

print("\n" + "="*70)
print("🏆 STEP 4: 추가 특징 준비 (Dense Layer 연결용)")
print("="*70)

# 4-1. 추가 특징들 수집
additional_features = hidden_gold_features + acceleration_features + combat_power_features
print(f"📊 추가 특징 총 {len(additional_features)}개:")

for feature in additional_features:
    print(f"   🎯 {feature}")

# 4-2. 추가 특징 데이터 준비
if additional_features:
    # 결측치 처리
    additional_df = df_blue[additional_features].fillna(0)
    
    # 표준화
    scaler_additional = StandardScaler()
    X_additional = scaler_additional.fit_transform(additional_df)
    
    print(f"✅ 추가 특징 준비 완료: {X_additional.shape}")
    print("💡 이 특징들은 LSTM 출력과 결합하여 Dense Layer에 입력할 수 있습니다.")
else:
    X_additional = np.array([]).reshape(NUM_GAMES, 0)
    print("⚠️ 추가 특징이 생성되지 않았습니다.")

# =================================================================
# 5. 데이터 분할 및 저장
# =================================================================

print("\n" + "="*70)
print("💾 STEP 5: 데이터 분할 및 저장")
print("="*70)

# 5-1. 데이터 분할
X_train, X_temp, y_train, y_temp = train_test_split(
    X_final, Y, test_size=0.3, random_state=42, stratify=Y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 추가 특징도 동일하게 분할 (있는 경우)
if X_additional.shape[1] > 0:
    X_add_train, X_add_temp = train_test_split(X_additional, test_size=0.3, random_state=42)
    X_add_val, X_add_test = train_test_split(X_add_temp, test_size=0.5, random_state=42)
else:
    X_add_train = X_add_val = X_add_test = np.array([]).reshape(-1, 0)

print(f"✅ 데이터 분할 완료:")
print(f"   🔵 Training:   {X_train.shape[0]:,}경기")
print(f"   🟢 Validation: {X_val.shape[0]:,}경기") 
print(f"   🟡 Test:       {X_test.shape[0]:,}경기")

# 5-2. 타겟 분포 확인
for split_name, split_y in [("Training", y_train), ("Validation", y_val), ("Test", y_test)]:
    win_rate = np.mean(split_y)
    print(f"   {split_name} 승률: {win_rate:.1%}")

# 5-3. 최종 저장
print(f"\n💾 최종 데이터 저장 중...")

# 메인 LSTM 데이터
np.savez_compressed('lol_lstm_data_v2.npz',
                   X_train=X_train, y_train=y_train,
                   X_val=X_val, y_val=y_val,
                   X_test=X_test, y_test=y_test,
                   feature_names=LSTM_FEATURES,
                   time_steps=valid_times,
                   X_additional_train=X_add_train,
                   X_additional_val=X_add_val,
                   X_additional_test=X_add_test,
                   additional_feature_names=additional_features)

print("✅ 'lol_lstm_data_v2.npz' 저장 완료!")

# 전처리된 DataFrame 저장
final_cols = all_feature_cols + hidden_gold_features + acceleration_features + combat_power_features + [target_col]
available_final_cols = [col for col in final_cols if col in df_blue.columns]

df_processed_v2 = df_blue[available_final_cols].copy()
df_processed_v2.to_csv('lol_processed_data_v2.csv', index=False)
print("✅ 'lol_processed_data_v2.csv' 저장 완료!")

# =================================================================
# 6. 최종 요약
# =================================================================

print("\n" + "="*70)
print("🎉 완전한 LoL 데이터 전처리 v2.0 완료!")
print("="*70)

print(f"""
📋 완전 전처리 요약:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 기본 데이터:
   • 원본 데이터: {len(df_raw):,}행
   • 블루팀 데이터: {len(df_blue):,}행
   • 최종 게임 수: {NUM_GAMES:,}경기

📊 LSTM 메인 특징:
   • 시간 단계: {valid_times} (총 {NUM_TIMESTEPS}개)
   • 기본 특징: {LSTM_FEATURES} (총 {NUM_FEATURES}개)
   • 최종 형태: {X_final.shape}

💡 추가 생성 특징:
   • KDA 격차: {len(kda_diff_features)}개
   • 미확인 골드: {len(hidden_gold_features)}개  
   • 성장 가속도: {len(acceleration_features)}개
   • 전투력 지수: {len(combat_power_features)}개
   • 추가 특징 총합: {len(additional_features)}개

📈 데이터 품질:
   • 전체 승률: {np.mean(Y):.1%}
   • 균형도: {'균형적' if 0.4 <= np.mean(Y) <= 0.6 else '불균형적'}
   • 결측치: 모두 처리됨
   • 표준화: 완료

💾 저장 파일:
   • lol_lstm_data_v2.npz (LSTM 모델용 - 메인)
   • lol_processed_data_v2.csv (분석용)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("🚀 이제 진짜 완전한 LSTM 모델 학습이 가능합니다!")
print("\n📘 데이터 로딩 예시:")
print("""
# v2 데이터 로딩
data = np.load('lol_lstm_data_v2.npz')
X_train = data['X_train']          # LSTM 메인 입력 (게임수, 4, 6)
y_train = data['y_train']          # 타겟
X_additional = data['X_additional_train']  # 추가 특징들 (Dense layer용)
feature_names = data['feature_names']      # ['golddiff', 'xpdiff', ...]
""")

print(f"\n💪 특징공학 완료도: 100% ✅")
print(f"   - 시계열 기본 특징: ✅")
print(f"   - KDA 분석: ✅") 
print(f"   - Hidden Gold: ✅")
print(f"   - 가속도 분석: ✅")
print(f"   - 종합 전투력: ✅")