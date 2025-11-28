"""
Progressive LSTM Data Preprocessing
점진적 LSTM 데이터 전처리 - 학습/추론 일관성 보장

작성 목적: 
- 기존 LSTM 모델의 학습/추론 분포 불일치 문제 해결
- 1개 경기 → 4개 학습 샘플로 점진적 데이터 증강
- 각 시점별 예측에 맞는 올바른 학습 데이터 생성
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

def create_progressive_lstm_dataset(csv_path='lol_processed_data_v2.csv', output_dir='processed_data'):
    """
    점진적 LSTM 학습 데이터셋 생성 - Oracle's Elixir 데이터 기반
    
    기존 문제:
    - 학습: 완전한 4시점 데이터 [10분, 15분, 20분, 25분]
    - 추론: 불완전한 데이터 [10분, 0, 0, 0] (제로 패딩)
    
    해결 방법:
    - 1개 경기 → 4개 학습 샘플 생성
    - 10분 예측용: [10분, 0, 0, 0]
    - 15분 예측용: [10분, 15분, 0, 0]  
    - 20분 예측용: [10분, 15분, 20분, 0]
    - 25분 예측용: [10분, 15분, 20분, 25분]
    """
    
    print("🚀 Progressive LSTM 데이터 전처리 시작 (Oracle's Elixir 기반)...")
    
    # 1. 원본 데이터 로딩
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 원본 데이터 로딩 완료: {len(df):,}경기")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        print("💡 먼저 lol_data_preprocessing_v2.py를 실행하여 lol_processed_data_v2.csv를 생성하세요.")
        return None
    
    # 2. Oracle's Elixir 데이터에 맞는 특성 컬럼 정의
    feature_columns = [
        'golddiffat10', 'xpdiffat10', 'csdiffat10', 'killsat10', 'deathsat10', 'assistsat10',
        'golddiffat15', 'xpdiffat15', 'csdiffat15', 'killsat15', 'deathsat15', 'assistsat15', 
        'golddiffat20', 'xpdiffat20', 'csdiffat20', 'killsat20', 'deathsat20', 'assistsat20',
        'golddiffat25', 'xpdiffat25', 'csdiffat25', 'killsat25', 'deathsat25', 'assistsat25'
    ]
    
    print("🔍 사용 가능한 컬럼 확인:")
    available_columns = [col for col in feature_columns if col in df.columns]
    missing_columns = [col for col in feature_columns if col not in df.columns]
    
    for col in available_columns:
        print(f"   ✅ {col}")
    for col in missing_columns:
        print(f"   ❌ {col} (없음)")
    
    # Oracle's Elixir 데이터에서는 kills, deaths, assists가 killsat*, deathsat*, assistsat* 형태일 수 있음
    if missing_columns:
        print("⚠️ 일부 컬럼이 없습니다. 대체 컬럼명을 확인합니다...")
        alt_feature_columns = []
        
        for time in [10, 15, 20, 25]:
            # 기본 차이 값들
            alt_feature_columns.extend([
                f'golddiffat{time}', f'xpdiffat{time}', f'csdiffat{time}'
            ])
            
            # KDA는 차이값으로 변경 (killsdiffat* 등이 있는지 확인)
            for stat in ['kills', 'deaths', 'assists']:
                diff_col = f'{stat}diffat{time}'
                original_col = f'{stat}at{time}'
                
                if diff_col in df.columns:
                    alt_feature_columns.append(diff_col)
                elif original_col in df.columns:
                    alt_feature_columns.append(original_col)
        
        feature_columns = [col for col in alt_feature_columns if col in df.columns]
        print(f"✅ 최종 사용 컬럼: {len(feature_columns)}개")
    
    # 3. 결측치가 있는 행 제거
    df_clean = df.dropna(subset=feature_columns + ['result'])
    print(f"✅ 결측치 제거 후: {len(df_clean):,}경기 (제거된 경기: {len(df) - len(df_clean):,}경기)")
    
    if len(df_clean) == 0:
        print("❌ 유효한 데이터가 없습니다. 컬럼명을 확인해주세요.")
        print("사용 가능한 컬럼:", df.columns.tolist()[:20])  # 처음 20개만 표시
        return None
    
    # 4. 점진적 샘플 생성
    progressive_samples = []
    targets = []
    sample_info = []  # 디버깅용 정보
    
    print("🔄 점진적 샘플 생성 중...")
    
    # 시점별로 6개 특성 추출 (Oracle's Elixir 구조에 맞게)
    feature_names = ['golddiff', 'xpdiff', 'csdiff', 'kills', 'deaths', 'assists']
    
    for idx, row in df_clean.iterrows():
        if idx % 1000 == 0:
            print(f"   진행률: {idx:,}/{len(df_clean):,} ({idx/len(df_clean)*100:.1f}%)")
        
        # 시점별 특성 추출 - 컬럼명 동적 확인
        features_by_time = {}
        
        for time in [10, 15, 20, 25]:
            time_features = []
            
            for feature in feature_names:
                # 여러 가능한 컬럼명 시도
                possible_cols = [
                    f'{feature}at{time}',        # golddiffat10
                    f'{feature}diffat{time}',    # killsdiffat10 (차이값)
                ]
                
                col_found = None
                for col in possible_cols:
                    if col in df_clean.columns:
                        col_found = col
                        break
                
                if col_found:
                    time_features.append(float(row[col_found]))
                else:
                    # 기본값 0 사용
                    time_features.append(0.0)
                    if time == 10:  # 첫 번째 시점에서만 경고
                        print(f"   ⚠️ {feature}at{time} 컬럼을 찾을 수 없어 0으로 설정합니다.")
            
            features_by_time[time] = time_features
        
        result = float(row['result'])
        
        # 1) 10분 예측용 샘플: [10분 데이터, 패딩, 패딩, 패딩]
        sample_10 = [
            features_by_time[10],
            [0.0] * 6,  # 패딩
            [0.0] * 6,  # 패딩  
            [0.0] * 6   # 패딩
        ]
        progressive_samples.append(sample_10)
        targets.append(result)
        sample_info.append(f"Game_{idx}_10min")
        
        # 2) 15분 예측용 샘플: [10분 데이터, 15분 데이터, 패딩, 패딩]
        sample_15 = [
            features_by_time[10],
            features_by_time[15],
            [0.0] * 6,  # 패딩
            [0.0] * 6   # 패딩
        ]
        progressive_samples.append(sample_15)
        targets.append(result)
        sample_info.append(f"Game_{idx}_15min")
        
        # 3) 20분 예측용 샘플: [10분 데이터, 15분 데이터, 20분 데이터, 패딩]
        sample_20 = [
            features_by_time[10],
            features_by_time[15],
            features_by_time[20],
            [0.0] * 6   # 패딩
        ]
        progressive_samples.append(sample_20)
        targets.append(result)
        sample_info.append(f"Game_{idx}_20min")
        
        # 4) 25분 예측용 샘플: [10분 데이터, 15분 데이터, 20분 데이터, 25분 데이터]
        sample_25 = [
            features_by_time[10],
            features_by_time[15],
            features_by_time[20],
            features_by_time[25]
        ]
        progressive_samples.append(sample_25)
        targets.append(result)
        sample_info.append(f"Game_{idx}_25min")
    
    print(f"✅ 점진적 샘플 생성 완료: {len(progressive_samples):,}샘플 (원본 {len(df_clean):,}경기 × 4)")
    
    # 5. NumPy 배열로 변환
    X = np.array(progressive_samples, dtype=np.float32)  # (samples, 4, 6)
    y = np.array(targets, dtype=np.float32)
    
    print(f"📊 최종 데이터 형태:")
    print(f"   - X shape: {X.shape} (samples, timepoints, features)")
    print(f"   - y shape: {y.shape}")
    print(f"   - 메모리 사용량: X={X.nbytes/1024/1024:.1f}MB, y={y.nbytes/1024:.1f}KB")
    
    # 6. 데이터 분포 확인
    print(f"📈 타겟 분포:")
    print(f"   - 승리(1): {np.sum(y == 1):,}샘플 ({np.mean(y)*100:.1f}%)")
    print(f"   - 패배(0): {np.sum(y == 0):,}샘플 ({(1-np.mean(y))*100:.1f}%)")
    
    # 7. 학습/검증 분할 (게임 단위로 분할하여 데이터 누출 방지)
    # 게임 인덱스 생성 (0~len(df_clean)-1, 각각이 4번 반복됨)
    game_indices = np.repeat(range(len(df_clean)), 4)
    
    # 게임 단위로 분할
    unique_game_indices = np.unique(game_indices)
    train_games, val_games = train_test_split(
        unique_game_indices, 
        test_size=0.2, 
        random_state=42,
        stratify=df_clean['result'].values  # 결과 비율 유지
    )
    
    # 인덱스를 샘플 단위로 확장
    train_mask = np.isin(game_indices, train_games)
    val_mask = np.isin(game_indices, val_games)
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    print(f"🔀 학습/검증 분할 완료:")
    print(f"   - 학습: {len(X_train):,}샘플 ({len(train_games):,}게임)")
    print(f"   - 검증: {len(X_val):,}샘플 ({len(val_games):,}게임)")
    print(f"   - 학습 승률: {np.mean(y_train)*100:.1f}%")
    print(f"   - 검증 승률: {np.mean(y_val)*100:.1f}%")
    
    # 8. 특성 정규화 (학습 데이터 기준)
    print("🔄 특성 정규화 중...")
    
    # 각 시점, 각 특성별로 정규화
    scalers = {}
    
    # 형태 변환: (samples, timepoints, features) → (samples*timepoints, features)
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    
    # 패딩이 아닌 실제 데이터만 스케일링 (0이 아닌 행만)
    non_zero_mask = np.any(X_train_flat != 0, axis=1)
    
    scaler = StandardScaler()
    scaler.fit(X_train_flat[non_zero_mask])
    
    # 전체 데이터에 적용 (패딩은 그대로 0 유지)
    X_train_scaled_flat = X_train_flat.copy()
    X_val_scaled_flat = X_val_flat.copy()
    
    X_train_scaled_flat[non_zero_mask] = scaler.transform(X_train_flat[non_zero_mask])
    
    non_zero_mask_val = np.any(X_val_flat != 0, axis=1)
    X_val_scaled_flat[non_zero_mask_val] = scaler.transform(X_val_flat[non_zero_mask_val])
    
    # 원래 형태로 복원
    X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
    
    print("✅ 특성 정규화 완료")
    
    # 9. 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 10. 데이터 저장
    print("💾 데이터 저장 중...")
    
    # NPZ 파일로 저장 (압축)
    np.savez_compressed(
        os.path.join(output_dir, 'progressive_lstm_data.npz'),
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        y_train=y_train,
        y_val=y_val,
        feature_names=['golddiff', 'xpdiff', 'csdiff', 'kills', 'deaths', 'assists']
    )
    
    # 스케일러 저장
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 메타데이터 저장
    metadata = {
        'original_games': len(df_clean),
        'total_samples': len(progressive_samples),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'feature_columns': feature_columns,
        'data_shape': {
            'timepoints': 4,
            'features_per_timepoint': 6,
            'input_shape': (4, 6)
        },
        'feature_order': ['golddiff', 'xpdiff', 'csdiff', 'kills', 'deaths', 'assists'],
        'preprocessing_info': {
            'method': 'progressive_augmentation',
            'padding_value': 0.0,
            'normalization': 'StandardScaler',
            'train_test_split': 0.8
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ 저장 완료:")
    print(f"   - 학습 데이터: {output_dir}/progressive_lstm_data.npz")
    print(f"   - 스케일러: {output_dir}/scaler.pkl")
    print(f"   - 메타데이터: {output_dir}/metadata.pkl")
    
    # 11. 샘플 데이터 확인 (디버깅용)
    print("\n🔍 샘플 데이터 확인:")
    print("첫 번째 10분 예측 샘플 (정규화 전):")
    print(X[0])  # 10분 예측용 샘플
    print("첫 번째 25분 예측 샘플 (정규화 전):")  
    print(X[3])  # 25분 예측용 샘플 (완전한 데이터)
    
    print("\n첫 번째 10분 예측 샘플 (정규화 후):")
    print(X_train_scaled[0])
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled, 
        'y_train': y_train,
        'y_val': y_val,
        'scaler': scaler,
        'metadata': metadata
    }

def load_progressive_data(data_dir='processed_data'):
    """
    저장된 점진적 LSTM 데이터 로딩
    """
    try:
        # 데이터 로딩
        data = np.load(os.path.join(data_dir, 'progressive_lstm_data.npz'))
        X_train = data['X_train']
        X_val = data['X_val']
        y_train = data['y_train'] 
        y_val = data['y_val']
        
        # 스케일러 로딩
        with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # 메타데이터 로딩
        with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✅ 점진적 데이터 로딩 완료:")
        print(f"   - 학습: {X_train.shape}")
        print(f"   - 검증: {X_val.shape}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'scaler': scaler,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        return None

def create_inference_sample(timepoint_data, scaler, target_timepoint=25):
    """
    추론용 샘플 생성 (학습 데이터와 동일한 형태)
    
    Args:
        timepoint_data: dict - {10: [features], 15: [features], ...}
        scaler: 학습 시 사용된 스케일러
        target_timepoint: 예측하고자 하는 시점 (10, 15, 20, 25)
    
    Returns:
        np.array: (1, 4, 6) 형태의 추론용 입력
    """
    
    # 타겟 시점에 따른 패딩 패턴 설정
    if target_timepoint == 10:
        # 10분 예측: [10분 데이터, 패딩, 패딩, 패딩]
        sample = [
            timepoint_data.get(10, [0]*6),
            [0]*6, [0]*6, [0]*6
        ]
    elif target_timepoint == 15:
        # 15분 예측: [10분 데이터, 15분 데이터, 패딩, 패딩]
        sample = [
            timepoint_data.get(10, [0]*6),
            timepoint_data.get(15, [0]*6),
            [0]*6, [0]*6
        ]
    elif target_timepoint == 20:
        # 20분 예측: [10분 데이터, 15분 데이터, 20분 데이터, 패딩]
        sample = [
            timepoint_data.get(10, [0]*6),
            timepoint_data.get(15, [0]*6), 
            timepoint_data.get(20, [0]*6),
            [0]*6
        ]
    elif target_timepoint == 25:
        # 25분 예측: [10분 데이터, 15분 데이터, 20분 데이터, 25분 데이터]
        sample = [
            timepoint_data.get(10, [0]*6),
            timepoint_data.get(15, [0]*6),
            timepoint_data.get(20, [0]*6),
            timepoint_data.get(25, [0]*6)
        ]
    else:
        raise ValueError(f"지원하지 않는 시점: {target_timepoint}")
    
    # NumPy 배열로 변환
    X = np.array([sample], dtype=np.float32)  # (1, 4, 6)
    
    # 정규화 적용 (패딩이 아닌 데이터만)
    X_flat = X.reshape(-1, 6)
    non_zero_mask = np.any(X_flat != 0, axis=1)
    
    if np.any(non_zero_mask):
        X_flat[non_zero_mask] = scaler.transform(X_flat[non_zero_mask])
    
    X_scaled = X_flat.reshape(1, 4, 6)
    
    return X_scaled

# 메인 실행부
if __name__ == "__main__":
    # 점진적 데이터 전처리 실행
    result = create_progressive_lstm_dataset()
    
    if result is not None:
        print("\n🎉 점진적 LSTM 데이터 전처리 완료!")
        print("\n다음 단계:")
        print("1. progressive_lstm_training.py로 모델 학습")
        print("2. 학습된 모델을 streamlit_ai_announcer.py에 적용")
        print("3. predict_with_lstm 함수 수정하여 올바른 추론 방식 사용")
        
        # 간단한 테스트
        print("\n🧪 추론 샘플 생성 테스트:")
        test_data = {
            10: [1000, 500, 3, 1, 4, 15],
            15: [2000, 1000, 5, 2, 8, 25],
            20: [3000, 1500, 8, 3, 12, 35],
            25: [4000, 2000, 12, 5, 18, 45]
        }
        
        for timepoint in [10, 15, 20, 25]:
            inference_sample = create_inference_sample(test_data, result['scaler'], timepoint)
            print(f"{timepoint}분 예측용 샘플 shape: {inference_sample.shape}")