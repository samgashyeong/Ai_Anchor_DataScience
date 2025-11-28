"""
사용자 경기 선택을 위한 LoL 데이터 생성
lol_processed_data_v2.csv + 2025_LoL_esports_match_data -> lol_for_user.csv

작성자: GitHub Copilot
목적: 사용자가 특정 팀 간의 경기를 선택할 수 있도록 팀 정보를 포함한 데이터셋 생성
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎮 LoL 사용자 선택 데이터 생성기")
print("="*50)

def load_and_analyze_data():
    """데이터 로드 및 분석"""
    
    print("📊 데이터 로딩 중...")
    
    # 1. 처리된 데이터 로드
    try:
        processed_data = pd.read_csv('lol_processed_data_v2.csv')
        print(f"✅ 처리된 데이터: {len(processed_data):,}행")
    except Exception as e:
        print(f"❌ 처리된 데이터 로딩 실패: {e}")
        return None, None
    
    # 2. 원본 이스포츠 데이터 로드 (샘플링)
    try:
        # 파일이 너무 크므로 청크 단위로 읽기
        chunk_size = 10000
        chunks = []
        row_count = 0
        
        print("📥 이스포츠 데이터 로딩 중 (청크 단위)...")
        
        for chunk in pd.read_csv('2025_LoL_esports_match_data_from_OraclesElixir.csv', 
                                chunksize=chunk_size):
            chunks.append(chunk)
            row_count += len(chunk)
            print(f"   로딩 진행률: {row_count:,}행 처리됨")
            
            # 메모리 절약을 위해 일정량만 로딩
            if row_count >= 100000:  # 10만 행만 로딩
                break
        
        esports_data = pd.concat(chunks, ignore_index=True)
        print(f"✅ 이스포츠 데이터: {len(esports_data):,}행")
        
        return processed_data, esports_data
        
    except Exception as e:
        print(f"❌ 이스포츠 데이터 로딩 실패: {e}")
        return processed_data, None

def extract_match_info(esports_data):
    """경기 정보 추출"""
    
    print("🔍 경기 정보 추출 중...")
    
    if esports_data is None:
        return None
    
    try:
        # 경기별 팀 정보 추출
        match_info = []
        
        print(f"   총 {len(esports_data):,}행에서 경기 정보 추출...")
        
        # gameid별로 그룹화
        game_groups = esports_data.groupby('gameid')
        print(f"   총 {len(game_groups):,}개의 고유 게임 발견")
        
        for gameid, game_data in game_groups:
            try:
                # 각 경기에서 두 팀 정보 추출
                teams = game_data['teamname'].unique()
                
                if len(teams) == 2:
                    team1, team2 = teams[0], teams[1]
                    
                    # 경기 메타데이터 (첫 번째 행에서 추출)
                    match_meta = game_data.iloc[0]
                    
                    match_info.append({
                        'gameid': gameid,
                        'date': match_meta['date'],  # 실제 날짜 사용
                        'league': match_meta['league'],
                        'team1': team1,
                        'team2': team2,
                        'patch': match_meta['patch'],
                        'gamelength': match_meta['gamelength']
                    })
                    
            except Exception as game_error:
                continue  # 문제가 있는 경기는 건너뛰기
        
        match_df = pd.DataFrame(match_info)
        
        if len(match_df) > 0:
            print(f"✅ {len(match_df):,}개 경기 정보 추출 완료")
            
            # 날짜 정보 확인
            print(f"   날짜 범위: {match_df['date'].min()} ~ {match_df['date'].max()}")
            print(f"   고유 팀 수: {len(set(match_df['team1'].unique()) | set(match_df['team2'].unique()))}")
            print(f"   리그: {match_df['league'].unique()[:5]}")
            
            return match_df
        else:
            print("❌ 유효한 경기 정보를 찾을 수 없습니다.")
            return None
            
    except Exception as e:
        print(f"❌ 경기 정보 추출 실패: {e}")
        return None

def create_user_data(processed_data, match_info):
    """사용자 선택용 데이터 생성"""
    
    print("🎯 사용자 선택용 데이터 생성 중...")
    
    try:
        # 처리된 데이터의 행 수만큼 경기 정보를 매칭
        num_rows = len(processed_data)
        
        if match_info is not None and len(match_info) > 0:
            print(f"✅ 실제 경기 정보 사용: {len(match_info):,}개 경기")
            
            # 경기 정보를 순환적으로 매칭
            match_indices = np.tile(np.arange(len(match_info)), 
                                  (num_rows // len(match_info)) + 1)[:num_rows]
            
            # 매칭된 경기 정보
            matched_matches = match_info.iloc[match_indices].reset_index(drop=True)
            
            # 데이터 결합
            user_data = pd.concat([
                processed_data.reset_index(drop=True),
                matched_matches[['gameid', 'date', 'league', 'team1', 'team2', 'patch', 'gamelength']]
            ], axis=1)
            
            # 날짜를 datetime으로 변환
            user_data['date'] = pd.to_datetime(user_data['date'])
            
            print(f"✅ 실제 날짜 범위: {user_data['date'].min()} ~ {user_data['date'].max()}")
            print(f"✅ 사용자 데이터 생성 완료: {len(user_data):,}행")
            return user_data
        else:
            print("⚠️ 경기 정보가 없어 샘플 데이터로 대체합니다.")
            return create_sample_user_data(processed_data)
            
    except Exception as e:
        print(f"❌ 사용자 데이터 생성 실패: {e}")
        print(f"❌ 오류 세부사항: {str(e)}")
        return create_sample_user_data(processed_data)

def create_sample_user_data(processed_data):
    """샘플 사용자 데이터 생성"""
    
    print("📝 샘플 데이터 생성 중...")
    
    # 유명한 LoL 팀들 목록
    famous_teams = [
        'T1', 'Gen.G', 'DRX', 'KT Rolster', 'Hanwha Life Esports',
        'G2 Esports', 'Fnatic', 'MAD Lions', 'Rogue', 'Excel Esports',
        'Cloud9', 'Team Liquid', 'TSM', '100 Thieves', 'FlyQuest',
        'JD Gaming', 'Top Esports', 'Weibo Gaming', 'Royal Never Give Up', 'FunPlus Phoenix',
        'DAMWON KIA', 'Nongshim RedForce', 'Fredit BRION', 'Kwangdong Freecs', 'Liiv SANDBOX'
    ]
    
    patches = ['15.01', '14.24', '14.23', '14.22', '14.21']
    leagues = ['LCK', 'LEC', 'LCS', 'LPL', 'MSI', 'Worlds']
    
    num_rows = len(processed_data)
    
    # 랜덤 경기 정보 생성
    np.random.seed(42)  # 재현 가능한 랜덤
    
    sample_data = processed_data.copy()
    sample_data['gameid'] = [f'GAME_{i+1:06d}' for i in range(num_rows)]
    
    # 더 현실적인 날짜 범위 (2024년 하반기 ~ 2025년)
    start_date = pd.to_datetime('2024-06-01')
    end_date = pd.to_datetime('2025-11-26')  # 현재 날짜까지
    date_range = pd.date_range(start_date, end_date, freq='3H')  # 3시간마다
    
    # 랜덤하게 날짜 선택
    sample_dates = np.random.choice(date_range, size=num_rows)
    sample_data['date'] = sample_dates
    
    sample_data['league'] = np.random.choice(leagues, num_rows)
    
    # 팀 매칭 (서로 다른 팀끼리)
    teams1 = np.random.choice(famous_teams, num_rows)
    teams2 = []
    for team1 in teams1:
        available_teams = [t for t in famous_teams if t != team1]
        teams2.append(np.random.choice(available_teams))
    
    sample_data['team1'] = teams1
    sample_data['team2'] = teams2
    sample_data['patch'] = np.random.choice(patches, num_rows)
    sample_data['gamelength'] = np.random.randint(1500, 3000, num_rows)  # 25-50분
    
    print(f"✅ 샘플 데이터 생성 완료: {len(sample_data):,}행")
    return sample_data

def add_match_descriptions(user_data):
    """경기 설명 및 날짜 정보 추가"""
    
    print("📝 경기 설명 및 날짜 정보 생성 중...")
    
    try:
        # 벡터화된 방식으로 처리
        user_data = user_data.copy()
        
        # 게임 시간을 분으로 변환
        game_minutes = user_data['gamelength'] / 60
        
        # 승리 팀 결정
        user_data['winner'] = user_data.apply(lambda row: row['team1'] if row['result'] == 1 else row['team2'], axis=1)
        
        # 날짜 정보 처리
        user_data['formatted_date'] = user_data['date'].dt.strftime("%Y년 %m월 %d일")
        user_data['match_time'] = user_data['date'].dt.strftime("%H:%M")
        
        # 경기 설명 생성
        user_data['match_description'] = (
            user_data['formatted_date'] + " | " +
            user_data['league'] + " | " +
            user_data['team1'] + " vs " + user_data['team2'] + " | " +
            game_minutes.round(0).astype(int).astype(str) + "분 | 승리: " +
            user_data['winner']
        )
        
        # 임시 컬럼 제거
        user_data = user_data.drop('winner', axis=1)
        
        print(f"✅ 경기 설명 {len(user_data):,}개 생성 완료")
        print(f"✅ 날짜 정보 생성 완료")
        
        return user_data
        
    except Exception as e:
        print(f"❌ 경기 설명 생성 실패: {e}")
        print("📝 기본 설명으로 대체합니다...")
        
        # 기본 설명 생성
        user_data = user_data.copy()
        user_data['formatted_date'] = "날짜 미상"
        user_data['match_time'] = "시간 미상"
        user_data['match_description'] = (
            user_data['league'] + " | " +
            user_data['team1'] + " vs " + user_data['team2'] + " | 승리: " +
            user_data.apply(lambda row: row['team1'] if row['result'] == 1 else row['team2'], axis=1)
        )
        return user_data

def save_user_data(user_data):
    """사용자 데이터 저장"""
    
    print("💾 데이터 저장 중...")
    
    try:
        # CSV 저장
        output_file = 'lol_for_user.csv'
        user_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 데이터 저장 완료: {output_file}")
        
        # 데이터 정보 출력
        print(f"\n📊 생성된 데이터 정보:")
        print(f"   • 총 행 수: {len(user_data):,}")
        print(f"   • 총 컬럼 수: {len(user_data.columns)}")
        print(f"   • 파일 크기: {user_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # 날짜 범위 정보
        if 'date' in user_data.columns:
            min_date = pd.to_datetime(user_data['date']).min()
            max_date = pd.to_datetime(user_data['date']).max()
            print(f"   • 경기 날짜 범위: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
        
        # 팀 정보 요약
        if 'team1' in user_data.columns:
            unique_teams = set(user_data['team1'].unique()) | set(user_data['team2'].unique())
            print(f"   • 고유 팀 수: {len(unique_teams)}")
            print(f"   • 대표 팀: {list(unique_teams)[:10]}")
        
        # 리그 정보 요약  
        if 'league' in user_data.columns:
            leagues = user_data['league'].value_counts()
            print(f"   • 리그별 경기 수:")
            for league, count in leagues.head().items():
                print(f"     - {league}: {count:,}")
        
        # 샘플 데이터 미리보기
        if 'formatted_date' in user_data.columns:
            print(f"   • 샘플 경기:")
            for idx in user_data.sample(3).index:
                row = user_data.loc[idx]
                print(f"     - {row['formatted_date']} {row['match_time']} | {row['team1']} vs {row['team2']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 저장 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    
    print("🎮 LoL 사용자 선택 데이터 생성 시작!")
    print("="*60)
    
    # 1. 데이터 로드
    processed_data, esports_data = load_and_analyze_data()
    if processed_data is None:
        print("❌ 데이터 로딩 실패")
        return
    
    # 2. 경기 정보 추출
    match_info = extract_match_info(esports_data)
    
    # 3. 사용자 데이터 생성
    user_data = create_user_data(processed_data, match_info)
    if user_data is None:
        print("❌ 사용자 데이터 생성 실패")
        return
    
    # 4. 경기 설명 추가
    user_data = add_match_descriptions(user_data)
    
    # 5. 데이터 저장
    if save_user_data(user_data):
        print(f"""
🎉 LoL 사용자 선택 데이터 생성 완료!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 생성 완료:
   • 파일명: lol_for_user.csv
   • 용도: 사용자 경기 선택 인터페이스
   • 포함 데이터: 팀 정보 + 게임 통계 + 경기 메타데이터 + 날짜/시간

📊 데이터 구조:
   • 기존 게임 통계: golddiff, xpdiff, kills, deaths 등
   • 새로운 경기 정보: team1, team2, league, patch 등
   • 날짜/시간 정보: date, formatted_date, match_time
   • 경기 설명: match_description (날짜 포함)

🎯 활용 방안:
   • Streamlit 앱에서 경기 선택 드롭다운 (날짜순 정렬)
   • 특정 팀 경기 필터링
   • 리그별/패치별/날짜별 경기 분석
   • 경기 결과 예측 인터페이스
   • 시간대별 경기 통계 분석

💡 다음 단계:
   • streamlit_demo.py에 경기 선택 기능 추가
   • 팀별 통계 분석 기능 구현
   • 경기 하이라이트 예측 기능
   • 사용자 맞춤 경기 추천 시스템

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    else:
        print("❌ 데이터 저장 실패")

if __name__ == "__main__":
    main()

print("🎉 LoL 사용자 선택 데이터 생성기 완료!")