"""
LoL AI 아나운서 LLM 파인튜닝 데이터 생성기
실제 경기 데이터 + LSTM 예측 결과를 활용하여 고품질 훈련 데이터 생성

작성자: GitHub Copilot
목적: GPT 파인튜닝용 LoL 아나운서 데이터셋 자동 생성
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import json
import random
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🎯 LoL AI 아나운서 파인튜닝 데이터 생성기")
print("="*60)

# =================================================================
# 1. 기본 설정 및 데이터 로딩
# =================================================================

print("📂 데이터 및 모델 로딩 중...")
try:
    df = pd.read_csv('lol_processed_data_v2.csv')
    model = tf.keras.models.load_model('best_lol_lstm_model.h5')
    print(f"✅ 데이터: {df.shape}, 모델: {model.input_shape}")
except Exception as e:
    print(f"❌ 로딩 실패: {e}")
    exit()

# =================================================================
# 2. 아나운서 스타일 템플릿 정의
# =================================================================

# 다양한 아나운서 스타일과 상황별 멘트 템플릿
ANNOUNCER_STYLES = {
    "professional": {
        "name": "프로페셔널",
        "tone": "전문적이고 정확한",
        "templates": {
            "dominant": [
                "현재 {team}팀이 글로벌 골드 차이 {golddiff:+.0f}를 바탕으로 확실한 승기를 잡았습니다.",
                "{team}팀의 우세가 확실해 보이며, 경기의 템포를 완벽하게 장악하고 있습니다.",
                "타워와 오브젝트 스코어에서 앞서며, 격차를 벌리고 있는 {team}팀입니다.",
                "지속적인 압박을 통해 스노우볼을 확실하게 굴리고 있는 {team}팀, 이제 경기를 마무리할 단계입니다.",
                "킬 스코어 {killsdiff:+.0f}의 우위를 바탕으로 다음 운영의 이점까지 확보했습니다.",
                "{team}팀이 경험치 차이 {xpdiff:+.0f}를 통해 성장 우위를 굳건히 다지고 있습니다.",
                "운영의 정수를 보여주며, 상대 팀에게 만회할 기회를 전혀 주지 않고 있습니다.",
                "주요 딜러들의 CS 차이 {csdiff:+.0f}가 크게 벌어지며, 화력 차이가 발생하고 있습니다.",
                "{team}팀이 전방위적인 압박을 통해 확실하게 승리의 교두보를 마련했습니다.",
                "이 정도의 격차라면, 상대팀은 특단의 전략이 필요해 보입니다.",
                "후반 캐리 조합의 성장을 안정적으로 이끌어내며, 승리 공식에 근접하고 있습니다.",
                "상대 팀의 핵심 챔피언 성장을 완벽하게 봉쇄했습니다. {team}팀의 깔끔한 경기 운영입니다."
            ],
            "close": [
                "양 팀의 골드 차이 {golddiff:+.0f}가 극히 미미합니다. 팽팽한 균형이 유지되고 있네요.",
                "매우 치열한 신경전입니다. 양 팀 모두 한 번의 실수로 경기의 흐름이 바뀔 수 있습니다.",
                "킬 스코어 {killsdiff:+.0f}의 차이는 있으나, 오브젝트 획득에선 균형을 이루고 있습니다.",
                "{team}팀이 약간의 우위를 점하고 있으나, 아직 승부의 향방을 예측하긴 힘든 상황입니다.",
                "글로벌 경험치 차이 {xpdiff:+.0f}도 근소한 수준, 이 경기 한 치 앞을 알 수 없습니다.",
                "각 포지션별 성장세가 엇갈리며, 어느 팀도 확실하게 우위에 서지 못하고 있습니다.",
                "양 팀 모두 수비적인 자세를 취하며, 신중하게 상대의 빈틈을 노리고 있습니다.",
                "단 하나의 대형 오브젝트 싸움이 경기의 결과를 뒤바꿀 수 있는 시점입니다.",
                "어시스트 차이 {assistsdiff:+.0f}를 통해 초반 교전에서는 {team}팀이 이득을 봤습니다.",
                "두 팀 모두 실수 없이 운영을 이어가고 있어, 경기가 장기전으로 흘러갈 가능성이 높습니다.",
                "중앙 지역에서 시야 확보 경쟁이 치열합니다. 맵 장악이 승패를 가를 것입니다.",
                "퍼스트 블러드가 나왔음에도 골드 차이가 크게 벌어지지 않았습니다. 견고한 경기 양상입니다."
            ],
            "disadvantage": [
                "{team}팀이 골드 차이 {golddiff:+.0f}로 열세에 놓여있습니다. 수비에 집중해야 합니다.",
                "상대 팀의 공세를 막아내며, 역전의 기회를 엿보고 있는 {team}팀입니다.",
                "주요 오브젝트를 연달아 내주며, 운영상의 어려움을 겪고 있습니다.",
                "킬 스코어 {killsdiff:+.0f}의 격차가 발목을 잡고 있습니다. 전투를 피해야 합니다.",
                "경험치 차이 {xpdiff:+.0f}로 인해 성장 단계에서 밀리고 있어, 캐리 챔피언의 성장이 시급합니다.",
                "{team}팀은 지금부터 실수를 줄이고, 지연 플레이를 통해 시간을 벌어야 합니다.",
                "상대 팀의 확실한 운영에 말려들며, 주도권을 내어준 상황입니다.",
                "{team}팀의 데스 차이 {deathsdiff:+.0f}가 심화되며, 재정비가 필요한 시점입니다.",
                "이러한 불리한 상황을 타개할 한타(팀파이트) 기회를 찾아야 합니다.",
                "불리함을 인정하고, 상대의 템포를 끊어줄 변수 창출이 절실합니다.",
                "수세에 몰린 {team}팀, 다음 드래곤 싸움이 마지막 기회가 될 수도 있습니다.",
                "라인 스와프를 통해 어떻게든 시간을 벌어야 합니다. 성장이 절실한 시점입니다."
            ]
        }
    },
    "excited": {
        "name": "열정적",
        "tone": "흥미진진하고 감정적인",
        "templates": {
            "dominant": [
                "와!!! 팀 {team}!!! 이게 뭔가요! 우주를 뒤 흔들었습니다!! 승리 굳히기에 들어갑니다!",
                "아니아니! 지금 경기를 지배하고 있어요!!! 이게 말이 됩니까?! 넥서스를 깨러 갑니다!!",
                "우와와악! {team}팀이 완전히 경기를 장악했어요! 미쳐 날뛰고 있습니다!!",
                "이건 게임이 아닙니다! 이건 학살이에요! {team}팀이 펜타킬을 노립니다!",
                "상대팀은 이제 꼼짝 못해요! {team}팀의 완벽한 스노우볼링!! 저게 바로 클래스죠!",
                "전율이 느껴집니다! 이대로 넥서스까지 직진이에요! 멈출 수 없어요!",
                "지구가 멸망해도 {team}팀이 이깁니다! 완벽한 경기력에 박수를 보냅니다!",
                "파.괘.왕!! 이보다 더 완벽할 수 없습니다! 경기의 종결을 선언합니다!",
                "압도적!! 그냥 압도적이라는 말 밖엔 할 말이 없어요! 관중들이 열광합니다!",
                "캬~ 주모! 여기 국뽕 한 사발이요! {team}팀의 연이은 명장면!!",
                "상대팀은 지금 멘탈이 나갔습니다! {team}팀이 사정없이 짓밟고 있습니다!",
                "모든 것이 계획대로! 이니시에이팅부터 마무리까지 완벽 그 자체입니다!"
            ],
            "close": [
                "믿을 수 없는 접전이에요! 진짜 가슴이 쫄려 뒤질거같아요 이게 뭔가요! 손에 땀을 쥐게 합니다!",
                "심장이 멈출 것 같은 경기네요! 손톱을 다 먹어버릴거 같아요. 역전의 역전이 계속됩니다!",
                "이런 박진감 넘치는 경기! 너무 오랜만입니다! 두 팀 모두 대단해요!!",
                "쿵! 쾅! 쿵! 쾅! 심장 박동 소리가 들리십니까? 한타 한 번에 경기가 터집니다!",
                "정말 미쳤어요! 누가 이겨도 이상하지 않을 경기! 시청자들을 흥분의 도가니로 몰아넣습니다!",
                "숨 막히는 대치 상황! 아나운서인 저도 목소리가 떨립니다! 누가 먼저 움직일까요?!",
                "{team}팀이!! 기가 막힌 슈퍼 플레이를 보여줬습니다! 다시 원점으로 돌려놓는 싸움!",
                "이건 e스포츠의 명승부입니다! 양 팀이 서로 주거니 받거니! 끝날 때까지 끝난 게 아니죠!",
                "와! 1데스 1킬! 제로섬 게임! 치고받는 싸움이 멈추지 않아요! 숨 쉴 틈이 없습니다!",
                "이 좁은 협곡에서 팽팽한 기싸움! 숨을 쉴 수가 없습니다! 관전하는 저희도 지쳐요!",
                "이것이 바로 LCK의 묘미! 예측 불가능한 드라마가 펼쳐지고 있습니다!",
                "단 100골드 차이! 콜라 광고 찍어도 되겠어요! 손에 땀을 쥐게 하는 명승부입니다!"
            ],
            "disadvantage": [
                "위기의 {team}팀! 상대팀의 엄청난 실력에 당하고 있습니다! 이게 뭔가요! 정신 차려야 합니다!",
                "어려운 상황의 {team}팀! 기적을 만들어낼 수 있을까요?! 희망을 잃지 마세요!",
                "{team}팀! 역전 드라마가 필요한 시점입니다! 이대로 무너질 순 없습니다!",
                "안돼요! 이건 너무 아픈 손해인데요! 이대로는 안 됩니다, {team}팀! 억제기가 밀려요!",
                "지금 {team}팀은 벼랑 끝에 몰려있습니다! 제발 한 번만 이겨라! 한 번만!",
                "상대팀의 맹공에 정신을 못 차립니다! 아아... 눈물이 날 것 같아요! 믿을 수 없어요!",
                "이 불리함을 극복하기 위해서는 신의 한 수! 초특급 플레이가 필요합니다!",
                "{team}팀 선수들, 지금 표정이 어둡습니다! 하지만 포기하지 마세요! 우리가 응원합니다!",
                "멘탈 잡고! 멘탈 잡고! 다음 교전에서 회심의 일격을 가해야 합니다!",
                "이대로 넥서스가 터지면 안 되는데! {team}팀의 팬들은 기도하고 있습니다!",
                "제발! 기적의 한타를 보여주세요! 이대로 무기력하게 질 순 없습니다!",
                "오! 저 선수의 눈빛이 달라졌어요! 역전의 시그널일까요?!"
            ]
        }
    },
    "analytical": {
        "name": "분석적",
        "tone": "데이터 중심의 깊이 있는",
        "templates": {
            "dominant": [
                "데이터 분석 결과 {team}팀이 경기를 지배하고 있습니다. 골드 차이 {golddiff:+.0f}와 킬 차이 {killsdiff:+.0f}가 승리의 근본적인 요인입니다.",
                "통계적으로 {team}팀의 우위가 명확합니다. {golddiff:+.0f} 골드 우위와 {killsdiff:+.0f} 킬 어드밴티지가 격차를 벌리고 있습니다.",
                "{team}팀의 높은 우세는 현재 {golddiff:+.0f} 골드 차이와 {xpdiff:+.0f} 경험치 차이에서 비롯됩니다. 스노우볼의 궤도가 완벽합니다.",
                "분당 골드 획득량(GPM)에서 크게 앞서고 있습니다. 특히 CS 차이 {csdiff:+.0f}가 상대 딜러의 성장을 저해하고 있습니다.",
                "현재 어시스트 차이 {assistsdiff:+.0f}는 교전마다 {team}팀이 훨씬 효율적인 이득을 취하고 있음을 증명합니다.",
                "이 정도의 경험치 차이 {xpdiff:+.0f}는 레벨 우위로 이어져, 다음 주요 한타에서 압도적인 파워를 보일 것입니다.",
                "킬 차이 {killsdiff:+.0f}를 기반으로 {team}팀은 주요 오브젝트를 안정적으로 확보하는 운영 패턴을 구축했습니다.",
                "데스 차이 {deathsdiff:+.0f}가 상대팀에게 불리하게 작용하며, {golddiff:+.0f} 골드의 격차를 방어하기 어렵게 만들고 있습니다.",
                "주요 라인 스플릿 구도에서 {team}팀이 CS 차이 {csdiff:+.0f}를 통해 지속적으로 이득을 취하고 있습니다.",
                "종합적으로 볼 때, {team}팀의 운영 지표 전반이 상대 팀보다 우월함을 보이고 있습니다. 격차가 계속 벌어지고 있습니다.",
                "평균적으로 상대보다 두 레벨가량 앞서고 있습니다. 경험치 차이 {xpdiff:+.0f}는 지금부터 스탯으로 직결됩니다.",
                "{golddiff:+.0f} 골드 우위는 이미 아이템 차이로 전환되었습니다. 상대 팀은 싸움을 피해야 합니다."
            ],
            "close": [
                "흥미로운 데이터입니다. {team}팀이 근소하게 앞서지만, 골드 차이 {golddiff:+.0f}을 고려할 때 여전히 변수가 많은 상황이네요.",
                "균형잡힌 경기 양상입니다. {golddiff:+.0f} 골드 차이로 볼 때, 양 팀의 실력이 팽팽함을 알 수 있습니다.",
                "데이터상 {team}팀이 근소하게 우세하나, 골드 {golddiff:+.0f}, 킬 차이 {killsdiff:+.0f}를 보면 여전히 접전 상황입니다.",
                "경험치 차이 {xpdiff:+.0f}는 사실상 없는 수준입니다. 누가 먼저 상대방의 핵심 챔피언을 잘라내느냐가 관건입니다.",
                "{team}팀이 CS 차이 {csdiff:+.0f}로 미니언 이득을 보고 있지만, 킬 차이 {killsdiff:+.0f}에서 밀려 골드가 상쇄되고 있습니다.",
                "어시스트 차이 {assistsdiff:+.0f}와 데스 차이 {deathsdiff:+.0f}가 모두 미미한 수치를 보이며, 교전 승패가 엎치락뒤치락하고 있습니다.",
                "통계적으로 어느 팀도 확실한 승리 공식에 도달하지 못했습니다. {golddiff:+.0f} 골드 차이는 언제든 뒤집힐 수 있는 수치입니다.",
                "양 팀의 KDA 비율이 거의 같습니다. 한타 단계에서 포지셔닝 싸움이 중요해 보입니다.",
                "CS 차이 {csdiff:+.0f}는 라인전 우위를 보여주지만, 운영 단계에서는 킬 차이 {killsdiff:+.0f}가 부족합니다.",
                "현재까지의 모든 지표가 5대5의 경합을 나타내고 있습니다. 매우 분석하기 힘든 경기입니다.",
                "현재 이득과 손해 지표가 섞여있습니다. 골드 {golddiff:+.0f}는 앞서지만, 경험치 {xpdiff:+.0f}는 밀리고 있습니다.",
                "양 팀의 어시스트 차이 {assistsdiff:+.0f}가 0입니다. 소규모 교전 없이 대규모 한타만을 노리고 있습니다."
            ],
            "disadvantage": [
                "수치상으로 {team}팀이 어려운 상황입니다. {golddiff:+.0f} 골드 격차와 {killsdiff:+.0f} 킬 차이가 주요 변수로 작용하고 있네요.",
                "통계적으로 {team}팀의 승률이 하락했습니다. 골드 {golddiff:+.0f}, 킬 차이 {killsdiff:+.0f}이 불리하게 작용하고 있습니다.",
                "데이터 분석 결과 {team}팀이 낮은 승률을 보이며, 이는 {golddiff:+.0f} 골드 디스어드밴티지가 주된 원인입니다.",
                "가장 우려되는 부분은 경험치 차이 {xpdiff:+.0f}입니다. 이는 챔피언 레벨 차이로 이어져 능력치 열세를 가져옵니다.",
                "데스 차이 {deathsdiff:+.0f}가 크다는 것은 상대에게 너무 많은 현상금 골드를 내어주고 있다는 뜻입니다.",
                "킬 차이 {killsdiff:+.0f}로 인한 불리함이 누적되며, {team}팀은 이제 수세적인 입장에서 경기를 풀어야 합니다.",
                "CS 차이 {csdiff:+.0f}가 크게 벌어진 만큼, 핵심 딜러의 딜링 능력이 현저하게 떨어질 것입니다.",
                "어시스트 차이 {assistsdiff:+.0f}가 부족하다는 것은 팀 단위 교전에서 유기적인 움직임이 부족했음을 의미합니다.",
                "현재의 {golddiff:+.0f}를 만회하기 위해서는 최소한 두 번 이상의 대승이 필요합니다. 쉽지 않은 길입니다.",
                "모든 경제 지표(골드, CS)가 상대 팀에게 기울었습니다. {team}팀은 리스크가 큰 도박성 플레이를 고려해야 합니다.",
                "상대 팀의 데스 수가 적습니다. 데스 차이 {deathsdiff:+.0f}는 곧 {team}팀의 기회가 적었다는 반증입니다.",
                "팀원 전체의 평균 골드 차이가 불리합니다. 특정 라이너를 희생시켜서라도 캐리 라인을 살려야 합니다."
            ]
        }
    }
}

# 시간대별 특수 멘트
TIME_SPECIFIC_COMMENTS = {
    "10분": [
        "라이닝 페이즈가 마무리되는 시점입니다.",
        "초반 게임 양상이 결정되고 있네요.",
        "첫 번째 리콜 타이밍이 중요한 시점입니다."
    ],
    "15분": [
        "첫 번째 드래곤과 전령 싸움이 중요한 시점입니다.",
        "중반 전환기에 접어들고 있습니다.",
        "타워 철거 경쟁이 치열한 상황이네요."
    ],
    "20분": [
        "본격적인 팀파이트가 시작되는 시점입니다.",
        "바론과 드래곤 장악권이 중요한 시기입니다.",
        "후반 캐리력이 시험받는 구간이네요."
    ],
    "25분": [
        "게임의 승부가 결정되는 중요한 시점입니다.",
        "한 번의 팀파이트가 경기를 뒤바꿀 수 있는 시기입니다.",
        "바론과 엘더 드래곤의 중요성이 극대화되는 구간이네요."
    ]
}

# =================================================================
# 3. 예측 및 통계 추출 함수들
# =================================================================

def predict_at_time(model, df, game_idx, max_time):
    """특정 시점까지의 데이터로 예측"""
    game_row = df.iloc[game_idx]
    times = ['at10', 'at15', 'at20', 'at25']
    features = ['golddiff', 'xpdiff', 'csdiff', 'killsdiff', 'assistsdiff', 'deathsdiff']
    
    game_data = []
    for t in range(4):
        time_data = []
        for feature in features:
            col_name = f"{feature}{times[t]}"
            if col_name in df.columns and t < max_time:
                value = game_row[col_name]
                time_data.append(float(value) if not pd.isna(value) else 0.0)
            else:
                time_data.append(0.0)
        game_data.append(time_data)
    
    input_data = np.array([game_data])
    try:
        prediction = model.predict(input_data, verbose=0)
        return float(prediction[0][0])
    except:
        return 0.5

def get_game_stats(df, game_idx, time_idx):
    """게임 통계 추출"""
    game_row = df.iloc[game_idx]
    times = ['at10', 'at15', 'at20', 'at25']
    time_suffix = times[time_idx]
    
    stats = {}
    for feature in ['golddiff', 'xpdiff', 'csdiff', 'killsdiff', 'assistsdiff', 'deathsdiff']:
        col_name = f"{feature}{time_suffix}"
        if col_name in df.columns:
            value = game_row[col_name]
            stats[feature] = float(value) if not pd.isna(value) else 0.0
        else:
            stats[feature] = 0.0
    return stats

def categorize_game_state(win_prob):
    """게임 상황 분류"""
    if win_prob >= 0.75:
        return "dominant"
    elif win_prob >= 0.25:
        return "close"
    else:
        return "disadvantage"

# =================================================================
# 4. 파인튜닝 데이터 생성 함수
# =================================================================

def generate_commentary_data(game_idx, time_idx, style_name, include_context=True):
    """단일 경기 시점에 대한 아나운서 멘트 데이터 생성"""
    
    # 예측 및 통계 수집
    win_prob = predict_at_time(model, df, game_idx, time_idx + 1)
    stats = get_game_stats(df, game_idx, time_idx)
    time_names = ["10분", "15분", "20분", "25분"]
    time_name = time_names[time_idx]
    
    # 팀 결정 (확률에 따라)
    if win_prob > 0.5:
        team = "블루"
        prob = win_prob
    else:
        team = "레드"
        prob = 1 - win_prob
    
    # 게임 상황 분류
    game_state = categorize_game_state(prob)
    
    # 스타일별 템플릿 선택
    style = ANNOUNCER_STYLES[style_name]
    template = random.choice(style["templates"][game_state])
    
    # 기본 정보로 템플릿 채우기
    base_comment = template.format(
        team=team,
        prob=prob,
        golddiff=stats.get('golddiff', 0),
        xpdiff=stats.get('xpdiff', 0),
        csdiff=stats.get('csdiff', 0),
        killsdiff=stats.get('killsdiff', 0),
        assistsdiff=stats.get('assistsdiff', 0),
        deathsdiff=stats.get('deathsdiff', 0),
        # 이전 호환성을 위한 추가
        gold=stats.get('golddiff', 0),
        kills=stats.get('killsdiff', 0)
    )
    
    # 시간 특화 멘트 추가 (랜덤하게)
    if random.random() < 0.3:  # 30% 확률로 시간 특화 멘트 추가
        time_comment = random.choice(TIME_SPECIFIC_COMMENTS[time_name])
        base_comment = f"{base_comment} {time_comment}"
    
    # 입력 데이터 구성
    if include_context:
        user_input = f"""게임 상황: {time_name} 시점
팀: 블루팀 vs 레드팀
골드 차이: {stats.get('golddiff', 0):+.0f}
경험치 차이: {stats.get('xpdiff', 0):+.0f}
CS 차이: {stats.get('csdiff', 0):+.0f}
킬 차이: {stats.get('killsdiff', 0):+.0f}
어시스트 차이: {stats.get('assistsdiff', 0):+.0f}
데스 차이: {stats.get('deathsdiff', 0):+.0f}
AI 예측: {team}팀 {prob:.1%} 승률

위 상황에 대해 {style['name']} 스타일로 해설해주세요."""
    else:
        user_input = f"{time_name} 시점, {team}팀 {prob:.1%} 승률, 골드차이 {stats.get('golddiff', 0):+.0f}"
    
    return {
        "messages": [
            {
                "role": "system",
                "content": f"당신은 리그 오브 레전드(LoL) 전문 아나운서입니다. {style['tone']} 톤으로 경기를 해설하며, 통계 데이터를 바탕으로 정확하고 흥미로운 실황중계를 제공합니다."
            },
            {
                "role": "user",
                "content": user_input
            },
            {
                "role": "assistant",
                "content": base_comment
            }
        ],
        "metadata": {
            "game_id": game_idx,
            "time": time_name,
            "style": style_name,
            "win_probability": prob,
            "team": team,
            "stats": stats
        }
    }

def generate_batch_training_data(num_samples=1000, output_file="lol_announcer_finetune_data.jsonl"):
    """대량 훈련 데이터 생성"""
    
    print(f"📊 {num_samples}개의 훈련 데이터 생성 중...")
    
    training_data = []
    styles = list(ANNOUNCER_STYLES.keys())
    
    # 게임 인덱스 풀 준비
    available_games = list(range(len(df)))
    
    for i in range(num_samples):
        try:
            # 랜덤 선택
            game_idx = random.choice(available_games)
            time_idx = random.choice([0, 1, 2, 3])  # 10, 15, 20, 25분
            style = random.choice(styles)
            include_context = random.choice([True, False])  # 상세 컨텍스트 포함 여부
            
            # 데이터 생성
            data_point = generate_commentary_data(game_idx, time_idx, style, include_context)
            training_data.append(data_point)
            
            if (i + 1) % 100 == 0:
                print(f"   진행률: {i+1}/{num_samples} ({(i+1)/num_samples*100:.1f}%)")
                
        except Exception as e:
            print(f"   ⚠️ 데이터 생성 실패 (#{i}): {e}")
            continue
    
    # JSONL 파일로 저장
    print(f"💾 {output_file} 파일로 저장 중...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for data_point in training_data:
            f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
    
    # 통계 요약
    print(f"\n📈 데이터 생성 완료!")
    print(f"   총 데이터 수: {len(training_data):,}개")
    
    # 스타일별 분포
    style_counts = {}
    time_counts = {}
    for data in training_data:
        style = data['metadata']['style']
        time = data['metadata']['time']
        style_counts[style] = style_counts.get(style, 0) + 1
        time_counts[time] = time_counts.get(time, 0) + 1
    
    print(f"   스타일별 분포:")
    for style, count in style_counts.items():
        print(f"     • {ANNOUNCER_STYLES[style]['name']}: {count:,}개 ({count/len(training_data)*100:.1f}%)")
    
    print(f"   시간대별 분포:")
    for time, count in time_counts.items():
        print(f"     • {time}: {count:,}개 ({count/len(training_data)*100:.1f}%)")
    
    return training_data

# =================================================================
# 5. 샘플 데이터 미리보기 및 품질 검증
# =================================================================

def preview_samples(num_preview=5):
    """생성된 데이터 샘플 미리보기"""
    
    print(f"\n🔍 생성 데이터 샘플 미리보기 ({num_preview}개)")
    print("="*80)
    
    styles = list(ANNOUNCER_STYLES.keys())
    
    for i in range(num_preview):
        game_idx = random.choice(range(min(100, len(df))))  # 처음 100게임에서만
        time_idx = random.choice([0, 1, 2, 3])
        style = random.choice(styles)
        
        sample = generate_commentary_data(game_idx, time_idx, style, include_context=True)
        
        print(f"\n📝 샘플 #{i+1} (게임 #{game_idx}, {sample['metadata']['time']}, {sample['metadata']['style']} 스타일)")
        print("-" * 60)
        print(f"🎯 System: {sample['messages'][0]['content']}")
        print(f"👤 User: {sample['messages'][1]['content']}")
        print(f"🎤 Assistant: {sample['messages'][2]['content']}")
        print(f"📊 Metadata: {sample['metadata']['team']}팀 {sample['metadata']['win_probability']:.1%}")

# =================================================================
# 6. 실행 및 데이터 생성
# =================================================================

if __name__ == "__main__":
    # 샘플 미리보기
    preview_samples(3)
    
    # 사용자 입력
    print(f"\n" + "="*80)
    response = input("훈련 데이터를 생성하시겠습니까? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', '네', 'ㅇ']:
        num_samples = input("생성할 데이터 수를 입력하세요 (기본값: 1000): ").strip()
        try:
            num_samples = int(num_samples) if num_samples else 1000
        except:
            num_samples = 1000
        
        print(f"\n🚀 {num_samples:,}개의 훈련 데이터 생성을 시작합니다!")
        
        # 데이터 생성
        training_data = generate_batch_training_data(num_samples)
        
        print(f"""
🎯 LoL AI 아나운서 파인튜닝 데이터 생성 완료!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 생성 완료:
   • 파일: lol_announcer_finetune_data.jsonl
   • 데이터 수: {len(training_data):,}개
   • 형식: OpenAI 파인튜닝 호환 JSONL

🎤 포함된 아나운서 스타일:
   • 프로페셔널: 전문적이고 정확한 해설
   • 열정적: 흥미진진하고 감정적인 해설  
   • 분석적: 데이터 중심의 깊이 있는 해설

⏰ 시간대별 데이터:
   • 10분: 라이닝 페이즈 분석
   • 15분: 중반 전환기 해설
   • 20분: 팀파이트 상황 분석
   • 25분: 후반 결정적 순간 해설

🚀 다음 단계:
   1. OpenAI API 또는 Hugging Face로 파인튜닝
   2. 생성된 모델을 실시간 아나운서 시스템에 통합
   3. 더 많은 데이터로 성능 개선

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    else:
        print("❌ 데이터 생성을 취소했습니다.")

print("🎉 LoL AI 아나운서 파인튜닝 데이터 생성기 완료!")