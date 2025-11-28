"""
OpenAI API를 사용한 LoL AI 아나운서 GPT 파인튜닝
생성된 JSONL 데이터를 활용하여 맞춤형 아나운서 모델 생성
목적: LoL 아나운서 전용 GPT 모델 파인튜닝 및 배포
"""

import openai
import json
import time
import os
from datetime import datetime

print("🤖 OpenAI GPT 파인튜닝 시스템")
print("="*50)

# =================================================================
# 1. OpenAI API 설정
# =================================================================

def setup_openai_client():
    """OpenAI 클라이언트 설정"""
    
    print("🔑 OpenAI API 설정 중...")
    
    # API 키 설정 (환경변수에서 읽기)
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        api_key = input("OpenAI API 키를 입력하세요: ").strip()
        
        if not api_key:
            print("❌ API 키가 필요합니다.")
            return None
    
    # 클라이언트 초기화
    client = openai.OpenAI(api_key=api_key)
    
    try:
        # API 연결 테스트
        models = client.models.list()
        print("✅ OpenAI API 연결 성공!")
        return client
    except Exception as e:
        print(f"❌ API 연결 실패: {e}")
        return None

# =================================================================
# 2. 훈련 데이터 검증 및 업로드
# =================================================================

def validate_training_data(file_path="lol_announcer_clean_data.jsonl"):
    """훈련 데이터 형식 검증"""
    
    print(f"📋 훈련 데이터 검증 중: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return False
    
    try:
        valid_count = 0
        total_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_count += 1
                try:
                    data = json.loads(line.strip())
                    
                    # 필수 필드 검증
                    if 'messages' in data:
                        messages = data['messages']
                        if len(messages) >= 2:
                            # system, user, assistant 메시지 확인
                            if any(msg.get('role') == 'system' for msg in messages):
                                if any(msg.get('role') == 'user' for msg in messages):
                                    if any(msg.get('role') == 'assistant' for msg in messages):
                                        valid_count += 1
                
                except json.JSONDecodeError:
                    print(f"⚠️ JSON 파싱 오류 (라인 {line_num})")
                
                if total_count % 100 == 0:
                    print(f"   검증 진행률: {total_count} 라인 처리됨")
        
        print(f"📊 검증 결과:")
        print(f"   • 총 데이터: {total_count:,}개")
        print(f"   • 유효 데이터: {valid_count:,}개")
        print(f"   • 유효율: {valid_count/total_count*100:.1f}%")
        
        return valid_count > 0
        
    except Exception as e:
        print(f"❌ 데이터 검증 실패: {e}")
        return False

def upload_training_file(client, file_path="lol_announcer_clean_data.jsonl"):
    """훈련 파일을 OpenAI에 업로드"""
    
    print(f"📤 훈련 파일 업로드 중: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        file_id = response.id
        print(f"✅ 파일 업로드 성공!")
        print(f"   • 파일 ID: {file_id}")
        print(f"   • 파일명: {response.filename}")
        print(f"   • 크기: {response.bytes:,} bytes")
        
        return file_id
        
    except Exception as e:
        print(f"❌ 파일 업로드 실패: {e}")
        return None

# =================================================================
# 3. 파인튜닝 작업 실행
# =================================================================

def start_fine_tuning(client, file_id, model_name="gpt-3.5-turbo"):
    """파인튜닝 작업 시작"""
    
    print(f"🚀 파인튜닝 작업 시작")
    print(f"   • 베이스 모델: {model_name}")
    print(f"   • 훈련 파일 ID: {file_id}")
    
    try:
        response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model_name,
            hyperparameters={
                "n_epochs": 3,  # 에포크 수
                "batch_size": 1,  # 배치 크기  
                "learning_rate_multiplier": 0.1  # 학습률
            }
        )
        
        job_id = response.id
        print(f"✅ 파인튜닝 작업 생성 성공!")
        print(f"   • 작업 ID: {job_id}")
        print(f"   • 상태: {response.status}")
        print(f"   • 생성 시간: {response.created_at}")
        
        return job_id
        
    except Exception as e:
        print(f"❌ 파인튜닝 작업 생성 실패: {e}")
        return None

def monitor_fine_tuning(client, job_id):
    """파인튜닝 진행 상황 모니터링"""
    
    print(f"👀 파인튜닝 진행 상황 모니터링: {job_id}")
    print("="*50)
    
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            
            print(f"\n📊 현재 상태: {status}")
            print(f"⏰ 업데이트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                print(f"📈 훈련된 토큰: {job.trained_tokens:,}")
            
            if status == "succeeded":
                print("🎉 파인튜닝 완료!")
                print(f"✅ 완료된 모델 ID: {job.fine_tuned_model}")
                return job.fine_tuned_model
                
            elif status == "failed":
                print("❌ 파인튜닝 실패!")
                if hasattr(job, 'error'):
                    print(f"오류 메시지: {job.error}")
                return None
                
            elif status in ["pending", "running"]:
                print("⏳ 파인튜닝 진행 중... 60초 후 다시 확인합니다.")
                time.sleep(60)
                
            else:
                print(f"⚠️ 알 수 없는 상태: {status}")
                time.sleep(30)
                
        except Exception as e:
            print(f"❌ 상태 확인 실패: {e}")
            time.sleep(30)

# =================================================================
# 4. 파인튜닝된 모델 테스트
# =================================================================

def test_fine_tuned_model(client, model_id):
    """파인튜닝된 모델 테스트"""
    
    print(f"🧪 파인튜닝된 모델 테스트: {model_id}")
    print("="*50)
    
    # 테스트 케이스들
    test_cases = [
        {
            "situation": "15분 시점, 블루팀 85% 승률, 골드차이 +2500",
            "style": "프로페셔널"
        },
        {
            "situation": "20분 시점, 레드팀 60% 승률, 골드차이 -1200", 
            "style": "열정적"
        },
        {
            "situation": "25분 시점, 블루팀 30% 승률, 골드차이 -3000",
            "style": "분석적"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 테스트 #{i}: {test_case['style']} 스타일")
        print(f"상황: {test_case['situation']}")
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": f"당신은 리그 오브 레전드(LoL) 전문 아나운서입니다. {test_case['style']} 스타일로 경기를 해설하며, 통계 데이터를 바탕으로 정확하고 흥미로운 실황중계를 제공합니다."
                    },
                    {
                        "role": "user", 
                        "content": test_case['situation']
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            result = response.choices[0].message.content
            print(f"🎤 AI 아나운서: {result}")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")

# =================================================================
# 5. 메인 파인튜닝 프로세스
# =================================================================

def main_fine_tuning_process():
    """전체 파인튜닝 프로세스 실행"""
    
    print("🎯 LoL AI 아나운서 GPT 파인튜닝 시작!")
    print("="*60)
    
    # 1. OpenAI 클라이언트 설정
    client = setup_openai_client()
    if not client:
        return
    
    # 2. 훈련 데이터 검증
    if not validate_training_data():
        return
    
    # 3. 사용자 확인
    print(f"\n" + "="*50)
    print("📋 파인튜닝 진행 안내:")
    print("• 파인튜닝에는 시간이 오래 걸릴 수 있습니다 (수십분~몇시간)")
    print("• 파인튜닝 비용이 발생합니다 (데이터량에 따라)")
    print("• 중간에 중단할 경우 비용은 이미 청구될 수 있습니다")
    
    response = input("\n파인튜닝을 진행하시겠습니까? (y/n): ").strip().lower()
    
    if response not in ['y', 'yes', '네', 'ㅇ']:
        print("❌ 파인튜닝을 취소했습니다.")
        return
    
    # 4. 파일 업로드
    file_id = upload_training_file(client)
    if not file_id:
        return
    
    # 5. 파인튜닝 작업 시작
    job_id = start_fine_tuning(client, file_id)
    if not job_id:
        return
    
    # 6. 진행 상황 모니터링
    model_id = monitor_fine_tuning(client, job_id)
    if not model_id:
        return
    
    # 7. 모델 테스트
    test_fine_tuned_model(client, model_id)
    
    # 8. 완료 안내
    print(f"""
🎉 LoL AI 아나운서 파인튜닝 완료!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 파인튜닝 성공:
   • 모델 ID: {model_id}
   • 상태: 사용 준비 완료
   • 기능: LoL 전용 아나운서 해설

🎤 모델 특징:
   • 프로페셔널 해설 스타일
   • 열정적 해설 스타일  
   • 분석적 해설 스타일
   • 시간대별 맞춤 해설
   • 실제 게임 데이터 기반

🚀 활용 방법:
   1. OpenAI API로 model="{model_id}" 사용
   2. 실시간 아나운서 시스템에 통합
   3. 웹/앱 서비스 배포

💡 다음 단계:
   • 실시간 스트리밍 연동
   • TTS(음성합성) 추가
   • 사용자 피드백 수집 및 개선

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# =================================================================
# 6. 실행
# =================================================================

if __name__ == "__main__":
    main_fine_tuning_process()

print("🎉 OpenAI GPT 파인튜닝 시스템 완료!")