from fastapi import FastAPI, HTTPException, Request
import sounddevice as sd
import torchaudio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

# 환경 변수 로드 (여기서는 환경 변수를 설정해야 하는 경우에만 사용)
load_dotenv()

# FastAPI 애플리케이션 생성
app = FastAPI()

# Whisper 모델 로드
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model_id = "openai/whisper-medium"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# ChatGPT 모델 설정 (LangChain 사용)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

# 텍스트 변환을 위한 PromptTemplate 설정
prompt_template = """
당신은 친구같은 AI로, 유저의 운동 목표와 상태에 맞춰 맞춤형 운동 계획을 제시하는 전문가야.
운동 계획은 각 운동 목표, 신체 상태, 운동 경험 수준을 고려해서 자세히 안내해줘. 운동을 진행할 때 필요한 운동 종목, 세트 수, 반복 횟수, 휴식 시간, 운동의 강도 등도 함께 알려줘.
운동 계획을 세울 때 다음 정보를 참고해야 해:

# 제공된 정보:
1. 운동수준: {운동수준}
2. 운동할 장소: {운동장소}
3. 일주일 운동 가능 횟수: {운동횟수}
4. 한 번 운동할 때 평균 시간: {운동시간}
5. 운동의 목표: {운동목표}
6. 초점을 두고 싶은 신체 부위: {초점부위}
7. 성별: {성별}
8. 나이: {나이}
9. 키: {키}
10. 몸무게: {몸무게}
11. 부상 여부: {부상여부}

# 운동 목표에 맞춘 운동 계획:
운동 목표는 {운동목표}이고, {초점부위}를 특히 신경 쓰고 싶은 거야. 이 정보를 바탕으로 네게 맞는 운동 프로그램을 짜줄게. 
예를 들어, 근육량을 늘리고 싶으면, 복합 운동(스쿼트, 데드리프트, 벤치프레스 등)을 추천할 수 있어. 부상이 있다면 부드럽게 시작하는 저강도 운동도 고려해야 해. 
운동을 할 장소는 {운동장소}에 맞게 실내/실외 운동을 제안할 수 있어. 

### 운동 계획 예시:
- **주간 운동 계획**: {운동횟수}일 동안 할 수 있고, 한 번의 운동 시간은 {운동시간}분 정도로 설정했을 때, 각 운동의 강도와 종류를 조절해야 해.
- **운동 강도**: {운동목표}에 따라 강도는 어떻게 할지, 부상 여부도 체크하며 진행할 수 있어.
- **운동 종류**: {초점부위}에 맞는 운동을 중심으로 세트 수, 반복 횟수, 휴식 시간 등을 설정할 거야. 

위 정보를 참고해 자세히 안내할게. 운동 목표에 맞춰 계획을 세우고, 자세한 세부사항까지 알려줘!

# Answer:
"""


prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

# 디렉토리 설정
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)  # 디렉토리가 없으면 생성
TEXT_DIR = Path("transcriptions")  # 텍스트 파일 저장 디렉토리
TEXT_DIR.mkdir(parents=True, exist_ok=True)  # 디렉토리가 없으면 생성

# 채팅 내역 저장용 변수
chat_history = []

# 파일명 생성 함수
def generate_audio_filename():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return AUDIO_DIR / f"audio_{timestamp}.wav"  # Path 객체로 반환

def generate_text_filename(audio_file: Path):
    timestamp = audio_file.stem.split("_")[-1]
    return TEXT_DIR / f"transcription_{timestamp}.txt"  # Path 객체로 반환

# 오디오 녹음 및 파일 저장 함수
def record_audio(duration: int = 5, sample_rate: int = 16000):
    """마이크 입력을 녹음하여 파일로 저장."""
    audio_file = generate_audio_filename()  # 고유한 파일명 생성
    print(f"녹음 중... {duration}초 동안")

    try:
        # 오디오 녹음
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # 녹음 완료까지 대기

        # 오디오 데이터를 torch.Tensor로 변환하여 저장
        audio_tensor = torch.tensor(audio).transpose(1, 0)  # [Samples, Channels] → [Channels, Samples]
        torchaudio.save(str(audio_file), audio_tensor, sample_rate)  # 파일 경로에 Path 객체를 str로 변환하여 전달
        print(f"녹음 완료: {audio_file}")
        return audio_file
    except Exception as e:
        print(f"녹음 중 오류 발생: {e}")
        return None

@app.post("/transcribe-microphone/")
async def transcribe_microphone(request: Request):
    try:
        # 오디오 녹음
        audio_file = record_audio(duration=5)
        if not audio_file:
            raise HTTPException(status_code=400, detail="녹음에 실패했습니다.")

        # 오디오 파일 로드
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"오디오 파일 로드 실패: {str(e)}")

        # 샘플링 레이트 16000Hz로 조정
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        # Whisper 모델로 텍스트 변환
        inputs = processor(waveform.squeeze(dim=0).numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_tokens = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        # 텍스트 파일로 저장
        text_file = generate_text_filename(audio_file)
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(transcription)

        # 채팅 내역 포맷팅
        formatted_chat_history = "\n".join([f"사용자: {msg['user']}\nAI: {msg['ai']}" for msg in chat_history]) if chat_history else "대화 내역이 없습니다."

        # 프롬프트 생성
        formatted_prompt = prompt.format(question=transcription, context=formatted_chat_history)

        # AI 응답 생성
        ai_answer = llm.predict(formatted_prompt)

        # 채팅 내역에 추가
        chat_history.append({"user": transcription, "ai": ai_answer})

        return {
            "transcription": transcription,
            "audio_file": str(audio_file),
            "text_file": str(text_file),
            "ai_answer": ai_answer,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")
    finally:
        print(f"녹음 파일 위치: {audio_file}, 텍스트 파일 위치: {text_file}, AI 응답: {ai_answer}")
