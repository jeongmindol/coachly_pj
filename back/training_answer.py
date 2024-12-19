from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# ChatGPT API 설정 (LangChain 사용)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.8)

# 운동 루틴을 위한 세 가지 프롬프트 템플릿 설정

prompt_template_1 = """
당신은 운동루틴 생성 AI입니다. 사용자의 정보를 바탕으로, 전반적인 체력 향상을 위한 운동 루틴을 추천해 주세요. 운동 루틴은 주 3~5회로 구성하며, 각 운동은 적절한 운동 시간과 세트 수, 반복 횟수를 포함해야 합니다.

# Question:
{question}

# Context:
{context}

# Output:
**전반적인 체력 향상을 위한 운동 루틴**  
- 유산소 운동과 근력 운동을 적절히 혼합한 루틴을 제공합니다.  
- 각 운동마다 운동 시간, 세트, 반복 횟수를 포함시키고, 운동 후 휴식 시간을 고려하여 설계합니다.  
- 다양한 운동을 포함하여 전신을 고르게 단련할 수 있도록 합니다.
"""

prompt_template_2 = """
당신은 운동루틴 생성 AI입니다. 사용자의 정보를 바탕으로, 근육 증가를 목표로 하는 운동 루틴을 추천해 주세요. 운동 루틴은 주 4~6회로 구성하며, 각 운동은 적절한 운동 시간과 세트 수, 반복 횟수를 포함해야 합니다.

# Question:
{question}

# Context:
{context}

# Output:
**근육 증가를 목표로 하는 운동 루틴**  
- 각 운동은 고강도의 근력 운동을 포함하며, 특히 중량 운동을 중심으로 설계합니다.  
- 하루에 특정 부위를 집중적으로 훈련하는 방식으로 세부적인 루틴을 제공합니다.  
- 각 운동마다 운동 시간, 세트, 반복 횟수를 포함하고, 운동 후 충분한 휴식 시간을 설정합니다.
"""

prompt_template_3 = """
당신은 운동루틴 생성 AI입니다. 사용자의 정보를 바탕으로, 체중 감량을 목표로 하는 운동 루틴을 추천해 주세요. 운동 루틴은 주 4~5회로 구성하며, 각 운동은 적절한 유산소 운동과 근력 운동을 포함해야 합니다.

# Question:
{question}

# Context:
{context}

# Output:
**체중 감량을 목표로 하는 운동 루틴**  
- 주로 유산소 운동과 복합적인 근력 운동을 포함시켜 칼로리 소모를 극대화합니다.  
- 운동 루틴은 30~60분 정도의 중간 강도로 설정하며, 운동 후 충분한 회복 시간을 두어야 합니다.  
- 체중 감량에 도움을 줄 수 있는 운동을 포함하고, 운동의 빈도와 강도를 점차적으로 늘려가야 합니다.
"""

# 프롬프트 템플릿 생성
prompt_1 = PromptTemplate(template=prompt_template_1, input_variables=["question", "context"])
prompt_2 = PromptTemplate(template=prompt_template_2, input_variables=["question", "context"])
prompt_3 = PromptTemplate(template=prompt_template_3, input_variables=["question", "context"])

# 채팅 내역 저장용 변수
chat_history = []

async def handle_chat_pt_request(request):
    body = await request.json()
    user_question = body.get("question", "기본 질문")

    # 채팅 내역 포맷팅
    formatted_chat_history = "\n".join(
        [f"사용자: {msg['user']}\nAI: {msg['ai']}" for msg in chat_history]
    ) if chat_history else "대화 내역이 없습니다."

    # 세 가지 다른 운동 루틴에 대한 응답을 생성
    ai_answers = []

    # 3가지 스타일의 운동 루틴을 각각 생성
    for prompt in [prompt_1, prompt_2, prompt_3]:
        formatted_prompt = prompt.format(question=user_question, context=formatted_chat_history)
        ai_answer = llm.predict(formatted_prompt)
        ai_answers.append(ai_answer)

    # 채팅 내역에 추가
    chat_history.append({"user": user_question, "ai": ai_answers[-1]})

    # 3가지 스타일의 운동 루틴을 배열로 반환
    return {"answer": ai_answers}

