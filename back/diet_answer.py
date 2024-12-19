# app/chat_handler.py
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# ChatGPT API 설정 (LangChain 사용)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.8)

prompt_template_1 = """
당신은 식단관리 AI입니다. 사용자의 정보를 바탕으로, 체중 감량을 목표로 하는 맞춤형 식단을 추천해 주세요. 식단은 아침, 점심, 저녁, 간식으로 나누어, 하루 총 칼로리와 영양소 비율을 포함해야 합니다.

# Question:
{question}

# Context:
{context}

# Output:
**체중 감량을 목표로 하는 식단**  
- 하루 총 칼로리를 적절한 칼로리 적자 상태로 설정해야 합니다.  
- 각 식사는 저칼로리 고단백 음식을 중심으로 구성하고, 탄수화물은 적게 포함시켜야 합니다.
- 아침, 점심, 저녁, 간식의 구체적인 식사 계획을 세우고, 식사 간격에 맞춰 영양소 비율을 정하세요.
- 간식은 칼로리를 적절히 조절하면서 포만감을 줄 수 있는 음식을 추천하세요.
"""

prompt_template_2 = """
당신은 식단관리 AI입니다. 사용자의 정보를 바탕으로, 근육 증가를 목표로 하는 맞춤형 식단을 추천해 주세요. 식단은 아침, 점심, 저녁, 간식으로 나누어, 하루 총 칼로리와 영양소 비율을 포함해야 합니다.

# Question:
{question}

# Context:
{context}

# Output:
**근육 증가를 목표로 하는 식단**  
- 하루 총 칼로리를 적절히 설정하여, 칼로리 흡수를 증가시킬 수 있도록 합니다.  
- 각 식사는 고단백 음식을 중심으로 구성하며, 운동 후 회복을 고려한 탄수화물과 단백질 비율을 포함해야 합니다.  
- 아침, 점심, 저녁, 간식의 구체적인 식사 계획을 세우고, 운동 계획에 맞춰 식사를 제시하세요.  
- 간식은 단백질 보충을 할 수 있는 음식을 추천합니다.
"""

prompt_template_3 = """
당신은 식단관리 AI입니다. 사용자의 정보를 바탕으로, 균형 잡힌 식사를 선호하는 맞춤형 식단을 추천해 주세요. 식단은 아침, 점심, 저녁, 간식으로 나누어, 하루 총 칼로리와 영양소 비율을 포함해야 합니다.

# Question:
{question}

# Context:
{context}

# Output:
**균형 잡힌 식단을 선호하는 식단**  
- 탄수화물, 단백질, 지방의 비율을 적절히 맞춰야 합니다.  
- 하루 총 섭취 칼로리를 사용자의 일상 활동 수준에 맞춰 설정하고, 각 식사마다 균형 잡힌 영양소를 제공해야 합니다.  
- 알레르기나 음식 선호도에 맞춰 식단을 개인화하고, 선호하는 음식을 포함시켜야 합니다.  
- 간식은 균형 잡힌 영양소가 포함된 음식을 제시하여 건강한 선택을 하도록 유도합니다.
"""

prompt_1 = PromptTemplate(template=prompt_template_1, input_variables=["question", "context"])
prompt_2 = PromptTemplate(template=prompt_template_2, input_variables=["question", "context"])
prompt_3 = PromptTemplate(template=prompt_template_3, input_variables=["question", "context"])

# 채팅 내역 저장용 변수
chat_history = []

async def handle_chat_diet_request(request):
    body = await request.json()
    user_question = body.get("question", "기본 질문")

    # 채팅 내역 포맷팅
    formatted_chat_history = "\n".join(
        [f"사용자: {msg['user']}\nAI: {msg['ai']}" for msg in chat_history]
    ) if chat_history else "대화 내역이 없습니다."

    # 프롬프트 생성
    # formatted_prompt = prompt.format(question=user_question, context=formatted_chat_history)

    # 세 가지 다른 응답을 생성
    ai_answers = []

    for prompt in [prompt_1, prompt_2, prompt_3]:
        formatted_prompt = prompt.format(question=user_question, context=formatted_chat_history)
        ai_answer = llm.predict(formatted_prompt)
        ai_answers.append(ai_answer)


    # 채팅 내역에 추가
    chat_history.append({"user": user_question, "ai": ai_answers[-1]})

    return {"answer": ai_answers}
