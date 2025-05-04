import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

with st.sidebar:
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    difficulty = st.selectbox("🧠 시험 난이도 선택", ["쉬움", "보통", "어려움"])

if not api_key:
    st.warning("OpenAI API 키를 입력해주세요.")
    st.stop()

st.title("📘 맞춤형 Quiz Generator")

quiz_function = {
    "name": "generate_quiz",
    "description": "Create a multiple choice quiz from the context with the specified difficulty",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"}
                                },
                                "required": ["answer", "correct"]
                            }
                        }
                    },
                    "required": ["question", "answers"]
                }
            }
        },
        "required": ["questions"]
    }
}

llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o",
    openai_api_key=api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a teacher creating multiple-choice quizzes."),
    ("human", "Make a {difficulty} quiz based on the following content:\n\n{context}")
])

chain = (
    RunnableMap({
        "difficulty": lambda x: x["difficulty"],
        "context": lambda x: x["context"]
    }) |
    prompt |
    llm.bind(functions=[quiz_function], function_call={"name": "generate_quiz"}) |
    JsonOutputFunctionsParser()
)

context = st.text_area("📄 시험에 사용할 내용 입력", height=200)
if context and st.button("🔍 시험 시작"):
    quiz_data = chain.invoke({"context": context, "difficulty": difficulty})

    if "questions" not in quiz_data:
        st.error("문제 생성 실패. 다시 시도해주세요.")
        st.stop()

    score = 0
    user_answers = []

    with st.form("quiz_form"):
        for idx, q in enumerate(quiz_data["questions"]):
            st.write(f"**Q{idx+1}. {q['question']}**")
            options = [a["answer"] for a in q["answers"]]
            user_choice = st.radio("", options, key=idx)
            user_answers.append((user_choice, q["answers"]))
        submitted = st.form_submit_button("제출")

    if submitted:
        for i, (user_answer, answers) in enumerate(user_answers):
            correct = next((a["answer"] for a in answers if a["correct"]), None)
            if user_answer == correct:
                score += 1
                st.success(f"Q{i+1}: 정답!")
            else:
                st.error(f"Q{i+1}: 오답 ❌ 정답은: **{correct}**")

        total = len(user_answers)
        st.markdown(f"## 🎯 점수: {score} / {total}")

        if score == total:
            st.balloons()
        else:
            if st.button("🔄 다시 풀기"):
                st.experimental_rerun()
