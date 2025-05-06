from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st

st.sidebar.title("🔧 설정")
openai_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
st.sidebar.markdown("[GitHub Repository](https://github.com/your-repo-link)")

sitemap_urls = [
    "https://developers.cloudflare.com/ai-gateway/sitemap.xml",
    "https://developers.cloudflare.com/vectorize/sitemap.xml",
    "https://developers.cloudflare.com/workers-ai/sitemap.xml",
]

if not openai_api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar.")
    st.stop()

llm = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke({"question": question, "context": doc.page_content}).content,
                "source": doc.metadata.get("source", "Unknown"),
                "date": doc.metadata.get("lastmod", "Unknown"),
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke({"question": question, "answers": condensed})

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ").replace("CloseSearch Submit Blog", "")

@st.cache_data(show_spinner="Loading Cloudflare documentation...")
def load_cloudflare_docs():
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(sitemap_urls, parsing_function=parse_page)
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store.as_retriever()

st.set_page_config(page_title="Cloudflare SiteGPT", page_icon="☁️")
st.title("☁️ Cloudflare SiteGPT")
st.markdown("""
Ask questions about Cloudflare's AI documentation:
- [AI Gateway](https://developers.cloudflare.com/ai-gateway/)
- [Vectorize](https://developers.cloudflare.com/vectorize/)
- [Workers AI](https://developers.cloudflare.com/workers-ai/)
""")

retriever = load_cloudflare_docs()
query = st.chat_input("질문을 입력하세요...")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if query:
    with st.spinner("Thinking..."):
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", result.content))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message.replace("$", "\\$"))
