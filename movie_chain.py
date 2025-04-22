from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# 예시 데이터: 영화 정보
examples = [
    {
        "movie": "Inception",
        "info": """
        Title: Inception
        Director: Christopher Nolan
        Cast: Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page
        Budget: $160 million
        Box Office: $836.8 million
        Genre: Sci-Fi, Thriller
        Synopsis: A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.
        """,
    },
    {
        "movie": "Parasite",
        "info": """
        Title: Parasite
        Director: Bong Joon-ho
        Cast: Song Kang-ho, Lee Sun-kyun, Cho Yeo-jeong
        Budget: $11 million
        Box Office: $266 million
        Genre: Drama, Thriller
        Synopsis: A poor family schemes to become employed by a wealthy household by infiltrating their domestic lives and posing as unrelated, highly qualified individuals.
        """,
    },
    {
        "movie": "The Godfather",
        "info": """
        Title: The Godfather
        Director: Francis Ford Coppola
        Cast: Marlon Brando, Al Pacino, James Caan
        Budget: $6 million
        Box Office: $250 million
        Genre: Crime, Drama
        Synopsis: The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.
        """,
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Tell me about the movie {movie}."),
        ("ai", "{info}"),
    ]
)

example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert who provides structured information about films."),
        example_prompt,
        ("human", "Tell me about the movie {movie}."),
    ]
)

chain = final_prompt | chat

chain.invoke({"movie": "Interstellar"})