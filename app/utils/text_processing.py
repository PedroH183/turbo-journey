from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)


def chunk_text(text, chunk_size=500):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def load_text_file(file_path):
    with open(file_path, "r") as f:
        return f.read()


def join_contexts(chunks, indices):
    return "\n".join([chunks[i] for i in indices])


def ask_question_to_gemini(context, pergunta):
    message = [SystemMessage(context), HumanMessage(pergunta)]
    return llm.invoke(message).content
