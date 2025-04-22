from dotenv import load_dotenv
from app.chat import RAGChat

load_dotenv()


def main():
    rag_chat = RAGChat()
    rag_chat.load_knowledge_base("./app/data/meuarquivo.txt")

    while True:
        question = input(">>")
        if question.lower() == "exit":
            break
        answer = rag_chat.get_answer(question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
