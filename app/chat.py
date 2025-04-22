from .models.embeddings import EmbeddingModel
from .models.vector_store import VectorStore
from .utils.text_processing import chunk_text, load_text_file, join_contexts, ask_question_to_gemini


class RAGChat:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = EmbeddingModel(model_name)
        self.chunks = []
        self.vector_store = None

    def load_knowledge_base(self, file_path):

        text = load_text_file(file_path)
        self.chunks = chunk_text(text)
        embeddings = self.embedding_model.encode_chunks(self.chunks)

        self.vector_store = VectorStore(embeddings[0].shape[0])
        self.vector_store.add_vectors(embeddings)

    def get_answer(self, question):
        question_embedding = self.embedding_model.encode_text([question])

        _, indices = self.vector_store.search(question_embedding)
        context = join_contexts(self.chunks, indices[0])

        # Get answer from LLM
        return ask_question_to_gemini(context, question)
