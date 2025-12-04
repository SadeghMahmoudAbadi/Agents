import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from multiprocessing import Pool
from tenacity import retry, wait_exponential
from sentence_transformers import SentenceTransformer
import pickle


load_dotenv(override=True)

MODEL = "x-ai/grok-4.1-fast:free"
DB_NAME = str(Path(__file__).parent.parent / "cv_preprocessed_db")
collection_name = "docs"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "cv_knowledge_base"
AVERAGE_CHUNK_SIZE = 500
wait = wait_exponential(multiplier=1, min=10, max=240)

chunks_path = str(Path(__file__).parent.parent / "cv_chunk_list.pkl")

WORKERS = 3

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


class Result(BaseModel):
    page_content: str
    metadata: dict

class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {"source": document["source"], "name": document["name"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )

class Chunks(BaseModel):
    chunks: list[Chunk]

def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""
    documents = []

    for file in KNOWLEDGE_BASE_PATH.rglob("*.md"):
        name = file.name
        with open(file, "r", encoding="utf-8") as f:
            documents.append({"name": name, "source": file.as_posix(), "text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents

def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""You are an expert in semantic segmentation and RAG preprocessing.
    Your task is to chunk documents into meaningful, coherent segments optimized for vector search and retrieval.

    Your Tasks
    1. Semantically Chunk Each Document
    For every document I provide:
    Break the content into meaningful, self-contained semantic units.
    Use the provided Approx Chunks only as a guideline — stay close, but do not destroy semantic coherence.
    Ensure that each chunk:
    Does not cut off sentences mid-way
    Does not split related bullet points or sections
    Preserves Markdown formatting where appropriate
    Represents a standalone idea that a vectorstore can index for retrieval

    Chunking Guidelines (Very Important)
    Semantic Rules
    Keep chunks conceptually complete.
    A chunk should represent a single idea, concept, or logical section.
    Where possible, align with Markdown structure:
    Headings
    Subheadings
    File sections
    Skill sections
    Project summaries
    Avoid creating excessively short or excessively long chunks.
    
    What Not To Do
    Do NOT rewrite, summarize, or reword content.
    Do NOT omit content.
    Do NOT merge unrelated sections just to match the approximate count.
    Do NOT break code blocks, bullet lists, or paragraphs in the middle.
    
    Input
    Document Name: {document["name"]}
    Source: {document["source"]}
    Approx Chunks: {how_many}
    Content:
    {document["text"]}
    """

def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]

@retry(wait=wait)
def process_document(document):
    messages = make_messages(document)
    response = openrouter.chat.completions.parse(model=MODEL, messages=messages, response_format=Chunks)
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]

def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks

def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    emb = SentenceTransformer(embedding_model).encode(texts)
    vectors = [e for e in emb]

    collection = chroma.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")

if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    create_embeddings(chunks)
    print("Ingestion complete")