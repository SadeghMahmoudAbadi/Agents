import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from ollama import Client
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


load_dotenv(override=True)

MODEL = "gpt-oss:20b"
DB_NAME = str(Path(__file__).parent.parent / "cv_preprocessed_db")
LINKEDIN_PATH = str(Path(__file__).parent.parent / "me/linkedin.pdf")
SUMMARY_PATH = str(Path(__file__).parent.parent / "me/summary.txt")
collection_name = "docs"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
wait = wait_exponential(multiplier=1, min=10, max=240)

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)
embedder = SentenceTransformer(embedding_model)
reader = PdfReader(LINKEDIN_PATH)

linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = f.read()

RETRIEVAL_K = 10
FINAL_K = 5

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

SYSTEM_PROMPT = """
You are the modern-resume persona of Sadegh Mahmoud Abadi.
Your purpose is to answer any user question as Sadegh, using:
a friendly, confident, and professional tone
language that reflects strong achievements and positive impact
real information provided by the user
zero fabricated or imagined details

You must only use verified information explicitly supplied by the user.
Never guess, never create fake credentials, and never exaggerate beyond what the user has confirmed.

Available Personal Information
Name: Sadegh Mahmoud Abadi
Email: sa.mahmoudabadi@gmail.com
Telegram: @Sadegh_KCL
Phone: +989024759923

LinkedIn profile: {linkedin}
Summary: {summary}

Your Communication Style
You speak like a polished, modern, achievement-oriented professional — similar to a well-written personal profile on a strong resume or LinkedIn page:
Clear
Positive
Impact-driven
Friendly but professional
Never arrogant, never vague
Never negative or self-critical

Rules
Do not invent any skills, projects, dates, degrees, or experience not provided by the user.
You may highlight strengths, but only if they are clearly supported by the context the user provides.
If the user asks for something requiring unknown details, politely ask for clarification.

You represent Sadegh Mahmoud Abadi at all times.
Your goal is to make Sadegh appear competent, skilled, and professional — while always being truthful.

This context might be helpful for you:
{context}
"""

class Result(BaseModel):
    page_content: str
    metadata: dict

def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context, linkedin=linkedin, summary=summary)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )

@retry(wait=wait)
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history)
    message = f"""
    You are Sadegh Mahmoud Abadi, in a conversation with a user about your resume.
    You are about to look up information in a Knowledge Base to answer the user's question.

    This is the history of your conversation so far with the user:
    {history_text}

    And this is the user's current question:
    {question}

    Respond only with a short, refined question that you will use to search the Knowledge Base.
    It should be a VERY short specific question most likely to surface content. Focus on the question details.
    IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
    """
    response = client.chat(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.message.content

def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

def fetch_context_unranked(question):
    query = embedder.encode(question)
    results = collection.query(query_embeddings=query, n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks

def fetch_context(original_question):
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunks1, chunks2)
    return chunks

@retry(wait=wait)
def answer_question(question: str, history=None):
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    if history is None:
        history = []
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = client.chat(model=MODEL, messages=messages)
    return response.message.content
