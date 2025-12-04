import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from ollama import Client
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential
from sentence_transformers import SentenceTransformer


load_dotenv(override=True)

MODEL = "gpt-oss:20b"
DB_NAME = str(Path(__file__).parent.parent / "cv_preprocessed_db")
collection_name = "docs"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
wait = wait_exponential(multiplier=1, min=10, max=240)

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)
embedder = SentenceTransformer(embedding_model)

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
LinkedIn: https://www.linkedin.com/in/sadegh-mahmoud-abadi-b3b4a9341
Telegram: @Sadegh_KCL
Phone: +989024759923

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

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

@retry(wait=wait)
def rerank(question, chunks):
    system_prompt = f"""
    You are a document re-ranker.
    You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
    The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
    You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
    Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
    Only reply with JSON in the format:
    {RankOrder.model_json_schema()}
    """
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat(model=MODEL, messages=messages, format=RankOrder.model_json_schema())
    reply = response.message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]

def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
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
    # reranked = rerank(original_question, chunks)
    # return reranked[:FINAL_K]

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
    return response.message.content, chunks