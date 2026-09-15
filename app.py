"""
app.py
Streamlit RAG assistant for hospital policy documents.
Retrieval: local FAISS index (built by ingest.py)
Generation: Groq API (fast, free-tier LLM inference)

Run locally:
    export GROQ_API_KEY="your_key_here"
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Add GROQ_API_KEY under Settings -> Secrets as:
    GROQ_API_KEY = "your_key_here"
"""

import os
import streamlit as st
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Config ----------
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a hospital policy assistant. Answer questions using ONLY the "
    "provided policy excerpts below. If the excerpts don't contain the answer, "
    "say you don't have that information in the knowledge base rather than "
    "guessing. Always mention which department/policy document your answer "
    "is based on. Keep answers clear and concise, suitable for hospital staff."
)


# ---------- Cached resources ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_vectorstore(_embeddings):
    if not os.path.isdir(INDEX_DIR):
        return None
    return FAISS.load_local(
        INDEX_DIR, _embeddings, allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    if not api_key:
        return None
    return Groq(api_key=api_key)


def get_departments(vectorstore):
    depts = set()
    for doc in vectorstore.docstore._dict.values():
        depts.add(doc.metadata.get("department", "root"))
    return sorted(depts)


def retrieve(vectorstore, query, department_filter, k=TOP_K):
    # Over-fetch, then filter by department client-side (simple + version-safe)
    results = vectorstore.similarity_search_with_score(query, k=k * 4)
    if department_filter and department_filter != "All":
        results = [
            (doc, score)
            for doc, score in results
            if doc.metadata.get("department") == department_filter
        ]
    return results[:k]


def build_context(results):
    blocks = []
    for i, (doc, _score) in enumerate(results, start=1):
        dept = doc.metadata.get("department", "unknown")
        src = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[{i}] Department: {dept} | Source: {src} (page {page})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


def ask_groq(client, question, context):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Policy excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content


# ---------- UI ----------
st.set_page_config(page_title="Hospital Policy Assistant", page_icon="🏥", layout="wide")
st.title("🏥 Hospital Policy Assistant")
st.caption("Ask questions about hospital policies. Answers are grounded in your uploaded policy documents.")

embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)
groq_client = load_groq_client()

if vectorstore is None:
    st.error(
        f"No index found at '{INDEX_DIR}/'. Run `python ingest.py` first, "
        "then make sure the faiss_index folder is included in this repo/deployment."
    )
    st.stop()

if groq_client is None:
    st.error(
        "GROQ_API_KEY not found. Set it as an environment variable locally, "
        "or add it under Settings -> Secrets on Streamlit Community Cloud."
    )
    st.stop()

with st.sidebar:
    st.header("Filters")
    departments = ["All"] + get_departments(vectorstore)
    selected_dept = st.selectbox("Department", departments)
    st.divider()
    st.caption(f"Model: `{GROQ_MODEL}`")
    st.caption(f"Retrieving top {TOP_K} chunks per question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['department']}** — {s['source_file']} (page {s['page']})")

question = st.chat_input("Ask about a hospital policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating answer..."):
            results = retrieve(vectorstore, question, selected_dept)

            if not results:
                answer = (
                    "I couldn't find anything relevant in the knowledge base "
                    "for this department/question. Try a different phrasing "
                    "or check the department filter."
                )
                sources = []
            else:
                context = build_context(results)
                answer = ask_groq(groq_client, question, context)
                sources = [
                    {
                        "department": doc.metadata.get("department", "unknown"),
                        "source_file": doc.metadata.get("source_file", "unknown"),
                        "page": doc.metadata.get("page", "?"),
                    }
                    for doc, _score in results
                ]

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- **{s['department']}** — {s['source_file']} (page {s['page']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
