from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp  # or use Ollama if preferred

import os

load_dotenv()

# -------------------------
# URLs of audio files
# -------------------------
URLs = [
    "https://storage.googleapis.com/aai-web-samples/langchain_agents_webinar.opus",
    "https://storage.googleapis.com/aai-web-samples/langchain_document_qna_webinar.opus",
    "https://storage.googleapis.com/aai-web-samples/langchain_retrieval_webinar.opus"
]


# -------------------------
# Function to transcribe audio files
# -------------------------
def create_docs(url_list):
    i = []
    for url in url_list:
        print(f"Transcribing {url}")
        i.append(AssemblyAIAudioTranscriptLoader(file_path=url).load()[0])
    return i


# -------------------------
# HuggingFace Embedder
# -------------------------
def make_embedder():
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


# -------------------------
# Replace OpenAI with LLaMA
# -------------------------
def make_qa_chain():
    llm = LlamaCpp(
        model_path="models/llama-2-7b-chat.ggmlv3.q4_0.bin",  # Update with your model path
        temperature=0.7,
        max_tokens=512,
        top_p=0.95,
        n_ctx=2048,
        verbose=True
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'fetch_k': 3}),
        return_source_documents=True
    )


# -------------------------
# Processing pipeline
# -------------------------
print('Transcribing files ...')
docs = create_docs(URLs)

print('Splitting documents...')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

for text in texts:
    text.metadata = {"audio_url": text.metadata["audio_url"]}

print('Embedding texts...')
hf = make_embedder()
db = Chroma.from_documents(texts, hf)

# -------------------------
# QA Chain Interaction
# -------------------------
print('\nEnter `e` to exit')
qa_chain = make_qa_chain()

while True:
    q = input('Enter your question: ')
    if q == 'e':
        break
    result = qa_chain({"query": q})
    print(f"\nQ: {result['query'].strip()}")
    print(f"A: {result['result'].strip()}")
    print("SOURCES:")
    for idx, elt in enumerate(result['source_documents']):
        print(f"    Source {idx}:")
        print(f"        Filepath: {elt.metadata['audio_url']}")
        print(f"        Contents: {elt.page_content}")
    print('\n')
