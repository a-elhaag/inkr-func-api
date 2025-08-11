import json
import logging
import os

import azure.functions as func
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()

# Azure OpenAI config
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://inkr-openai.openai.azure.com/")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# LangChain LLM + Embeddings
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
    temperature=0.7,
)

embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=embedding_model,
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
)


# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Vector store (in-memory FAISS for now)
VECTORSTORE_PATH = "notes_index"
if os.path.exists(VECTORSTORE_PATH):
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    # Create an empty FAISS index with no documents, but with correct embedding dimension
    # You can create an empty index manually or add a dummy doc with empty content:
    dummy_doc = Document(page_content="")  # dummy doc to get embedding dimension
    vectorstore = FAISS.from_documents([dummy_doc], embeddings)
    # Then optionally remove that dummy doc immediately if you want (not necessary)

# Azure Functions app
app = func.FunctionApp()

@app.route(route="save_note", auth_level=func.AuthLevel.FUNCTION)
def save_note(req: func.HttpRequest) -> func.HttpResponse:
    """
    Save a note into the vector store
    """
    try:
        req_body = req.get_json()
        note_text = req_body.get("note", "")
        if not note_text:
            return func.HttpResponse(
                json.dumps({"error": "Note content is required"}),
                status_code=400,
                mimetype="application/json"
            )

        doc = Document(page_content=note_text)
        vectorstore.add_documents([doc])
        vectorstore.save_local(VECTORSTORE_PATH)

        return func.HttpResponse(
            json.dumps({"status": "Note saved successfully"}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error saving note: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="search_notes", auth_level=func.AuthLevel.FUNCTION)
def search_notes(req: func.HttpRequest) -> func.HttpResponse:
    """
    Search notes by semantic similarity
    """
    try:
        req_body = req.get_json()
        query = req_body.get("query", "")
        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Search query is required"}),
                status_code=400,
                mimetype="application/json"
            )

        results = vectorstore.similarity_search(query, k=3)
        notes = [doc.page_content for doc in results]

        return func.HttpResponse(
            json.dumps({"results": notes}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error searching notes: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="chat", auth_level=func.AuthLevel.FUNCTION)
def chat_with_notes(req: func.HttpRequest) -> func.HttpResponse:
    """
    Chat with AI using memory and note context
    """
    try:
        req_body = req.get_json()
        user_input = req_body.get("message", "")
        if not user_input:
            return func.HttpResponse(
                json.dumps({"error": "Message is required"}),
                status_code=400,
                mimetype="application/json"
            )

        # Search for relevant notes
        search_results = vectorstore.similarity_search(user_input, k=3)
        context_notes = "\n".join([doc.page_content for doc in search_results])

        # Prompt with memory + note context
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant for note-taking. Use the provided notes if relevant."),
            ("system", f"Relevant notes:\n{context_notes}"),
            ("human", "{user_input}")
        ])

        chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
        response = chain.run(user_input=user_input)

        return func.HttpResponse(
            json.dumps({"response": response}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Chat error: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="summarize", auth_level=func.AuthLevel.FUNCTION)
def summarize_note(req: func.HttpRequest) -> func.HttpResponse:
    """
    Summarize a note
    """
    try:
        req_body = req.get_json()
        note = req_body.get("note", "")
        if not note:
            return func.HttpResponse(
                json.dumps({"error": "Note content is required"}),
                status_code=400,
                mimetype="application/json"
            )

        prompt = ChatPromptTemplate.from_template(
            "Summarize this note clearly and concisely:\n\n{note}"
        )

        chain = LLMChain(llm=llm, prompt=prompt)
        summary = chain.run(note=note)

        return func.HttpResponse(
            json.dumps({"summary": summary}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({"status": "healthy", "service": "inkr-func-api"}),
        status_code=200,
        mimetype="application/json"
    )
