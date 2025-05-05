config = {}
if os.path.exists('config.json'):
    with open('config.json', 'r') as conf:
        config = json.load(conf)
else:
    config['OPENAI_API_KEY'] = ''
    config['DATA_SOURCE_DIR'] = 'data-sources'
    config['DB_PATH'] = '/home/trana/dev/rag-database'

api_key = config['OPENAI_API_KEY']
data_source_dir = config['DATA_SOURCE_DIR']
database_dir = config['DB_PATH']

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter

# Load markdown documents
markdown_dir = os.path.join(data_source_dir, "markdowns")
markdown_loader = DirectoryLoader(
    path=markdown_dir,
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
markdown_documents = markdown_loader.load()

# Load email documents
email_dir = os.path.join(data_source_dir, "emails")
email_loader = DirectoryLoader(
    path=email_dir,
    glob="**/*.eml",
    loader_cls=UnstructuredEmailLoader
)
email_documents = email_loader.load()

documents = markdown_documents + email_documents

# Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(api_key=api_key)

# Create a Chroma vector store
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=embedding_model,
#     persist_directory=database_dir  # Directory to persist the database
# )
# vectorstore.persist()

vectorstore = Chroma(persist_directory=database_dir, embedding_function=embedding_model)



retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7, "k": 1})

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=api_key)

template = """\nAnswer the question and base on the context below. 
If the context doesn't contain any relevant information to the question, just say "I don't know".

Context:
{context}

Question: {input}
"""
prompt = PromptTemplate.from_template(template)

doc_chain = create_stuff_documents_chain(llm,prompt)
qa_chain = create_retrieval_chain(retriever, doc_chain)

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/wiki-message', methods=['POST'])
def wiki_message():
    user_query = request.json.get('userQuery', '')
    if not user_query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Use the qa_chain to get the response
    response = qa_chain.invoke({"input": user_query})
    context = [
    {
        "content": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in response["context"]
    ]

    return jsonify({
        "rag-response": response["answer"],
        "context": context
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
