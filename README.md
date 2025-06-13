📖 Gsy_Yoga_AI_Assistant 🧘‍♂️
An AI-powered assistant for Govardhan School of Yoga using RAG (Retrieval Augmented Generation) to answer queries from yoga-related documents via a web app.

📑 How It Works
1️⃣ Vectorizing Documents (vectorize_doc.py)

Reads PDFs from Data/

Splits text into chunks (to handle large texts efficiently)

Converts chunks into vector embeddings via TogetherEmbeddings

Stores them in Chroma vector DB (vector_db_dir_2/)

2️⃣ Running the Assistant (main.py)

Start app with:

bash
Copy
Edit
streamlit run main.py
User asks a query

Relevant document chunks fetched from vector DB

LLM (LLaMA 3 70B) generates an answer using these chunks

Logs chat history in both text and JSON

📄 Prompt (prompt_template.py)
Sets assistant’s polite, helpful tone. If a query is beyond knowledge base, suggests contacting:

css
Copy
Edit
yoga@ecovillage.org.in
📦 Tech Used
Python, Streamlit, LangChain, ChromaDB, TogetherEmbeddings, ChatGroq

