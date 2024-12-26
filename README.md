# Conversational PDFs with Chat History

This project implements a conversational AI system that allows users to interact with PDF documents using natural language queries. The system uses LLM and RAG to provide accurate and context-aware responses based on the content of uploaded PDF files.

## Features

- PDF Upload: Users can upload multiple PDF files for analysis and querying
- Conversational Interface: Interact with the system using natural language questions
- Context-Aware Responses: The system considers chat history to provide more accurate and relevant answers
  
## How It Works

1. Users upload PDF files through the Streamlit interface
2. The system processes and indexes the PDF content using advanced text splitting and embedding techniques
3. Users can ask questions about the uploaded documents
4. The system retrieves relevant information from the indexed documents, considering the chat history for context
5. Responses are generated using a large language model (LLM) and displayed to the user

## Tech Stack Used

- LangChain
- Streamlit
- Groq
- Hugging Face
- ChromaDB
- PyPDF
- Python


## Note

This project requires a valid Groq API key to function. Ensure you have the necessary credentials before using the application.


The Sample output of the application deployed on Streamlit looks like: 


![Sample output](https://github.com/user-attachments/assets/517e15b6-9496-4a98-aeb6-b9d1c9ee35fc)
