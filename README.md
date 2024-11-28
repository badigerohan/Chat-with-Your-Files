# Chat with Your Files 📂

This project is a Streamlit-based application that allows users to interact with their PDF, CSV, and DOCX files using natural language queries. The app processes uploaded files, extracts their content, and creates an AI-powered conversational interface for exploring the documents.

## Features
- **Multi-format support:** Works with PDF, CSV, and DOCX files.
- **Natural Language Interaction:** Ask questions about your documents and receive context-aware answers.
- **File Processing:** Automatically extracts and processes text from uploaded files.
- **Conversational Memory:** Keeps track of chat history for contextual understanding.

## How It Works
1. Upload your files (PDF, CSV, DOCX) via the sidebar.
2. The app extracts and splits the content into manageable chunks.
3. Creates vector embeddings for efficient information retrieval.
4. Uses a conversational AI model to answer your queries based on the uploaded files.
