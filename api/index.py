import os
import json
import re
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
from api.config import *  # Ensure you have the OPENROUTER_API_KEY and other configurations

# Define the UltraSimpleRAG class first
class UltraSimpleRAG:
    def __init__(self):
        self.document_chunks = []
        self.chunk_metadata = []

    def load_documents(self):
        """Load and process OwlVest documents"""
        print("Loading OwlVest documents...")

        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        files = [
            "owlvest_master_data.txt",
            "clean_whitepaper.txt", 
            "clean_pic_data.txt",
            "website_data.txt"
        ]

        all_text = ""
        for file_name in files:
            file_path = os.path.join(data_path, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    all_text += f"\n\n--- {file_name} ---\n\n{content}"
                    print(f"Loaded {file_name} ({len(content)} characters)")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

        return all_text
    
    def split_text_into_chunks(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        print("Splitting text into chunks...")

        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()

        chunks = []
        metadata = []

        # Simple sentence-based splitting
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        chunk_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                metadata.append({
                    "chunk_id": chunk_count,
                    "length": len(current_chunk),
                    "type": "owlvest_data"
                })
                chunk_count += 1

                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            metadata.append({
                "chunk_id": chunk_count,
                "length": len(current_chunk),
                "type": "owlvest_data"
            })

        print(f"Created {len(chunks)} chunks")
        return chunks, metadata
    
    def simple_text_similarity(self, query, text):
        """Calculate simple text similarity using word overlap"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))

        if not query_words:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))

        if union == 0:
            return 0.0

        return intersection / union
    
    def search_similar_chunks(self, query, k=3):
        """Search for similar chunks using simple text similarity"""
        if not self.document_chunks:
            return []

        try:
            # Calculate similarity scores for all chunks
            chunk_scores = []
            for i, chunk in enumerate(self.document_chunks):
                score = self.simple_text_similarity(query, chunk)
                chunk_scores.append({
                    'index': i,
                    'score': score,
                    'content': chunk
                })

            # Sort by score (highest first) and take top k
            chunk_scores.sort(key=lambda x: x['score'], reverse=True)
            top_chunks = chunk_scores[:k]

            # Filter out chunks with very low similarity
            relevant_chunks = []
            for chunk_info in top_chunks:
                if chunk_info['score'] > 0.01:  # Minimum similarity threshold
                    relevant_chunks.append({
                        'content': chunk_info['content'],
                        'score': chunk_info['score'],
                        'metadata': self.chunk_metadata[chunk_info['index']] if chunk_info['index'] < len(self.chunk_metadata) else {}
                    })

            return relevant_chunks

        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []
    
    def initialize(self):
        """Initialize the complete RAG system"""
        print("Initializing OwlVest Ultra-Simple RAG system...")

        # Load documents
        text = self.load_documents()
        if not text:
            return False

        # Split into chunks
        self.document_chunks, self.chunk_metadata = self.split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

        print("Ultra-Simple RAG system initialized successfully!")
        return True
    
    def get_relevant_context(self, query, k=None):
        """Get relevant context for a query"""
        k = k or SIMILARITY_SEARCH_K

        relevant_chunks = self.search_similar_chunks(query, k)

        if not relevant_chunks:
            return "No relevant information found in the knowledge base."

        # Combine relevant chunks
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(f"[Relevance: {chunk['score']:.3f}]\n{chunk['content']}")

        return "\n\n---\n\n".join(context_parts)

# Initialize RAG system here at the top of the file, before any routes or Flask app setup
rag_system = UltraSimpleRAG()

# Flask setup
app = Flask(__name__)
# Allow CORS for all origins for testing (in production, restrict this)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for API routes
CORS(app)

# Global variables (you can keep them)
chat_history = []
document_chunks = []
chunk_metadata = []

# RAG initialization happens at import time
try:
    rag_system.initialize()
except Exception as e:
    print(f"RAG init warning: {e}")

@app.after_request
def after_request(response):
    """Add CORS headers for Vercel deployment"""
    response.headers.add('Access-Control-Allow-Origin', '*')  # Allow all origins (or restrict as needed)
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

def query_openrouter_api(prompt, context=""):
    """Query OpenRouter API with context"""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "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":
        return "Please configure your OpenRouter API key in config.py"
    
    # Prepare the full prompt with context
    if context:
        full_prompt = f"""You are OwlVest AI Assistant. Use this context from OwlVest's knowledge base to answer the user's question accurately and helpfully:

Context Information:
{context}

User Question: {prompt}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Be professional, helpful, and accurate
- Focus on OwlVest-specific information
- If asked about something not in the context, politely redirect to relevant OwlVest topics

Please provide a helpful answer:"""
    else:
        full_prompt = prompt
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-3.5-turbo",  # Update model name based on the OpenRouter model you're using
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        elif "error" in data:
            return f"API Error: {data['error'].get('message', 'Unknown error')}"
        else:
            return f"Unexpected response: {data}"
    except requests.exceptions.Timeout:
        return "Request timeout. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except Exception as e:
        return f"Error: {e}"

# Flask routes
@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint with RAG"""
    try:
        user_query = request.json.get("query")
        if not user_query:
            return jsonify({"response": "No query provided."})
        
        # Get relevant context from RAG system
        context = rag_system.get_relevant_context(user_query)
        
        # Query OpenRouter API with context
        response = query_openrouter_api(user_query, context)
        
        # Store in chat history
        chat_history.append({
            "user": user_query,
            "bot": response,
            "context_used": context[:200] + "..." if len(context) > 200 else context
        })
        
        return jsonify({"response": response})
        
    except Exception as e:
        return jsonify({"response": f"Server error: {str(e)}"})

@app.route("/status")
def status():
    """Status endpoint"""
    return jsonify({
        "status": "running",
        "rag_ready": len(rag_system.document_chunks) > 0,
        "api_key_configured": OPENROUTER_API_KEY != "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",
        "chat_history_count": len(chat_history),
        "documents_loaded": len(rag_system.document_chunks),
        "system_type": "Ultra-Simple RAG (Text Similarity)"
    })

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "rag_system": "ready" if len(rag_system.document_chunks) > 0 else "initializing",
        "api_configured": OPENROUTER_API_KEY != "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"
    })

if __name__ == "__main__":
    print("Starting OwlVest Ultra-Simple RAG Chatbot...")
    print(f"Configure your API key in config.py")
    print(f"Will be available at http://{FLASK_HOST}:{FLASK_PORT}")
    
    # Initialize RAG system
    if rag_system.initialize():
        print("Ultra-Simple RAG system initialized successfully!")
        print("Ready to answer questions about OwlVest!")
        print("Using text similarity for document retrieval")
    else:
        print("Ultra-Simple RAG system initialization failed, running in basic mode")
    
    # Start Flask app
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
