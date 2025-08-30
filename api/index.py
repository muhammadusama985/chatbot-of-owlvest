import os
import json
import re
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
from api.config import *

# Flask setup
app = Flask(__name__)
CORS(app)

# Global variables
chat_history = []
document_chunks = []
chunk_metadata = []

class UltraSimpleRAG:
    def __init__(self):
        self.document_chunks = []
        self.chunk_metadata = []
        
    def load_documents(self):
        """Load and process OwlVest documents"""
        print(" Loading OwlVest documents...")
        
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
                    print(f" Loaded {file_name} ({len(content)} characters)")
            except Exception as e:
                print(f" Error loading {file_name}: {e}")
        
        return all_text
    
    def split_text_into_chunks(self, text, chunk_size=1000, overlap=200):
        """Split text into overlapping chunks"""
        print(" Splitting text into chunks...")
        
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
        
        print(f" Created {len(chunks)} chunks")
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
            print(f" Error searching chunks: {e}")
            return []
    
    def initialize(self):
        """Initialize the complete RAG system"""
        print(" Initializing OwlVest Ultra-Simple RAG system...")
        
        # Load documents
        text = self.load_documents()
        if not text:
            return False
        
        # Split into chunks
        self.document_chunks, self.chunk_metadata = self.split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        print(" Ultra-Simple RAG system initialized successfully!")
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

# Initialize RAG system
rag_system = UltraSimpleRAG()

def query_openrouter_api(prompt, context=""):
    """Query OpenRouter API with context"""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        return " Please configure your OpenRouter API key in config.py"
    
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
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": TEMPERATURE,
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
            return f" Unexpected response: {data}"
    except requests.exceptions.Timeout:
        return " Request timeout. Please try again."
    except requests.exceptions.RequestException as e:
        return f" Request failed: {e}"
    except Exception as e:
        return f" Error: {e}"

# Flask routes
@app.route("/")
def home():
    """Professional chat interface"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OwlVest Knowledge Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            min-height: 100vh;
            box-shadow: 0 0 0 1px #e2e8f0;
        }

        .header {
            background: #1e293b;
            color: white;
            padding: 24px 32px;
            border-bottom: 1px solid #334155;
        }

        .header-content {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo {
            width: 48px;
            height: 48px;
            background: #3b82f6;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: bold;
        }

        .header-text h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .header-text p {
            font-size: 14px;
            color: #94a3b8;
        }

        .status-bar {
            background: #059669;
            color: white;
            padding: 12px 32px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .status-indicator {
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-content {
            display: flex;
            height: calc(100vh - 140px);
        }

        .sidebar {
            width: 280px;
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
            padding: 24px;
            overflow-y: auto;
        }

        .sidebar-section {
            margin-bottom: 32px;
        }

        .sidebar-section h3 {
            font-size: 14px;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 16px;
        }

        .feature-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 8px 0;
            font-size: 14px;
            color: #64748b;
        }

        .feature-icon {
            width: 16px;
            height: 16px;
            background: #e2e8f0;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
        }

        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px 32px;
            background: white;
        }

        .chat-container::-webkit-scrollbar {
            width: 6px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: #f1f5f9;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 3px;
        }

        .message {
            margin-bottom: 24px;
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }

        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            font-weight: 500;
        }

        .user-avatar {
            background: #3b82f6;
            color: white;
        }

        .bot-avatar {
            background: #059669;
            color: white;
        }

        .message-meta {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .message-author {
            font-size: 14px;
            font-weight: 600;
            color: #1e293b;
        }

        .message-badge {
            background: #059669;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }

        .message-content {
            margin-left: 44px;
            color: #475569;
            font-size: 15px;
            line-height: 1.6;
        }

        .user-message {
            text-align: right;
        }

        .user-message .message-header {
            justify-content: flex-end;
        }

        .user-message .message-content {
            margin-left: 0;
            margin-right: 44px;
            background: #f1f5f9;
            padding: 16px 20px;
            border-radius: 16px 16px 4px 16px;
            display: inline-block;
            max-width: 80%;
        }

        .bot-message .message-content {
            background: #f8fafc;
            padding: 16px 20px;
            border-radius: 4px 16px 16px 16px;
            border-left: 3px solid #059669;
            max-width: 85%;
        }

        .error-message .message-content {
            background: #fef2f2;
            border-left-color: #ef4444;
            color: #dc2626;
        }

        .typing-indicator {
            display: none;
            margin-left: 44px;
            color: #64748b;
            font-style: italic;
            font-size: 14px;
        }

        .typing-dots {
            display: inline-flex;
            gap: 4px;
            margin-left: 8px;
        }

        .typing-dot {
            width: 6px;
            height: 6px;
            background: #64748b;
            border-radius: 50%;
            animation: typing 1.4s ease-in-out infinite;
        }

        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typing {
            0%, 60%, 100% { transform: scale(1); opacity: 0.5; }
            30% { transform: scale(1.2); opacity: 1; }
        }

        .input-section {
            border-top: 1px solid #e2e8f0;
            padding: 24px 32px;
            background: white;
        }

        .input-container {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        #userInput {
            width: 100%;
            padding: 16px 20px;
            border: 1px solid #d1d5db;
            border-radius: 12px;
            font-size: 15px;
            outline: none;
            resize: none;
            min-height: 52px;
            max-height: 120px;
            transition: border-color 0.2s ease;
            font-family: inherit;
            line-height: 1.5;
        }

        #userInput:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        #userInput::placeholder {
            color: #9ca3af;
        }

        .send-btn {
            padding: 16px 24px;
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 100px;
            justify-content: center;
        }

        .send-btn:hover {
            background: #2563eb;
        }

        .send-btn:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }

        .welcome-message {
            text-align: center;
            color: #64748b;
            padding: 48px 24px;
        }

        .welcome-message h2 {
            font-size: 20px;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 8px;
        }

        @media (max-width: 768px) {
            .main-content {
                flex-direction: column;
                height: auto;
            }
            
            .sidebar {
                width: 100%;
                border-right: none;
                border-bottom: 1px solid #e2e8f0;
                padding: 16px 24px;
            }
            
            .chat-container {
                height: 400px;
                padding: 16px 24px;
            }
            
            .input-section {
                padding: 16px 24px;
            }
            
            .header {
                padding: 16px 24px;
            }
            
            .message-content,
            .user-message .message-content,
            .bot-message .message-content {
                max-width: 95%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <div class="logo">OV</div>
                <div class="header-text">
                    <h1>OwlVest Knowledge Assistant</h1>
                    <p>Enterprise AI-powered knowledge retrieval system</p>
                </div>
            </div>
        </div>
        
        <div class="status-bar" id="statusBar">
            <div class="status-indicator"></div>
            <span id="statusText">System operational - Ready to assist with OwlVest queries</span>
        </div>
        
        <div class="main-content">
            <div class="sidebar">
                <div class="sidebar-section">
                    <h3>System Features</h3>
                    <div class="feature-item">
                        <div class="feature-icon"></div>
                        <span>RAG-powered responses</span>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon"></div>
                        <span>Knowledge base search</span>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon"></div>
                        <span>Context-aware answers</span>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon"></div>
                        <span>Real-time processing</span>
                    </div>
                </div>
                
                <div class="sidebar-section">
                    <h3>Query Guidelines</h3>
                    <div class="feature-item">
                        <div class="feature-icon"></div>
                        <span>Ask specific questions</span>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon"></div>
                        <span>Reference company data</span>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon"></div>
                        <span>Request clarifications</span>
                    </div>
                </div>
            </div>
            
            <div class="chat-area">
                <div class="chat-container" id="chatContainer">
                    <div class="message bot-message">
                        <div class="message-header">
                            <div class="message-avatar bot-avatar">AI</div>
                            <div class="message-meta">
                                <span class="message-author">OwlVest Assistant</span>
                                <span class="message-badge">Online</span>
                            </div>
                        </div>
                        <div class="message-content">
                            Welcome to the OwlVest Knowledge Assistant. I have access to comprehensive company documentation and can help you find information about OwlVest's services, team, and operations.<br><br>
                            Please feel free to ask any questions about the company.
                        </div>
                    </div>
                </div>
                
                <div class="typing-indicator" id="typingIndicator">
                    Assistant is typing
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
                
                <div class="input-section">
                    <div class="input-container">
                        <div class="input-wrapper">
                            <textarea id="userInput" placeholder="Enter your question about OwlVest..." onkeypress="handleKeyPress(event)" rows="1"></textarea>
                        </div>
                        <button class="send-btn" onclick="sendMessage()" id="sendBtn">
                            <span>Send</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isProcessing = false;
        
        // Auto-resize textarea
        const textarea = document.getElementById('userInput');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
        
        function addMessage(message, isUser = false, isError = false) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}${isError ? ' error-message' : ''}`;
            
            if (isUser) {
                messageDiv.innerHTML = `
                    <div class="message-header">
                        <div class="message-meta">
                            <span class="message-author">You</span>
                        </div>
                        <div class="message-avatar user-avatar">U</div>
                    </div>
                    <div class="message-content">${message}</div>
                `;
            } else {
                messageDiv.innerHTML = `
                    <div class="message-header">
                        <div class="message-avatar bot-avatar">AI</div>
                        <div class="message-meta">
                            <span class="message-author">OwlVest Assistant</span>
                            <span class="message-badge">${isError ? 'Error' : 'Active'}</span>
                        </div>
                    </div>
                    <div class="message-content">${message}</div>
                `;
            }
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        function showTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'block';
            const container = document.getElementById('chatContainer');
            container.scrollTop = container.scrollHeight;
        }
        
        function hideTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'none';
        }
        
        function setLoading(loading) {
            isProcessing = loading;
            const sendBtn = document.getElementById('sendBtn');
            const userInput = document.getElementById('userInput');
            
            if (loading) {
                showTypingIndicator();
                sendBtn.innerHTML = '<span>Sending...</span>';
            } else {
                hideTypingIndicator();
                sendBtn.innerHTML = '<span>Send</span>';
            }
            
            sendBtn.disabled = loading;
            userInput.disabled = loading;
        }
        
        function updateStatus(text, type = 'success') {
            const statusBar = document.getElementById('statusBar');
            const statusText = document.getElementById('statusText');
            
            statusText.textContent = text;
            
            if (type === 'error') {
                statusBar.style.background = '#dc2626';
            } else if (type === 'warning') {
                statusBar.style.background = '#d97706';
            } else {
                statusBar.style.background = '#059669';
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey && !isProcessing) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        async function sendMessage() {
            if (isProcessing) return;
            
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage(message, true);
            input.value = '';
            input.style.height = 'auto';
            
            // Show loading
            setLoading(true);
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: message })
                });
                
                const data = await response.json();
                const isError = data.response.startsWith(' ') || data.response.startsWith(' ');
                
                addMessage(data.response, false, isError);
                
                // Update status
                if (isError) {
                    updateStatus('Error processing request - Please try again', 'error');
                } else {
                    updateStatus('Response generated successfully - Ready for next query', 'success');
                }
                
            } catch (error) {
                addMessage('Connection error: Unable to reach the assistant. Please check your connection and try again.', false, true);
                updateStatus('Connection error - Please check network', 'error');
            } finally {
                setLoading(false);
            }
        }
        
        // Initialize on load
        window.addEventListener('load', async () => {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                if (data.rag_ready && data.api_key_configured) {
                    updateStatus('System operational - Ready to assist with OwlVest queries', 'success');
                } else if (!data.api_key_configured) {
                    updateStatus('API configuration required - Please contact administrator', 'warning');
                } else {
                    updateStatus('System initializing - Please wait', 'warning');
                }
            } catch (error) {
                updateStatus('System status unavailable - Limited functionality', 'error');
            }
        });
        
        // Auto-focus input
        document.getElementById('userInput').focus();
    </script>
</body>
</html>"""
    return html

@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint with RAG"""
    try:
        user_query = request.json.get("query")
        if not user_query:
            return jsonify({"response": " No query provided."})
        
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
        return jsonify({"response": f" Server error: {str(e)}"})

@app.route("/status")
def status():
    """Status endpoint"""
    return jsonify({
        "status": "running",
        "rag_ready": len(rag_system.document_chunks) > 0,
        "api_key_configured": OPENROUTER_API_KEY != "your_openrouter_api_key_here",
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
        "api_configured": OPENROUTER_API_KEY != "your_openrouter_api_key_here"
    })

if __name__ == "__main__":
    print(" Starting OwlVest Ultra-Simple RAG Chatbot...")
    print(f" Configure your API key in config.py")
    print(f" Will be available at http://{FLASK_HOST}:{FLASK_PORT}")
    
    # Initialize RAG system
    if rag_system.initialize():
        print(" Ultra-Simple RAG system initialized successfully!")
        print(" Ready to answer questions about OwlVest!")
        print(" Using text similarity for document retrieval")
    else:
        print(" Ultra-Simple RAG system initialization failed, running in basic mode")
    
    # Start Flask app
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
