# 🦉 OwlVest Chatbot

A Flask-based chatbot that uses OpenRouter API with DeepSeek model and can be enhanced with vector search capabilities.

## 🚀 Quick Start (Basic Testing)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Edit `config.py` and replace `your_openrouter_api_key_here` with your actual OpenRouter API key.

### 3. Run Basic Chatbot
```bash
python basic_chatbot.py
```

### 4. Open in Browser
Navigate to `http://127.0.0.1:5001` to test the basic chatbot.

## 🔧 Configuration

The `config.py` file contains all configuration settings:
- **API Key**: Your OpenRouter API key
- **Flask Settings**: Host, port, and debug mode
- **Model Settings**: AI model name and temperature
- **Vector Store Settings**: Chunk size and overlap

## 📁 File Structure

- `basic_chatbot.py` - Simple Flask chatbot for testing
- `app.py` - Full-featured chatbot with vector search (requires data files)
- `config.py` - Configuration file (replaces .env)
- `data/` - Directory containing knowledge base files
- `requirements.txt` - Python dependencies

## 🐛 Issues Fixed

1. **Missing .env file** → Replaced with `config.py`
2. **Missing dependencies** → Updated `requirements.txt`
3. **Import errors** → Fixed langchain imports
4. **File loading errors** → Added error handling

## 🧪 Testing

### Basic Chatbot
- Simple web interface
- No vector search (faster for testing)
- In-memory chat history
- Status endpoint at `/status`

### Full Chatbot (app.py)
- Vector search with FAISS
- Knowledge base integration
- Context-aware responses
- Requires data files in `data/` directory

## 🔄 Next Steps

1. Test the basic chatbot first
2. Configure your API key
3. Test API connectivity
4. Once working, enhance with vector search features

## 📞 API Endpoints

- `GET /` - Web interface
- `POST /chat` - Chat endpoint
- `GET /status` - Server status

## ⚠️ Notes

- The basic chatbot runs on `127.0.0.1:5001` by default
- Change the IP in `config.py` if you need external access
- Vector search requires the data files to be present
- API key must be configured before use
