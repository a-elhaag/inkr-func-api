# Inkr Function API - Azure OpenAI Integration

This Azure Functions app provides a REST API for Azure OpenAI chat completions.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your actual values:
   ```
   AZURE_OPENAI_API_KEY=your-actual-api-key
   AZURE_OPENAI_ENDPOINT=https://inkr-openai.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   ```

3. **Test Locally**
   ```bash
   # Test the OpenAI integration
   python test_openai.py
   
   # Start the Azure Functions host
   func start
   ```

## API Endpoints

### POST /api/chat
Chat completion endpoint that supports both streaming and non-streaming responses.

**Request Body:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_completion_tokens": 800,
  "temperature": 1.0,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stream": false
}
```

**Example with curl:**
```bash
# Basic chat
curl -X POST http://localhost:7071/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is AI?"}
    ]
  }'

# Streaming response
curl -X POST http://localhost:7071/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "inkr-func-api"
}
```

## Features

- ✅ Azure OpenAI integration with proper error handling
- ✅ Support for streaming and non-streaming responses
- ✅ Configurable chat parameters (temperature, max tokens, etc.)
- ✅ Multi-turn conversation support
- ✅ Health check endpoint
- ✅ Environment variable configuration
- ✅ Comprehensive test suite

## Deployment

To deploy to Azure:

1. **Create Azure Function App**
2. **Configure Application Settings** with your environment variables
3. **Deploy using Azure Functions Core Tools:**
   ```bash
   func azure functionapp publish <your-function-app-name>
   ```

## Testing

Run the test script to verify your Azure OpenAI configuration:
```bash
python test_openai.py
```

This will test:
- Basic chat completion
- Multi-turn conversations  
- Streaming responses
