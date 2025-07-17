import os
import json
import logging
import azure.functions as func
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://inkr-openai.openai.azure.com/")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

app = func.FunctionApp()

@app.route(route="chat", auth_level=func.AuthLevel.FUNCTION)
def chat_completion(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function for chat completions using Azure OpenAI
    """
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Get request body
        req_body = req.get_json()
        
        if not req_body:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract parameters from request
        messages = req_body.get("messages", [{"role": "user", "content": "What is AI?"}])
        max_tokens = req_body.get("max_completion_tokens", 800)
        temperature = req_body.get("temperature", 1.0)
        top_p = req_body.get("top_p", 1.0)
        frequency_penalty = req_body.get("frequency_penalty", 0.0)
        presence_penalty = req_body.get("presence_penalty", 0.0)
        stream = req_body.get("stream", False)
        
        # Create chat completion
        response = client.chat.completions.create(
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            model=deployment,
            stream=stream
        )
        
        if stream:
            # Handle streaming response
            def generate():
                for chunk in response:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content or ""
                        yield f"data: {json.dumps({'content': content})}\n\n"
                yield "data: [DONE]\n\n"
            
            return func.HttpResponse(
                generate(),
                status_code=200,
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            # Handle regular response
            result = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            
            return func.HttpResponse(
                json.dumps(result),
                status_code=200,
                mimetype="application/json"
            )
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Internal server error: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint
    """
    return func.HttpResponse(
        json.dumps({"status": "healthy", "service": "inkr-func-api"}),
        status_code=200,
        mimetype="application/json"
    )
