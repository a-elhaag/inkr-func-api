"""
Test script to demonstrate Azure OpenAI usage patterns
Run this script to test your Azure OpenAI integration
"""

import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://inkr-openai.openai.azure.com/")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if not api_key:
    print("Please set AZURE_OPENAI_API_KEY in your .env file")
    exit(1)

# Initialize client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

def test_basic_completion():
    """Test basic chat completion"""
    print("Testing basic chat completion...")
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?",
            }
        ],
        max_completion_tokens=800,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment
    )
    
    print(f"Response: {response.choices[0].message.content}\n")
    return response

def test_multi_turn_conversation():
    """Test multi-turn conversation"""
    print("Testing multi-turn conversation...")
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ]
    
    # First response
    response1 = client.chat.completions.create(
        messages=messages,
        max_completion_tokens=400,
        temperature=1.0,
        model=deployment
    )
    
    # Add assistant response to conversation
    messages.append({
        "role": "assistant", 
        "content": response1.choices[0].message.content
    })
    
    # Add follow-up question
    messages.append({
        "role": "user",
        "content": "What is so great about #1?"
    })
    
    # Second response
    response2 = client.chat.completions.create(
        messages=messages,
        max_completion_tokens=400,
        temperature=1.0,
        model=deployment
    )
    
    print(f"First response: {response1.choices[0].message.content}")
    print(f"Follow-up response: {response2.choices[0].message.content}\n")

def test_streaming():
    """Test streaming response"""
    print("Testing streaming response...")
    
    response = client.chat.completions.create(
        stream=True,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Tell me a short story about AI.",
            }
        ],
        max_completion_tokens=400,
        temperature=1.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model=deployment,
    )
    
    print("Streaming response: ", end="")
    for update in response:
        if update.choices:
            content = update.choices[0].delta.content or ""
            print(content, end="", flush=True)
    
    print("\n")

if __name__ == "__main__":
    print("=== Azure OpenAI Test Suite ===\n")
    
    try:
        # Run tests
        test_basic_completion()
        test_multi_turn_conversation()
        test_streaming()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your .env file is configured correctly with valid Azure OpenAI credentials.")
    
    finally:
        client.close()
