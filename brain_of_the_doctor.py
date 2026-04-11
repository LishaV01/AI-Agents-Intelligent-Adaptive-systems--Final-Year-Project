from groq import Groq

def analyze_image_with_query(query, model, encoded_image, api_key):
    """
    Analyzes an image using Groq Cloud for high accuracy.
    """
    client = Groq(api_key=api_key)  
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    res = chat_completion.choices[0].message.content
    print(f"DEBUG - Doctor's Response: {res}")
    return res

def encode_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
