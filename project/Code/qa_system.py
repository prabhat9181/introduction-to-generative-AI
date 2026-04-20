import openai

openai.api_key = "sk-xxxxxxxxxxxx"

def generate_answer(context, question):
    prompt = f"Context:\n{context}\n\nQuestion:{question}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response['choices'][0]['message']['content']
