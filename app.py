import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.json["message"]  
    
    prompt = user_message
    messages = [{"role": "user", "content": prompt}]
    tokenized_message = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    
    response_token_ids = model.generate(tokenized_message['input_ids'].cuda(),
                                        attention_mask=tokenized_message['attention_mask'].cuda(),
                                        max_new_tokens=4096,
                                        pad_token_id=tokenizer.eos_token_id)
    
    generated_tokens = response_token_ids[:, len(tokenized_message['input_ids'][0]):]
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    return jsonify({"response": generated_text})

if __name__ == "__main__":
    app.run(debug=True)
