# 5. Coding an AI Chatbot

After understanding how an AI chatbot works and what components it consists of, we will build a simple demo chatbot that runs on Google Colab.

Unlike the common approach of calling APIs from external services, this blog demonstrates a chatbot that loads and runs an AI model locally within the Google Colab environment. This approach helps us better understand how the model works internally and is well suited for research, experimentation, and learning—without relying on third-party APIs.

## 5.1. Installing required libraries

First, we need to install several essential libraries to load and run a language model directly on Google Colab:

- ***transformers:*** Hugging Face’s library for loading and working with large language models

- ***torch:*** the core deep learning framework used for tensor computation and model execution

- ***accelerate:*** helps optimize model execution by managing CPU/GPU configuration, resource allocation, and inference acceleration with minimal setup

- ***bitsandbytes:*** enables loading models in compressed formats (8-bit or 4-bit), significantly reducing memory usage on limited hardware

```
!pip install -q -U torch transformers accelerate bitsandbytes
```

## 5.2. Loading the language model

In this demo, we use the following model:

***Qwen2.5-1.5B-Instruct***

This model is:

- Lightweight (~1.5B parameters)

- Fine-tuned for conversational tasks

- Suitable for demo

You can also explore and replace it with other suitable models available on ***[Hugging Face](https://huggingface.co/models)***

```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# Set to True if you want to use GPU
use_gpu = False

print("⏳ Loading model ...")
if use_gpu==True:
    nf4_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_mem_usage =True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage =True
    )
tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
print("⏳ Model successfully loaded")
```
## 5.3 A simple chatbot function

The processing flow of this function follows exactly the design principles discussed in the previous sections:

Receive input from the user

Wrap the input into a prompt

Send the prompt to the model

Receive and print the generated response
```
def local_chatbot():
    user_input = input("\n👤 User: ")
    if user_input.lower() in ['bye', 'exit']: return
    
    promt = f"""<|im_start|>system
              You are a helpful AI assistant. Keep your answers concise and to the point.
              <|im_end|>
              <|im_start|>user
              {user_input}
              <|im_end|>
              <|im_start|>assistant
            """
    # Tokenize
    inputs = tokenizer(promt, return_tensors="pt")
    
    # Generate
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    # Decode
    response = tokenizer.decode(outputs[0])
    
    # Simple string processing for cleaner output
    print(f"🤖 Bot: {response.split("<|im_start|>assistant")[-1].strip().replace("<|im_end|>","")}")
    return response

response = local_chatbot()
```

***Full source code : [Google Colab](https://colab.research.google.com/drive/1vpn7lnZbX3niohOM_7jMayMYqrmBVlIT?usp=sharing)***

# 6. Conclusion: Building a chatbot is a design problem before it is a coding problem

From this article, one key takeaway stands out:

***Building an AI chatbot does not start with code but with design thinking.***

Before writing any code, you should clearly answer a few fundamental questions:

- What problem is the chatbot designed to solve?

- Who are the primary users?

- What is the scope of questions and answers?

- Does it require domain-specific or private data?

When these questions are not clearly defined, starting to code too early often leads to:

- Complex systems with low practical impact

- Chatbots that respond vaguely and are hard to control

- Higher deployment costs without addressing real user needs

On the other hand, when the design thinking is clear:

- Technology choices become simpler and more purposeful

- Code becomes merely the implementation of ideas

- The system is easier to extend, optimize, and maintain in the long term

*In the next blog post, we will build upon this simple demo to develop a more complete chatbot, and then deploy it on free platforms to run as a real, working demo product.*
