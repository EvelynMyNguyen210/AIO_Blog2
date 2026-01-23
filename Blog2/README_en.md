# 4. Why do you want to create a chatbot?
## 4.1 Defining the purpose of the chatbot
In reality, most AI chatbots today can be categorized into one of four main groups.

**FAQ Bot – Frequently Asked Questions Bot**  
This is the most common type of chatbot, often used in customer service.  
- Answers repetitive questions: working hours, policies, user guides  
- No need for long conversations  
- Content is relatively fixed  

This type of chatbot is suitable for reducing human workload, especially in customer support systems.

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn/static/uploads/20260123_152134_f48bddef.png" style="margin: 0 auto; display: block;"><br/>
  <em>Figure 4.1. FAQ chatbot</em>
</p>

**Task-oriented Bot – Chatbot for task execution**  
Unlike FAQ Bots, this type not only answers but also **guides users through a process**.  

Examples:  
- Scheduling appointments  
- Booking services  
- Step-by-step information lookup  

The focus of this type of chatbot is logic and conversation flow, not natural chatting style.

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn//static/uploads/20260123_153156_3c398178.jpeg" style="margin: 0 auto; display: block;"><br/>
  <em>Figure 4.2. Task-oriented Bot</em>
</p>

**Conversational Bot – Natural conversation chatbot**  
This type of chatbot is like a “chatting companion.”  
- Goal is to maintain conversation  
- Responses need to be natural and flexible  
- Not necessarily “absolutely correct”  

This type is often used for entertainment, emotional support, or social interaction.  

However, note: Conversational bots are harder to build than other types, because they require handling context and long conversation history.

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn//static/uploads/20260123_153349_1ce0facd.webp" style="margin: 0 auto; display: block;"><br/>
  <em>Figure 4.3. Conversational Bot</em>
</p>

**Domain-specific Bot – Chatbot for a specific field**  

Chatbots designed for a particular domain such as:  
- Healthcare  
- Education  
- Sales  

Characteristics of this type:  
- Requires domain-specific data  
- Must strictly control content  
- Mistakes can cause serious consequences  

<p align="center">
  <img src="https://aioconquer.aivietnam.edu.vn//static/uploads/20260123_155702_32fc2bbb.jpeg" style="margin: 0 auto; display: block;"><br/>
  <em>Figure 4.4. Domain-specific Bot</em>
</p>

**Mandatory questions before coding**  
After identifying the type of chatbot, you need to clearly answer the following questions:  
- Who is this chatbot for?  
- What kind of questions will it answer?  
- Does it need to remember conversation history or just answer individual questions?  
- Does it require private data, or only general knowledge?  

If these questions are not clearly answered, coding will easily go **“off track”**, making features harder to fix and expand.

## 4.2 Common mistakes when starting to build a chatbot

When first building a chatbot, many people encounter the same mistakes:

**Expecting the chatbot to “understand” like a human**  
Chatbots have no awareness or emotions. They only process language and predict answers based on learned data. Expecting them to think like humans will lead to disappointment.  

Example: You create a product consulting chatbot and ask:  
“I want to buy a phone for my parents for convenience.”  

Humans will naturally understand:  
- Elderly users  
- Prioritize ease of use, good battery, large text  

But the chatbot may only latch onto the keyword “phone” and provide a list of popular products, not suitable for the context. This happens because chatbots lack life experience or social reasoning, and only analyze language patterns from training data.

**Trusting the chatbot 100%**  
AI chatbots can give wrong answers but sound very convincing. Without control mechanisms, they may generate misleading information that users cannot easily detect.  

Example: A learning chatbot is asked:  
“In which case is this formula applied?”  

It may respond with detailed explanations and technical terms, but the content could be wrong or outdated. If users don’t verify, this false information may be taken as fact.  

The issue is not that the chatbot “lies,” but that it **does not verify information**, only predicts the most likely answer.

**Not limiting scope**  
Wanting a chatbot to “answer everything” is a common mistake. The narrower the scope, the more effective and controllable the chatbot becomes.  

Example: You say:  
“My chatbot answers questions, gives advice, chats, and acts as a personal assistant.”  

The result is usually:  
- Rambling answers  
- Unclear strengths  
- Hard to control quality  

In practice, a chatbot only performs well when its task scope is clearly defined. An FAQ chatbot is different from a scheduling chatbot, and both differ from a conversational chatbot.

**Ignoring cost and security**  
When starting, many people focus only on “making it run,” forgetting backend issues.  

Examples:  
- Placing API keys directly in code and uploading to GitHub  
- Not limiting the number of requests  
- Not monitoring API usage costs  

Consequences:  
- API key leaks  
- Unauthorized account usage  
- Costs rising unexpectedly  

These problems often appear after deployment, and fixing them later is much more costly.

## 4.3 When should you start making a demo?

A demo **should not** be the first step, but rather a way to test whether your idea is truly effective.  

After clearly defining what the chatbot is for and who it serves, you can then think about making a small demo. A demo should begin once the chatbot’s purpose is clear and its scope narrowed. This is the time to check one simple but crucial question: *Does this chatbot solve the problem I set out to address?*  

A good demo doesn’t need full features, beautiful UI, or perfect UX. Instead, it should focus on the chatbot’s core functionality. If the chatbot is meant to answer questions, test whether it answers correctly and consistently. If it is designed to support a task, check whether it completes that task smoothly.  

The goal of a demo is not to create a finished product, but to help you detect early issues in the idea, scope, or approach. A simple but focused demo will save you a lot of time and effort when moving to full chatbot development.

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
