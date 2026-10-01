import os
import time
import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = None

def get_client():
    """获取OpenAI客户端，如果未初始化则初始化"""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY 环境变量未设置。"
                "请创建 .env 文件并设置 OPENAI_API_KEY 和 OPENAI_BASE_URL，"
                "或者设置环境变量。"
            )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    return client

refine_query_prompt = '\nYou are an experienced software developer specializing in code retrieval and understanding.\n\n<raw_query>\n{RAW_QUERY}\n</raw_query>\n\nTask:\nThe provided <raw_query> combines a functional requirement with its arguments.\nYour task is to **rewrite** this query into a natural language description that is **ACCURATE, DETAILED, and INFORMATIVE**.\nThe goal is to produce a refined query that captures the complete coding intent by seamlessly integrating the functional requirement with **ALL** its arguments and specific values.\n\nInstructions:\n1. Analysis:\n- Analyze the <raw_query> to understand the context.\n- Identify all technical keywords, library names, algorithm names, configuration flags, and specific data values.\n- **JUDGMENT (CRITICAL):** **Preserve EVERYTHING.** Do NOT filter out specific data elements, file paths, numbers, or string literals. In a code retrieval context, specific values (e.g., `\'<temporary_directory>/logs\'`, `epoch=50`) often correspond to default parameter values or hardcoded constants in the target function. These are high-value search terms and must be retained.\n\n2. Rewriting (Semantic Synthesis):\n- Rewrite the query into a clear, grammatically correct sentence or paragraph.\n- **Integrate Specifics Naturally:** Weave the "Functionality" and the specific "Arguments" together.\n- Instead of saying "a specified directory", say "the \'<temporary_directory>/logs\' directory".\n- Instead of saying "a specific threshold", say "a threshold of 0.5".\n- Ensure the final query clearly describes what the code does using the **exact constraints and values** provided.\n\nPlease answer in the following format:\n\n[start_of_analysis]\n<analysis_of_intent_and_confirmation_of_specific_values>\n[end_of_analysis]\n\n[start_of_rewritten_query]\n<refined_query_with_specific_values_preserved>\n[end_of_rewritten_query]\n\nNotes:\n- The output must be **SPECIFIC**: Retain literal values from the arguments (paths, IDs, constants).\n- The output must be **ACCURATE**: Do not hallucinate features not implied by the raw query.\n- The output should be **NATURAL**: Do not just list the arguments; make them part of the narrative flow of the sentence.\n\nExample 1:\nRaw: "train model arguments: epochs=50, optimizer=\'adam\', learning_rate=0.001, early_stopping=True, log_dir=\'<temporary_directory>/logs\'"\nRewritten: "Train a model for 50 epochs using the Adam optimizer with a learning rate of 0.001, incorporating early stopping and saving logs specifically to the \'<temporary_directory>/logs\' directory."\n\nExample 2:\nRaw: "parse log arguments: file_path=\'<system_log_directory>/syslog\', pattern=\'ERROR\', output_format=\'json\', max_retries=3"\nRewritten: "Parse the log file located at \'<system_log_directory>/syslog\' to extract entries matching the \'ERROR\' pattern, performing up to 3 retries, and serialize the output to JSON format."\n'

def call_with_retry(fn, retries=5, base_delay=0.5, max_delay=8.0):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "RateLimit" not in msg and "upstream" not in msg:
                raise
            delay = min(max_delay, base_delay * (2 ** i)) * (1 + random.random() * 0.25)
            time.sleep(delay)
    return fn()

def refine_query(raw_query: str, modelname: str) -> str:
    prompt = refine_query_prompt.format(RAW_QUERY=raw_query)
    try:
        response = call_with_retry(lambda: get_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelname,
            temperature=0.3,
            top_p=0.95,
        ))
        content = response.choices[0].message.content or ""
        
        start_tag = "[start_of_rewritten_query]"
        end_tag = "[end_of_rewritten_query]"
        
        start_index = content.find(start_tag)
        end_index = content.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            refined_query = content[start_index + len(start_tag):end_index].strip()
            return refined_query
        else:
            return raw_query

    except Exception as e:
        print(f"Error refining query: {e}")
        return raw_query
