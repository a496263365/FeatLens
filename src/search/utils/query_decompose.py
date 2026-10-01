import time
import json
import random
from openai import OpenAI
import os
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

decompose_query_prompt = """
You are an expert Software Architect and Code Retrieval Specialist.

<raw_query>
{RAW_QUERY}
</raw_query>

Task:
The provided <raw_query> describes a complex function that performs multiple sequential steps or logic operations.
Your task is to **decompose** this composite description into 3-5 distinct, atomic **Search Queries**.
Each search query will be used to retrieve specific functions from a codebase via semantic similarity search.

Instructions:
1. **Analyze the Logic Flow:** Break down the raw query into its constituent steps (e.g., Validation -> Retrieval -> Processing -> Logging -> Return).
2. **Formulate Atomic Summaries:** For each step, write a query that sounds like a **Function Summary** (Docstring).
    - Use code-oriented verbs: "Retrieve...", "Validate...", "Iterate...", "Check...", "Yield...".
3. **Entity Handling Strategy (CRITICAL):**
    - **PRESERVE Specific Names:** If the query mentions specific class names (e.g., `HadoopJobRunner`), module names, or unique system identifiers, **you MUST include them** in the sub-queries. These are critical search keywords.
    - **Generalize Generic Variables:** For common variable names (e.g., `output_dir`, `flag`, `i`), generalize them to their concepts (e.g., "target directory", "boolean flag").

Please answer in the following format:

[start_of_analysis]
<breakdown_of_logic_steps_and_key_entities>
[end_of_analysis]

[start_of_sub_queries]
1. <search_query_for_step_1>
2. <search_query_for_step_2>
...
[end_of_sub_queries]

Notes:
- **Max 3 queries.**
- Each query must describe a single functional responsibility.
- The queries should be optimized for finding function summaries in a vector database.
- **Do not** describe the parameters block (like `:param self:`) separately. Instead, weave the *type information* (e.g., "of the HadoopJobRunner instance") naturally into the functional description.

Example:
Raw: "This function yields lists of directories. It iterates over unique log directories obtained from the hadoop log directories. :param self: HadoopJobRunner."
Sub-queries:
1. Yield a generator of directory lists for processing.
2. Retrieve and iterate over unique log directories associated with the **HadoopJobRunner**.
"""

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

def decompose_query(raw_query: str, modelname: str) -> list[str]:
    prompt = decompose_query_prompt.format(RAW_QUERY=raw_query)
    try:
        response = call_with_retry(lambda: get_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=modelname,
            temperature=0.3,
            top_p=0.95,
        ))
        content = response.choices[0].message.content or ""
        
        start_tag = "[start_of_sub_queries]"
        end_tag = "[end_of_sub_queries]"
        
        start_index = content.find(start_tag)
        end_index = content.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            sub_queries_section = content[start_index + len(start_tag):end_index].strip()
            # Extract numbered list items
            queries = []
            for line in sub_queries_section.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() and line[1] == '.'):
                    # Remove the number and the dot
                    query = line.split('.', 1)[1].strip()
                    queries.append(query)
            return queries if queries else [raw_query]
        else:
            return [raw_query]

    except Exception as e:
        print(f"Error decomposing query: {e}")
        return [raw_query]
