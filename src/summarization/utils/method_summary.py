"""
这个模块用于生成方法/函数描述
"""
import pandas as pd
import os
import numpy as np
import json
import time
import random
import re
import ast
import tokenize
from io import StringIO
from typing import List
from pydantic import BaseModel, ValidationError
from openai import OpenAI
from transformers import RobertaTokenizer, T5ForConditionalGeneration

from model.models import Function
from utils.function_clustering import set_func_adj_matrix

# Load API configuration from environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the client lazily to avoid import-time failures
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

# Prompt template for generating function summaries
Func_summary_template = """\nYou are a software engineer who is reverse engineering the code in a system to extract its design requirements and functional descriptions. 
code is below:
{code}
A. Project context positioning
- Module Attribution analysis:
{folder_structure}
- Dependency graph:
{dependence}
B. Function functionality destructuring
- Input/output parameter analysis: List and explain the data structure and purpose of all inputs and outputs
- Core logic flowchart: flowchart describing key processing steps in natural language
- Exception handling mechanism: error handling logic and boundary conditions are identified and illustrated
- If you can figure out the action object of the function, specify it; if you can't, leave it out
# Task:
-Give a description of the function based on the AB step, and extrapolate the functional requirements from the code implementation
-Non-functional requirements identification: infer non-functional requirements such as performance and security as reflected by code constraints
-Answer exactly as the code says. Don't introduce extra information
-Use the following format to answer:
Please return the response in the following JSON format:
{{
    "func_desc": "Function description",
    "func_flow": "Function flow",
    "func_notf": "Non-functional requirements",
}}
"""
class func_response(BaseModel):
    func_desc: str
    func_flow: str
    func_notf: str
    class Config:
        extra = "forbid"  # Strictly forbid extra fields

def extract_comments_from_code(code: str, language: str="python") -> str:
    """
    从代码中提取注释
    支持Python和Java的常见注释格式
    
    Args:
        code: 函数代码字符串
        language: 代码语言，支持"python"或"java"
        
    Returns:
        提取的注释文本，多个注释用换行符连接
    """
    # Type checking and conversion: ensure code is a string
    if code is None:
        return ""
    if not isinstance(code, str):
        # Try converting to a string
        try:
            code = str(code)
        except Exception:
            return ""
    
    # Handle an empty or whitespace-only string after conversion
    if not code or not code.strip():
        return ""
    
    comments = []
    
    if language == "python":
        # 1. Extract Python docstrings with ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        comments.append(docstring)
        except (SyntaxError, ValueError):
            # If the code is incomplete and cannot be parsed, continue with other extraction methods
            pass
        
        # 2. Extract Python single-line comments (# comment)
        try:
            tokens = tokenize.generate_tokens(StringIO(code).readline)
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    # Remove the comment marker (#) and surrounding whitespace
                    comment_text = tok.string.strip().lstrip('#').strip()
                    if comment_text and comment_text not in comments:
                        comments.append(comment_text)
        except (tokenize.TokenError, SyntaxError):
            # If tokenize fails, use a regular expression as a fallback
            single_line_comments = re.findall(r'#\s*(.+?)(?=\n|$)', code)
            for comment in single_line_comments:
                comment = comment.strip()
                if comment and comment not in comments:
                    comments.append(comment)
                
    elif language == "java":
        # 1. Extract JavaDoc comments (/** ... */)
        javadoc_comments = re.findall(r'/\*\*\s*(.+?)\s*\*/', code, re.DOTALL)
        for comment in javadoc_comments:
            # Clean JavaDoc comments by removing leading asterisks and indentation from each line
            lines = []
            for line in comment.split('\n'):
                cleaned_line = line.strip().lstrip('*').strip()
                if cleaned_line:
                    lines.append(cleaned_line)
            cleaned_comment = '.'.join(lines).strip()
            if cleaned_comment and cleaned_comment not in comments:
                comments.append(cleaned_comment)
        
        # 2. Extract multiline comments (/* ... */)
        multiline_comments = re.findall(r'/\*\s*(.+?)\s*\*/', code, re.DOTALL)
        for comment in multiline_comments:
            # Clean multiline comments by removing leading asterisks from each line
            lines = []
            for line in comment.split('\n'):
                cleaned_line = line.strip().lstrip('*').strip()
                if cleaned_line:
                    lines.append(cleaned_line)
            cleaned_comment = '.'.join(lines).strip()
            if cleaned_comment and cleaned_comment not in comments:
                comments.append(cleaned_comment)
        
        # 3. Extract single-line comments (// comment)
        java_single_comments = re.findall(r'//\s*(.+?)(?=\n|$)', code)
        for comment in java_single_comments:
            comment = comment.strip()
            if comment and comment not in comments:
                comments.append(comment)
    
    # Merge all comments into a string
    if comments:
        return '.'.join(comments)
    else:
        return ""



def generate_function_description_by_comment(functions: List[Function], language:str="python") -> List[Function]:
    """
    从函数代码中提取注释作为函数描述
    
    Args:
        functions: 函数列表
        language: 代码语言，支持"python"或"java"
        
    Returns:
        更新后的函数列表，func_desc字段包含提取的注释
    """
    for function in functions:
        function.func_desc = extract_comments_from_code(function.func_code, language=language)
        print(f"Function ID: {function.func_id}, Name: {function.func_name}, Description: {function.func_desc}")
    return functions


def generate_function_descriptions(functions:List[Function], modelname:str="deepseek-v3.2", method_adj_matrix: np.ndarray=None, language:str="python") -> List[Function]:
    
    for function in functions:
        function.func_desc = extract_comments_from_code(function.func_code, language=language)
        if function.func_desc != "":
            continue
        row = method_adj_matrix[function.func_id]
        dependence_parts = []
        for j in np.nonzero(row)[0]:
            if j == function.func_id:
                continue
            dependence_parts.append(
                functions[j].func_fullName + "\n" + functions[j].func_code + "\n"
            )
        dependence = "".join(dependence_parts)
        prompt = Func_summary_template.format(
            code=function.func_code,
            folder_structure=function.func_fullName,
            dependence=dependence
        )
        try:
            response = call_with_retry(lambda: get_client().chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=modelname,
                response_format={"type": "json_object"},
                temperature=0.3, 
                top_p=0.95, 
                frequency_penalty=0.5, 
                presence_penalty=0.2 
            ))
            json_str = response.choices[0].message.content
            json_str = json_str.replace("```json", "").replace("```", "")
            result = func_response.model_validate(json.loads(json_str)) 
            function.func_desc = result.func_desc
            function.func_flow = result.func_flow
            function.func_notf = result.func_notf
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            function.func_desc = function.func_name.split(".")[-1]
            function.func_flow = function.func_code
            function.func_notf = ""
        except ValidationError as e:
            print(f"Pydantic验证失败: {e}")
            function.func_desc = function.func_name.split(".")[-1]
            function.func_flow = function.func_code
            function.func_notf = ""
        except Exception as e:
            print(f"其他错误: {e}")
            function.func_desc = function.func_name.split(".")[-1]
            function.func_flow = function.func_code
            function.func_notf = ""
        # Print function descriptions
        print(f"Function ID: {function.func_id}, Name: {function.func_name}")
        print(f"Description: {function.func_desc}")
        print(f"Flow: {function.func_flow}")
        print(f"Non-functional requirements: {function.func_notf}")

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

def function_name_summary(functions: List[Function]) -> List[Function]:
    for function in functions:
        function.func_desc = function.func_fullName
    return functions

def function_file_name_summary(functions: List[Function]) -> List[Function]:
    for function in functions:
        # Use file name, function name, and parameter names
        params = '('+function.func_fullName.split("(")[-1]
        method_name = function.func_fullName.split("(")[0].split(".")[-1]
        file_name = function.func_fullName.split("(")[0].split(".")[-2]
        function.func_desc = f"{file_name}.{method_name}{params}"
        print(function.func_desc)  # DEBUG
    return functions

def code_t5_summary_all_gen(functions, language="python", batch_size=16):
    from utils.content_cache import cache_path, read, write
    pending = []
    paths = []
    for f in functions:
        payload = dict(model=os.getenv('FEATLENS_CODET5_MODEL', 'Salesforce/codet5-base-multi-sum'),
                       protocol='all-code-max-input512-output40-v1', language=language,
                       text=f.func_code if str(f.func_code).strip() else f.func_name)
        p = cache_path('codet5', payload)
        cached = read(p)
        if cached is not None:
            f.func_desc = cached['description']
        else:
            pending.append(f); paths.append(p)
    if pending:
        _code_t5_summary_all_gen_uncached(pending, language=language, batch_size=batch_size)
        for f,p in zip(pending,paths): write(p, {'description': f.func_desc})
    print(f"CodeT5 cache: {len(functions)-len(pending)} hits, {len(pending)} generated", flush=True)
    return functions

def _code_t5_summary_all_gen_uncached(functions: List[Function], language: str = "python", batch_size: int = 16) -> List[Function]:
    import torch
    tokenizer = RobertaTokenizer.from_pretrained(os.getenv('FEATLENS_CODET5_MODEL', 'Salesforce/codet5-base-multi-sum'))
    model = T5ForConditionalGeneration.from_pretrained(os.getenv('FEATLENS_CODET5_MODEL', 'Salesforce/codet5-base-multi-sum'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Collect all functions and generate in one batch; use the function name as model input when code is empty
    to_generate_indices: List[int] = []
    texts: List[str] = []
    max_length = 512  # Common input limit for CodeT5

    for idx, function in enumerate(functions):
        text = function.func_code
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if not text.strip():
            # Still call the model, using the function name as the input prompt
            text = function.func_name or ""
        to_generate_indices.append(idx)
        texts.append(text)

    if not texts:
        return functions

    # Warn about length truncation, using the length excluding special tokens as the more accurate measure
    lengths = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for i, ids in enumerate(lengths):
        if len(ids) > max_length:
            fn = functions[to_generate_indices[i]].func_name
            print(f"警告: 函数 {fn} 的代码过长 ({len(ids)} tokens)，已截断至 {max_length} tokens")

    # Encode, infer, and decode in batches
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            batch_indices = to_generate_indices[start:end]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=40,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, desc in enumerate(decoded):
                functions[batch_indices[i]].func_desc = desc
                print(desc)

    return functions

def code_t5_summary(functions: List[Function], language: str = "python", batch_size: int = 16) -> List[Function]:
    import torch
    tokenizer = RobertaTokenizer.from_pretrained(os.getenv('FEATLENS_CODET5_MODEL', 'Salesforce/codet5-base-multi-sum'))
    model = T5ForConditionalGeneration.from_pretrained(os.getenv('FEATLENS_CODET5_MODEL', 'Salesforce/codet5-base-multi-sum'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preliminarily fill descriptions with the comment extractor and collect the functions that still need generation
    to_generate_indices: List[int] = []
    texts: List[str] = []
    max_length = 512  # Common input limit for CodeT5

    for idx, function in enumerate(functions):
        function.func_desc = extract_comments_from_code(function.func_code, language=language)
        if function.func_desc:
            continue
        text = function.func_code
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if not text or not text.strip():
            # Use the function name as a fallback when code is empty
            function.func_desc = function.func_name
            continue
        to_generate_indices.append(idx)
        texts.append(text)

    if not texts:
        return functions

    # Warn about length truncation, using the length excluding special tokens as the more accurate measure
    lengths = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for i, ids in enumerate(lengths):
        if len(ids) > max_length:
            fn = functions[to_generate_indices[i]].func_name
            print(f"警告: 函数 {fn} 的代码过长 ({len(ids)} tokens)，已截断至 {max_length} tokens")

    # Encode, infer, and decode in batches
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            batch_indices = to_generate_indices[start:end]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=40,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for i, desc in enumerate(decoded):
                functions[batch_indices[i]].func_desc = desc
                print(desc)

    return functions

def method_summary(output_dir: str, strategy: str, language:str="python") -> List[Function]:
    method_df = pd.read_csv(os.path.join(output_dir, "methods.csv"))
    functions = []

    for index, row in method_df.iterrows():
        function_fullName = row["method_signature"]
        function_name = function_fullName.split("(")[0].split(".")[-1]
        # Extract the class or module name, which may be the penultimate or final segment
        parts = function_fullName.split("(")[0].split(".")
        func_file = parts[-2] if len(parts) >= 2 else parts[-1]
        
        # Ensure method_code is a string
        method_code = row["method_code"]
        if pd.isna(method_code):
            method_code = ""
        elif not isinstance(method_code, str):
            method_code = str(method_code)
        
        function = Function(
            func_id=row["ID"],
            func_name=function_name,
            func_desc="",
            func_file=func_file,
            func_flow="",
            func_notf="",
            func_code=method_code,
            func_fullName=function_fullName,
            func_txt_vector=[]
        )
        functions.append(function)
    print(f"Total functions: {len(functions)}")
    #input("please input to continue")
    # Load the adjacency matrix
    # Read CSV into a NumPy array and ensure integer type
    try:
        func_adj_matrix_df = pd.read_csv(os.path.join(output_dir, 'method_adj_matrix.csv'), header=None)
        # Convert to integer type, handling NaN or string values
        func_adj_matrix_df = func_adj_matrix_df.fillna(0).astype(int)
        func_adj_matrix = func_adj_matrix_df.to_numpy()
        
        # Validate that the matrix dimensions match the number of functions
        expected_size = len(functions)
        if func_adj_matrix.shape[0] != expected_size or func_adj_matrix.shape[1] != expected_size:
            print(f"警告: 矩阵大小 ({func_adj_matrix.shape}) 与函数数量 ({expected_size}) 不匹配，将调整矩阵大小")
            # Truncate an oversized matrix or zero-pad an undersized one
            if func_adj_matrix.shape[0] > expected_size:
                func_adj_matrix = func_adj_matrix[:expected_size, :expected_size]
            elif func_adj_matrix.shape[0] < expected_size:
                # Create a new matrix filled with zeros
                new_matrix = np.zeros((expected_size, expected_size), dtype=int)
                new_matrix[:func_adj_matrix.shape[0], :func_adj_matrix.shape[1]] = func_adj_matrix
                func_adj_matrix = new_matrix
        
        set_func_adj_matrix(func_adj_matrix)
    except Exception as e:
        print(f"警告: 加载 method_adj_matrix.csv 失败: {e}")
        # Create an empty zero matrix if loading fails
        func_adj_matrix = np.zeros((len(functions), len(functions)), dtype=int)
        set_func_adj_matrix(func_adj_matrix)

    # Before generating function descriptions, check whether the current function code already has one
    for function in functions:
        if function.func_desc != "":
            continue
        else:
            function.func_desc = extract_comments_from_code(function.func_code, language=language)
            if function.func_desc != "":
                continue
            else:
                function.func_desc = function.func_name

    if strategy == "function_name":
        functions = function_name_summary(functions)
    elif strategy == "function_file_name":
        functions = function_file_name_summary(functions)
    elif strategy == "code_t5":
        functions = code_t5_summary_all_gen(functions, language=language)
    elif strategy == "llm":
        functions = generate_function_descriptions(functions, modelname=modelname, method_adj_matrix=method_adj_matrix, language=language)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    # Save functions to CSV
    functions_df = pd.DataFrame([function.__dict__ for function in functions])
    functions_df.to_csv(os.path.join(output_dir, "methods_with_desc.csv"), index=False)
    return functions