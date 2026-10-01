
# Locate the package from this file, not from the caller's working directory.
from pathlib import Path as _PortablePath
import sys as _portable_sys
_PORTABLE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents
                      if (p / "featlens_paths.py").is_file())
if str(_PORTABLE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_ROOT))
from featlens_paths import (package_path as _package_path,
                            external_path as _external_path,
                            model_location as _model_location,
                            evaluation_python as _evaluation_python,
                            temporary_path as _temporary_path)

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from structure_analsis.java.java_import_analyzer import JavaImportAnalyzer
from structure_analsis.java.java_method_analyzer import JavaMethodAnalyzer
from structure_analsis.python.ENRE_py.enre.__main__ import main as enre_main
from structure_analsis.python.python_analsis import PythonMethodAnalyzer

from model.models import Function, method_Cluster
from utils.file_operations import create_directory_summary, add_functions_to_files
from utils.file_clustering import find_best_resolution, save_to_file_cluster
from utils.function_clustering import cluster_all_functions_to_features, set_func_adj_matrix
from utils.feature_generation import generate_feature_description, merge_features_by_method_cluster, features_to_csv, generate_feature_description_parallel
from utils.method_summary import method_summary
import json
import logging


# Semantic strategy used during clustering; currently function_name, function_file_name, code_t5, or llm
STRATEGY = "code_t5"
# Whether to call the LLM concurrently when generating features
IS_PARALLEL = True
# Whether to generate feature descriptions
GENERATE_DESCRIPTION = True


def main(project_root: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Allow either a local path or a web URL; this example uses a local path
    if(project_root.startswith("http")):
        print("使用网页地址")
        # Clone repositories retrieved from the web into the repository directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repository_dir = os.path.join(script_dir, "repository")
        os.makedirs(repository_dir, exist_ok=True)
        repo_name = project_root.split("/")[-1].rstrip(".git")  # Remove the .git suffix when present
        repo_path = os.path.join(repository_dir, repo_name)
        
        # Check whether the directory already exists
        if os.path.exists(repo_path) and os.listdir(repo_path):
            print(f"项目目录已存在，使用已有目录: {repo_path}")
        else:
            # Run git clone and check its return code
            exit_code = os.system(f"git clone {project_root} {repo_path}")
            if exit_code != 0:
                print(f"错误: Git clone 失败 (退出码: {exit_code})")
                return {
                    "project_root": project_root,
                    "output_dir": output_dir,
                    "status": "failed",
                    "reason": f"git clone failed with exit code {exit_code}",
                    "total_functions": 0,
                    "total_features": 0,
                }
            # Check whether cloning succeeded and the directory is non-empty
            if not os.path.exists(repo_path) or not os.listdir(repo_path):
                print(f"错误: 克隆后目录为空或不存在: {repo_path}")
                return {
                    "project_root": project_root,
                    "output_dir": output_dir,
                    "status": "failed",
                    "reason": "cloned repository path is empty",
                    "total_functions": 0,
                    "total_features": 0,
                }
        
        project_root = repo_path
        print(f"项目已拉取到{repo_path}")
    else:
        print("使用本地地址")
        project_root = os.path.abspath(project_root)

    # Detect the project language
    has_java = any(f.endswith('.java') for root, _, files in os.walk(project_root) for f in files)
    has_python = any(f.endswith('.py') for root, _, files in os.walk(project_root) for f in files)
    
    # Analyze project structure
    if has_java:
        print("分析Java项目")
        file_analyzer = JavaImportAnalyzer()
        file_analyzer.analyze_project(project_root, output_dir)
        method_analyzer = JavaMethodAnalyzer()
        method_analyzer.analyze_project(project_root, output_dir)
    elif has_python:
        print("分析Python项目")
        # Analyze Python projects with ENRE-py
        enre_main([project_root, output_dir])
        # method_analyzer = PythonMethodAnalyzer()
        # method_analyzer.analyze_project(project_root, output_dir)
    else:
        print("未找到支持的代码文件（.java 或 .py）")
        return {
            "project_root": project_root,
            "output_dir": output_dir,
            "status": "skipped",
            "reason": "no supported source files (.java or .py)",
            "total_functions": 0,
            "total_features": 0,
        }

    # Generate function descriptions using function names (function_name), CodeT5 (code_t5), or an LLM (llm)
    language = "python" if has_python else "java"
    functions = method_summary(output_dir, strategy=STRATEGY, language=language)
    # Generate file descriptions; file names are used here
    files = create_directory_summary(project_root)
    add_functions_to_files(files, functions, language=language)
    # Optionally run in parallel; parallel execution may be unreliable on some servers, so disabling it is recommended
    is_parallel = IS_PARALLEL

    # Print selected file and function information
    for file in files[:5]:
        print(f"File ID: {file.file_id}, Name: {file.file_name}, Path: {file.file_path}, Description: {file.file_desc}")
        for function in file.func_list[:5]:
            print(f"  Function ID: {function.func_id}, Name: {function.func_name}, Description: {function.func_desc}")
        print("\n")
    
    # Generate text embeddings
    model = SentenceTransformer('all-mpnet-base-v2')
    for file in files:
        file.file_txt_vector = model.encode(file.file_desc).tolist()

    # File clustering
    best_gamma, best_labels, results = find_best_resolution(
        files,
        a=0.5,
        n_points=25,
        gamma_min=0.01, gamma_max=0.4,
        seeds_per_gamma=8,
        use_knn=True, knn_k=20,
        use_threshold=False, threshold_tau=0.0,
        min_clusters=3, max_clusters_ratio=0.15,
        min_cluster_size=3,
        use_silhouette=False,
    )

    # Set the log file
    logging.basicConfig(
        filename=os.path.join(output_dir, 'cluster_results.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        force=True,
    )

    def save_results_to_file(feature_list, summary, file_path):
        """保存聚类结果到文件"""
        results = {
            "features": [
                f.to_dict()
                for f in feature_list
            ]
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    clusters = save_to_file_cluster(files, best_labels)

    for c in clusters:
        log_message = f"Cluster ID: {c.cluster_id}, {len(c.cluster_file_list)} Files: {[file.file_name for file in c.cluster_file_list]}"
        print(log_message)
        logging.info(log_message)
    
    # Expand clusters to the function level
    method_clusters = []
    for cluster in clusters:
        func_list = []
        for file in cluster.cluster_file_list:
            for function in file.func_list:
                function.func_txt_vector = model.encode(function.func_desc).tolist()
                func_list.append(function)
                # 7. Create a method-cluster object (method_Cluster)
                #    - cluster.cluster_id: inherited from the file cluster ID
                #    - "": description is initially empty
                #    - func_list: all functions contained in this file cluster
        method_cluster = method_Cluster(cluster.cluster_id, "", func_list)
        method_clusters.append(method_cluster)
    
    for method_cluster in method_clusters:
        log_message = f"Cluster ID: {method_cluster.cluster_id}, Functions: {[f.func_name for f in method_cluster.cluster_func_list]}"
        print(log_message)
        logging.info(log_message)

    print("begin to cluster all functions to features")
    # Function clustering
    feature_list, summary = cluster_all_functions_to_features(
        method_clusters,
        weight_parameter=0.25,
        # gamma_min=0.005, gamma_max=0.15, n_points=24, #mrjob
        gamma_min=0.05, gamma_max=0.15, n_points=24, #boto
        #gamma_min=0.1, gamma_max=0.85, n_points=40,
        seeds_per_gamma=8,
        use_knn=True, knn_k=20,
        use_threshold=False, threshold_tau=0.0,
        min_clusters=2, max_clusters_ratio=0.15,
        use_silhouette=False, silhouette_sample_size=None,
        objective="CPM",
        consensus_tau=0.6, consensus_gamma=0.1,
        rng_seed=2025,
        target_total_features=None,
    )
    log_message = f"Total Features: {len(feature_list)}"
    print(log_message)
    #input("confirm the num of features")
    logging.info(log_message)
    
    for f in feature_list:
        log_message = f"Feature ID {f.feature_id}: {f.cluster_id} {len(f.feature_func_list)} {[x.func_fullName for x in f.feature_func_list]}"
        print(log_message)
        logging.info(log_message)
    
    # Save clustering results to a file
    save_results_to_file(feature_list, summary, os.path.join(output_dir, 'cluster_results.json'))
    
    modelname = "deepseek-v3.2"

    description_stats = {
        "failed_feature_ids": [],
        "failed_feature_count": 0,
        "fallback_feature_ids": [],
        "fallback_feature_count": 0,
    }

    # # Generate feature descriptions
    if GENERATE_DESCRIPTION:
        if is_parallel:
            description_stats = generate_feature_description_parallel(feature_list, modelname=modelname, max_workers=8)
        else:
            description_stats = generate_feature_description(feature_list, modelname=modelname)

    desc_log_message = (
        f"Description generation issues: {description_stats['failed_feature_count']} features had generation errors "
        f"(fallback used for {description_stats['fallback_feature_count']} features)."
    )
    print(desc_log_message)
    logging.info(desc_log_message)
    if description_stats["failed_feature_ids"]:
        failed_ids_message = f"Description generation error feature IDs: {description_stats['failed_feature_ids']}"
        print(failed_ids_message)
        logging.info(failed_ids_message)

    # Merge features
    merge_features_by_method_cluster(feature_list, method_clusters, modelname=modelname)

    # Save to CSV
    features_to_csv(feature_list, method_clusters, os.path.join(output_dir, "features.csv"))

    return {
        "project_root": project_root,
        "output_dir": output_dir,
        "status": "success",
        "reason": "",
        "total_functions": len(functions),
        "total_features": len(feature_list),
        "description_issue_features": description_stats["failed_feature_count"],
        "description_issue_feature_ids": description_stats["failed_feature_ids"],
        "description_fallback_features": description_stats["fallback_feature_count"],
    }

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    # Project path to summarize
    project_root = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', 'Multimedia/mingus'))
    # Directory in which to save RepoSummary results
    output_dir = str(_package_path('results/retrieval/deveval/featlens_and_rag/Multimedia/mingus'))

    main(
        project_root=project_root,
        output_dir=output_dir
    )
