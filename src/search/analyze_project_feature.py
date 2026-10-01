
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
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Path to generated features
FEATURE_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/Internet/boto/features.csv'))
# Path to the DevEval case JSON file, not to the dataset repository itself
DATA_JSONL = str(_package_path('data/deveval/tasks_with_dependencies_1146.jsonl'))


def analyze_project(project_path, output_path):
    # Create output directory under output_path using last two segments of project_path
    norm_path = os.path.normpath(project_path)
    parts = [p for p in norm_path.split(os.sep) if p]
    last_two = parts[-2:] if len(parts) >= 2 else parts
    dir_name = "_".join(last_two)
    output_dir = os.path.join(output_path, dir_name)
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Filter by project_path
    filtered_file = f"{output_dir}/filtered.jsonl"
    with open(DATA_JSONL, 'r') as infile, open(filtered_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get('project_path') == '/'.join(last_two):
                outfile.write(line)
    
    # Step 2: Find similar clusters
    df = pd.read_csv(FEATURE_CSV)
    clusters = df.groupby('id')['desc'].first().reset_index()
    # Normalize method_name by removing arguments and retaining the part after the first dot, e.g., mrjob.hadoop.main -> hadoop.main
    base_names = df['method_name'].astype(str).str.split('(').str[0]
    df['method_name_norm'] = base_names.str.split('.', n=1).str[1].fillna(base_names)
    method_names = df['method_name_norm'].unique().tolist()
    print(method_names)

    #model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_embeddings = model.encode(clusters['desc'].tolist())
    
    with_clusters_file = f"{output_dir}/with_clusters.jsonl"
    with open(filtered_file, 'r') as f, open(with_clusters_file, 'w') as outfile:
        # Record P/R/F1 for query top1, top3, and top5
        top_1_match = 0
        top_3_match = 0
        top_5_match = 0
        top_1_pred = 0
        top_3_pred = 0
        top_5_pred = 0
        top_gt = 0

        for line in f:
            data = json.loads(line.strip())
            deps = []
            deps.extend(data['dependency']['intra_class'])
            deps.extend(data['dependency']['intra_file'])
            deps.extend(data['dependency']['cross_file'])
            deps = [dep for dep in deps if dep in method_names]

            top_gt += len(deps)

            query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]
            top_1_indices = np.argsort(similarities)[-1:][::-1]

            top_3_indices = np.argsort(similarities)[-3:][::-1]

            top_5_indices = np.argsort(similarities)[-5:][::-1]

            # Compute top1: obtain IDs from clusters, then method names from the original dataframe
            top_1_cluster_ids = clusters.iloc[top_1_indices]['id'].tolist()
            top_1_methods = []
            for cluster_id in top_1_cluster_ids:
                top_1_methods.extend(df[df['id'] == cluster_id]['method_name_norm'].tolist())
            top_1_pred += len(top_1_methods)
            for dep in deps:
                for method in top_1_methods:
                    if dep == method:
                        top_1_match += 1
                        break
            # Compute top3: obtain IDs from clusters, then method names from the original dataframe
            top_3_cluster_ids = clusters.iloc[top_3_indices]['id'].tolist()
            top_3_methods = []
            for cluster_id in top_3_cluster_ids:
                top_3_methods.extend(df[df['id'] == cluster_id]['method_name_norm'].tolist())
            top_3_pred += len(top_3_methods)
            for dep in deps:
                for method in top_3_methods:
                    if dep == method:
                        top_3_match += 1
                        break
            # Compute top5: obtain IDs from clusters, then method names from the original dataframe
            top_5_cluster_ids = clusters.iloc[top_5_indices]['id'].tolist()
            top_5_methods = []
            for cluster_id in top_5_cluster_ids:
                top_5_methods.extend(df[df['id'] == cluster_id]['method_name_norm'].tolist())
            top_5_pred += len(top_5_methods)
            for dep in deps:
                for method in top_5_methods:
                    if dep == method:
                        top_5_match += 1
                        break
            
            
        print(f"Top 1 Match: {top_1_match}")
        print(f"Top 1 Pred: {top_1_pred}")
        print(f"Top 1 P={(top_1_match/top_1_pred)*100:.2f}%")
        print(f"Top 1 R={(top_1_match/top_gt)*100:.2f}%")
        print(f"Top 1 F1={(2* (top_1_match/top_1_pred) * (top_1_match/top_gt) / ((top_1_match/top_1_pred) + (top_1_match/top_gt)))*100:.2f}%")
        print("--------------------------------")
        print(f"Top 3 Match: {top_3_match}")
        print(f"Top 3 Pred: {top_3_pred}")
        print(f"Top 3 P={(top_3_match/top_3_pred)*100:.2f}%")
        print(f"Top 3 R={(top_3_match/top_gt)*100:.2f}%")
        print(f"Top 3 F1={(2* (top_3_match/top_3_pred) * (top_3_match/top_gt) / ((top_3_match/top_3_pred) + (top_3_match/top_gt)))*100:.2f}%")
        print("--------------------------------")
        print(f"Top 5 Match: {top_5_match}")
        print(f"Top 5 Pred: {top_5_pred}")
        print(f"Top 5 P={(top_5_match/top_5_pred)*100:.2f}%")
        print(f"Top 5 R={(top_5_match/top_gt)*100:.2f}%")
        print(f"Top 5 F1={(2* (top_5_match/top_5_pred) * (top_5_match/top_gt) / ((top_5_match/top_5_pred) + (top_5_match/top_gt)))*100:.2f}%")
        print("--------------------------------")
    # # Step 3: Count dependency matches
    # results = []
    # with open(with_clusters_file, 'r') as f:
    # total_gt_unfiltered = 0  # number of ground-truth dependencies
    # total_gt = 0  # number of ground-truth dependencies
    # total_pred = 0  # number of dependencies predicted by top1, top2, and top3

    # total_match = 0  # number of matches for top1, top2, and top3

    #     for line in f:
    #         data = json.loads(line.strip())
            
    #         deps = []
    #         deps.extend(data['dependency']['intra_class'])
    #         deps.extend(data['dependency']['intra_file'])
    #         deps.extend(data['dependency']['cross_file'])
    #         total_gt_unfiltered += len(deps)
    # Map dependencies to method_names and keep only names that exist
    #         deps = [dep for dep in deps if dep in method_names]
    #         total_gt += len(deps)

    #         length_deps = len(deps)
    #         cluster_coverage = {}
            
    #         for cluster_id in data['top_3_clusters']:
    #             cluster_methods = df[df['id'] == cluster_id]['method_name'].tolist()
    # Function names in dep here are [mrjob.hadoop.HadoopJobRunner._hadoop_log_dirs, mrjob.hadoop.HadoopJobRunner.fs]
    # Function names in cluster_methods here are [mrjob.hadoop.HadoopJobRunner._hadoop_log_dirs(self, output_dir), mrjob.hadoop.HadoopJobRunner.fs(self)]
    #             match_count = 0
    #             total_pred += len(cluster_methods)
    #             for dep in deps:
    #                 for method in cluster_methods:
    #                     if dep in method:
    #                         match_count += 1
    #                         break
    #             total_match += match_count
    #             coverage_ratio = match_count / length_deps if length_deps > 0 else 0
    #             cluster_coverage[cluster_id] = coverage_ratio
            
    #         result = {
    #             'namespace': data['namespace'],
    #             'total_dependencies': length_deps,
    #             'top_3_clusters': data['top_3_clusters'],
    #             'cluster_coverage_ratio': cluster_coverage
    #         }
    #         results.append(result)
    # print(f"Total GT Unfiltered: {total_gt_unfiltered}")
    # print(f"Total GT: {total_gt}")
    # print(f"Total Pred: {total_pred}")
    # print(f"Total Match: {total_match}")
    # print(f"P={(total_match/total_pred)*100:.2f}%")
    # print(f"R={(total_match/total_gt)*100:.2f}%")
    # print(f"F1={(2* (total_match/total_pred) * (total_match/total_gt) / ((total_match/total_pred) + (total_match/total_gt)))*100:.2f}%")
    # # Save final results
    # analysis_file = f"{output_dir}/analysis.jsonl"
    # with open(analysis_file, 'w') as f:
    #     for result in results:
    #         f.write(json.dumps(result) + '\n')
    
    print(f"Analysis completed for {project_path}")
    print(f"Results saved in {output_dir}/ directory")


if __name__ == "__main__":
    # Path to the project source code
    project_path = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', 'Internet/boto'))
    # Output path; currently unused
    output_path = str(_package_path('results/devEvalSearchOut'))
    analyze_project(project_path, output_path)
