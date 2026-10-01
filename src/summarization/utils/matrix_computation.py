import numpy as np
import re
from typing import List
from model.models import File

def compute_similarity_matrix(files: List[File]):
    # Extract file embeddings
    txt_vectors = [file.file_txt_vector for file in files]
    normalized_vectors = txt_vectors / np.linalg.norm(txt_vectors, axis=1, keepdims=True)
    # Compute the similarity matrix
    similarity_matrix = np.dot(normalized_vectors, np.transpose(normalized_vectors))
    # Normalize similarity-matrix values to the range 0-1
    similarity_matrix = (similarity_matrix + 1) / 2
    return similarity_matrix

def _extract_identifiers(code: str):
    # Extract Java/Python identifiers beginning with a letter or underscore
    return set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', code))

def compute_link_matrix(files: List[File]):
    link_matrix = np.zeros((len(files), len(files)))
    id_sets = {f.file_id: _extract_identifiers(f.file_code) for f in files}
    for file in files:
        name = file.file_name
        for other_file in files:
            if file.file_id == other_file.file_id:
                continue
            if (name in id_sets[other_file.file_id]) or (name == other_file.file_name):
                link_matrix[file.file_id][other_file.file_id] = 1
                link_matrix[other_file.file_id][file.file_id] = 1
    return link_matrix

