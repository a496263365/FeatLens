
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
import networkx as nx
from pyvis.network import Network
from flask import Flask, render_template_string, request, send_file
import glob
import re
import json

app = Flask(__name__)

# BASE_DIR = '<package_root>/results'
BASE_DIR = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/graphs'))
# Hardcoded path to the filtered jsonl file containing ground truth
FILTERED_JSONL_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))

def load_ground_truth(task_id):
    """
    Loads ground truth methods for a specific task ID from the JSONL file.
    Returns a set of ground truth signatures.
    """
    gt_sigs = set()
    if not os.path.exists(FILTERED_JSONL_PATH):
        print(f"Warning: Filtered JSONL file not found at {FILTERED_JSONL_PATH}")
        return gt_sigs

    try:
        target_line_num = int(task_id)
        current_line_num = 0
        with open(FILTERED_JSONL_PATH, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                current_line_num += 1
                
                # Check if this is the target line (1-based index)
                if current_line_num == target_line_num:
                    data = json.loads(line)
                    dependency = data.get('dependency', {})
                    gt_sigs.update(dependency.get('intra_class', []))
                    gt_sigs.update(dependency.get('intra_file', []))
                    gt_sigs.update(dependency.get('cross_file', []))
                    break
    except ValueError:
        print(f"Error: Invalid task_id '{task_id}' - must be an integer.")
    except Exception as e:
        print(f"Error reading filtered JSONL: {e}")
        
    return gt_sigs

def find_gml_files():
    gml_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.endswith('.gml'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, BASE_DIR)
                gml_files.append(rel_path)
    return sorted(gml_files)

@app.route('/')
def index():
    files = find_gml_files()
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GML Visualizer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            ul { list-style-type: none; padding: 0; }
            li { margin: 5px 0; }
            a { text-decoration: none; color: #007bff; }
            a:hover { text-decoration: underline; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Select a GML file to visualize</h1>
            <ul>
                {% for file in files %}
                <li><a href="/view/{{ file }}">{{ file }}</a></li>
                {% endfor %}
            </ul>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, files=files)

@app.route('/view/<path:filename>')
def view_graph(filename):
    file_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(file_path):
        return "File not found", 404

    try:
        G = nx.read_gml(file_path)
    except Exception as e:
        return f"Error reading GML file: {e}", 500

    # Extract Task ID from filename (e.g., task_77_mid.gml -> 77)
    basename = os.path.basename(filename)
    match = re.search(r'task_(\d+)_', basename)
    task_id = match.group(1) if match else None
    print(f"task_id: {task_id}")
    
    gt_sigs = set()
    if task_id:
        gt_sigs = load_ground_truth(task_id)
        print(f"gt_sigs: {gt_sigs}")

    # Metrics for matching
    matched_gt = set()
    total_gt = len(gt_sigs)
    print(f"total_gt: {total_gt}")

    # Create Pyvis network
    # Using CDN for resources to avoid local dependency issues and ensure TomSelect is available
    # Changed background to white and font color to black as requested
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", select_menu=True, filter_menu=True, cdn_resources='remote')
    
    # Process nodes for visualization
    for node_id in G.nodes():
        node = G.nodes[node_id]
        
        # Determine if node matches Ground Truth
        is_gt_match = False
        node_sig = node.get('sig')
        
        # Normalize signature logic (similar to analyze script, removing prefix if needed)
        # For simplicity, we check if node_sig is in gt_sigs or if a suffix matches
        # Assume an exact match for now, following the request to extract and match the value of ...sig
        if node_sig:
             candidates = {node_sig}
             if '(' in node_sig:
                 candidates.add(node_sig.split('(')[0])

             expanded_candidates = set()
             for cand in candidates:
                 expanded_candidates.add(cand)
                #  if '.' in cand:
                #      parts = cand.split('.')
                #      if len(parts) > 1:
                #          expanded_candidates.add('.'.join(parts[1:]))
                #      if len(parts) > 2 and parts[0] == parts[1]:
                #          expanded_candidates.add('.'.join(parts[2:]))

             matched_sig = next((c for c in expanded_candidates if c in gt_sigs), None)
             if matched_sig is not None:
                 is_gt_match = True
                 matched_gt.add(matched_sig)

        # Set color for GT nodes
        if is_gt_match:
            node['color'] = 'red'
            node['title'] = (node.get('title', '') + "<br><b>STATUS:</b> GROUND TRUTH MATCH").strip()
            # Make GT nodes slightly larger
            node['size'] = 15
        else:
            node['color'] = '#97c2fc' # Default blue-ish
            node['size'] = 10

        
        # Set label to sig if available, else use existing label or id
        if 'sig' in node:
            sig_str = str(node['sig'])
            # Simplify sig: split by '.' and keep last two parts
            parts = sig_str.split('.')
            if len(parts) >= 2:
                node['label'] = f"{parts[-2]}.{parts[-1]}"
            else:
                node['label'] = sig_str
        
        # Construct title (tooltip) with all attributes
        title_parts = []
        for key, value in node.items():
            # Format long values (like code) to be more readable if needed
            val_str = str(value)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            title_parts.append(f"<b>{key}:</b> {val_str}")
        
        node['title'] = "<br>".join(title_parts)
        
        # Update node in graph
        G.nodes[node_id].update(node)

    # Process edges for visualization
    for u, v, data in G.edges(data=True):
        # Add edge type label if available (assuming 'type' or similar attribute exists, or just show all attributes in title)
        # If you specifically want to show a label on the edge, pyvis uses 'label' attribute
        # Let's check if there's a specific attribute for edge type. 
        # If not specified, we might just look for common ones or just put everything in title.
        # But user asked to "show edge type on edge". Let's assume there is an attribute that represents type.
        # Common GML edge attributes might be 'label', 'relation', 'type', etc.
        # Without seeing edge data, I will try to find a likely candidate or just iterate all to find something meaningful.
        # For now, let's put all data in title, and if there is a 'type' or 'relation' or 'label', use it as label.
        
        edge_title_parts = []
        label_candidate = None
        
        for key, value in data.items():
             edge_title_parts.append(f"<b>{key}:</b> {value}")
             if key.lower() in ['type', 'relation', 'rel', 'label']:
                 label_candidate = str(value)
        
        data['title'] = "<br>".join(edge_title_parts)
        if label_candidate:
            data['label'] = label_candidate
            
        G.edges[u, v].update(data)

    net.from_nx(G)
    
    # Add stats to the UI (using a custom heading injection)
    match_count = len(matched_gt)
    missing_gt = gt_sigs - matched_gt
    missing_count = len(missing_gt)
    
    # Helper to format list for HTML
    def format_list(items):
        if not items:
            return "<i>None</i>"
        return "<ul style='margin: 5px 0; padding-left: 20px; font-size: 10px; max-height: 100px; overflow-y: auto;'>" + \
               "".join([f"<li>{item}</li>" for item in sorted(items)]) + "</ul>"

    # Set options for better physics/layout
    net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 12
        },
        "shape": "dot"
      },
      "edges": {
        "color": {
          "color": "#848484",
          "inherit": false
        },
        "smooth": false
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      }
    }
    """)
    
    html_content = net.generate_html()
    
    # Inject stats into the HTML body
    # Moved to bottom-left (bottom: 10px, left: 10px)
    # Added collapsible details (<details>)
    stats_html = f"""
    <div style="position: absolute; bottom: 10px; left: 10px; z-index: 1000; background: rgba(255, 255, 255, 0.9); padding: 10px; border: 1px solid #ccc; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); max-width: 300px; max-height: 80vh; overflow-y: auto;">
        <h3 style="margin-top: 0; font-size: 16px;">Analysis for Task {task_id}</h3>
        
        <details>
            <summary style="cursor: pointer; font-weight: bold;">Ground Truth Total: {total_gt}</summary>
            {format_list(gt_sigs)}
        </details>
        
        <details>
            <summary style="cursor: pointer; font-weight: bold; color: red;">Matched (In Graph): {match_count}</summary>
            {format_list(matched_gt)}
        </details>
        
        <details open>
            <summary style="cursor: pointer; font-weight: bold;">Missing: {missing_count}</summary>
            {format_list(missing_gt)}
        </details>
    </div>
    """
    
    # Insert stats after <body> tag
    html_content = html_content.replace('<body>', '<body>' + stats_html)
    
    return html_content

if __name__ == '__main__':
    # Use 0.0.0.0 to be accessible if needed, though usually localhost is fine in this env
    app.run(host='0.0.0.0', port=5090, debug=True)
