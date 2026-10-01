import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

# Add the parent directory to sys.path so the enre module can be imported
# Obtain the directory of the current file, the enre directory
current_dir = Path(__file__).parent
# Obtain the parent directory, the ENRE-py directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from enre.analysis.analyze_manager import AnalyzeManager
from enre.cfg.Resolver import Resolver
from enre.cfg.module_tree import Scene
from enre.passes.aggregate_control_flow_info import aggregate_cfg_info
from enre.vis.representation import DepRepr
from enre.vis.summary_repr import from_summaries, call_graph_representation
from enre.ent.EntKind import EntKind


def main(args=None) -> None:
    """
    主函数，支持从命令行调用和从 Python 代码调用
    
    Args:
        args: 可选参数列表，如果为 None 则从 sys.argv 读取
              如果提供列表，则使用该列表作为参数（例如: [project_root, output_dir]）
    """
    # Temporarily modify sys.argv when an argument list is provided
    original_argv = sys.argv[:]
    if args is not None:
        # Convert the args list to sys.argv format
        sys.argv = [sys.argv[0]] + [str(arg) for arg in args]
    
    # Extract output_dir first as the second positional argument to avoid argparse errors
    output_dir = None
    if args is not None and len(args) >= 2:
        output_dir = Path(args[1]) if args[1] else None
        # Remove the second argument from sys.argv to avoid argparse errors
        if len(sys.argv) > 2:
            sys.argv = sys.argv[:2]
    elif len(sys.argv) > 2:
        # Treat the second argument as output_dir when it does not start with --
        if not sys.argv[2].startswith('--'):
            output_dir = Path(sys.argv[2])
            # Temporarily remove the second argument to avoid argparse errors
            sys.argv = sys.argv[:2] + sys.argv[3:]
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("root path", type=str, nargs='?',
                            help="root package path")
        parser.add_argument("--profile", action="store_true", help="output consumed time in json format")
        parser.add_argument("--cfg", action="store_true",
                            help="run control flow analysis and output module summaries")
        parser.add_argument("--compatible", action="store_true", help="output compatible format")
        parser.add_argument("--builtins", action="store", help="builtins module path")
        parser.add_argument("--cg", action="store_true", help="dump call graph in json")
        config = parser.parse_args()
        
        # Obtain root_path
        if args is not None and len(args) >= 1:
            root_path = Path(args[0]).resolve()
        elif len(sys.argv) > 1:
            root_path = Path(sys.argv[1]).resolve()
        else:
            print("错误: 需要提供项目根路径")
            return
        
        start = time.time()
        manager = enre_wrapper(root_path, config.compatible, config.cfg, config.cg, config.builtins, output_dir=output_dir)
        end = time.time()

        if config.profile:
            time_in_json = json.dumps({
                "analyzed files": len(manager.root_db.tree),
                "analysing time": end - start})
            print(time_in_json)
            # print(f"analysing time: {end - start}s")
    finally:
        # Restore the original sys.argv
        sys.argv = original_argv


def dump_call_graph(project_name: str, resolver: Resolver) -> None:
    call_graph = call_graph_representation(resolver)
    out_path = f"{project_name}-call-graph-enre.json"
    with open(out_path, "w") as file:
        json.dump(call_graph, file, indent=4)


def enre_wrapper(root_path: Path, compatible_format: bool, need_cfg: bool, need_call_graph: bool,
                 builtin_module: str, output_dir: Path = None) -> AnalyzeManager:
    project_name = root_path.name
    builtins_path = Path(builtin_module) if builtin_module else None
    manager = AnalyzeManager(root_path, builtins_path)
    manager.work_flow()

    # The ENRE report is saved as report-enre.json under output_dir without a project-name prefix to avoid later case-sensitivity issues
    out_path = output_dir / "report-enre.json"

    if need_cfg:
        print("dependency analysis finished, now running control flow analysis")
        resolver = cfg_wrapper(root_path, manager.scene)
        print("control flow analysis finished")
        aggregate_cfg_info(manager.root_db, resolver)
        if need_call_graph:
            dump_call_graph(project_name, resolver)

    with open(out_path, "w") as file:
        if not compatible_format:
            json.dump(DepRepr.from_package_db(manager.root_db).to_json_1(), file, indent=4)
        else:
            repr = DepRepr.from_package_db(manager.root_db).to_json()
            json.dump(repr, file, indent=4)

    # Generate the CSV file
    dep_repr = DepRepr.from_package_db(manager.root_db)
    # root_path is the project root used during analysis and should contain the source files
    generate_csv_files(dep_repr, manager.root_db, root_path, output_dir)

    return manager


def cfg_wrapper(root_path: Path, scene: Scene) -> Resolver:
    resolver = Resolver(scene)
    resolver.resolve_all()
    out_path = Path(f"{root_path.name}-report-cfg.txt")
    with open(out_path, "w") as file:
        summary_repr = from_summaries(scene.summaries)
        file.write(summary_repr)
    return resolver


def extract_method_code(file_path: str, start_line: int, end_line: int, project_root: Path = None) -> str:
    """从文件中提取指定行号的代码"""
    if start_line <= 0:
        return ""
    
    try:
        # Try several path-resolution strategies
        path = None
        tried_paths = []
        
        # 1. Try the direct path, either absolute or relative to the current directory
        test_path = Path(file_path)
        tried_paths.append(str(test_path.absolute()))
        if test_path.exists():
            path = test_path
        
        # 2. Try a path relative to the project root
        if not path and project_root:
            potential_path = project_root / file_path
            print(f"potential_path: {potential_path}")
            tried_paths.append(str(potential_path.absolute()))
            if potential_path.exists():
                path = potential_path
        
        # 3. Try normalizing the path separators
        if not path and project_root:
            # Normalize according to the project root's path separator
            if "\\" in str(project_root):
                normalized_path = file_path.replace("/", "\\")
            else:
                normalized_path = file_path.replace("\\", "/")
            potential_path = project_root / normalized_path
            tried_paths.append(str(potential_path.absolute()))
            if potential_path.exists():
                path = potential_path
        
        if not path or not path.exists():
            # If all paths fail, try finding the file case-insensitively
            if project_root:
                file_name = Path(file_path).name
                for py_file in project_root.rglob("*.py"):
                    if py_file.name.lower() == file_name.lower():
                        # Check whether paths match while ignoring case and path separators
                        rel_path = str(py_file.relative_to(project_root)).replace("\\", "/")
                        if rel_path.lower() == file_path.lower().replace("\\", "/"):
                            path = py_file
                            break
        
        if not path or not path.exists():
            # If the file does not exist, its path may be relative to the project root used during analysis
            # Recursively search under the project root for a matching file name
            if project_root and project_root.exists():
                file_name = Path(file_path).name
                # Search only under the project root to avoid scanning the entire filesystem
                for py_file in project_root.rglob(file_name):
                    # Check whether the relative path matches while ignoring case and separators
                    rel_path = str(py_file.relative_to(project_root)).replace("\\", "/")
                    expected_rel_path = file_path.replace("\\", "/")
                    if rel_path.lower() == expected_rel_path.lower():
                        path = py_file
                        break
                
                # If still not found, try matching only the file name
                if (not path or not path.exists()) and file_name:
                    for py_file in project_root.rglob(file_name):
                        path = py_file
                        break
            
            if not path or not path.exists():
                return ""
        
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if start_line > len(lines):
                return ""
            
            # If end_line is invalid (<=0 or -1), try extracting through the end of the function
            # First find the function end by locating the next non-empty line with the same or lower indentation
            if end_line <= 0 or end_line > len(lines):
                # If no valid end_line exists, find the function end starting from start_line
                # Obtain the indentation of the function definition line
                if start_line <= len(lines):
                    def_line = lines[start_line - 1]
                    # Compute indentation, assuming spaces or tabs
                    indent_level = len(def_line) - len(def_line.lstrip())
                    
                    # Search from the next line until a non-empty line with the same or lower indentation is found
                    end_line = start_line
                    for i in range(start_line, len(lines)):
                        line = lines[i]
                        if line.strip():  # Non-empty line
                            line_indent = len(line) - len(line.lstrip())
                            if line_indent <= indent_level:
                                end_line = i  # Found the function end, excluding this line
                                break
                    else:
                        # If not found, extract through the end of the file
                        end_line = len(lines)
            
            # Ensure end_line is within the valid range
            if end_line > len(lines):
                end_line = len(lines)
            if end_line < start_line:
                end_line = start_line
            
            # Python line numbers start at 1 while array indexes start at 0
            code_lines = lines[start_line - 1:end_line]
            return "".join(code_lines).rstrip()
    except Exception as e:
        # Debug: uncomment to inspect the error
        # print(f"Error extracting code from {file_path}: {e}")
        return ""


def extract_method_code_from_entity(node, package_db, project_root: Path = None) -> str:
    """从原始 Entity 中尝试提取方法代码（优先方法）"""
    from enre.ent.EntKind import EntKind
    
    # Try to find the corresponding Entity in package_db
    func_entity = None
    for rel_path, module_db in package_db.tree.items():
        for ent in module_db.dep_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                break
        if func_entity:
            break
    
    if not func_entity:
        # If not found locally, also try the global database
        for ent in package_db.global_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                break
    
    if not func_entity:
        return ""
    
    # Use the Entity location information, which is more accurate
    try:
        file_path = str(func_entity.location.file_path)
        # Ensure paths use forward slashes to match the Node format
        file_path = file_path.replace("\\", "/")
        start_line = func_entity.location.code_span.start_line
        end_line = func_entity.location.code_span.end_line
        
        # If the Entity path is absolute, try using it
        code = extract_method_code(file_path, start_line, end_line, project_root)
        
        # If extraction fails, try the relative path obtained from Node
        if not code and node.file_path != file_path:
            code = extract_method_code(node.file_path, start_line, end_line, project_root)
        
        return code
    except Exception:
        # If accessing Entity location information fails, fall back to Node information
        return extract_method_code(node.file_path, node.start_line, node.end_line, project_root)


def extract_method_code_direct(node, package_db, project_root: Path = None) -> str:
    """直接从 package_db 中查找文件并提取代码"""
    from enre.ent.EntKind import EntKind
    
    # Try to find the corresponding Entity in package_db
    func_entity = None
    for rel_path, module_db in package_db.tree.items():
        for ent in module_db.dep_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                # Also obtain the module path information
                module_path = rel_path
                break
        if func_entity:
            break
    
    if not func_entity:
        return ""
    
    # Try several file paths
    file_paths_to_try = []
    
    # 1. Use the Entity file_path
    try:
        file_paths_to_try.append(str(func_entity.location.file_path))
    except:
        pass
    
    # 2. Use the Node file_path
    file_paths_to_try.append(node.file_path)
    
    # 3. Try the module path plus the file name
    try:
        if module_path:
            file_name = Path(node.file_path).name
            file_paths_to_try.append(str(module_path / file_name))
    except:
        pass
    
    # 4. Try the project root plus the relative path
    if project_root:
        file_paths_to_try.append(str(project_root / node.file_path))
    
    start_line = func_entity.location.code_span.start_line
    end_line = func_entity.location.code_span.end_line
    
    # If the Entity line number is invalid, use the Node line number
    if start_line <= 0:
        start_line = node.start_line
    if end_line <= 0:
        end_line = node.end_line
    
    # Try every possible path
    for file_path in file_paths_to_try:
        if file_path:
            code = extract_method_code(file_path, start_line, end_line, project_root)
            if code:
                return code
    
    return ""


def get_function_signature_with_params(node, package_db) -> str:
    """获取包含参数信息的函数签名"""
    from enre.ent.EntKind import EntKind, RefKind
    
    # First try to find the corresponding Entity in package_db
    func_entity = None
    for rel_path, module_db in package_db.tree.items():
        for ent in module_db.dep_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                break
        if func_entity:
            break
    
    if not func_entity:
        # If not found locally, also try the global database
        for ent in package_db.global_db.ents:
            if ent.id == node.id and ent.kind() == EntKind.Function:
                func_entity = ent
                break
    
    if not func_entity:
        # If no Entity is found, return the original longname
        return node.longname
    
    # Find parameters in the Entity refs, using the Parameter type of DefineKind
    params = []
    for ref in func_entity.refs():
        if (ref.ref_kind == RefKind.DefineKind and 
            ref.target_ent.kind() == EntKind.Parameter):
            # Extract parameter names from longname, usually function_name.param_name
            param_name = ref.target_ent.longname.name
            params.append((ref.lineno, ref.col_offset, param_name))
    
    # Sort parameters by line and column to preserve their source order
    params.sort(key=lambda x: (x[0], x[1]))
    param_names = [p[2] for p in params]
    
    # Build the signature with parameters
    if param_names:
        return f"{node.longname}({', '.join(param_names)})"
    else:
        return f"{node.longname}()"


def generate_csv_files(dep_repr: DepRepr, package_db = None, project_root: Path = None, output_dir: Path = None) -> None:
    """生成 CSV 文件: files.csv, methods.csv, file_adj_matrix.csv, func_adj_matrix.csv"""
    
    # Handle output_dir: use the current directory when it is None
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect file and method nodes
    file_nodes = []  # Module nodes
    func_nodes = []  # Function nodes
    node_id_to_index = {}  # Mapping from node IDs to indexes
    
    # Separate file and function nodes
    for node in dep_repr._node_list:
        if node.ent_type == EntKind.Module.value:
            file_nodes.append(node)
            node_id_to_index[node.id] = len(file_nodes) - 1
        elif node.ent_type == EntKind.Function.value:
            func_nodes.append(node)
            node_id_to_index[node.id] = len(func_nodes) - 1
    
    # Generate files.csv under output_dir
    # with open(output_dir / "files.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id", "file_path", "longname", "start_line", "end_line", "start_col", "end_col"])
    #     for idx, node in enumerate(file_nodes):
    #         writer.writerow([
    #             idx,  # use indexes starting at 0
    #             node.file_path,
    #             node.longname,
    #             node.start_line,
    #             node.end_line,
    #             node.start_col,
    #             node.end_col
    #         ])
    
    # Generate methods.csv
    project_root = project_root.parent
    with open(output_dir / "methods.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "method_signature", "func_file", "method_code"])
        for idx, node in enumerate(func_nodes):
            # Obtain method signatures with parameters
            method_signature = get_function_signature_with_params(node, package_db) if package_db else node.longname
            
            # Prefer code from the original Entity because its location information is more accurate
            method_code = ""
            if package_db:
                method_code = extract_method_code_from_entity(node, package_db, project_root)
            
            # If extraction from Entity fails, try using the Node location
            if not method_code:
                method_code = extract_method_code(node.file_path, node.start_line, node.end_line, project_root)
            
            # Debug: if still empty, try other methods
            if not method_code and package_db:
                # Try finding the file directly in package_db
                method_code = extract_method_code_direct(node, package_db, project_root)
            
            writer.writerow([
                idx,  # Use indexes starting at 0
                method_signature,  # method_signature with parameters
                node.file_path,  # func_file
                method_code  # method_code
            ])
    
    # Build a mapping from node IDs to nodes
    node_id_to_node = {node.id: node for node in dep_repr._node_list}
    
    # Build a mapping from file IDs to file paths for the file adjacency matrix
    file_id_to_path = {node.id: node.file_path for node in file_nodes}
    
    # Build a mapping from function IDs to file paths to derive file dependencies from function dependencies
    func_id_to_file_path = {}
    for node in func_nodes:
        func_id_to_file_path[node.id] = node.file_path
    
    # Build the file adjacency matrix
    file_id_to_index = {node.id: idx for idx, node in enumerate(file_nodes)}
    file_adj_matrix = defaultdict(lambda: defaultdict(int))
    
    # Build the function adjacency matrix
    func_id_to_index = {node.id: idx for idx, node in enumerate(func_nodes)}
    func_adj_matrix = defaultdict(lambda: defaultdict(int))
    
    # Process all edges
    for edge in dep_repr._edge_list:
        src_id = edge.src
        dest_id = edge.dest
        
        src_node = node_id_to_node.get(src_id)
        dest_node = node_id_to_node.get(dest_id)
        
        if src_node and dest_node:
            # File adjacency matrix: add a relation when two files depend on each other or a function in file A depends on a function in file B
            if src_node.ent_type == EntKind.Module.value and dest_node.ent_type == EntKind.Module.value:
                # Direct module-to-module dependency
                if src_id in file_id_to_index and dest_id in file_id_to_index:
                    src_idx = file_id_to_index[src_id]
                    dest_idx = file_id_to_index[dest_id]
                    file_adj_matrix[src_idx][dest_idx] = 1
            elif src_node.ent_type == EntKind.Function.value and dest_node.ent_type == EntKind.Function.value:
                # Derive file dependencies from function dependencies
                src_file_path = func_id_to_file_path.get(src_id)
                dest_file_path = func_id_to_file_path.get(dest_id)
                if src_file_path and dest_file_path and src_file_path != dest_file_path:
                    # Find the corresponding file node ID
                    src_file_node = next((n for n in file_nodes if n.file_path == src_file_path), None)
                    dest_file_node = next((n for n in file_nodes if n.file_path == dest_file_path), None)
                    if src_file_node and dest_file_node:
                        src_file_idx = file_id_to_index[src_file_node.id]
                        dest_file_idx = file_id_to_index[dest_file_node.id]
                        file_adj_matrix[src_file_idx][dest_file_idx] = 1
            
            # Function adjacency matrix
            if src_node.ent_type == EntKind.Function.value and dest_node.ent_type == EntKind.Function.value:
                if src_id in func_id_to_index and dest_id in func_id_to_index:
                    src_idx = func_id_to_index[src_id]
                    dest_idx = func_id_to_index[dest_id]
                    func_adj_matrix[src_idx][dest_idx] = 1
    
    # Generate file_adj_matrix.csv
    with open(output_dir / "file_adj_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write the matrix
        for i, node in enumerate(file_nodes):
            row = [file_adj_matrix[i].get(j, 0) for j in range(len(file_nodes))]
            writer.writerow(row)
    
    # Generate method_adj_matrix.csv
    with open(output_dir / "method_adj_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write the matrix
        for i, node in enumerate(func_nodes):
            row = [func_adj_matrix[i].get(j, 0) for j in range(len(func_nodes))]
            writer.writerow(row)

if __name__ == '__main__':
    main()
