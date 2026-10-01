import os
import csv
import ast
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

class PythonMethodAnalyzer:
    def __init__(self):
        # Store function information: ID -> (function signature, function code, class/module name, file name, start line, end line)
        self.methods: Dict[int, Tuple[str, str, str, str, int, int]] = {}
        # Mapping from class/module names to method IDs
        self.class_methods: Dict[str, Set[int]] = defaultdict(set)
        # Call relationships: caller ID -> set of callee IDs
        self.call_relations: Dict[int, Set[int]] = defaultdict(set)
        # Store file ASTs
        self.file_trees: Dict[str, ast.AST] = {}
        # Current method ID counter
        self.current_id = 1
        # Store import mappings, one per file
        self.import_mapping: Dict[str, Dict[str, str]] = {}
        # Store module names by file
        self.module_names: Dict[str, str] = {}
        # Current parse context, represented by the enclosing function ID
        self.current_context: List[int] = []
        # File dependencies: file -> set of dependency files
        self.file_dependencies: Dict[str, Set[str]] = defaultdict(set)
        # Mapping from file names to IDs
        self.file_to_id: Dict[str, int] = {}
        # List of all files
        self.all_files: List[str] = []
        # Project root directory
        self.project_root: Optional[str] = None
        # Mapping from files to method IDs for fast lookup
        self.file_to_method_ids: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)  # file_path -> [(method_id, start, end), ...]
    
    def analyze_project(self, project_root: str, output_dir: str = "."):
        if not os.path.isdir(project_root):
            print(f"无效的项目目录: {project_root}")
            return

        self.project_root = os.path.abspath(project_root)
        python_files = self._collect_python_files(project_root)
        print(f"找到 {len(python_files)} 个Python文件")

        # First pass: collect function information and import relationships
        print("开始解析文件信息...")
        for i, file_path in enumerate(python_files, 1):
            if i % 10 == 0 or i == len(python_files):
                print(f"  解析进度: {i}/{len(python_files)} ({i*100//len(python_files)}%)")
            self._parse_file_info(file_path)
        print(f"解析完成，共找到 {len(self.methods)} 个函数/方法")

        # Second pass: analyze call relationships
        print("开始分析调用关系...")
        for i, file_path in enumerate(python_files, 1):
            if i % 10 == 0 or i == len(python_files):
                print(f"  分析进度: {i}/{len(python_files)} ({i*100//len(python_files)}%)")
            self._analyze_call_relations(file_path)
        print(f"调用关系分析完成，共找到 {sum(len(v) for v in self.call_relations.values())} 条调用关系")

        # Output
        print("正在生成输出文件...")
        os.makedirs(output_dir, exist_ok=True)
        self._write_method_csv(os.path.join(output_dir, "method.csv"))
        self._generate_call_matrix(os.path.join(output_dir, "method_adj_matrix.csv"))
        self._generate_file_adj_matrix(os.path.join(output_dir, "file_adj_matrix.csv"))
        print("分析完成！")
    
    def _collect_python_files(self, root_dir: str) -> List[str]:
        python_files = []
        # DEBUG: count test files
        test_file_num = 0
        for root, _, files in os.walk(root_dir):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), root_dir)
                # If the path contains "test", treat it as test code and exclude it from analysis
                if "tests" in relative_path.split(os.sep) or "test" in relative_path.split(os.sep):
                    test_file_num += 1
                    continue
                # If this is a Python file, add it to the result list
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        print(f"找到 {test_file_num} 个测试文件，过滤")
        return python_files
    
    def _parse_file_info(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                source_code = file.read()
            
            tree = ast.parse(source_code)
            self.file_trees[file_path] = tree
            
            # Compute the module name relative to the project root
            if self.project_root:
                try:
                    rel_path = os.path.relpath(file_path, self.project_root)
                except ValueError:
                    # If a relative path cannot be computed, use the basename of the absolute path
                    rel_path = os.path.basename(file_path)
            else:
                rel_path = os.path.basename(file_path)
            
            module_name = os.path.splitext(rel_path)[0].replace(os.sep, '.')
            if module_name.endswith('.__init__'):
                module_name = module_name.replace('.__init__', '')
            if module_name.startswith('.'):
                module_name = module_name[1:]
            
            self.module_names[file_path] = module_name
            
            # Create import mappings
            import_map: Dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.asname if alias.asname else alias.name.split('.')[0]
                        import_map[import_name] = alias.name
                elif isinstance(node, ast.ImportFrom):
                    module_name_import = node.module or ''
                    if node.level > 0:  # Relative imports
                        base_module = module_name
                        for _ in range(node.level - 1):
                            if '.' in base_module:
                                base_module = '.'.join(base_module.split('.')[:-1])
                        if module_name_import:
                            module_name_import = f"{base_module}.{module_name_import}" if base_module else module_name_import
                    for alias in node.names:
                        import_name = alias.asname if alias.asname else alias.name
                        if module_name_import:
                            import_map[import_name] = f"{module_name_import}.{alias.name}"
                        else:
                            import_map[import_name] = alias.name
            self.import_mapping[file_path] = import_map
            
            # Collect the file list
            if file_path not in self.file_to_id:
                file_id = len(self.all_files)
                self.file_to_id[file_path] = file_id
                self.all_files.append(file_path)
            
            # Process classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._parse_class_methods(file_path, module_name, node, source_code)
                elif isinstance(node, ast.FunctionDef):
                    # Module-level functions
                    self._record_method(file_path, module_name, node, source_code)
                    
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"解析文件 {file_path} 时出错: {str(e)}")
    
    def _parse_class_methods(self, file_path: str, module_name: str, class_node: ast.ClassDef, source_code: str):
        class_name = f"{module_name}.{class_node.name}" if module_name else class_node.name
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                self._record_method(file_path, class_name, node, source_code, class_node)
    
    def _record_method(self, file_path: str, class_name: str, func_node: ast.FunctionDef, source_code: str, class_node: Optional[ast.ClassDef] = None):
        # Function name
        method_name = func_node.name
        
        # Parameter list
        params = []
        for arg in func_node.args.args:
            param_name = arg.arg
            # Try to obtain parameter type annotations
            if arg.annotation:
                # ast.unparse is available in Python 3.9+
                try:
                    if hasattr(ast, 'unparse'):
                        param_type = ast.unparse(arg.annotation)
                    else:
                        param_type = self._get_annotation_str(arg.annotation)
                    params.append(f"{param_type} {param_name}")
                except:
                    params.append(param_name)
            else:
                params.append(param_name)
        
        # Handle *args and **kwargs
        if func_node.args.vararg:
            params.append(f"*{func_node.args.vararg.arg}")
        if func_node.args.kwarg:
            params.append(f"**{func_node.args.kwarg.arg}")
        
        param_str = ", ".join(params)
        
        # Signature; class methods include self/cls parameters
        method_signature = f"{class_name}.{method_name}( {param_str} )"
        
        # Location information
        start_line = func_node.lineno
        end_line = func_node.end_lineno if hasattr(func_node, 'end_lineno') else self._get_method_end_line(func_node, source_code)
        
        # Source code
        method_code = self._extract_method_code(source_code, start_line, end_line)
        
        # ID assignment and storage
        method_id = self.current_id
        self.current_id += 1
        
        self.methods[method_id] = (method_signature, method_code, class_name, file_path, start_line, end_line)
        self.class_methods[class_name].add(method_id)
        # Add file-to-method-ID mappings for fast lookup
        self.file_to_method_ids[file_path].append((method_id, start_line, end_line))
    
    def _get_annotation_str(self, annotation) -> str:
        """将类型注解转换为字符串"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_annotation_str(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            base = self._get_annotation_str(annotation.value)
            if isinstance(annotation.slice, ast.Index) and hasattr(annotation.slice, 'value'):
                args = self._get_annotation_str(annotation.slice.value)
            elif hasattr(annotation, 'slice'):
                args = self._get_annotation_str(annotation.slice)
            else:
                args = "?"
            return f"{base}[{args}]"
        return "?"
    
    def _get_method_end_line(self, func_node: ast.FunctionDef, source_code: str) -> int:
        """获取方法结束行号"""
        # end_lineno is available in Python 3.8+
        if hasattr(func_node, 'end_lineno') and func_node.end_lineno:
            return func_node.end_lineno
        
        start_line = func_node.lineno
        lines = source_code.split('\n')
        
        # Find the end position of the function body
        if func_node.body:
            last_stmt = func_node.body[-1]
            if hasattr(last_stmt, 'end_lineno') and last_stmt.end_lineno:
                return last_stmt.end_lineno
        
        # If end_lineno is unavailable, try inferring the end from indentation
        if len(lines) >= start_line:
            if start_line > 0:
                start_indent = len(lines[start_line - 1]) - len(lines[start_line - 1].lstrip())
                for i in range(start_line, len(lines)):
                    line = lines[i]
                    if line.strip() and len(line) - len(line.lstrip()) <= start_indent:
                        return i + 1
            return len(lines)
        
        return start_line
    
    def _extract_method_code(self, source_code: str, start_line: int, end_line: int) -> str:
        """按行号提取方法代码"""
        lines = source_code.split('\n')
        if start_line <= 0:
            return ""
        if end_line <= 0 or end_line < start_line:
            end_line = start_line
        start_idx = max(0, min(start_line - 1, len(lines) - 1))
        end_idx = max(start_idx, min(end_line - 1, len(lines) - 1))
        return '\n'.join(lines[start_idx:end_idx + 1])
    
    def _analyze_call_relations(self, file_path: str):
        """分析调用关系"""
        try:
            tree = self.file_trees[file_path]
            module_name = self.module_names[file_path]
            import_map = self.import_mapping[file_path]
            
            # Recursively traverse all nodes
            self._traverse_and_analyze(tree, file_path, module_name, import_map)
            
        except Exception as e:
            print(f"分析调用关系时出错: {file_path} - {str(e)}")
    
    def _traverse_and_analyze(self, node, file_path: str, module_name: str, import_map: Dict[str, str]):
        """递归遍历AST节点并分析调用关系"""
        # Update context
        if hasattr(node, 'lineno'):
            self._update_context(file_path, node.lineno)
        
        # Process function calls
        if isinstance(node, ast.Call):
            self._process_function_call(file_path, node, module_name, import_map)
        
        # Process imports for file dependency relationships
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._process_import_for_dependencies(file_path, node, module_name)
        
        # Recursively traverse child nodes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self._traverse_and_analyze(item, file_path, module_name, import_map)
            elif isinstance(value, ast.AST):
                self._traverse_and_analyze(value, file_path, module_name, import_map)
    
    def _update_context(self, file_path: str, line: int):
        """根据位置更新当前解析上下文（优化版：使用文件索引）"""
        if not line:
            self.current_context = []
            return
        
        # Optimization: inspect only methods in the current file by using the file index
        matching_methods = []
        
        # Use the file index to find methods in the current file quickly
        if file_path in self.file_to_method_ids:
            for method_id, start, end in self.file_to_method_ids[file_path]:
                if start <= line <= end:
                    matching_methods.append((method_id, start, end))
        else:
            # Fallback: traverse all methods if the index is missing; this should not occur, but is kept for safety
            for method_id, (_, _, _, m_file_path, start, end) in self.methods.items():
                if m_file_path == file_path and start <= line <= end:
                    matching_methods.append((method_id, start, end))
        
        if not matching_methods:
            self.current_context = []
            return
        
        # Sort by nesting depth, with the innermost method first
        matching_methods.sort(key=lambda x: (x[2] - x[1], -x[1]), reverse=True)
        self.current_context = [m[0] for m in matching_methods]
    
    def _process_function_call(self, file_path: str, node: ast.Call, module_name: str, import_map: Dict[str, str]):
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        
        # Resolve the called function
        func_name = None
        target_module = None
        
        if isinstance(node.func, ast.Name):
            # Direct function call func()
            func_name = node.func.id
            # Check whether this is an imported function
            if func_name in import_map:
                full_name = import_map[func_name]
                if '.' in full_name:
                    parts = full_name.rsplit('.', 1)
                    target_module = parts[0]
                    func_name = parts[1]
                else:
                    target_module = None
        elif isinstance(node.func, ast.Attribute):
            # Attribute call obj.method() or module.func()
            func_name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                # module.func() or obj.method()
                value_name = node.func.value.id
                if value_name in import_map:
                    # This is an imported module
                    target_module = import_map[value_name]
                elif value_name == "self":
                    # Instance-method call; use the caller's class
                    caller_class = self.methods[caller_id][2]
                    target_module = caller_class
                else:
                    # It may be a local variable or field; try to infer its type
                    target_module = self._infer_expression_type(caller_id, value_name, file_path, module_name, import_map)
        
        # Obtain the number of parameters
        arg_count = len(node.args) + (1 if node.keywords else 0)
        
        # Find matching functions
        callee_ids = []
        
        if target_module:
            # Search in the target module or class
            callee_ids = self._find_matching_methods(target_module, func_name, arg_count)
        
        # If no match is found, try fuzzy matching while limiting the search for performance
        if not callee_ids:
            # Search in the current class or module
            caller_class = self.methods[caller_id][2]
            for method_id in self.class_methods.get(caller_class, []):
                signature = self.methods[method_id][0]
                if self._simple_method_match(signature, func_name, arg_count):
                    if method_id not in callee_ids:
                        callee_ids.append(method_id)
            
            # Search only modules in the current file rather than all classes/modules for performance
            # This avoids traversing all functions in large projects
            if not callee_ids and module_name:
                # Try looking up from the current module name
                if module_name in self.class_methods:
                    for method_id in self.class_methods[module_name]:
                        signature = self.methods[method_id][0]
                        if self._simple_method_match(signature, func_name, arg_count):
                            if method_id not in callee_ids:
                                callee_ids.append(method_id)
            
            # Search all known classes/modules last as the slowest fallback
            # For very large projects, this section can be disabled to reduce search time
            if not callee_ids and len(self.methods) < 10000:  # Search only when the number of methods is not too large
                for class_name in self.class_methods:
                    if len(callee_ids) >= 10:  # Limit the number of matches
                        break
                    for method_id in self.class_methods[class_name]:
                        signature = self.methods[method_id][0]
                        if self._simple_method_match(signature, func_name, arg_count):
                            if method_id not in callee_ids:
                                callee_ids.append(method_id)
                                if len(callee_ids) >= 10:
                                    break
        
        # Record call relationships
        for callee_id in callee_ids:
            if callee_id != caller_id:
                self.call_relations[caller_id].add(callee_id)
    
    def _infer_expression_type(self, caller_id: int, var_name: str, file_path: str, module_name: str, import_map: Dict[str, str]) -> Optional[str]:
        """推断变量类型（简化）"""
        # Check whether this is an imported module
        if var_name in import_map:
            return import_map[var_name]
        
        # Check whether this is an instance of the current class
        caller_class = self.methods[caller_id][2]
        if var_name == "self":
            return caller_class
        
        return None
    
    def _find_matching_methods(self, class_name: str, method_name: str, arg_count: int) -> List[int]:
        """查找匹配的方法"""
        method_ids = []
        
        if class_name in self.class_methods:
            for method_id in self.class_methods[class_name]:
                signature = self.methods[method_id][0]
                if self._simple_method_match(signature, method_name, arg_count):
                    method_ids.append(method_id)
        
        return method_ids
    
    def _simple_method_match(self, signature: str, method_name: str, arg_count: int) -> bool:
        """简单的方法名匹配（不考虑类型）"""
        # Check whether the method name appears in the signature
        pattern = f".{method_name}("
        if pattern not in signature:
            return False
        
        # Extract the parameter section
        match = re.search(r'\((.*?)\)', signature)
        if not match:
            return arg_count == 0
        
        params_str = match.group(1).strip()
        if not params_str:
            return arg_count == 0
        
        # Count parameters while excluding *args and **kwargs
        param_list = [p.strip() for p in params_str.split(',') if p.strip() and not p.strip().startswith('*')]
        # Simplified approach: Python may have default parameters, so check only the minimum parameter count
        return len(param_list) <= arg_count or arg_count == 0
    
    def _process_import_for_dependencies(self, file_path: str, node, module_name: str):
        """处理导入以构建文件依赖关系"""
        try:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_module = alias.name
                    target_file = self._resolve_module_to_file(imported_module)
                    if target_file and target_file != file_path:
                        self.file_dependencies[file_path].add(target_file)
            elif isinstance(node, ast.ImportFrom):
                imported_module = node.module or ''
                if node.level > 0:  # Relative imports
                    base_module = module_name
                    for _ in range(node.level - 1):
                        if '.' in base_module:
                            base_module = '.'.join(base_module.split('.')[:-1])
                    if imported_module:
                        imported_module = f"{base_module}.{imported_module}" if base_module else imported_module
                
                if imported_module:
                    target_file = self._resolve_module_to_file(imported_module)
                    if target_file and target_file != file_path:
                        self.file_dependencies[file_path].add(target_file)
        except Exception as e:
            pass
    
    def _resolve_module_to_file(self, module_name: str) -> Optional[str]:
        """将模块名解析为文件路径"""
        # Search among known module names
        for file_path, mod_name in self.module_names.items():
            if mod_name == module_name or mod_name.endswith('.' + module_name):
                return file_path
            # Check whether this is part of a module
            if module_name.startswith(mod_name + '.'):
                return file_path
        return None
    
    def _write_method_csv(self, output_file: str):
        """写入方法信息到 CSV"""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['ID', 'method_signature', 'method_code', 'class_name', 'file_path', 'start_line', 'end_line'])
            for method_id, (signature, code, class_name, file_path, start_line, end_line) in self.methods.items():
                clean_code = code.replace('\n', '\\n').replace('\r', '\\r')
                writer.writerow([method_id, signature, clean_code, class_name, file_path, start_line, end_line])
        print(f"方法信息已写入: {output_file}")
    
    def _generate_call_matrix(self, output_file: str):
        """生成调用关系邻接矩阵"""
        if not self.methods:
            print("未找到可分析的方法")
            return
        
        method_ids = sorted(self.methods.keys())
        id_index = {method_id: idx for idx, method_id in enumerate(method_ids)}
        size = len(method_ids)
        
        adj_matrix = [[0] * size for _ in range(size)]
        
        for caller_id, callee_ids in self.call_relations.items():
            if caller_id in id_index:
                caller_idx = id_index[caller_id]
                for callee_id in callee_ids:
                    if callee_id in id_index:
                        callee_idx = id_index[callee_id]
                        adj_matrix[caller_idx][callee_idx] = 1
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Caller/Callee'] + [str(mid) for mid in method_ids])
            for i, method_id in enumerate(method_ids):
                row = [str(method_id)] + adj_matrix[i]
                writer.writerow(row)
        print(f"方法调用邻接矩阵已写入: {output_file}")
    
    def _generate_file_adj_matrix(self, output_file: str):
        """生成文件依赖关系邻接矩阵"""
        if not self.all_files:
            print("未找到可分析的文件")
            return
        
        # Create a mapping from file paths to indexes
        file_index = {file_path: idx for idx, file_path in enumerate(self.all_files)}
        size = len(self.all_files)
        
        adj_matrix = [[0] * size for _ in range(size)]
        
        for src_file, deps in self.file_dependencies.items():
            if src_file in file_index:
                src_idx = file_index[src_file]
                for dep_file in deps:
                    if dep_file in file_index:
                        dep_idx = file_index[dep_file]
                        adj_matrix[src_idx][dep_idx] = 1
        
        # Use the file name as the identifier
        file_names = [os.path.basename(f) for f in self.all_files]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([''] + file_names)
            for i, file_name in enumerate(file_names):
                row = [file_name] + adj_matrix[i]
                writer.writerow(row)
        print(f"文件依赖邻接矩阵已写入: {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python python_analsis.py <project_root> [output_dir]")
        sys.exit(1)
    
    project_root = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    analyzer = PythonMethodAnalyzer()
    analyzer.analyze_project(project_root, output_dir)
