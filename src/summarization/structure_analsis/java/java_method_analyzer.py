import os
import csv
import re
import javalang
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

class JavaMethodAnalyzer:
    def __init__(self):
        # Store method information: ID -> (method signature, method code, class name, file name, start line, end line)
        self.methods: Dict[int, Tuple[str, str, str, str, int, int]] = {}
        # Mapping from class names to method IDs
        self.class_methods: Dict[str, Set[int]] = defaultdict(set)
        # Call relationships: caller ID -> set of callee IDs
        self.call_relations: Dict[int, Set[int]] = defaultdict(set)
        # Store file ASTs
        self.file_trees: Dict[str, Any] = {}
        # Current method ID counter
        self.current_id = 1
        # Store import mappings, one per file
        self.import_mapping: Dict[str, Dict[str, str]] = {}
        # Store package names by file
        self.package_names: Dict[str, str] = {}
        # Class inheritance/implementation relationships: class or interface -> [parent classes/interfaces...]
        self.class_hierarchy: Dict[str, List[str]] = defaultdict(list)
        # Mapping from method signatures to IDs
        self.signature_to_id: Dict[str, List[int]] = defaultdict(list)
        # Current parse context: the stack of enclosing method IDs, supporting nesting
        self.current_context: List[int] = []
        # Store field type information: class name -> field name -> type
        self.field_types: Dict[str, Dict[str, str]] = defaultdict(dict)
        # Store local variable type information: method ID -> variable name -> type
        self.local_vars: Dict[int, Dict[str, str]] = defaultdict(dict)
        # Store method return types
        self.method_return_types: Dict[int, Optional[str]] = {}
    
    def analyze_project(self, project_root: str, output_dir: str = "."):
        if not os.path.isdir(project_root):
            print(f"无效的项目目录: {project_root}")
            return

        java_files = self._collect_java_files(project_root)

        # First pass: collect classes, inheritance relationships, and method information
        for file_path in java_files:
            self._parse_file_info(file_path)

        # Second pass: analyze call relationships
        for file_path in java_files:
            self._analyze_call_relations(file_path)

        # Output
        os.makedirs(output_dir, exist_ok=True)
        self._write_method_csv(os.path.join(output_dir, "method.csv"))
        self._generate_call_matrix(os.path.join(output_dir, "method_adj_matrix.csv"))
    
    def _collect_java_files(self, root_dir: str) -> List[str]:
        java_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".java"):
                    java_files.append(os.path.join(root, file))
        return java_files
    
    def _parse_file_info(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                source_code = file.read()
            
            tree = javalang.parse.parse(source_code)
            self.file_trees[file_path] = tree
            package_name = tree.package.name if tree.package else ""
            self.package_names[file_path] = package_name
            
            # Create import mappings
            import_map: Dict[str, str] = {}
            for imp in tree.imports:
                if not imp.wildcard:
                    # import x.y.Z -> map["Z"] = "x.y.Z"
                    import_name = imp.path.split('.')[-1]
                    import_map[import_name] = imp.path
                else:
                    # Record wildcard-import package prefixes for reference; exact matching still primarily uses class names
                    import_map[imp.path.rstrip('.*')] = imp.path
            self.import_mapping[file_path] = import_map
            
            # Process inheritance and implementation relationships
            for _, node in tree.filter(javalang.tree.ClassDeclaration):
                self._process_class_declaration(file_path, node, package_name)
            for _, node in tree.filter(javalang.tree.InterfaceDeclaration):
                self._process_interface_declaration(file_path, node, package_name)
            
            # Process methods
            for _, node in tree.filter(javalang.tree.ClassDeclaration):
                full_class_name = f"{package_name}.{node.name}" if package_name else node.name
                self._parse_class_methods(file_path, full_class_name, node)
                # Collect field type information
                self._collect_field_types(file_path, full_class_name, node, package_name, import_map)
            for _, node in tree.filter(javalang.tree.InterfaceDeclaration):
                full_class_name = f"{package_name}.{node.name}" if package_name else node.name
                self._parse_class_methods(file_path, full_class_name, node)
                
        except (javalang.parser.JavaSyntaxError, UnicodeDecodeError) as e:
            print(f"解析文件 {file_path} 时出错: {str(e)}")
    
    def _process_class_declaration(self, file_path: str, node, package_name: str):
        full_class_name = f"{package_name}.{node.name}" if package_name else node.name
        import_map = self.import_mapping.get(file_path, {})

        # Process inheritance
        if node.extends:
            base_class = self._resolve_class_name(node.extends.name, package_name, import_map)
            if base_class:
                self.class_hierarchy[full_class_name].append(base_class)
        
        # Process implementation
        if node.implements:
            for impl in node.implements:
                interface_name = self._resolve_class_name(impl.name, package_name, import_map)
                if interface_name:
                    self.class_hierarchy[full_class_name].append(interface_name)
    
    def _process_interface_declaration(self, file_path: str, node, package_name: str):
        full_interface_name = f"{package_name}.{node.name}" if package_name else node.name
        import_map = self.import_mapping.get(file_path, {})
        
        # Process interface inheritance
        if node.extends:
            for ext in node.extends:
                base_interface = self._resolve_class_name(ext.name, package_name, import_map)
                if base_interface:
                    self.class_hierarchy[full_interface_name].append(base_interface)
    
    def _parse_class_methods(self, file_path: str, class_name: str, class_node):
        # Process all methods in a class or interface
        for _, node in class_node.filter(javalang.tree.MethodDeclaration):
            self._record_method(file_path, class_name, node)
        # Process constructors
        for _, node in class_node.filter(javalang.tree.ConstructorDeclaration):
            self._record_method(file_path, class_name, node)
        # Process inner classes declared directly to avoid infinite recursion
        # Access class body members directly instead of using recursive filter lookup
        if hasattr(class_node, 'body') and class_node.body:
            for member in class_node.body:
                if isinstance(member, javalang.tree.ClassDeclaration):
                    inner_class_name = f"{class_name}${member.name}"
                    self._parse_class_methods(file_path, inner_class_name, member)
    
    def _record_method(self, file_path: str, class_name: str, method_node):
        # Method name; constructors are normalized to <init>
        method_name = method_node.name
        if isinstance(method_node, javalang.tree.ConstructorDeclaration):
            method_name = "<init>"
        
        # Parameter list: type plus name
        params = []
        for param in method_node.parameters:
            param_type = self._get_type_name(param.type)
            param_name = getattr(param, 'name', '')
            # Unified format: Type name; retain only the type when the name is missing
            if param_name:
                params.append(f"{param_type} {param_name}")
            else:
                params.append(f"{param_type}")
        param_str = ", ".join(params)
        
        # Signature including parameter names
        method_signature = f"{class_name}.{method_name}( {param_str} )"
        
        # Location information
        start_line = method_node.position.line if method_node.position else 0
        end_line = self._get_method_end_line(method_node, file_path)
        
        # Source code
        method_code = self._extract_method_code(file_path, start_line, end_line)
        
        # ID assignment and storage
        method_id = self.current_id
        self.current_id += 1
        
        # Store the return type
        if isinstance(method_node, javalang.tree.MethodDeclaration):
            return_type = self._get_type_name(method_node.return_type) if method_node.return_type else "void"
            self.method_return_types[method_id] = return_type
        else:
            self.method_return_types[method_id] = class_name  # Constructors return their owning class
        
        self.methods[method_id] = (method_signature, method_code, class_name, file_path, start_line, end_line)
        self.class_methods[class_name].add(method_id)
        self.signature_to_id[method_signature].append(method_id)
        
        # Collect local variable type information
        self._collect_local_variables(method_id, method_node, class_name)
    
    def _collect_field_types(self, file_path: str, class_name: str, class_node, package_name: str, import_map: Dict[str, str]):
        """收集类的字段类型信息"""
        for _, field_node in class_node.filter(javalang.tree.FieldDeclaration):
            field_type = self._get_type_name(field_node.type)
            # Resolve to a fully qualified name
            resolved_type = self._resolve_class_name(field_type, package_name, import_map) or field_type
            for declarator in field_node.declarators:
                field_name = declarator.name
                self.field_types[class_name][field_name] = resolved_type
    
    def _collect_local_variables(self, method_id: int, method_node, class_name: str):
        """收集方法的局部变量类型信息"""
        # Collect method parameters
        for param in method_node.parameters:
            param_type = self._get_type_name(param.type)
            param_name = getattr(param, 'name', '')
            if param_name:
                self.local_vars[method_id][param_name] = param_type
        
        # Collect local variable declarations in method bodies with simplified handling
        if hasattr(method_node, 'body') and method_node.body:
            self._extract_local_vars_from_body(method_id, method_node.body, class_name)
    
    def _extract_local_vars_from_body(self, method_id: int, body, class_name: str):
        """从方法体中提取局部变量声明"""
        if isinstance(body, javalang.tree.BlockStatement):
            for stmt in body.statements or []:
                if isinstance(stmt, javalang.tree.LocalVariableDeclaration):
                    var_type = self._get_type_name(stmt.type)
                    for declarator in stmt.declarators:
                        var_name = declarator.name
                        self.local_vars[method_id][var_name] = var_type

    def _get_method_end_line(self, method_node, file_path: str) -> int:
        
        start_line = method_node.position.line if method_node.position else 0
        if start_line <= 0:
            return 0

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception:
            return start_line  # Fallback

        n = len(lines)
        i = start_line - 1  # 0-based

        brace_depth = 0
        in_single_line_comment = False
        in_block_comment = False

        def iterate_chars(line):
            nonlocal in_single_line_comment, in_block_comment
            in_string = False
            in_char = False
            escape = False
            j = 0
            L = len(line)
            while j < L:
                ch = line[j]
                nxt = line[j+1] if j+1 < L else ''

                # Line comment
                if not in_string and not in_char and not in_block_comment and ch == '/' and nxt == '/':
                    in_single_line_comment = True
                    break
                # Block-comment start
                if not in_string and not in_char and not in_block_comment and ch == '/' and nxt == '*':
                    in_block_comment = True
                    j += 2
                    continue
                # Block-comment end
                if in_block_comment and ch == '*' and nxt == '/':
                    in_block_comment = False
                    j += 2
                    continue

                if in_block_comment:
                    j += 1
                    continue

                # String
                if not in_char and ch == '"' and not in_string:
                    in_string = True
                    escape = False
                    j += 1
                    continue
                elif in_string:
                    if ch == '\\' and not escape:
                        escape = True
                    elif ch == '"' and not escape:
                        in_string = False
                    else:
                        escape = False
                    j += 1
                    continue

                # Character literal
                if not in_string and ch == "'" and not in_char:
                    in_char = True
                    escape = False
                    j += 1
                    continue
                elif in_char:
                    if ch == '\\' and not escape:
                        escape = True
                    elif ch == "'" and not escape:
                        in_char = False
                    else:
                        escape = False
                    j += 1
                    continue

                # Ordinary character
                yield ch
                j += 1

        # Stage 1: find '{' or ';'
        found_open_brace = False
        while i < n:
            line = lines[i]
            in_single_line_comment = False
            for ch in iterate_chars(line):
                if in_single_line_comment:
                    break
                if ch == '{':
                    found_open_brace = True
                    brace_depth = 1
                    break
                if ch == ';':
                    # Declaration without a method body, ending at this line
                    return i + 1
            if found_open_brace:
                break
            i += 1

        if not found_open_brace:
            # Neither '{' nor ';' was found; use a fallback
            return start_line

        # Stage 2: balance parentheses until the depth reaches zero
        while i < n:
            line = lines[i]
            in_single_line_comment = False
            for ch in iterate_chars(line):
                if in_single_line_comment:
                    break
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        return i + 1
            i += 1

        # Balance remains nonzero at end of file; fall back to the final line
        return n
    
    def _extract_method_code(self, file_path: str, start_line: int, end_line: int) -> str:
        """按行号提取方法代码；若 end_line < start_line，则只取 start_line 那一行避免空串。"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                lines = file.readlines()
            if start_line <= 0:
                return ""
            if end_line <= 0 or end_line < start_line:
                end_line = start_line
            start_idx = max(0, min(start_line - 1, len(lines) - 1))
            end_idx = max(start_idx, min(end_line - 1, len(lines) - 1))
            return ''.join(lines[start_idx:end_idx + 1])
        except Exception as e:
            print(f"提取方法代码出错: {str(e)}")
            return ""
    
    def _analyze_call_relations(self, file_path: str):
        """递归遍历AST并分析调用关系"""
        try:
            tree = self.file_trees[file_path]
            package_name = self.package_names[file_path]
            import_map = self.import_mapping[file_path]
            
            # Recursively traverse all nodes
            self._traverse_and_analyze(tree, file_path, package_name, import_map)
            
        except Exception as e:
            print(f"分析调用关系时出错: {file_path} - {str(e)}")
    
    def _traverse_and_analyze(self, node, file_path: str, package_name: str, import_map: Dict[str, str]):
        """递归遍历AST节点并分析调用关系"""
        # Update context
        if hasattr(node, 'position') and node.position:
            self._update_context(file_path, node.position)
        
        # Process method calls without elif because child nodes still need recursive processing
        if isinstance(node, javalang.tree.MethodInvocation):
            self._process_method_invocation(file_path, node, package_name, import_map)
        
        # Process constructor calls
        if isinstance(node, javalang.tree.ClassCreator):
            self._process_constructor_call(file_path, node, package_name, import_map)
        
        # Process method references
        if isinstance(node, javalang.tree.MethodReference):
            self._process_method_reference(file_path, node, package_name, import_map)
        
        # Process explicit constructor calls (super()/this())
        if isinstance(node, javalang.tree.ExplicitConstructorInvocation):
            self._process_explicit_constructor(file_path, node, package_name, import_map)
        
        # Lambda expression: recursively process its body
        if isinstance(node, javalang.tree.LambdaExpression):
            self._process_lambda_expression(file_path, node, package_name, import_map)
        
        # Recursively traverse child nodes regardless of whether a particular node type matched
        for child in self._get_children(node):
            if child is not None:
                self._traverse_and_analyze(child, file_path, package_name, import_map)
    
    def _get_children(self, node):
        """获取AST节点的所有子节点"""
        children = []
        if hasattr(node, '__dict__'):
            for key, value in node.__dict__.items():
                if key == 'position' or key.startswith('_'):
                    continue
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, javalang.tree.Node):
                            children.append(item)
                elif isinstance(value, javalang.tree.Node):
                    children.append(value)
        return children
    
    def _update_context(self, file_path: str, position):
        """根据位置更新当前解析上下文（支持嵌套方法）"""
        if not position:
            return
        
        line = position.line
        
        # Find all methods containing the current line, including nested ones
        matching_methods = []
        for method_id, (_, _, _, m_file_path, start, end) in self.methods.items():
            if m_file_path == file_path and start <= line <= end:
                matching_methods.append((method_id, start, end))
        
        if not matching_methods:
            self.current_context = []
            return
        
        # Sort by nesting depth, with the innermost method first
        matching_methods.sort(key=lambda x: (x[2] - x[1], -x[1]), reverse=True)
        self.current_context = [m[0] for m in matching_methods]
    
    def _process_method_invocation(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        method_name = node.member
        
        # Resolve the target class
        target_class = None
        if node.qualifier:
            qualifier = node.qualifier
            if isinstance(qualifier, str):
                if qualifier == "this":
                    target_class = self.methods[caller_id][2]
                elif qualifier == "super":
                    # super.method() calls a parent-class method
                    caller_class = self.methods[caller_id][2]
                    if caller_class in self.class_hierarchy:
                        # Take the first parent class, as Java normally allows only one
                        parents = self.class_hierarchy[caller_class]
                        if parents:
                            target_class = parents[0]
                else:
                    target_class = self._resolve_class_name(qualifier, package_name, import_map)
                    # It may be field access; try to infer the field type
                    if not target_class:
                        target_class = self._resolve_field_type(caller_id, qualifier)
            elif isinstance(qualifier, javalang.tree.MemberReference):
                # Process chained calls such as obj.field.method()
                expr_type = self._infer_expression_type(qualifier, caller_id, package_name, import_map)
                if expr_type:
                    target_class = self._resolve_class_name(expr_type, package_name, import_map)
            elif isinstance(qualifier, javalang.tree.MethodInvocation):
                # Python 3.9+ supports ast.unparse
                expr_type = self._infer_expression_type(qualifier, caller_id, package_name, import_map)
                if expr_type:
                    target_class = self._resolve_class_name(expr_type, package_name, import_map)
            elif hasattr(qualifier, 'name'):
                target_class = self._resolve_class_name(qualifier.name, package_name, import_map)
            elif hasattr(qualifier, 'type'):
                target_class = self._resolve_class_name(self._get_type_name(qualifier.type), package_name, import_map)
            else:
                # Attempt to infer the expression type
                expr_type = self._infer_expression_type(qualifier, caller_id, package_name, import_map)
                if expr_type:
                    target_class = self._resolve_class_name(expr_type, package_name, import_map)
        else:
            # Unqualified name: could be an instance method (this.method()) or a static method
            # First check whether it is a static import
            target_class = self._resolve_static_import(file_path, method_name, import_map)
            if not target_class:
                # Use the context class
                target_class = self._find_context_class(file_path, node.position)
        
        # Obtain parameter types
        arg_types: List[Optional[str]] = []
        for arg in node.arguments:
            arg_type = self._infer_expression_type(arg, caller_id, package_name, import_map)
            if arg_type == "this":
                arg_type = self.methods[caller_id][2]
            if arg_type:
                arg_types.append(arg_type)
        
        callee_ids = []
        
        # If the target class is found, try exact matching
        if target_class:
            callee_ids = self._find_matching_methods(target_class, method_name, arg_types)
        
        # If no exact match is found, try fuzzy matching
        if not callee_ids:
            # First try to find a method with the same name in the current class
            caller_class = self.methods[caller_id][2]
            for method_id in self.class_methods.get(caller_class, []):
                signature = self.methods[method_id][0]
                # Check whether the method name matches, accounting for parameter count
                if self._simple_method_match(signature, method_name, len(arg_types)):
                    if method_id not in callee_ids:
                        callee_ids.append(method_id)
            
            # Search the inheritance chain
            if caller_class in self.class_hierarchy:
                for parent in self.class_hierarchy[caller_class]:
                    for method_id in self.class_methods.get(parent, []):
                        signature = self.methods[method_id][0]
                        if self._simple_method_match(signature, method_name, len(arg_types)):
                            if method_id not in callee_ids:
                                callee_ids.append(method_id)
            
            # If still not found, search all known classes for the same method name using relaxed matching
            if not callee_ids:
                for class_name in self.class_methods:
                    for method_id in self.class_methods[class_name]:
                        signature = self.methods[method_id][0]
                        # Simple match: the method name appears in the signature and the parameter count matches
                        if self._simple_method_match(signature, method_name, len(arg_types)):
                            if method_id not in callee_ids:
                                callee_ids.append(method_id)
        
        # Record call relationships
        for callee_id in callee_ids:
            if callee_id != caller_id:  # Avoid self-calls unless this is a recursive method
                self.call_relations[caller_id].add(callee_id)
    
    def _process_constructor_call(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        
        # Resolve the class that owns the constructor
        class_type = node.type
        class_name = self._get_type_name(class_type)
        target_class = self._resolve_class_name(class_name, package_name, import_map)
        
        if not target_class:
            return
        
        # Obtain parameter types
        arg_types: List[Optional[str]] = []
        if node.arguments:
            for arg in node.arguments:
                t = self._infer_expression_type(arg, caller_id, package_name, import_map)
                if t == "this":
                    t = self.methods[caller_id][2]
                if t:
                    arg_types.append(t)
        
        # Find constructors
        callee_ids = self._find_matching_methods(target_class, "<init>", arg_types)
        
        # Record call relationships
        for callee_id in callee_ids:
            self.call_relations[caller_id].add(callee_id)
    
    def _process_method_reference(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        method_name = node.method
        
        # Resolve the class that owns the method
        target_class = None
        if node.expression:
            expr_type = self._infer_expression_type(node.expression, caller_id, package_name, import_map)
            if expr_type == "this":
                expr_type = self.methods[caller_id][2]
            if expr_type:
                target_class = self._resolve_class_name(expr_type, package_name, import_map)
        
        if not target_class:
            return
        
        # Find a matching method; method references usually pass no arguments, but their signatures still need to match
        callee_ids = self._find_matching_methods(target_class, method_name, [])
        
        # Record call relationships
        for callee_id in callee_ids:
            self.call_relations[caller_id].add(callee_id)
    
    def _process_lambda_expression(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        """处理 Lambda 表达式中的方法调用"""
        # Code inside a lambda expression should be processed in the enclosing context
        # No special handling is needed here; recursive traversal will process it
        pass
    
    def _process_explicit_constructor(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        """处理显式构造函数调用（如 super() 或 this()）"""
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        _, _, class_name, _, _, _ = self.methods[caller_id]
        
        if node.type == 'super':
            if class_name in self.class_hierarchy:
                parent_classes = self.class_hierarchy[class_name]
                for parent in parent_classes:
                    arg_types = []
                    for arg in node.arguments or []:
                        t = self._infer_expression_type(arg, caller_id, package_name, import_map)
                        if t == "this":
                            t = class_name
                        if t:
                            arg_types.append(t)
                    callee_ids = self._find_matching_methods(parent, "<init>", arg_types)
                    for callee_id in callee_ids:
                        self.call_relations[caller_id].add(callee_id)
        else:
            # this() calls another constructor of the same class
            arg_types = []
            for arg in node.arguments or []:
                t = self._infer_expression_type(arg, caller_id, package_name, import_map)
                if t == "this":
                    t = class_name
                if t:
                    arg_types.append(t)
            callee_ids = self._find_matching_methods(class_name, "<init>", arg_types)
            for callee_id in callee_ids:
                # Avoid self-calls
                if callee_id != caller_id:
                    self.call_relations[caller_id].add(callee_id)
    
    def _process_chained_call(self, file_path: str, node, package_name: str, import_map: Dict[str, str]):
        """处理链式方法调用（非常简化的推断）"""
        if not self.current_context:
            return
        
        caller_id = self.current_context[0]
        
        # Recursively process each part of a chained call
        chain: List[str] = []
        current = node
        while current and isinstance(current, javalang.tree.MemberReference):
            chain.insert(0, current.member)
            current = getattr(current, 'qualifier', None) if hasattr(current, 'qualifier') else None
        
        # Resolve the start of the chain
        target_class = None
        if isinstance(current, javalang.tree.MemberReference):
            if current.member == "this":
                target_class = self.methods[caller_id][2]
            else:
                target_class = self._resolve_class_name(current.member, package_name, import_map)
        elif hasattr(current, 'type'):
            target_class = self._resolve_class_name(self._get_type_name(current.type), package_name, import_map)
        
        if not target_class:
            return
        
        # Process each method call in the chain
        for method_name in chain:
            callee_ids = self._find_matching_methods(target_class, method_name, [])
            if callee_ids:
                return_type = self._get_method_return_type(callee_ids[0])
                if return_type:
                    target_class = self._resolve_class_name(return_type, package_name, import_map) or target_class
                for callee_id in callee_ids:
                    self.call_relations[caller_id].add(callee_id)
    
    def _get_method_return_type(self, method_id: int) -> Optional[str]:
        """获取方法的返回类型"""
        return self.method_return_types.get(method_id)
    
    def _infer_expression_type(self, expr, caller_id: Optional[int] = None, package_name: str = "", import_map: Dict[str, str] = None) -> Optional[str]:
        """推断表达式类型（改进版）"""
        if import_map is None:
            import_map = {}
        
        # Literal
        if isinstance(expr, javalang.tree.Literal):
            if "'" in str(expr.value) or '"' in str(expr.value):
                return "String"
            elif '.' in str(expr.value) and not expr.value.startswith('.'):
                return "double"
            elif str(expr.value).replace('-', '').replace('+', '').isdigit():
                return "int"
            elif str(expr.value) in ['true', 'false']:
                return "boolean"
            elif str(expr.value).endswith('L') or str(expr.value).endswith('l'):
                return "long"
            elif str(expr.value).endswith('F') or str(expr.value).endswith('f'):
                return "float"
            elif str(expr.value).endswith('D') or str(expr.value).endswith('d'):
                return "double"
        
        # this reference
        elif isinstance(expr, javalang.tree.This):
            return "this"
        
        # super reference
        elif isinstance(expr, javalang.tree.SuperMemberReference):
            if caller_id:
                caller_class = self.methods[caller_id][2]
                if caller_class in self.class_hierarchy:
                    parents = self.class_hierarchy[caller_class]
                    if parents:
                        return parents[0]
            return None
        
        # Field access
        elif isinstance(expr, javalang.tree.MemberReference):
            if caller_id and hasattr(expr, 'member'):
                # Try to find it among local variables or fields
                var_type = self.local_vars[caller_id].get(expr.member)
                if var_type:
                    return var_type
                caller_class = self.methods[caller_id][2]
                field_type = self.field_types[caller_class].get(expr.member)
                if field_type:
                    return field_type
                # It may be the result of a method call; try to infer the type
                return expr.member
        
        # Method call
        elif isinstance(expr, javalang.tree.MethodInvocation):
            return self._infer_method_return_type(expr, caller_id, package_name, import_map)
        
        # Constructor call
        elif isinstance(expr, javalang.tree.ClassCreator):
            class_type = expr.type
            class_name = self._get_type_name(class_type)
            return self._resolve_class_name(class_name, package_name, import_map)
        
        # Array access
        elif isinstance(expr, javalang.tree.ArraySelector):
            if hasattr(expr, 'postfix_operators'):
                # Array type; obtain the element type
                if hasattr(expr, 'primary'):
                    primary_type = self._infer_expression_type(expr.primary, caller_id, package_name, import_map)
                    if primary_type and primary_type.endswith('[]'):
                        return primary_type[:-2]
                    return primary_type
        
        # Type cast
        elif isinstance(expr, javalang.tree.Cast):
            if hasattr(expr, 'type'):
                return self._get_type_name(expr.type)
        
        # Node with type information
        elif hasattr(expr, 'type'):
            return self._get_type_name(expr.type)
        
        # Ternary operator
        elif isinstance(expr, javalang.tree.TernaryExpression):
            # Return the type of the then expression
            if hasattr(expr, 'if_true'):
                return self._infer_expression_type(expr.if_true, caller_id, package_name, import_map)
        
        return None
    
    def _infer_method_return_type(self, method_invocation, caller_id: Optional[int] = None, package_name: str = "", import_map: Dict[str, str] = None) -> Optional[str]:
        """推断方法调用的返回类型（改进版）"""
        if import_map is None:
            import_map = {}
        
        method_name = method_invocation.member
        
        # Resolve the target class
        target_class = None
        if method_invocation.qualifier:
            qualifier = method_invocation.qualifier
            if isinstance(qualifier, str):
                if qualifier == "this" and caller_id:
                    target_class = self.methods[caller_id][2]
                else:
                    target_class = self._resolve_class_name(qualifier, package_name, import_map)
            else:
                expr_type = self._infer_expression_type(qualifier, caller_id, package_name, import_map)
                if expr_type:
                    target_class = self._resolve_class_name(expr_type, package_name, import_map)
        else:
            if caller_id:
                target_class = self.methods[caller_id][2]
        
        if target_class:
            # Find the method and obtain its return type
            arg_types = []
            for arg in method_invocation.arguments:
                arg_type = self._infer_expression_type(arg, caller_id, package_name, import_map)
                if arg_type:
                    arg_types.append(arg_type)
            
            method_ids = self._find_matching_methods(target_class, method_name, arg_types)
            if method_ids:
                # Return the return type of the first matching method
                return_type = self.method_return_types.get(method_ids[0])
                if return_type:
                    return return_type
        
        # Default return-type inference
        if method_invocation.qualifier:
            qualifier = method_invocation.qualifier
            if isinstance(qualifier, str):
                return qualifier
            elif hasattr(qualifier, 'name'):
                return qualifier.name
        
        return None
    
    def _resolve_field_type(self, caller_id: int, field_name: str) -> Optional[str]:
        """解析字段类型"""
        if caller_id in self.methods:
            caller_class = self.methods[caller_id][2]
            # Search in the current class
            if caller_class in self.field_types:
                field_type = self.field_types[caller_class].get(field_name)
                if field_type:
                    return field_type
            # Recursively search in parent classes
            if caller_class in self.class_hierarchy:
                for parent in self.class_hierarchy[caller_class]:
                    if parent in self.field_types:
                        field_type = self.field_types[parent].get(field_name)
                        if field_type:
                            return field_type
        return None
    
    def _resolve_static_import(self, file_path: str, method_name: str, import_map: Dict[str, str]) -> Optional[str]:
        """解析静态导入的方法"""
        # Check whether this is a statically imported method
        # Simplified handling: search known classes for the static method
        for class_name in self.class_methods:
            for method_id in self.class_methods[class_name]:
                signature = self.methods[method_id][0]
                # Check whether this is a static method, simplified by matching the method name
                if method_name in signature:
                    # Check whether this is an imported class
                    short_name = class_name.split('.')[-1]
                    if short_name in import_map and import_map[short_name] == class_name:
                        return class_name
        return None
    
    def _resolve_class_name(self, name: str, package_name: str, import_map: Dict[str, str]) -> Optional[str]:
        """解析类名到完全限定名（基于当前文件 import_map 与已知类集合）"""
        if '.' in name and name in self.class_methods:
            return name
        
        if name in import_map:
            return import_map[name]
        
        full_name = f"{package_name}.{name}" if package_name else name
        if full_name in self.class_methods:
            return full_name
        
        java_lang_name = f"java.lang.{name}"
        if java_lang_name in self.class_methods:
            return java_lang_name
        
        for cls in self.class_methods:
            if cls.endswith(f".{name}"):
                return cls
        
        return None
    
    def _find_context_class(self, file_path: str, position) -> Optional[str]:
        """根据位置查找上下文类（使用当前行所在的方法的类名）"""
        if not position:
            return None
        for _, (_, _, class_name, _, start, end) in self.methods.items():
            if start <= position.line <= end:
                return class_name
        return None
    
    def _find_matching_methods(self, class_name: str, method_name: str, arg_types: List[str]) -> List[int]:
        """查找匹配的方法，递归查找继承链和接口（改进版）"""
        method_ids: List[int] = []
        visited_classes = set()
        
        def find_recursive(cls_name: str):
            if cls_name in visited_classes:
                return
            visited_classes.add(cls_name)
            
            # Find methods of the current class
            if cls_name in self.class_methods:
                for method_id in self.class_methods[cls_name]:
                    signature = self.methods[method_id][0]
                    if self._match_signature(signature, method_name, arg_types):
                        method_ids.append(method_id)
            
            # Recursively search parent classes
            if cls_name in self.class_hierarchy:
                for parent in self.class_hierarchy[cls_name]:
                    find_recursive(parent)
        
        find_recursive(class_name)
        return method_ids
    
    def _simple_method_match(self, signature: str, method_name: str, arg_count: int) -> bool:
        """简单的.method名匹配（不考虑类型）"""
        # Check whether the method name appears in the signature
        pattern = f".{method_name}("
        if pattern not in signature:
            # Special handling for constructors
            if method_name == "<init>" and "<init>(" in signature:
                pattern = "<init>("
            else:
                return False
        
        # Extract the parameter section starting from the last '(' to avoid matching parentheses in generics
        last_paren = signature.rfind('(')
        if last_paren == -1:
            return arg_count == 0
        
        params_str = signature[last_paren + 1:].rstrip(')').strip()
        if not params_str:
            return arg_count == 0
        
        # Count parameters by commas while accounting for generics and nested parentheses
        # Simplified approach: split by commas and remove empty strings
        param_list = [p.strip() for p in params_str.split(',') if p.strip()]
        return len(param_list) == arg_count
    
    def _match_signature(self, signature: str, method_name: str, arg_types: List[str]) -> bool:
        """检查方法签名是否匹配（改进版，支持类型兼容性匹配）"""
        match = re.match(r".*?\.(\w+)\((.*?)\)", signature)
        if not match:
            return False
        
        sig_method_name = match.group(1)
        sig_params = match.group(2)
        
        if sig_method_name != method_name and not (sig_method_name == "<init>" and method_name == "<init>"):
            return False
        
        # Allow "Type name" in signatures; compare only the type before the first space or the full generic type
        def extract_type(p: str) -> str:
            p = p.strip()
            if not p:
                return ""
            # Handle forms such as "List<String> ids" or "Map<K, V> m"
            # Extract the part before the generic parameters
            parts = p.split()
            if len(parts) > 0:
                return parts[0]
            return p
        
        sig_param_list_raw = [p.strip() for p in sig_params.split(',')] if sig_params else []
        sig_param_types = [extract_type(p) for p in sig_param_list_raw if p != ""]
        
        # Return False when the parameter counts do not match
        if len(sig_param_types) != len(arg_types):
            return False
        
        # If the parameter list is empty, match directly
        if len(sig_param_types) == 0:
            return True
        
        # Check type compatibility
        primitive_map = {
            "int": "Integer",
            "long": "Long",
            "double": "Double",
            "float": "Float",
            "boolean": "Boolean",
            "char": "Character",
            "byte": "Byte",
            "short": "Short"
        }
        
        for i, param_type in enumerate(arg_types):
            # Skip the check when the parameter type is unknown, allowing a match
            if not param_type:
                continue
            
            sig_t = sig_param_types[i]
            if not sig_t:
                return False
            
            # Exact match
            if sig_t == param_type:
                continue
            
            # Primitive and wrapper types match
            if param_type in primitive_map and sig_t == primitive_map[param_type]:
                continue
            if sig_t in primitive_map and primitive_map[sig_t] == param_type:
                continue
            
            # Package-name matching; allow same-named types from different packages for this simplified check
            sig_base = sig_t.split('.')[-1] if '.' in sig_t else sig_t
            param_base = param_type.split('.')[-1] if '.' in param_type else param_type
            if sig_base == param_base:
                continue
            
            # Partial fully qualified name match
            if sig_t.endswith(f".{param_base}") or param_type.endswith(f".{sig_base}"):
                continue
            
            # Simplified generic type match that ignores generic parameters
            sig_generic = sig_t.split('<')[0] if '<' in sig_t else sig_t
            param_generic = param_type.split('<')[0] if '<' in param_type else param_type
            if sig_generic == param_generic:
                continue
            
            return False
        
        return True
    
    def _get_type_name(self, type_node) -> str:
        """获取类型名称，处理泛型（仅保留主类型名与泛型参数的主类型名）"""
        if isinstance(type_node, javalang.tree.ReferenceType):
            base_name = '.'.join(type_node.name) if isinstance(type_node.name, list) else type_node.name
            if type_node.arguments:
                arg_names = []
                for arg in type_node.arguments:
                    if hasattr(arg, 'type') and arg.type is not None:
                        arg_names.append(self._get_type_name(arg.type))
                    else:
                        arg_names.append("?")
                return f"{base_name}<{', '.join(arg_names)}>"
            return base_name
        elif hasattr(type_node, 'name'):
            return type_node.name
        return ""
    
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print