import os
import csv
import javalang
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class JavaImportAnalyzer:
    def __init__(self):
        self.class_to_file: Dict[str, str] = {}
        self.package_classes: Dict[str, Set[str]] = defaultdict(set)
        self.all_classes: List[str] = []
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
    
    def analyze_project(self, project_root: str, output_file: str = "file_adj_matrix.csv"):
        # Validate the project path
        if not os.path.isdir(project_root):
            print(f"无效的项目目录: {project_root}")
            return
        
        # Collect all Java files
        java_files = self._collect_java_files(project_root)
        
        # First parsing pass: collect class information
        for file_path in java_files:
            self._parse_class_info(file_path)
        
        # Second parsing pass: analyze dependencies
        for file_path in java_files:
            self._analyze_file_dependencies(file_path)
        
        # Generate the adjacency matrix and write it to CSV
        self._generate_adjacency_matrix(output_file)
    
    def _collect_java_files(self, root_dir: str) -> List[str]:
        java_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".java"):
                    java_files.append(os.path.join(root, file))
        return java_files
    
    def _parse_class_info(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
            
            tree = javalang.parse.parse(source_code)
            package_name = tree.package.name if tree.package else ""
            
            # Process all class declarations in the file
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                full_class_name = f"{package_name}.{node.name}" if package_name else node.name
                self.all_classes.append(full_class_name)
                self.class_to_file[full_class_name] = file_path
                self.package_classes[package_name].add(full_class_name)
            
            # Process interface declarations
            for path, node in tree.filter(javalang.tree.InterfaceDeclaration):
                full_class_name = f"{package_name}.{node.name}" if package_name else node.name
                self.all_classes.append(full_class_name)
                self.class_to_file[full_class_name] = file_path
                self.package_classes[package_name].add(full_class_name)
                
        except (javalang.parser.JavaSyntaxError, UnicodeDecodeError) as e:
            print(f"解析文件 {file_path} 时出错: {str(e)}")
    
    def _analyze_file_dependencies(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                source_code = file.read()
            
            tree = javalang.parse.parse(source_code)
            package_name = tree.package.name if tree.package else ""
            
            # Obtain the main class of the current file
            main_class = None
            for path, node in tree.filter((javalang.tree.ClassDeclaration, javalang.tree.InterfaceDeclaration)):
                if not main_class:
                    main_class = f"{package_name}.{node.name}" if package_name else node.name
            
            if not main_class:
                return
            
            # Process imported dependencies
            for imp in tree.imports:
                imported = imp.path
                
                # Process wildcard imports
                if imp.wildcard:
                    imported_pkg = imported.rstrip('.*')
                    # Add all local classes under that package
                    for cls in self.package_classes.get(imported_pkg, []):
                        self.dependencies[main_class].add(cls)
                # Process single-class imports
                else:
                    if imported in self.all_classes:
                        self.dependencies[main_class].add(imported)
            
            # Process implicit imports from the same package
            if package_name in self.package_classes:
                for cls in self.package_classes[package_name]:
                    if cls != main_class:
                        self.dependencies[main_class].add(cls)
            
            # Process fully qualified name references
            for _, node in tree.filter(javalang.tree.ReferenceType):
                # Try to resolve fully qualified names
                if hasattr(node, 'name') and '.' in node.name:
                    possible_class = node.name
                    if possible_class in self.all_classes:
                        self.dependencies[main_class].add(possible_class)
            
        except (javalang.parser.JavaSyntaxError, UnicodeDecodeError) as e:
            print(f"分析依赖 {file_path} 时出错: {str(e)}")
    
    def _generate_adjacency_matrix(self, output_dir: str):
        if not self.all_classes:
            print("未找到可分析的类")
            return
        
        # Sort class names for consistent ordering
        sorted_classes = sorted(set(self.all_classes))
        class_index = {cls: idx for idx, cls in enumerate(sorted_classes)}
        size = len(sorted_classes)
        
        # Initialize the adjacency matrix
        adj_matrix = [[0] * size for _ in range(size)]
        
        # Populate dependency relationships
        for src_class, deps in self.dependencies.items():
            if src_class not in class_index:
                continue
            src_idx = class_index[src_class]
            for dep_class in deps:
                if dep_class in class_index:
                    dep_idx = class_index[dep_class]
                    adj_matrix[src_idx][dep_idx] = 1
        
        # Write the CSV file
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "file_adj_matrix.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header row
            writer.writerow([''] + sorted_classes)
            
            # Write matrix rows
            for i, cls in enumerate(sorted_classes):
                row = [cls] + adj_matrix[i]
                writer.writerow(row)
        
        print(f"邻接矩阵已写入: {output_file}")


if __name__ == "__main__":
    import sys
    
    project_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "out"

    analyzer = JavaImportAnalyzer()
    analyzer.analyze_project(project_path, output_dir)