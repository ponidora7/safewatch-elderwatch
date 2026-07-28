import os

def generate_structure_file(startpath, output_filename="workspace_structure.txt"):
    # Folder yang diabaikan agar struktur tetap rapi
    ignore_dirs = {'.git', 'venv', '__pycache__', '.vscode', '.idea', 'node_modules', '.next', 'out'}
    
    # Resolve output path relative to startpath
    output_path = os.path.join(startpath, output_filename)
    
    # Buka file txt dengan enkoding utf-8 agar ikon folder/file muncul dengan benar
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"📁 Workspace: {os.path.basename(os.path.abspath(startpath))}\n")
        
        for root, dirs, files in os.walk(startpath):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            level = root.replace(startpath, '').count(os.sep)
            indent = '│   ' * level
            
            if root != startpath:
                f.write(f"{indent[:-4]}├── 📁 {os.path.basename(root)}/\n")
                
            sub_indent = '│   ' * (level + 1)
            for i, file_name in enumerate(files):
                # Jangan masukkan skrip ini dan file output ke dalam daftar teks
                if file_name in {output_filename, 'tree_structure.py'}:
                    continue
                    
                is_last = (i == len(files) - 1)
                branch = '└── 📄 ' if is_last else '├── 📄 '
                f.write(f"{sub_indent[:-4]}{branch}{file_name}\n")
                
    print(f"Success! Workspace structure saved to: {output_path}")

if __name__ == "__main__":
    # Get the parent directory of this script (repository root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    generate_structure_file(repo_root)
