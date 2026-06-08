import os

def generate_structure_file(startpath, output_filename="workspace_structure.txt"):
    # Folder yang diabaikan agar struktur tetap rapi
    ignore_dirs = {'.git', 'venv', '__pycache__', '.vscode', '.idea'}
    
    # Buka file txt dengan enkoding utf-8 agar ikon folder/file muncul dengan benar
    with open(output_filename, 'w', encoding='utf-8') as f:
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
                
    print(f"✅ Sukses! Struktur workspace telah disimpan di file: {output_filename}")

if __name__ == "__main__":
    generate_structure_file('.')
