import os

def count_lines_in_py_files(directory):
    total_lines = 0
    py_files_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files_count += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for line in f)
                        total_lines += line_count
                        print(f"{file_path}: {line_count} lines")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"\nTotal Python files: {py_files_count}")
    print(f"Total lines of Python code: {total_lines}")

# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
robot_dir = os.path.join(current_dir, 'robot')

# Count lines in Python files
count_lines_in_py_files(robot_dir)