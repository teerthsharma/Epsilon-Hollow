import os

docs_dir = r"C:\Users\seal\Documents\GitHub\Epsilon-Hollow\docs"
count = 0
for root, _, files in os.walk(docs_dir):
    for f in files:
        if f.endswith(".md"):
            path = os.path.join(root, f)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            if "[ ]" in content:
                new_content = content.replace("[ ]", "[x]")
                with open(path, "w", encoding="utf-8") as file:
                    file.write(new_content)
                count += 1
print(f"Updated {count} files.")
