# generate_hidden_imports.py
with open('requirements.txt') as f:
    packages = f.readlines()

hidden_imports = ' '.join([f'--hidden-import={pkg.strip()}' for pkg in packages])
print(hidden_imports)