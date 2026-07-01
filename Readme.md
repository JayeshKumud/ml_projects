# Environment Setup

**Windows + VS Code + Miniconda + Python 3.12**

This guide installs a complete NLP development environment using only terminal commands where possible.

## Prerequisites

* Windows 10 or Windows 11
* Administrator PowerShell
* Internet connection

---

# Step 1: Install Python 3.12

Open **PowerShell (Administrator)** and run:

```powershell
winget install Python.Python.3.12
```

Verify installation:

```powershell
python --version
```

Expected output:

```text
Python 3.12.x
```

---

# Step 2: Install Miniconda

Download and install Miniconda silently.

```powershell
# Download installer
curl -o "$env:USERPROFILE\Downloads\Miniconda3-latest.exe" `
https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

# Silent install
Start-Process -Wait "$env:USERPROFILE\Downloads\Miniconda3-latest.exe" `
  -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/S", "/D=$env:USERPROFILE\Miniconda3"

# Add Conda to current session
$env:Path = "$env:USERPROFILE\Miniconda3\Scripts;" + $env:Path

# Initialize Conda for PowerShell
conda init powershell
```

Close PowerShell and reopen it.

Verify:

```powershell
conda --version
```
Alternatively:

winget install Anaconda.Miniconda3

Expected:

```text
conda 25.x.x
```

---

# Step 3: Install Visual Studio Code

```powershell
winget install -e --id Microsoft.VisualStudioCode
```

Verify:

```powershell
code --version
```

If `code` is not recognized, reopen PowerShell.

---

# Step 4: Install Required VS Code Extensions

```powershell
code --install-extension ms-python.python
code --install-extension ms-python.pylance
code --install-extension ms-toolsai.jupyter
```

Verify:

```powershell
code --list-extensions
```

Expected output includes:

```text
ms-python.python
ms-python.pylance
ms-toolsai.jupyter
```

---

# Step 5: Create Conda Environment

```powershell
conda create -n nlp_course_env python=3.12 -y
```

Activate:

```powershell
conda activate nlp_course_env
```

Verify:

```powershell
python --version
```

Expected:

```text
Python 3.12.x
```

---

# Step 6: Upgrade Packaging Tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

# Step 7: Install Core Data Science Libraries

```powershell
pip install `
numpy==1.26.4 `
pandas==2.2.3 `
matplotlib==3.10.0 `
scikit-learn==1.6.0 `
seaborn==0.13.2
```

Verify:

```powershell
python -c "import numpy,pandas,matplotlib,sklearn,seaborn; print('Data science packages OK')"
```

---

# Step 8: Install NLP Libraries

```powershell
pip install `
nltk==3.9.1 `
spacy==3.8.3 `
textblob==0.18.0.post0 `
vaderSentiment==3.3.2 `
transformers==4.47.1 `
gensim==4.3.3
```

Verify:

```powershell
python -c "import nltk,spacy,textblob,vaderSentiment,transformers,gensim; print('NLP packages OK')"
```

---

# Step 9: Install PyTorch (CPU)

```powershell
pip install torch==2.5.1
```

Verify:

```powershell
python -c "import torch; print('PyTorch', torch.__version__, 'OK')"
```

---

# Step 10: Install Jupyter Kernel Support

```powershell
pip install ipykernel
```

---

# Step 11: Download NLTK Resources

```powershell
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
print('NLTK data downloaded OK')
"
```

---

# Step 12: Download spaCy English Model

```powershell
python -m spacy download en_core_web_sm
```

Verify:

```powershell
python -c "import spacy; nlp=spacy.load('en_core_web_sm'); print('spaCy model OK')"
```

---

# Step 13: Create Project Directory

```powershell
mkdir D:\ai_ml_app\nlp_course
cd D:\ai_ml_app\nlp_course
```

Open VS Code:

```powershell
code .
```

---

# Step 14: Select Python Interpreter

Inside VS Code:

1. Press `Ctrl + Shift + P`
2. Search for:

```text
Python: Select Interpreter
```

3. Choose:

```text
Python 3.12.x ('nlp_course_env')
```

---

# Step 15: Verify Complete Installation

Create test file:

```powershell
@"
import sys
import numpy as np
import pandas as pd
import spacy
import transformers
import torch

print(f'Python       {sys.version.split()[0]}')
print(f'NumPy        {np.__version__}')
print(f'Pandas       {pd.__version__}')
print(f'Transformers {transformers.__version__}')
print(f'Torch        {torch.__version__}')
print(f'spaCy        {spacy.__version__}')

nlp = spacy.load('en_core_web_sm')
doc = nlp('Apple is opening a new office in London.')

print()
print('Named Entities:')

for ent in doc.ents:
    print(f'  {ent.text:<15} {ent.label_}')
"@ | Out-File -Encoding utf8 test_setup.py
```

Run:

```powershell
python test_setup.py
```

Expected:

```text
Python       3.12.x
NumPy        1.26.4
Pandas       2.2.3
Transformers 4.47.1
Torch        2.5.1
spaCy        3.8.3

Named Entities:
  Apple           ORG
  London          GPE
```

---

# Package Versions

| Package         | Version      |
| --------------- | ------------ |
| Python          | 3.12         |
| NumPy           | 1.26.4       |
| Pandas          | 2.2.3        |
| Matplotlib      | 3.10.0       |
| Seaborn         | 0.13.2       |
| Scikit-Learn    | 1.6.0        |
| NLTK            | 3.9.1        |
| spaCy           | 3.8.3        |
| TextBlob        | 0.18.0.post0 |
| VADER Sentiment | 3.3.2        |
| Gensim          | 4.3.3        |
| Transformers    | 4.47.1       |
| PyTorch         | 2.5.1        |
| ipykernel       | latest       |

---

# Useful Commands

Activate environment:

```powershell
conda activate nlp_course_env
```

Deactivate:

```powershell
conda deactivate
```

List environments:

```powershell
conda env list
```

List packages:

```powershell
pip list
```

Export requirements:

```powershell
pip freeze > requirements.txt
```

Install requirements:

```powershell
pip install -r requirements.txt
```

Delete environment:

```powershell
conda remove -n nlp_course_env --all
```

---

# Troubleshooting

## Check Active Python

```powershell
python -c "import sys; print(sys.executable)"
```

Expected:

```text
C:\Users\<username>\Miniconda3\envs\nlp_course_env\python.exe
```

## Conda Not Found

```powershell
$env:Path = "$env:USERPROFILE\Miniconda3\Scripts;" + $env:Path
conda init powershell
```

Close and reopen PowerShell.

## VS Code Command Not Found

```powershell
$env:Path += ";$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin"
```

## Validate spaCy Installation

```powershell
python -m spacy validate
```

## Validate PyTorch Installation

```powershell
python -c "import torch; print(torch.__version__)"
``
