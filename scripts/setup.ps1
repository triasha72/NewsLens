$ErrorActionPreference = "Stop"

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3.12 -m venv .venv
} elseif (Get-Command python3.12 -ErrorAction SilentlyContinue) {
    python3.12 -m venv .venv
} else {
    throw "Python 3.12 was not found."
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\python.exe -m ruff check .

Write-Host "NewsLens setup complete."
Write-Host "Activate it later with: .\.venv\Scripts\Activate.ps1"
