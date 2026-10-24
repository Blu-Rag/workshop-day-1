# Getting Started

## Clone the repository

Replace `<repository-url>` with the URL for this repository:

```bash
git clone <repository-url>
cd workshop-day-1
```

## Create and activate a virtual environment

Create the environment from the repository directory:

```bash
python -m venv .venv
```

Activate it using the command for your shell:

| Shell | Activation command |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.\.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

When the environment is active, install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure environment variables

Some examples use API credentials. Create a `.env` file in the repository root and add the variable names needed by the example you want to run. This template contains no credentials:

```dotenv
GOOGLE_API_KEY=
GROQ_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
MISTRAL_API_KEY=
OPENWEATHER_API_KEY=
```

Fill in only the credentials for services you use. Keep `.env` private; it is ignored by Git and should not be committed or shared.
