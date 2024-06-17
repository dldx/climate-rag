# Climate RAG

## RAG pipeline to identify useful web pages and reports on the internet, scrape and ingest them to collect better energy and climate data

Acknowledgements:
* [Pixegami](https://github.com/pixegami/rag-tutorial-v2) for initial RAG workflow
* [Greg Kamradt](https://www.youtube.com/watch?v=8OJC21T2SL4) for chunking strategies


## Installation

Create a virtual environment and install the required packages using the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Add all the relevant API keys to a .env file in the root directory.

```bash
cp .env.example .env
```

## Usage

To run the RAG pipeline, first start a ChromaDB server:

```bash
chroma run --path chroma/
```

Then run the following command:

```bash
python query_data.py "Give me a list of coal power plants in Vietnam"
```

To add new urls to the database, run the following command:

```bash
python populate_database.py --url "https://www.example.com"
```

To add local markdown files to the database, put your files in data/ and run the following command:

```bash
python populate_database.py
```
