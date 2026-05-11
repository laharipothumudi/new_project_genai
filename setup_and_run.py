"""
One-shot setup: scrape catalog → build FAISS index → start server.
Run: python setup_and_run.py
"""
import os, subprocess, sys

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, check=True)
    return result

if __name__ == "__main__":
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()

    if not os.path.exists("catalog.json"):
        run(f"{sys.executable} scraper.py")
    else:
        print("catalog.json already exists, skipping scrape.")

    if not os.path.exists("catalog_index.pkl"):
        run(f"{sys.executable} vector_store.py")
    else:
        print("catalog_index.pkl already exists, skipping index build.")

    run(f"{sys.executable} -m flask --app main run --host 0.0.0.0 --port 8000")
