from tools import add_urls_to_db, get_vector_store
import logging
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from rich.progress import Progress

logging.basicConfig(level=logging.INFO)
db = get_vector_store()
urls = [] # Add URLs here

partial_func = partial(add_urls_to_db, db=db, use_gemini=True)

def process_url(url, progress, task_id):
    result = partial_func([url])
    progress.update(task_id, advance=1)
    return result

with Progress() as progress:
    task_id = progress.add_task("[green]Processing URLs...", total=len(urls))
    with Pool(4) as pool:
        pool.starmap(process_url, [(url, progress, task_id) for url in urls])