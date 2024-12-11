from typing import List
from pathvalidate import sanitize_filename
import requests
from typing_extensions import TypedDict
from playwright.sync_api import sync_playwright, Download
from playwright.sync_api import Page
from pathlib import Path
from functools import partial
import tempfile

class DownloadedURL(TypedDict):
    url: str
    local_path: str

def download_started(downloaded_urls: List[DownloadedURL], download_dir: str, download: Download):
    download_filename = download.suggested_filename
    download_filename = sanitize_filename(download_filename).replace(' ', '_')

    final_location = Path(download_dir) / download_filename
    download.save_as(final_location)

    print('download saved to:', final_location)
    downloaded_urls.append({"url": download.url, "local_path": str(final_location)})

def download_urls(page: Page, urls: List[str], download_dir: str) -> List[DownloadedURL]:
    downloaded_urls = []
    for url in urls:
        # use tmp html to download
        page.on("download", partial(download_started, downloaded_urls, download_dir))
        with page.expect_download() as download_info:
            try:
                page.goto(url, wait_until="networkidle")
            except Exception as e:
                if "Download is starting" not in str(e):
                    print('Error:', e)

    return downloaded_urls

def download_urls_in_headed_chrome(urls: List[str], download_dir: str) -> List[DownloadedURL]:
    with sync_playwright() as p:
        browser = p.firefox.launch(headless = False)
        context = browser.new_context(ignore_https_errors = True)

        page = context.new_page()
        return download_urls(page, urls=urls, download_dir=download_dir)

def download_urls_with_requests(urls: List[str], download_dir: str) -> List[DownloadedURL]:
    downloaded_urls = []
    for url in urls:
        response = requests.get(url)
        filename = url.split('/')[-1]
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, dir=download_dir, prefix=filename, suffix=suffix) as tmp_file:
            tmp_file.write(response.content)
            local_path = Path(tmp_file.name)
        downloaded_urls.append(DownloadedURL(url=url, local_path=str(local_path)))

    return downloaded_urls