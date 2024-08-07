from typing import List
from pathvalidate import sanitize_filename
from typing_extensions import TypedDict
from playwright.sync_api import sync_playwright, Download
from playwright.sync_api import Page
from pathlib import Path
from functools import partial

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

        page = browser.new_page()
        return download_urls(page, urls=urls, download_dir=download_dir)
