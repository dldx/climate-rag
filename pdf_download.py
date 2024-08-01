from typing import List
from pathvalidate import sanitize_filename
from typing_extensions import TypedDict
from playwright.sync_api import sync_playwright
from playwright.sync_api import Page
import os
import shutil
import tempfile


class DownloadedURL(TypedDict):
    url: str
    local_path: str

def download_urls(page: Page, urls: List[str], download_dir: str) -> List[DownloadedURL]:
    # generate a temporary html file
    html_links = []
    for url in urls:
        html_links.append(f'<a href="{url}">{url}</a>')
    tmp_html = f"""<html><body>{''.join(html_links)}</body></html>"""


    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = tempfile.NamedTemporaryFile(dir=tmp_dir.name, suffix=".html")
    tmpfile_path = tmp_file.name
    with open(tmpfile_path, 'w', encoding='utf8') as f:
        f.write(tmp_html)

    # use tmp html to download
    page.goto("file://" + tmpfile_path)

    downloaded_urls = []

    for link in page.query_selector_all('//a'):
        url = link.get_attribute('href')
        print('download starting:', url)
        with page.expect_download() as download_info:
            link.click(modifiers=["Alt", ])

        try:

            download_filename = download_info.value.suggested_filename
            download_filename = sanitize_filename(download_filename).replace(' ', '_')

            final_location = os.path.join(download_dir, download_filename)
            shutil.copyfile(download_info.value.path(), final_location)

            print('download saved to:', final_location)
            downloaded_urls.append({"url": url, "local_path": final_location})
        except Exception as e:
            print('Download failed:', url, e)

    tmp_file.close()
    tmp_dir.cleanup()
    return downloaded_urls


def download_urls_in_headed_chrome(urls: List[str], download_dir: str) -> List[DownloadedURL]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = False)

        page = browser.new_page()
        return download_urls(page, urls=urls, download_dir=download_dir)