import asyncio
from playwright.async_api import async_playwright
import json
from anyio import Path
from aiofiles.tempfile import TemporaryDirectory

preference = {
    "plugins": {
        "always_open_pdf_externally": True,
    },
}


async def handle(route):
    response = await route.fetch()
    if 'content-type' in response.headers and response.headers['content-type'] == 'application/pdf':
        response.headers['Content-Disposition'] = 'attachment'
    await route.fulfill(response=response, headers=response.headers)


async def main():
    async with TemporaryDirectory() as d:
        preference_dir = Path(d) / "Default"
        await preference_dir.mkdir(777, parents=True, exist_ok=True)
        # breakpoint()
        # await (preference_dir / "Preferences.json").write_text(json.dumps(preference))

        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(d, headless=False, accept_downloads=True)
            try:
                await context.route("*", handle)
                page = await context.new_page()
                async with page.expect_download() as download_info:
                    try:
                        await page.goto("https://www.uniper.energy/sites/default/files/2023-11/2022-11_Uniper_Capital_Markets_Story.pdf")
                    except:
                        download = await download_info.value
                        print(await download.path())
            finally:
                await context.close()

asyncio.run(main())