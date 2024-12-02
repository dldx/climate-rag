import asyncio
import logging
from typing import AsyncIterator, Iterator, List, Literal

from langchain_core.documents import Document

from langchain_core.document_loaders import BaseLoader

logger = logging.getLogger(__name__)

from ua_parser import user_agent_parser
from functools import cached_property
from typing import List
from time import time
import random

class UserAgent:
    """container for a User-Agent"""

    def __init__(self, string) -> None:
        self.string: str = string
        # Parse the User-Agent string
        self.parsed_string: str = user_agent_parser.Parse(string)
        self.last_used: int = time()

    # Get the browser name
    @cached_property
    def browser(self) -> str:
        return self.parsed_string["user_agent"]["family"]

    # Get the browser version
    @cached_property
    def browser_version(self) -> int:
        return int(self.parsed_string["user_agent"]["major"])

    # Get the operation system
    @cached_property
    def os(self) -> str:
        return self.parsed_string["os"]["family"]

    # Return the actual user agent string
    def __str__(self) -> str:
        return self.string

class Rotator:
    """weighted random user agent rotator"""

    def __init__(self, user_agents: List[UserAgent]):
        # Add User-Agent strings to the UserAgent container
        user_agents = [UserAgent(ua) for ua in user_agents]
        self.user_agents = user_agents

    # Add weight for each User-Agent
    def weigh_user_agent(self, user_agent: UserAgent):
        weight = 1_000
        # Add higher weight for less used User-Agents
        if user_agent.last_used:
            _seconds_since_last_use = time() - user_agent.last_used
            weight += _seconds_since_last_use
        # Add higher weight based on the browser
        if user_agent.browser == "Chrome":
            weight += 100
        if user_agent.browser == "Firefox" or "Edge":
            weight += 50
        if user_agent.browser == "Chrome Mobile" or "Firefox Mobile":
            weight += 0
        # Add higher weight for higher browser versions
        if user_agent.browser_version:
            weight += user_agent.browser_version * 10
        # Add higher weight based on the OS type
        if user_agent.os == "Windows":
            weight += 150
        if user_agent.os == "Mac OS X":
            weight += 100
        if user_agent.os == "Linux" or "Ubuntu":
            weight -= 50
        if user_agent.os == "Android":
            weight -= 100
        return weight

    def get(self):
        # Weigh all User-Agents
        user_agent_weights = [
            self.weigh_user_agent(user_agent) for user_agent in self.user_agents
        ]
        # Select a random User-Agent
        user_agent = random.choices(
            self.user_agents,
            weights=user_agent_weights,
            k=1,
        )[0]
        # Update the last used time when selecting a User-Agent
        user_agent.last_used = time()
        return user_agent

from collections import Counter

# Some user agents from the list we created earlier
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
    "Mozilla/5.0 (Windows NT 6.1; rv:109.0) Gecko/20100101 Firefox/113.0",
    "Mozilla/5.0 (Android 12; Mobile; rv:109.0) Gecko/113.0 Firefox/113.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/113.0",
]

class AsyncChromiumLoader(BaseLoader):
    """Scrape HTML pages from URLs using a
    headless instance of the Chromium."""

    def __init__(self, urls: List[str], *, headless: bool = True, excluded_resource_types: List[Literal["stylesheet", "script", "image", "font"]] = ["stylesheet", "image", "font"]):
        """Initialize the loader with a list of URL paths.

        Args:
            urls: A list of URLs to scrape content from.
            headless: Whether to run browser in headless mode.

        Raises:
            ImportError: If the required 'playwright' package is not installed.
        """
        self.urls = urls
        self.headless = headless
        self.excluded_resource_types = excluded_resource_types

        try:
            import playwright  # noqa: F401
        except ImportError:
            raise ImportError(
                "playwright is required for AsyncChromiumLoader. "
                "Please install it with `pip install playwright`."
            )

    async def block_aggressively(self, route):
        if (route.request.resource_type in self.excluded_resource_types):
            await route.abort()
        else:
            await route.continue_()
            from collections import Counter

# Some user agents from the list we created earlier

    async def ascrape_playwright(self, url: str) -> str:
        """
        Asynchronously scrape the content of a given URL using Playwright's async API.

        Args:
            url (str): The URL to scrape.

        Returns:
            str: The scraped HTML content or an error message if an exception occurs.

        """
        from playwright.async_api import async_playwright

        logger.info("Starting scraping...")
        results = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                rotator = Rotator(user_agents)
                context = await browser.new_context(user_agent=str(rotator.get()))
                page = await context.new_page()
                await page.route("**/*", self.block_aggressively)
                await page.goto(url, wait_until="domcontentloaded")
                # Scroll down to load more content
                for i in range(5): #make the range as long as needed
                    await page.mouse.wheel(0, 15000)
                    time.sleep(2)
                results = await page.content()  # Simply get the HTML content
                logger.info("Content scraped")
            except Exception as e:
                results = f"Error: {e}"
            await browser.close()
        return results

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load text content from the provided URLs.

        This method yields Documents one at a time as they're scraped,
        instead of waiting to scrape all URLs before returning.

        Yields:
            Document: The scraped content encapsulated within a Document object.

        """
        for url in self.urls:
            html_content = asyncio.run(self.ascrape_playwright(url))
            metadata = {"source": url}
            yield Document(page_content=html_content, metadata=metadata)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """
        Asynchronously load text content from the provided URLs.

        This method leverages asyncio to initiate the scraping of all provided URLs
        simultaneously. It improves performance by utilizing concurrent asynchronous
        requests. Each Document is yielded as soon as its content is available,
        encapsulating the scraped content.

        Yields:
            Document: A Document object containing the scraped content, along with its
            source URL as metadata.
        """
        tasks = [self.ascrape_playwright(url) for url in self.urls]
        results = await asyncio.gather(*tasks)
        for url, content in zip(self.urls, results):
            metadata = {"source": url}
            yield Document(page_content=content, metadata=metadata)
