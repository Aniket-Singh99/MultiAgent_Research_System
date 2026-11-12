
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
import os

load_dotenv()

# Initialize Tavily client
tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def tavily_search(query: str) -> str:
    """   
    Search the web for research information using Tavily.
    Returns relevant search results with URLs and content.
    """

    response = tavily.search(
        query=query,
        search_depth="basic",
        max_results=3
    )

    results = []

    for result in response.get("results", []):
        results.append(
            f"""
            Title: {result.get('title')}
            URL: {result.get('url')}
            Content: {result.get('content')}
            """
        )

    return "\n".join(results)


@tool
def scrape_url(url: str) -> str:
    """
    Scrape a webpage and extract its readable text content.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary HTML elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract readable text
        content = soup.get_text(
            separator=" ",
            strip=True
        )

        return content[:6000]

    except requests.exceptions.RequestException as e:
        return f"Error scraping URL: {str(e)}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"



research_tools = [
    tavily_search,
    scrape_url
]
