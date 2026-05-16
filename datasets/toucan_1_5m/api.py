from typing import Optional


def web_search_exa(query: str, numResults: int = 5) -> dict:
    """Search the web using Exa AI - performs real-time web searches and can scrape content from specific URLs.

    Args:
    ----
        query: Search query
        numResults: Number of search results to return (default: 5)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def research_paper_search_exa(query: str, numResults: int = 5) -> dict:
    """Search for academic papers and research using Exa AI - specializes in finding scholarly articles, research papers, and academic content.

    Args:
    ----
        query: Research paper search query
        numResults: Number of research papers to return (default: 5)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def company_research_exa(companyName: str, numResults: int = 5) -> dict:
    """Research companies using Exa AI - finds comprehensive information about businesses, organizations, and corporations.

    Args:
    ----
        companyName: Name of the company to research
        numResults: Number of search results to return (default: 5)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def crawling_exa(url: str, maxCharacters: int = 3000) -> dict:
    """Extract and crawl content from specific URLs using Exa AI - retrieves full text content, metadata, and structured information from web pages.

    Args:
    ----
        url: URL to crawl and extract content from
        maxCharacters: Maximum characters to extract (default: 3000)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def competitor_finder_exa(
    companyName: str, industry: Optional[str] = None, numResults: int = 5
) -> dict:
    """Find competitors for a business using Exa AI - identifies similar companies, competitive landscape analysis, and market positioning.

    Args:
    ----
        companyName: Name of the company to find competitors for
        industry: Industry sector (optional, helps narrow search)
        numResults: Number of competitors to find (default: 5)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def linkedin_search_exa(query: str, searchType: str = "all", numResults: int = 5) -> dict:
    """Search LinkedIn profiles and companies using Exa AI - finds professional profiles, company pages, and business-related content on LinkedIn.

    Args:
    ----
        query: LinkedIn search query (e.g., person name, company, job title)
        searchType: Type of LinkedIn content to search (default: all)
        numResults: Number of LinkedIn results to return (default: 5)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def wikipedia_search_exa(query: str, numResults: int = 5) -> dict:
    """Search Wikipedia articles using Exa AI - finds comprehensive, factual information from Wikipedia entries.

    Args:
    ----
        query: Wikipedia search query (topic, person, place, concept, etc.)
        numResults: Number of Wikipedia articles to return (default: 5)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def github_search_exa(query: str, searchType: str = "all", numResults: int = 5) -> dict:
    """Search GitHub repositories and code using Exa AI - finds repositories, code snippets, documentation, and developer profiles on GitHub.

    Args:
    ----
        query: GitHub search query (repository name, programming language, username, etc.)
        searchType: Type of GitHub content to search (default: all)
        numResults: Number of GitHub results to return (default: 5)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def deep_researcher_start(instructions: str, model: str = "exa-research") -> dict:
    """Start a comprehensive AI-powered deep research task for complex queries.

    Args:
    ----
        instructions: Complex research question or detailed instructions for the AI researcher
        model: Research model: 'exa-research' or 'exa-research-pro' (default: exa-research)

    Returns:
    -------
        dict: Response from the API

    """
    pass


def deep_researcher_check(taskId: str) -> dict:
    """Check the status and retrieve results of a deep research task.

    Args:
    ----
        taskId: The task ID returned from deep_researcher_start tool

    Returns:
    -------
        dict: Response from the API

    """
    pass
