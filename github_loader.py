import json
import logging
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)
GITHUB_API_URL = "https://api.github.com"

MAX_FILE_SIZE_BYTES = int(os.getenv("GITHUB_MAX_FILE_SIZE_KB", "200")) * 1024
MAX_FILES = int(os.getenv("GITHUB_MAX_FILES", "200"))
MAX_DEPTH = int(os.getenv("GITHUB_MAX_DEPTH", "10"))
DOWNLOAD_WORKERS = int(os.getenv("GITHUB_DOWNLOAD_WORKERS", "8"))

REQUEST_TIMEOUT = 30          # seconds per HTTP request
MAX_RETRIES = 3               # retries for network errors / 5xx responses
PER_PAGE = 100                # page size used for paginated list responses

SUPPORTED_EXTENSIONS = {
    ".py", ".ipynb",
    ".js", ".jsx", ".ts", ".tsx", ".html", ".htm", ".css", ".scss", ".vue",
    ".java", ".kt", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb",
    ".php", ".swift", ".scala", ".r", ".sql", ".sh",
    ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".xml", ".gradle",
}

SUPPORTED_FILENAMES = {"dockerfile", "makefile", "procfile", "readme", "requirements"}

IGNORED_DIRECTORIES = {
    "node_modules", "venv", "env", "__pycache__", "dist", "build", "target",
    "vendor", "site-packages", "bower_components", "coverage", "htmlcov",
    "migrations", "chroma_db", "images", "img", "assets", "static", "fonts",
}

IGNORED_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "pipfile.lock", "composer.lock", "cargo.lock", ".ds_store",
}

IGNORED_SUFFIXES = (".min.js", ".min.css", ".map", ".lock")
class GitHubLoaderError(Exception):
    """Any problem loading a repository. ``status_code`` is a suggested HTTP code."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class GitHubRateLimitError(GitHubLoaderError):
    """GitHub API rate limit exceeded."""

    def __init__(self, message: str):
        super().__init__(message, status_code=429)
def parse_github_url(url: str) -> Tuple[str, str, Optional[str], str]:
    """
    Return ``(owner, repo, ref, subpath)`` for a GitHub URL.

    Supports:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      https://github.com/owner/repo/tree/<branch>/<sub/folder>
    ``ref`` is ``None`` when the URL does not name a branch (default branch used).
    """
    if not url or not url.strip():
        raise GitHubLoaderError("Please enter a GitHub repository URL.")

    url = url.strip()
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    if parsed.netloc.lower() not in ("github.com", "www.github.com"):
        raise GitHubLoaderError(
            "Invalid GitHub URL. Expected something like "
            "https://github.com/<owner>/<repository>"
        )

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise GitHubLoaderError(
            "Invalid GitHub URL. Expected something like "
            "https://github.com/<owner>/<repository>"
        )

    owner, repo = parts[0], parts[1]
    if repo.lower().endswith(".git"):
        repo = repo[:-4]

    ref: Optional[str] = None
    subpath = ""
    if len(parts) >= 4 and parts[2] in ("tree", "blob"):
        ref = parts[3]
        subpath = "/".join(parts[4:])

    return owner, repo, ref, subpath


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def _build_session(token: Optional[str]) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-repository-qa-chatbot",
        }
    )
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def _error_message(response: requests.Response) -> str:
    try:
        return response.json().get("message", response.text[:200])
    except ValueError:
        return response.text[:200]


def _raise_for_github_error(response: requests.Response, has_token: bool) -> None:
    """Translate a non-200 GitHub response into a clear exception."""
    status = response.status_code
    message = _error_message(response)

    if status == 401:
        raise GitHubLoaderError(
            "GitHub rejected the token (401). Check GITHUB_TOKEN in your .env file.",
            status_code=401,
        )

    if status in (403, 429):
        remaining = response.headers.get("X-RateLimit-Remaining")
        is_rate_limited = (
            remaining == "0"
            or status == 429
            or "rate limit" in message.lower()
        )
        if is_rate_limited:
            reset = response.headers.get("X-RateLimit-Reset")
            when = ""
            if reset and reset.isdigit():
                when = " It resets at " + datetime.fromtimestamp(
                    int(reset), tz=timezone.utc
                ).strftime("%H:%M:%S UTC") + "."
            hint = (
                ""
                if has_token
                else " Add a GITHUB_TOKEN to your .env file to raise the limit "
                "from 60 to 5000 requests per hour."
            )
            raise GitHubRateLimitError(
                f"GitHub API rate limit exceeded.{when}{hint}"
            )
        raise GitHubLoaderError(f"GitHub denied access (403): {message}", status_code=403)

    if status == 404:
        raise GitHubLoaderError(
            "Repository, branch or path not found (404). If the repository is "
            "private, set GITHUB_TOKEN in your .env file.",
            status_code=404,
        )

    raise GitHubLoaderError(
        f"GitHub API error {status}: {message}",
        status_code=502 if status >= 500 else 400,
    )


def _request(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> requests.Response:
    """GET with retries for network errors and 5xx responses."""
    has_token = "Authorization" in session.headers
    last_problem = "unknown error"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            last_problem = str(exc)
            logger.warning("GitHub request failed (attempt %d): %s", attempt, exc)
            time.sleep(2 ** (attempt - 1))
            continue

        if response.status_code == 200:
            return response

        if response.status_code >= 500:
            last_problem = f"HTTP {response.status_code}"
            logger.warning("GitHub returned %d (attempt %d)", response.status_code, attempt)
            time.sleep(2 ** (attempt - 1))
            continue

        _raise_for_github_error(response, has_token)

    raise GitHubLoaderError(
        f"Could not reach the GitHub API: {last_problem}", status_code=502
    )


def _get_json(session: requests.Session, url: str, params: Optional[dict] = None):
    """GET JSON; if the result is a list, follow ``rel="next"`` pagination links."""
    response = _request(session, url, params=params)
    data = response.json()

    if isinstance(data, list):
        while "next" in response.links:
            response = _request(session, response.links["next"]["url"])
            page = response.json()
            if not isinstance(page, list):
                break
            data.extend(page)
    return data


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------
def _is_ignored_directory(name: str) -> bool:
    lowered = name.lower()
    if lowered in IGNORED_DIRECTORIES:
        return True
    # Hidden folders (.git, .idea, .vscode, .venv ...) except .github
    return lowered.startswith(".") and lowered != ".github"


def _is_supported_file(name: str, size: int) -> bool:
    lowered = name.lower()
    if lowered in IGNORED_FILENAMES or lowered.endswith(IGNORED_SUFFIXES):
        return False
    if size <= 0 or size > MAX_FILE_SIZE_BYTES:
        return False

    extension = os.path.splitext(lowered)[1]
    if extension in SUPPORTED_EXTENSIONS:
        return True
    stem = lowered if not extension else os.path.splitext(lowered)[0]
    return lowered in SUPPORTED_FILENAMES or stem in SUPPORTED_FILENAMES
def _list_repository_files(
    session: requests.Session,
    owner: str,
    repo: str,
    ref: Optional[str],
    start_path: str,
) -> List[Dict]:
    """
    Breadth-first walk of the repository through the Contents API.
    Returns the metadata (not the content) of every supported file.
    """
    params = {"per_page": PER_PAGE}
    if ref:
        params["ref"] = ref

    candidates: Dict[str, Dict] = {}   # keyed by path -> no duplicates
    skipped_large = 0
    queue = deque([(start_path, 0)])

    while queue:
        path, depth = queue.popleft()
        url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}".rstrip("/")
        data = _get_json(session, url, params=params)
        if isinstance(data, dict):
            data = [data]

        for item in data:
            item_type = item.get("type")
            name = item.get("name", "")

            if item_type == "dir":
                if depth + 1 > MAX_DEPTH:
                    continue
                if _is_ignored_directory(name):
                    continue
                queue.append((item["path"], depth + 1))

            elif item_type == "file":
                size = item.get("size", 0)
                if size > MAX_FILE_SIZE_BYTES:
                    skipped_large += 1
                    continue
                if _is_supported_file(name, size):
                    candidates[item["path"]] = item

    if skipped_large:
        logger.info("Skipped %d file(s) larger than %d bytes", skipped_large, MAX_FILE_SIZE_BYTES)
    return list(candidates.values())


def _priority(item: Dict) -> Tuple[int, int, str]:
    """README files first, then shallow files, then alphabetical."""
    path = item.get("path", "")
    is_readme = 0 if os.path.basename(path).lower().startswith("readme") else 1
    return (is_readme, path.count("/"), path.lower())

def _notebook_to_text(raw: str) -> str:
    """Keep only the code and markdown cells of a Jupyter notebook (no outputs)."""
    try:
        notebook = json.loads(raw)
    except ValueError:
        return raw

    parts = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = source.strip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(source)
        else:
            parts.append("# [code cell]\n" + source)
    return "\n\n".join(parts)


def _fetch_file_text(session: requests.Session, item: Dict) -> Optional[str]:
    """
    Download one file. Uses ``download_url`` first (does not consume the REST
    API rate limit) and falls back to the Contents API.
    Returns ``None`` for binary / empty / unreadable files.
    """
    content: Optional[bytes] = None

    download_url = item.get("download_url")
    if download_url:
        try:
            content = _request(session, download_url, headers={"Accept": "*/*"}).content
        except GitHubRateLimitError:
            raise
        except GitHubLoaderError as exc:
            logger.warning("download_url failed for %s: %s", item.get("path"), exc)

    if content is None and item.get("url"):
        try:
            response = _request(
                session,
                item["url"],
                headers={"Accept": "application/vnd.github.raw+json"},
            )
            content = response.content
        except GitHubRateLimitError:
            raise
        except GitHubLoaderError as exc:
            logger.warning("Skipping %s: %s", item.get("path"), exc)
            return None

    if not content:
        return None
    if b"\x00" in content[:8000]:  # binary file
        return None

    text = content.decode("utf-8", errors="ignore").strip()
    if not text:
        return None

    if item.get("name", "").lower().endswith(".ipynb"):
        text = _notebook_to_text(text)
    return text or None
def load_repository_documents(
    github_url: str, token: Optional[str] = None
) -> List[Document]:
    """
    Fetch the supported files of a GitHub repository and return one
    ``Document`` per file. Metadata: source (file path), file_name,
    file_type, repo, url, size.
    """
    token = token or os.getenv("GITHUB_TOKEN") or None
    owner, repo, ref, subpath = parse_github_url(github_url)
    session = _build_session(token)

    logger.info("Listing files of %s/%s", owner, repo)
    candidates = _list_repository_files(session, owner, repo, ref, subpath)
    if not candidates:
        raise GitHubLoaderError(
            "No supported source or documentation files were found in this repository."
        )

    candidates.sort(key=_priority)
    if len(candidates) > MAX_FILES:
        logger.warning(
            "Repository has %d supported files; loading at most %d (README and "
            "top-level files first).", len(candidates), MAX_FILES,
        )

    def download(item: Dict) -> Optional[Document]:
        text = _fetch_file_text(session, item)
        if text is None:
            return None
        return Document(
            page_content=text,
            metadata={
                "source": item["path"],
                "file_name": item["name"],
                "file_type": os.path.splitext(item["name"])[1].lower() or item["name"].lower(),
                "repo": f"{owner}/{repo}",
                "url": item.get("html_url", ""),
                "size": item.get("size", 0),
            },
        )

    documents: List[Document] = []
    position = 0
    with ThreadPoolExecutor(max_workers=max(1, DOWNLOAD_WORKERS)) as executor:
    
        while position < len(candidates) and len(documents) < MAX_FILES:
            batch = candidates[position: position + (MAX_FILES - len(documents))]
            position += len(batch)
            documents.extend(doc for doc in executor.map(download, batch) if doc is not None)

    if not documents:
        raise GitHubLoaderError("Files were found but none of them could be read as text.")

    logger.info("Loaded %d file(s) from %s/%s", len(documents), owner, repo)
    return documents