import requests
import time
import re
from urllib.parse import urlparse
from typing import List, Dict


def extract_github_code(repo_urls: List[str], allowed_extensions: List[str]) -> Dict[str, str]:
    """
    Extract code files from public GitHub repositories without authentication.
    
    Behavior:
        - For each URL in `repo_urls` (e.g. "https://github.com/user/repo" or "https://github.com/user/repo/"),
          this function:
            1. Parses owner and repo name.
            2. Queries the GitHub REST API to find the repository's default branch.
            3. Calls the Git Trees API on that branch with ?recursive=1 to list all entries.
            4. Filters 'blob' entries by `allowed_extensions`.
            5. Downloads matching files via raw.githubusercontent.com using the detected branch.
            6. Returns a dict mapping "<owner>/<repo>/<path_in_repo>" -> file contents (str).
        - Uses only public endpoints, no authentication.
        - Uses `requests` only (no external dependencies beyond the standard library).
        - Handles multiple repos in one run.
        - Gracefully handles missing repos, rate limits, unreachable files, symlinks, empty repos.
        - Skips non-matching file extensions and entries that are not blobs (e.g., trees, submodules).
        - Implements conservative retry/backoff logic and respects rate-limit reset headers if present.
        - Deterministic: processes repos in the order provided and files in the order returned by GitHub.
    
    Parameters:
        repo_urls (List[str]): List of GitHub repository URLs, e.g. ["https://github.com/user/repo"].
        allowed_extensions (List[str]): List of file extensions to include, e.g. [".py", ".js"].
            Extensions are matched case-insensitively. If the extensions provided lack a leading '.',
            the function will handle them (".py" and "py" both work).
    
    Returns:
        Dict[str, str]: Mapping where keys are "<owner>/<repo>/<path>" and values are file contents as strings.
            Only files that matched the allowed extensions and were successfully downloaded are included.
    
    Notes and limitations:
        - No authentication: subject to GitHub's unauthenticated rate limits (typically low).
        - For extremely large repositories, the git/trees recursive endpoint may return very large responses;
          the function implements retries and will skip repos it cannot fully fetch.
        - Binary files are filtered only by extension; the function does not attempt to detect binary contents.
    
    Example:
        result = extract_github_code(
            ["https://github.com/psf/requests"],
            [".py", ".md"]
        )
        # result keys look like "psf/requests/requests/sessions.py"
    """
    # --- Helper setup ---
    session = requests.Session()
    session.headers.update({
        "User-Agent": "hf-space-github-extractor/1.0 (python-requests)"
    })
    
    # Normalize allowed extensions: ensure they start with dot, case-insensitive matching later
    normalized_exts = set()
    for ext in allowed_extensions:
        e = ext.strip().lower()
        if not e:
            continue
        if not e.startswith('.'):
            e = '.' + e
        normalized_exts.add(e)
    
    # Some useful constants for retries/backoff
    MAX_RETRIES = 5
    BACKOFF_BASE = 0.8  # seconds, will multiply exponentially
    REQUEST_TIMEOUT = 15  # seconds per HTTP call
    
    def _sleep_backoff(attempt: int):
        """Sleep with exponential backoff (capped)."""
        delay = BACKOFF_BASE * (2 ** (attempt - 1))
        # add a small jitter
        delay = delay + (0.1 * (attempt % 3))
        time.sleep(min(delay, 30))
    
    def _parse_github_url(url: str):
        """
        Parse a GitHub repo URL and return (owner, repo) or (None, None) if invalid.
        Accepts formats like:
            - https://github.com/owner/repo
            - https://github.com/owner/repo/
            - https://github.com/owner/repo.git
            - git@github.com:owner/repo.git  (not common in HTTP lists, but we try to support)
        """
        if url.startswith("git@github.com:"):
            # git@github.com:owner/repo.git
            try:
                path = url.split(":", 1)[1]
            except Exception:
                return None, None
            if path.endswith(".git"):
                path = path[:-4]
            parts = path.strip("/").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
            return None, None
        # Parse HTTP/HTTPS
        try:
            p = urlparse(url)
        except Exception:
            return None, None
        if p.netloc.lower() not in ("github.com", "www.github.com"):
            return None, None
        parts = p.path.strip("/").split("/")
        if len(parts) < 2:
            return None, None
        owner, repo = parts[0], parts[1]
        if repo.endswith(".git"):
            repo = repo[:-4]
        return owner, repo
    
    def _get_json_with_retries(url: str, params=None):
        """GET a JSON endpoint with retries and some basic rate-limit handling. Returns (json, status_code, headers)"""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None, None, {}
                _sleep_backoff(attempt)
                continue
            # If rate-limited, GitHub often returns 403 with relevant headers
            if resp.status_code == 403:
                # Check for rate limit headers
                reset = resp.headers.get("X-RateLimit-Reset")
                if reset:
                    try:
                        wait_seconds = max(0, int(reset) - int(time.time()))
                        # If wait_seconds is small, sleep; otherwise skip after one sleep to avoid long waits.
                        if wait_seconds <= 10:
                            time.sleep(wait_seconds + 1)
                            continue
                    except Exception:
                        pass
                # transient 403 - backoff and retry a few times
                if attempt == MAX_RETRIES:
                    try:
                        return resp.json(), resp.status_code, resp.headers
                    except Exception:
                        return None, resp.status_code, resp.headers
                _sleep_backoff(attempt)
                continue
            # For other 5xx codes, backoff
            if 500 <= resp.status_code < 600:
                if attempt == MAX_RETRIES:
                    try:
                        return resp.json(), resp.status_code, resp.headers
                    except Exception:
                        return None, resp.status_code, resp.headers
                _sleep_backoff(attempt)
                continue
            # Otherwise return result (including 404 or 200)
            try:
                return resp.json(), resp.status_code, resp.headers
            except ValueError:
                return None, resp.status_code, resp.headers
        return None, None, {}
    
    def _get_text_with_retries(url: str):
        """GET raw text content with retries. Returns (text or None, status_code, headers)."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    return None, None, {}
                _sleep_backoff(attempt)
                continue
            if resp.status_code == 200:
                # Return decoded text. Requests will decode based on headers; for safety, access text.
                return resp.text, resp.status_code, resp.headers
            if resp.status_code == 403:
                # rate-limit handling similar to JSON
                reset = resp.headers.get("X-RateLimit-Reset")
                if reset:
                    try:
                        wait_seconds = max(0, int(reset) - int(time.time()))
                        if wait_seconds <= 10:
                            time.sleep(wait_seconds + 1)
                            continue
                    except Exception:
                        pass
                if attempt == MAX_RETRIES:
                    return None, resp.status_code, resp.headers
                _sleep_backoff(attempt)
                continue
            if 500 <= resp.status_code < 600:
                if attempt == MAX_RETRIES:
                    return None, resp.status_code, resp.headers
                _sleep_backoff(attempt)
                continue
            # For 404 or other client errors, do not retry beyond a couple attempts
            if attempt == MAX_RETRIES:
                return None, resp.status_code, resp.headers
            _sleep_backoff(attempt)
        return None, None, {}
    
    results: Dict[str, str] = {}
    
    # Pre-compiled regex to extract file extension
    ext_regex = re.compile(r"(\.[A-Za-z0-9_+-]+)$")
    
    for repo_url in repo_urls:
        owner, repo = _parse_github_url(repo_url)
        if not owner or not repo:
            # invalid URL, skip
            continue
        repo_prefix = f"{repo}"
        
        # Step 1: Get repo metadata to discover default branch
        repo_api_url = f"https://api.github.com/repos/{owner}/{repo}"
        repo_meta, status, headers = _get_json_with_retries(repo_api_url)
        if status is None:
            # network problem - skip repo
            continue
        if status == 404:
            # repo not found or private - skip gracefully
            continue
        if status != 200 or not isinstance(repo_meta, dict):
            # unexpected response - skip
            continue
        default_branch = repo_meta.get("default_branch")
        if not default_branch:
            # fallback to main/master heuristic
            default_branch = "main"
        
        # Step 2: Attempt to fetch the repo tree recursively
        # We'll try the default_branch; if that fails (e.g., branch doesn't exist), try "main" then "master"
        branches_to_try = [default_branch]
        if default_branch not in ("main", "master"):
            branches_to_try.extend(["main", "master"])
        seen_tree = None
        used_branch = None
        for branch in branches_to_try:
            tree_api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}"
            params = {"recursive": "1"}
            tree_json, tree_status, tree_headers = _get_json_with_retries(tree_api, params=params)
            if tree_status == 200 and isinstance(tree_json, dict) and "tree" in tree_json:
                seen_tree = tree_json
                used_branch = branch
                break
            # If we got a 404 for this branch, try next
            if tree_status in (404, 422):
                continue
            # For other issues (403 rate-limit), try next after backoff - the helper already performed backoff
            # but do not loop forever
        if seen_tree is None:
            # Could not fetch tree for any branch; skip repo
            continue
        
        entries = seen_tree.get("tree", [])
        if not isinstance(entries, list) or len(entries) == 0:
            # empty repo or no files
            continue
        
        # Step 3: Iterate entries and download blobs that match allowed extensions
        # Only consider entries where "type" == "blob" (file). Skip "tree" (dir) and "commit" (submodule).
        for entry in entries:
            try:
                entry_type = entry.get("type")
                if entry_type != "blob":
                    continue
                path = entry.get("path")
                if not path:
                    continue
                # Determine extension
                m = ext_regex.search(path)
                if not m:
                    continue
                ext = m.group(1).lower()
                if ext not in normalized_exts:
                    continue
                # Build raw URL and fetch
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{used_branch}/{path}"
                text, txt_status, txt_headers = _get_text_with_retries(raw_url)
                if txt_status == 200 and text is not None:
                    key = f"{repo_prefix}/{path}"
                    results[key] = text
                else:
                    # failed to fetch file content; skip
                    continue
            except Exception:
                # In case of unexpected entry structure, continue to next
                continue
    return results
