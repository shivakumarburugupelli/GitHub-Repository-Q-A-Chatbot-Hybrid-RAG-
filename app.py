import os
from urllib.parse import urlparse

import requests
import streamlit as st

# Defaults to the local API. Set API_URL to use a deployed API.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("GitHub Repository Q&A Chatbot")

github_url = st.text_input("Enter GitHub repository URL")


def is_valid_github_repo_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    parts = [part for part in parsed.path.strip("/").split("/") if part]

    return (
        parsed.scheme in ("http", "https")
        and parsed.netloc.lower() in ("github.com", "www.github.com")
        and len(parts) >= 2
    )


if github_url:
    if is_valid_github_repo_url(github_url):
        st.success("Valid GitHub repository URL")
    else:
        st.error("Enter a valid GitHub repository URL, such as https://github.com/owner/repository.")


if st.button("Load repository"):
    if not github_url or not is_valid_github_repo_url(github_url):
        st.warning("Please enter a valid GitHub repository URL.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/load",
                json={"github_url": github_url.strip()},
                timeout=60,
            )

            if response.ok:
                result = response.json()
                st.success(result.get("message", "Repository loaded successfully."))
            else:
                st.error(
                    f"Failed to load repository. "
                    f"Status code: {response.status_code}. {response.text}"
                )
        except requests.RequestException as exc:
            st.error(f"Could not connect to the API: {exc}")
        except ValueError:
            st.error("The API returned an invalid JSON response.")

st.subheader("Ask a question")

question = st.text_input("Ask a question about the repository")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question.strip()},
                timeout=(10,300),
            )

            if response.ok:
                result = response.json()
                answer = result.get("answer")

                if answer is not None:
                    st.write(answer)
                else:
                    st.error("The API response did not contain an answer.")
            else:
                st.error(f"Error: {response.status_code}. {response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not connect to the API: {exc}")
        except ValueError:
            st.error("The API returned an invalid JSON response.")