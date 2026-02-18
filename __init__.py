import os

if not os.environ.get("STREAMLIT_APP_CONTEXT"):
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_CACHE_TYPE"] = "memory"