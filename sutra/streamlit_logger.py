import streamlit as st
import time
from tqdm.notebook import tqdm

# -----------------------------------------------------------------
# 1️⃣  Minimal logger implementation (identical to the one you already use)
# -----------------------------------------------------------------
class _StreamLitLogger:
    def __init__(self, placeholder, max_lines: int = 3):
        self._ph = placeholder
        self.max_lines = max_lines
        self.buffer = []

    def write(self, msg):
        # Keep only the newest `max_lines` messages
        self.buffer.append(msg)
        if len(self.buffer) > self.max_lines:
            self.buffer = self.buffer[-self.max_lines :]
        # Join with line‑breaks and push to the placeholder
        self._ph.text("\n".join(self.buffer))

    def flush(self):
        # Streamlit may call `flush` when the cache is cleared – just ignore it
        pass


# -----------------------------------------------------------------
# 2️⃣  Factory that returns a **mutable** logger and tells Streamlit
#     “don’t replay this object”.  This is the only place where we use
#     `allow_output_mutation=True`.
# -----------------------------------------------------------------
@st.experimental_memo(allow_output_mutation=True)
def get_shared_logger():
    """Create the placeholder once and keep returning the same logger."""
    placeholder = st.empty()                     # UI element lives globally
    return _StreamLitLogger(placeholder, max_lines=5)
