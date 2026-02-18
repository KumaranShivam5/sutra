# utils/streamlit_logger.py  (copy‑paste this file)
import streamlit as st
from typing import List

import traceback
import sys

# def message(msg, type=1):
#     tp = {
#         1 : '>>>>> [INFO] >>>>>> ' , 
#         2 : '>>>>> [DEBUG] >>>>> ' , 
#         3 : '>>>>> [PROCESS] >>>>> ' , 
#     }
#     print(f'{tp[type]} {msg}')
#     # st.write(f'{tp[type]} {msg}')
#     return None

def message(msg, type=1):
    tp = {
        1 : '>>>>> [INFO] >>>>>> ' , 
        2 : '>>>>> [DEBUG] >>>>> ' , 
        3 : '>>>>> [PROCESS] >>>>> ' , 
        'i' : '>>>>> [INFO] >>>>>> ' , 
        'd' : '>>>>> [DEBUG] >>>>> ' , 
        'p' : '>>>>> [PROCESS] >>>>> ' , 
        'w' : '>>>>> [WARNING] >>>>> ' , 
        'e' : '>>>>> [ERROR] >>>>> ' , 
        'fe' : '>>>>> [Failed-FIT] >>>>> ' , 
    }
    if type == 'e':

        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_traceback is not None:
            file_name = exc_traceback.tb_frame.f_code.co_filename
            line_number = exc_traceback.tb_lineno
            msg = f"{msg} | File: {file_name}, Line: {line_number}"
    print(f'{tp[type]} {msg}')
    # st.write(f'{tp[type]} {msg}')
    return None

class StreamlitWriter:
    """
    Simple file‑like object that forwards text to a Streamlit placeholder.
    It works for `print`, `tqdm`, or any library that writes to a file.
    """

    def __init__(self, placeholder: st.delta_generator.DeltaGenerator):
        self._placeholder = placeholder
        self._buffer = ""
        self._history: List[str] = []

    # ------------------------------------------------------------------
    # Required file‑like methods
    # ------------------------------------------------------------------
    def write(self, s: str) -> None:
        self._buffer += s
        if "\n" in self._buffer:
            # split on newlines – everything except the last piece is a full line
            lines = self._buffer.split("\n")
            for line in lines[:-1]:
                self._emit(line)
            self._buffer = lines[-1]          # keep the incomplete tail

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    # ------------------------------------------------------------------
    # Private helper that updates Streamlit
    # ------------------------------------------------------------------
    def _emit(self, line: str) -> None:
        line = line.rstrip("\r")
        self._history.append(line)
        # `write` updates the same placeholder – no new widget each call.
        self._placeholder.write("\n".join(self._history))

    # ------------------------------------------------------------------
    # Optional convenience API
    # ------------------------------------------------------------------
    def get_log(self) -> str:
        """Return the whole captured log (useful for download)."""
        return "\n".join(self._history)

    def clear(self) -> None:
        """Erase UI and internal buffer."""
        self._history.clear()
        self._placeholder.empty()