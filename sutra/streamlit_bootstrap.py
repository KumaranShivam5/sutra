# utils/streamlit_bootstrap.py   (add this *instead* of the class above, or side‑by‑side)
def _install_tqdm_as_streamlit_bar():
    """
    Replaces tqdm with a very thin wrapper that updates only a native
    Streamlit progress widget.  The textual bar is completely suppressed.
    """
    from tqdm.notebook import tqdm as _BaseTqdm

    class StreamlitBarOnly(_BaseTqdm):
        def __init__(self, *args, **kwargs):
            # We do NOT pass a `file=` argument – we suppress the textual output.
            # Extract an optional placeholder for the progress widget.
            self._st_placeholder = kwargs.pop("st_progress", None)
            super().__init__(*args, **kwargs)

        def display(self, msg: str | None = None, pos: int | None = None):
            # `self.n` is the current counter, `self.total` the max.
            if self._st_placeholder is not None and self.total:
                fraction = self.n / self.total
                self._st_placeholder.progress(fraction)
            # Do **not** call the parent `display` – that would write the textual bar.

    # Replace the name in the tqdm module.
    import tqdm
    tqdm.tqdm = StreamlitBarOnly