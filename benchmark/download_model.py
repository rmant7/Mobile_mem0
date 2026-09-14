"""
Resolves and downloads the actual GGUF file from cstr/multilingual-e5-base-GGUF
at run time — never a hardcoded filename. Quantized-model repos get
restructured on a timescale of weeks (files renamed, added, removed); this
mirrors the same resolve-at-download-time approach
ai.localstudio.app.models.HuggingFaceResolver (in the AI app repo) uses for
exactly that reason, just in Python instead of Kotlin.

Never commits the GGUF anywhere — huggingface_hub's own cache directory
(~/.cache/huggingface by default, or Colab's ephemeral disk) is where it
lands; nothing here writes it into this repo or into any git-tracked path.
"""
from __future__ import annotations

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "cstr/multilingual-e5-base-GGUF"

# Q4_K_M first — the standard quality/size tradeoff for llama.cpp on a
# laptop or Colab's own (limited, shared) CPU quota; Q4_K_S/plain Q4_K as
# fallbacks if this repo only offers those variants. Deliberately not IQ4_XS
# or F16: Base's Q4_K is the reference quant this benchmark exists to
# measure, not the smallest or the most precise one available.
QUANT_PRIORITY = ["Q4_K_M", "Q4_K_S", "Q4_K"]


def resolve_filename(repo_id: str = REPO_ID) -> str:
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.lower().endswith(".gguf")]
    if not files:
        raise RuntimeError(f"no .gguf file found in {repo_id}")

    for quant in QUANT_PRIORITY:
        for f in files:
            if quant.lower() in f.rfilename.lower():
                return f.rfilename

    # Unrecognised naming convention: fall back to the smallest file by
    # actual size — same fallback ai.localstudio.core.registry.ArtifactResolver
    # uses on the Kotlin side, the safer default when the quant/size tradeoff
    # of what's actually offered can't be read from the filename.
    return min(files, key=lambda f: f.size or float("inf")).rfilename


def download(repo_id: str = REPO_ID, filename: str | None = None) -> str:
    """Downloads (or reuses huggingface_hub's local cache for) the resolved file; returns its local path."""
    filename = filename or resolve_filename(repo_id)
    return hf_hub_download(repo_id=repo_id, filename=filename)


if __name__ == "__main__":
    path = download()
    print(path)
