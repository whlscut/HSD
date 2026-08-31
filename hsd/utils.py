"""Shared utilities for the HSD pipelines."""

import re
from io import BytesIO

import requests
from PIL import Image


def primarily_clean_draft(draft_text):
    """Clean a draft block produced by the pipeline drafter (PP-StructureV3):
    strip whitespace, collapse repeated spaces, and normalize full-width
    ASCII characters to half-width so that the draft tokenizes consistently
    with the target model's output.
    """
    def str_q2b_ascii(ustring: str) -> str:
        res = []
        for uchar in ustring:
            code = ord(uchar)
            # full-width space -> half-width space
            if code == 0x3000:
                res.append(" ")
            # full-width ASCII characters (! to ~)
            elif 0xFF01 <= code <= 0xFF5E:
                res.append(chr(code - 0xFEE0))
            else:
                res.append(uchar)
        return "".join(res)

    draft_text = draft_text.strip()

    # collapse repeated whitespace into a single space
    draft_text = re.sub(r"\s{2,}", " ", draft_text)

    draft_text = str_q2b_ascii(draft_text)

    draft_text = draft_text.replace("。 ", "。")
    draft_text = draft_text.replace("[ ", "[")
    draft_text = draft_text.replace("] ", "]")

    draft_text = draft_text.replace("<html>", "")
    draft_text = draft_text.replace("</html>", "")
    draft_text = draft_text.replace("<body>", "")
    draft_text = draft_text.replace("</body>", "")

    return draft_text


def clean_repeated_substrings(text):
    """Remove degenerate repetitions that a model occasionally produces at the
    end of long generations (e.g. the same token/phrase looped many times)."""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[: n - length * (count - 1)]

    return text


def get_image(input_source):
    """Load an image from a local path or URL."""
    if input_source.startswith(("http://", "https://")):
        response = requests.get(input_source)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(input_source)
