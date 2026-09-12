#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Run docformatter with correct multiline string token positions."""

from __future__ import annotations

import tokenize
from collections.abc import Callable, Iterator


_original_generate_tokens = tokenize.generate_tokens


def _generate_tokens(readline: Callable[[], str]) -> Iterator[tokenize.TokenInfo]:
    for token in _original_generate_tokens(readline):
        if token.type == tokenize.STRING and token.start[0] != token.end[0]:
            # Python 3.12.0 can undercount this column after non-ASCII text.
            # untokenize otherwise copies the apparent gap after the string,
            # duplicating source characters, including its closing quotes.
            end_column = len(token.string.rsplit("\n", 1)[-1])
            token = token._replace(end=(token.end[0], end_column))
        yield token


def main() -> int:
    import docformatter

    original = tokenize.generate_tokens
    tokenize.generate_tokens = _generate_tokens
    try:
        return docformatter.main()
    finally:
        tokenize.generate_tokens = original


if __name__ == "__main__":
    raise SystemExit(main())
