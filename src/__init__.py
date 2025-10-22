"""Project package initialisation for :mod:`mini-llm`."""

from __future__ import annotations

import os

# NOTE: HuggingFace tokenizers spawn worker pools that do not interact well with
# the fork semantics used in some of our multiprocessing utilities.  When a
# tokenizer instance is created before a process fork happens, the default
# behaviour is to emit a noisy warning about parallelism being disabled.  We
# avoid the warning – and the associated risk of a deadlock – by explicitly
# disabling tokenizers parallelism globally.  The environment variable must be
# set before any tokenizer code is imported, so we do it at package import time.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
