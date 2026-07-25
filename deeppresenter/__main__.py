#!/usr/bin/env python3
"""Slidex command-line entry point.

Use ``slidex onboard`` for setup and ``slidex generate`` to create a deck.
The historical ``pptagent`` and ``deeppresenter`` commands remain compatibility aliases.
"""

from deeppresenter.cli import main

if __name__ == "__main__":
    main()
