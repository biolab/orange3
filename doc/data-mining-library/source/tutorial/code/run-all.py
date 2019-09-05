"""
Runs all the scripts in the directory.
"""

import glob
import sys

scripts = set(glob.glob("*.py")).difference([sys.argv[0]])
for script in scripts:
    print("-" * 70)
    print(script)
    print()
    exec(open(script).read())
