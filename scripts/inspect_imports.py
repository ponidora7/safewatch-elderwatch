#!/usr/bin/env python
import traceback
def try_import(name):
    try:
        m = __import__(name)
        print(f"IMPORT OK: {name} -> {getattr(m, '__file__', 'no __file__')}")
    except Exception as e:
        print(f"IMPORT FAIL: {name} -> {e}")
        traceback.print_exc()

try_import('dateutil')
print('---')
try_import('mediapipe')
