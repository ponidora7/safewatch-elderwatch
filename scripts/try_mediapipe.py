import importlib, traceback
try:
    m = importlib.import_module('mediapipe')
    print('MEDIAPIPE OK ->', getattr(m, '__file__', 'no file'))
except Exception as e:
    print('MEDIAPIPE FAIL ->', e)
    traceback.print_exc()
