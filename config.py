PDP = 'PeriodicDerivativePredictor'
MAP = 'MovingAveragePredictor'
MARPS = 'MovingAverageRollingStdPredictor'

paramsAllKPIs = \
    {'cfg_PDP':{'predictor': PDP, 'params':'width=1, period=86400, sigma=3','preprocess': False}} #NOTE PDP actually breaks when you just mix all the timestamps :P This just as an example

paramsPerKPI = \
    {'cfg_all_PDP':
         {'02e99bd4f6cfb33f': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          '046ec29ddf80d62e': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          '07927a9a18fa19ae': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': True},
          '09513ae3e75778a3': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.5', 'preprocess': False},
          '18fbb1d5a5dc099d': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          '1c35dbf57f55f5e4': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          '40e25005ff8992bd': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          '54e8a140f6237526': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.5', 'preprocess': False},
          '71595dd7171f4540': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.5', 'preprocess': False},  # try
          '769894baefea4e9e': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.8', 'preprocess': False},
          '76f4550c43334374': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.5', 'preprocess': False},
          '7c189dd36f048a6c': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.5', 'preprocess': False},
          '88cf3a776ba00e7c': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          '8a20c229e9860d0c': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.5', 'preprocess': False},
          '8bef9af9a922e0b3': {'predictor': PDP, 'params':'width=1, period=86400, sigma=4', 'preprocess': False},
          '8c892e5525f3e491': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          '9bd90500bfd11edb': {'predictor': PDP, 'params':'width=1, period=86400, sigma=5.5', 'preprocess': False},  # 3.5 already better
          '9ee5879409dccef9': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          'a40b1df87e3f1c87': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.5', 'preprocess': False},  # GOOD
          'a5bf5d65261d859a': {'predictor': PDP, 'params':'width=1, period=86400, sigma=4.5', 'preprocess': False},
          'affb01ca2b4f0b45': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3.5', 'preprocess': False},
          'b3b2e6d1a791d63a': {'predictor': PDP, 'params':'width=1, period=86400, sigma=4', 'preprocess': False},
          'c58bfcbacb2822d1': {'predictor': PDP, 'params':'width=1, period=86400, sigma=4', 'preprocess': False},
          'cff6d3c01e6a6bfa': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          'da403e4e3f87c9e0': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False},
          'e0770391decc44ce': {'predictor': PDP, 'params':'width=1, period=86400, sigma=3', 'preprocess': False}}
     }
