import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.trend import WMAIndicator, MACD, CCIIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator, \
                        WilliamsRIndicator, ROCIndicator
from ta.volume import AccDistIndexIndicator
from ta.volatility import AverageTrueRange


class TiCalculator:
    FIELDS = ['date', 'sma_5', 'sma_8', 'sma_10', 'wma_5', 'wma_8',
              'wma_10', 'macd', 'cci', 'stoch_k', 'stoch_d', 'williams',
              'rsi', 'roc', 'ad', 'atr']
    MA_WINDOWS = [5, 8, 10]


    def calc_tis(prices):
        df = add_all_ta_features(pd.DataFrame(prices), open='open', high='high',
                                 low='low', close='close', volume='volume')

        tis_np = [df.date] + \
                 TiCalculator._calc_trend_indicators(df) + \
                 TiCalculator._calc_momentum_indicators(df) + \
                 TiCalculator._calc_acc_dist(df) + \
                 TiCalculator._calc_atr(df)

        tis_df = pd.DataFrame(np.vstack(tis_np).T, columns=TiCalculator.FIELDS)
        return tis_df.dropna().to_dict(orient = 'records')


    def _calc_trend_indicators(df):
        smas, wmas = [], []
        for win in TiCalculator.MA_WINDOWS:
            smas.append(df.close.rolling(window=win).mean())
            wmas.append(WMAIndicator(close=df.close, window=win).wma())

        macd = MACD(close=df.close, window_fast=14, window_slow=28)
        cci = CCIIndicator(high=df.high, low=df.low, close=df.close,
                           window=14, constant=0.015)
        return smas + wmas + [ macd.macd(), cci.cci() ]


    def _calc_momentum_indicators(df):
        stochastic = StochRSIIndicator(close=df.close, window=14, smooth1=1, smooth2=10)
        williams = WilliamsRIndicator(high=df.high, low=df.low, close=df.close, lbp=14)
        rsi = RSIIndicator(close=df.close, window=14)
        roc = ROCIndicator(close=df.close, window=20)
        return [ stochastic.stochrsi_k(), stochastic.stochrsi_d(),
                 williams.williams_r(), rsi.rsi(), roc.roc() ]


    def _calc_acc_dist(df):
        ad = AccDistIndexIndicator(high=df.high, low=df.low, close=df.close, volume=df.volume)
        return [ ad.acc_dist_index() ]


    def _calc_atr(df):
        atr = AverageTrueRange(high=df.high, low=df.low, close=df.close, window=14)
        return [ atr.average_true_range() ]
