#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
from utils.diff import add_diff


#abschg
def signal(*args):
    # https://bbs.quantclass.cn/thread/9776

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = abs(df['close'].pct_change(16))

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Acs 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['adx'] = ta.ADX(df['high'], df['low'], df['close'], n)
    df['adx_close'] = df['adx'] / df['close']
    df[factor_name] = df['adx_close'].rolling(n).std()

    del df['adx'], df['adx_close']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff


def signal(*args):
    '''
    AdaptBollingv3
    以原版v3择时中的mtm_mean作为选B因子
    '''

    df, n, diff_num, factor_name = args

    n1 = int(n)

    # ==============================================================

    df['mtm'] = df['close'] / df['close'].shift(n1) - 1
    df['mtm_mean'] = df['mtm'].rolling(window=n1, min_periods=1).mean()

    # 基于价格atr，计算波动率因子wd_atr
    df['c1'] = df['high'] - df['low']
    df['c2'] = abs(df['high'] - df['close'].shift(1))
    df['c3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['c1', 'c2', 'c3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=n1, min_periods=1).mean()
    df['avg_price_'] = df['close'].rolling(window=n1, min_periods=1).mean()
    df['wd_atr'] = df['atr'] / df['avg_price_']

    # 参考ATR，对MTM指标，计算波动率因子
    df['mtm_l'] = df['low'] / df['low'].shift(n1) - 1
    df['mtm_h'] = df['high'] / df['high'].shift(n1) - 1
    df['mtm_c'] = df['close'] / df['close'].shift(n1) - 1
    df['mtm_c1'] = df['mtm_h'] - df['mtm_l']
    df['mtm_c2'] = abs(df['mtm_h'] - df['mtm_c'].shift(1))
    df['mtm_c3'] = abs(df['mtm_l'] - df['mtm_c'].shift(1))
    df['mtm_tr'] = df[['mtm_c1', 'mtm_c2', 'mtm_c3']].max(axis=1)
    df['mtm_atr'] = df['mtm_tr'].rolling(window=n1, min_periods=1).mean()

    # 参考ATR，对MTM mean指标，计算波动率因子
    df['mtm_l_mean'] = df['mtm_l'].rolling(window=n1, min_periods=1).mean()
    df['mtm_h_mean'] = df['mtm_h'].rolling(window=n1, min_periods=1).mean()
    df['mtm_c_mean'] = df['mtm_c'].rolling(window=n1, min_periods=1).mean()
    df['mtm_c1'] = df['mtm_h_mean'] - df['mtm_l_mean']
    df['mtm_c2'] = abs(df['mtm_h_mean'] - df['mtm_c_mean'].shift(1))
    df['mtm_c3'] = abs(df['mtm_l_mean'] - df['mtm_c_mean'].shift(1))
    df['mtm_tr'] = df[['mtm_c1', 'mtm_c2', 'mtm_c3']].max(axis=1)
    df['mtm_atr_mean'] = df['mtm_tr'].rolling(window=n1, min_periods=1).mean()

    indicator = 'mtm_mean'

    # mtm_mean指标分别乘以三个波动率因子
    df[indicator] = df[indicator] * df['mtm_atr']
    df[indicator] = df[indicator] * df['mtm_atr_mean']
    df[indicator] = df[indicator] * df['wd_atr']

    df[factor_name] = df[indicator] * 100000000

    drop_col = [
        'mtm', 'mtm_mean', 'c1', 'c2', 'c3', 'tr', 'atr', 'wd_atr', 'mtm_l',
        'mtm_h', 'mtm_c', 'mtm_c1', 'mtm_c2', 'mtm_c3', 'mtm_tr', 'mtm_atr',
        'mtm_l_mean', 'mtm_h_mean', 'mtm_c_mean', 'mtm_atr_mean', 'avg_price_'
    ]
    df.drop(columns=drop_col, inplace=True)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/18309

def signal(*args):
    '''
    AdaptBollingv3
    使用Sroc_v2 代替 mtm_mean 作为选B因子
    '''

    df, n, diff_num, factor_name = args

    n1 = int(n)

    # ==============================================================

    # df['mtm'] = df['close'] / df['close'].shift(n1) - 1
    # df['mtm_mean'] = df['mtm'].rolling(window=n1, min_periods=1).mean()

    ema = ta.KAMA(df['close'], n)
    ref = ema.shift(2 * n)
    df['sorc'] = (ema - ref) / (ref + eps)  

    # 基于价格atr，计算波动率因子wd_atr
    df['c1'] = df['high'] - df['low']
    df['c2'] = abs(df['high'] - df['close'].shift(1))
    df['c3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['c1', 'c2', 'c3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=n1, min_periods=1).mean()
    df['avg_price_'] = df['close'].rolling(window=n1, min_periods=1).mean()
    df['wd_atr'] = df['atr'] / df['avg_price_']

    # 参考ATR，对MTM指标，计算波动率因子
    df['mtm_l'] = df['low'] / df['low'].shift(n1) - 1
    df['mtm_h'] = df['high'] / df['high'].shift(n1) - 1
    df['mtm_c'] = df['close'] / df['close'].shift(n1) - 1
    df['mtm_c1'] = df['mtm_h'] - df['mtm_l']
    df['mtm_c2'] = abs(df['mtm_h'] - df['mtm_c'].shift(1))
    df['mtm_c3'] = abs(df['mtm_l'] - df['mtm_c'].shift(1))
    df['mtm_tr'] = df[['mtm_c1', 'mtm_c2', 'mtm_c3']].max(axis=1)
    df['mtm_atr'] = df['mtm_tr'].rolling(window=n1, min_periods=1).mean()

    # 参考ATR，对MTM mean指标，计算波动率因子
    df['mtm_l_mean'] = df['mtm_l'].rolling(window=n1, min_periods=1).mean()
    df['mtm_h_mean'] = df['mtm_h'].rolling(window=n1, min_periods=1).mean()
    df['mtm_c_mean'] = df['mtm_c'].rolling(window=n1, min_periods=1).mean()
    df['mtm_c1'] = df['mtm_h_mean'] - df['mtm_l_mean']
    df['mtm_c2'] = abs(df['mtm_h_mean'] - df['mtm_c_mean'].shift(1))
    df['mtm_c3'] = abs(df['mtm_l_mean'] - df['mtm_c_mean'].shift(1))
    df['mtm_tr'] = df[['mtm_c1', 'mtm_c2', 'mtm_c3']].max(axis=1)
    df['mtm_atr_mean'] = df['mtm_tr'].rolling(window=n1, min_periods=1).mean()

    indicator = 'sorc'

    # mtm_mean指标分别乘以三个波动率因子
    df[indicator] = df[indicator] * df['mtm_atr']
    df[indicator] = df[indicator] * df['mtm_atr_mean']
    df[indicator] = df[indicator] * df['wd_atr']

    df[factor_name] = df[indicator] * 100000000

    drop_col = [
        'c1', 'c2', 'c3', 'tr', 'atr', 'wd_atr', 'mtm_l',
        'mtm_h', 'mtm_c', 'mtm_c1', 'mtm_c2', 'mtm_c3', 'mtm_tr', 'mtm_atr',
        'mtm_l_mean', 'mtm_h_mean', 'mtm_c_mean', 'mtm_atr_mean', 'avg_price_'
    ]
    df.drop(columns=drop_col, inplace=True)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # ADOSC 指标
    """
    AD=CUM_SUM(((CLOSE-LOW)-(HIGH-CLOSE))*VOLUME/(HIGH-LOW))
    AD_EMA1=EMA(AD,N1)
    AD_EMA2=EMA(AD,N2) 
    ADOSC=AD_EMA1-AD_EMA2
    ADL（收集派发线）指标是成交量的加权累计求和，其中权重为 CLV
    指标。ADL 指标可以与 OBV 指标进行类比。不同的是 OBV 指标只
    根据价格的变化方向把成交量分为正、负成交量再累加，而 ADL 是 用 CLV 指标作为权重进行成交量的累加。我们知道，CLV 指标衡量
    收盘价在最低价和最高价之间的位置，CLV>0(<0),则收盘价更靠近最
    高（低）价。CLV 越靠近 1(-1)，则收盘价越靠近最高（低）价。如
    果当天的 CLV>0，则 ADL 会加上成交量*CLV（收集）；如果当天的
    CLV<0，则 ADL 会减去成交量*CLV（派发）。
    ADOSC 指标是 ADL（收集派发线）指标的短期移动平均与长期移动
    平均之差。如果 ADOSC 上穿 0，则产生买入信号；如果 ADOSC 下 穿 0，则产生卖出信号。
    """
    df['AD'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) * df['volume'] / (df['high'] - df['low'])
    df['AD_sum'] = df['AD'].cumsum()
    df['AD_EMA1'] = df['AD_sum'].ewm(n, adjust=False).mean()
    df['AD_EMA2'] = df['AD_sum'].ewm(n * 2, adjust=False).mean()
    df['ADOSC'] = df['AD_EMA1'] - df['AD_EMA2']

    # 标准化
    df[factor_name] = (df['ADOSC'] - df['ADOSC'].rolling(n).min()) / (df['ADOSC'].rolling(n).max() - df['ADOSC'].rolling(n).min())

    del df['AD']
    del df['AD_sum']
    del df['AD_EMA2']
    del df['AD_EMA1']
    del df['ADOSC']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff, eps


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # ADTM 指标
    """
    N=20
    DTM=IF(OPEN>REF(OPEN,1),MAX(HIGH-OPEN,OPEN-REF(OP
    EN,1)),0)
    DBM=IF(OPEN<REF(OPEN,1),MAX(OPEN-LOW,REF(OPEN,1)-O
    PEN),0)
    STM=SUM(DTM,N)
    SBM=SUM(DBM,N)
    ADTM=(STM-SBM)/MAX(STM,SBM)
    ADTM 通过比较开盘价往上涨的幅度和往下跌的幅度来衡量市场的
    人气。ADTM 的值在-1 到 1 之间。当 ADTM 上穿 0.5 时，说明市场
    人气较旺；当 ADTM 下穿-0.5 时，说明市场人气较低迷。我们据此构
    造交易信号。
    当 ADTM 上穿 0.5 时产生买入信号；
    当 ADTM 下穿-0.5 时产生卖出信号。

    """
    df['h_o'] = df['high'] - df['open']
    df['diff_open'] = df['open'] - df['open'].shift(1)
    max_value1 = df[['h_o', 'diff_open']].max(axis=1)
    df.loc[df['open'] > df['open'].shift(1), 'DTM'] = max_value1
    df['DTM'].fillna(value=0, inplace=True)

    df['o_l'] = df['open'] - df['low']
    max_value2 = df[['o_l', 'diff_open']].max(axis=1)
    # DBM = pd.where(df['open'] < df['open'].shift(1), max_value2, 0)
    df.loc[df['open'] < df['open'].shift(1), 'DBM'] = max_value2
    df['DBM'].fillna(value=0, inplace=True)

    df['STM'] = df['DTM'].rolling(n).sum()
    df['SBM'] = df['DBM'].rolling(n).sum()
    max_value3 = df[['STM', 'SBM']].max(axis=1)
    df[factor_name] = (df['STM'] - df['SBM']) / max_value3

    del df['h_o']
    del df['diff_open']
    del df['o_l']
    del df['STM']
    del df['SBM']
    del df['DBM']
    del df['DTM']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Adtm 指标
    """
    N=20
    DTM=IF(OPEN>REF(OPEN,1),MAX(HIGH-OPEN,OPEN-REF(OP
    EN,1)),0)
    DBM=IF(OPEN<REF(OPEN,1),MAX(OPEN-LOW,REF(OPEN,1)-O
    PEN),0)
    STM=SUM(DTM,N)
    SBM=SUM(DBM,N)
    Adtm=(STM-SBM)/MAX(STM,SBM)
    Adtm 通过比较开盘价往上涨的幅度和往下跌的幅度来衡量市场的
    人气。Adtm 的值在-1 到 1 之间。当 Adtm 上穿 0.5 时，说明市场
    人气较旺；当 Adtm 下穿-0.5 时，说明市场人气较低迷。我们据此构
    造交易信号。
    当 Adtm 上穿 0.5 时产生买入信号；
    当 Adtm 下穿-0.5 时产生卖出信号。

    """
    tmp1_s = df['high'] - df['open']  # HIGH-OPEN
    tmp2_s = df['open'] - df['open'].shift(1)  # OPEN-REF(OPEN,1)
    tmp3_s = df['open'] - df['low']  # OPEN-LOW
    tmp4_s = df['open'].shift(1) - df['open']  # REF(OPEN,1)-OPEN

    dtm = np.where(df['open'] > df['open'].shift(1), np.maximum(tmp1_s, tmp2_s), 0)
    dbm = np.where(df['open'] < df['open'].shift(1), np.maximum(tmp3_s, tmp4_s), 0)
    stm = pd.Series(dtm).rolling(n, min_periods=1).sum()
    sbm = pd.Series(dbm).rolling(n, min_periods=1).sum()

    signal = (stm - sbm) / (1e-9 + pd.Series(stm).combine(pd.Series(sbm), max).values)
    df[factor_name] = scale_01(signal, n)



    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

    


    

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Adtm 指标
    """
    N=20
    DTM=IF(OPEN>REF(OPEN,1),MAX(HIGH-OPEN,OPEN-REF(OP
    EN,1)),0)
    DBM=IF(OPEN<REF(OPEN,1),MAX(OPEN-LOW,REF(OPEN,1)-O
    PEN),0)
    STM=SUM(DTM,N)
    SBM=SUM(DBM,N)
    Adtm=(STM-SBM)/MAX(STM,SBM)
    Adtm 通过比较开盘价往上涨的幅度和往下跌的幅度来衡量市场的
    人气。Adtm 的值在-1 到 1 之间。当 Adtm 上穿 0.5 时，说明市场
    人气较旺；当 Adtm 下穿-0.5 时，说明市场人气较低迷。我们据此构
    造交易信号。
    当 Adtm 上穿 0.5 时产生买入信号；
    当 Adtm 下穿-0.5 时产生卖出信号。

    """
    tmp1_s = df['high'] - df['open']  # HIGH-OPEN
    tmp2_s = df['open'] - df['open'].shift(1)  # OPEN-REF(OPEN,1)
    tmp3_s = df['open'] - df['low']  # OPEN-LOW
    tmp4_s = df['open'].shift(1) - df['open']  # REF(OPEN,1)-OPEN

    dtm = np.where(df['open'] > df['open'].shift(1), np.maximum(tmp1_s, tmp2_s), 0)
    dbm = np.where(df['open'] < df['open'].shift(1), np.maximum(tmp3_s, tmp4_s), 0)
    stm = pd.Series(dtm).rolling(n, min_periods=1).sum()
    sbm = pd.Series(dbm).rolling(n, min_periods=1).sum()

    signal = (stm - sbm) / (1e-9 + pd.Series(stm).combine(pd.Series(sbm), max).values)
    signal = df['close'] - signal
    df[factor_name] = scale_01(signal, n)


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


    


    

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18446

def signal(*args):

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['max_high'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)

    df['max_low'] = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    df['XPDM'] = np.where(df['max_high'] > df['max_low'], df['high'] - df['high'].shift(1), 0)
    df['PDM'] = df['XPDM'].rolling(n).sum()

    df['c1'] = abs(df['high'] - df['low'])
    df['c2'] = abs(df['high'] - df['close'])
    df['c3'] = abs(df['low'] - df['close'])
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)

    df['TR_sum'] = df['TR'].rolling(n).sum()
    df['DI+'] = df['PDM'] / df['TR_sum']

    df['mtm'] = (df['close'] / df['close'].shift(n) - 1).rolling(
        window=n, min_periods=1).mean()

    df[factor_name] = df['DI+'] * df['mtm']

    del df['max_high']
    del df['max_low']
    del df['XPDM']
    del df['PDM']
    del df['mtm']
    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['TR_sum']
    del df['DI+']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18446

def signal(*args):

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['max_high'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    df['max_low'] = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)

    df['XNDM'] = np.where(df['max_low'] > df['max_high'], df['low'].shift(1) - df['low'], 0)
    df['NDM'] = df['XNDM'].rolling(n).sum()
    df['c1'] = abs(df['high'] - df['low'])
    df['c2'] = abs(df['high'] - df['close'])
    df['c3'] = abs(df['low'] - df['close'])
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)

    df['TR_sum'] = df['TR'].rolling(n).sum()

    df['DI-'] = df['NDM'] / df['TR_sum']
    df['mtm'] = (df['close'] / df['close'].shift(n) - 1).rolling(
        window=n, min_periods=1).mean()

    df[factor_name] = df['DI-'] * df['mtm']


    del df['max_high']
    del df['max_low']
    del df['mtm']
    del df['XNDM']
    del df['NDM']
    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['TR_sum']
    del df['DI-']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    #该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
        N1=14
        MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
        MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
        XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
        PDM=SUM(XPDM,N1)
        XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
        NDM=SUM(XNDM,N1)
        TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
        TR=SUM(TR,N1)
        DI+=PDM/TR
        DI-=NDM/TR
        ADX 指标计算过程中的 DI+与 DI-指标用相邻两天的最高价之差与最
        低价之差来反映价格的变化趋势。当 DI+上穿 DI-时，产生买入信号；
        当 DI+下穿 DI-时，产生卖出信号。
        """
    # MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
    df['max_high'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    # MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
    df['max_low'] = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    # XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
    df['XPDM'] = np.where(df['max_high'] > df['max_low'], df['high'] - df['high'].shift(1), 0)
    # PDM=SUM(XPDM,N1)
    df['PDM'] = df['XPDM'].rolling(n).sum()
    # XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
    df['XNDM'] = np.where(df['max_low'] > df['max_high'], df['low'].shift(1) - df['low'], 0)
    # NDM=SUM(XNDM,N1)
    df['NDM'] = df['XNDM'].rolling(n).sum()
    # ABS(HIGH-LOW)
    df['c1'] = abs(df['high'] - df['low'])
    # ABS(HIGH-CLOSE)
    df['c2'] = abs(df['high'] - df['close'])
    # ABS(LOW-CLOSE)
    df['c3'] = abs(df['low'] - df['close'])
    # TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)
    # TR=SUM(TR,N1)
    df['TR_sum'] = df['TR'].rolling(n).sum()
    # DI+=PDM/TR
    df['DI+'] = df['PDM'] / df['TR']
    # DI-=NDM/TR
    df['DI-'] = df['NDM'] / df['TR']

    df[f'ADX_DI+_bh_{n}'] = df['DI+'].shift(1)
    df[f'ADX_DI-_bh_{n}'] = df['DI-'].shift(1)
    # 去量纲
    df[factor_name] = (df['PDM'] + df['NDM']) / df['TR']


    # 删除中间过程数据
    del df['max_high']
    del df['max_low']
    del df['XPDM']
    del df['PDM']
    del df['XNDM']
    del df['NDM']
    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['TR_sum']
    del df['DI+']
    del df['DI-']








    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    #该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
        N1=14
        MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
        MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
        XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
        PDM=SUM(XPDM,N1)
        XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
        NDM=SUM(XNDM,N1)
        TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
        TR=SUM(TR,N1)
        DI+=PDM/TR
        DI-=NDM/TR
        ADX 指标计算过程中的 DI+与 DI-指标用相邻两天的最高价之差与最
        低价之差来反映价格的变化趋势。当 DI+上穿 DI-时，产生买入信号；
        当 DI+下穿 DI-时，产生卖出信号。
        """
    # MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
    df['max_high'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    # MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
    df['max_low'] = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    # XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
    df['XPDM'] = np.where(df['max_high'] > df['max_low'], df['high'] - df['high'].shift(1), 0)
    # PDM=SUM(XPDM,N1)
    df['PDM'] = df['XPDM'].rolling(n).sum()
    # XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
    df['XNDM'] = np.where(df['max_low'] > df['max_high'], df['low'].shift(1) - df['low'], 0)
    # NDM=SUM(XNDM,N1)
    df['NDM'] = df['XNDM'].rolling(n).sum()
    # ABS(HIGH-LOW)
    df['c1'] = abs(df['high'] - df['low'])
    # ABS(HIGH-CLOSE)
    df['c2'] = abs(df['high'] - df['close'])
    # ABS(LOW-CLOSE)
    df['c3'] = abs(df['low'] - df['close'])
    # TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)
    # TR=SUM(TR,N1)
    df['TR_sum'] = df['TR'].rolling(n).sum()
    # DI+=PDM/TR
    df[factor_name] = df['PDM'] / df['TR'] #DI+
    # DI-=NDM/TR
    # df['DI-'] = df['NDM'] / df['TR'] #DI-

    # df[f'ADX_DI+_bh_{n}'] = df['DI+'].shift(1)
    # df[f'ADX_DI-_bh_{n}'] = df['DI-'].shift(1)
    # 去量纲
    # df[factor_name] = (df['PDM'] + df['NDM']) / df['TR']

    # 删除中间过程数据
    del df['max_high']
    del df['max_low']
    del df['XPDM']
    del df['PDM']
    del df['XNDM']
    del df['NDM']
    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['TR_sum']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    #该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
        N1=14
        MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
        MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
        XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
        PDM=SUM(XPDM,N1)
        XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
        NDM=SUM(XNDM,N1)
        TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
        TR=SUM(TR,N1)
        DI+=PDM/TR
        DI-=NDM/TR
        ADX 指标计算过程中的 DI+与 DI-指标用相邻两天的最高价之差与最
        低价之差来反映价格的变化趋势。当 DI+上穿 DI-时，产生买入信号；
        当 DI+下穿 DI-时，产生卖出信号。
        """
    # MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
    df['max_high'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    # MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
    df['max_low'] = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    # XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
    df['XPDM'] = np.where(df['max_high'] > df['max_low'], df['high'] - df['high'].shift(1), 0)
    # PDM=SUM(XPDM,N1)
    df['PDM'] = df['XPDM'].rolling(n).sum()
    # XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
    df['XNDM'] = np.where(df['max_low'] > df['max_high'], df['low'].shift(1) - df['low'], 0)
    # NDM=SUM(XNDM,N1)
    df['NDM'] = df['XNDM'].rolling(n).sum()
    # ABS(HIGH-LOW)
    df['c1'] = abs(df['high'] - df['low'])
    # ABS(HIGH-CLOSE)
    df['c2'] = abs(df['high'] - df['close'])
    # ABS(LOW-CLOSE)
    df['c3'] = abs(df['low'] - df['close'])
    # TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)
    # TR=SUM(TR,N1)
    df['TR_sum'] = df['TR'].rolling(n).sum()
    # DI+=PDM/TR
    # df[factor_name] = df['PDM'] / df['TR'] #DI+
    # DI-=NDM/TR
    df[factor_name] = df['NDM'] / df['TR'] #DI-

    # df[f'ADX_DI+_bh_{n}'] = df['DI+'].shift(1)
    # df[f'ADX_DI-_bh_{n}'] = df['DI-'].shift(1)
    # 去量纲
    # df[factor_name] = (df['PDM'] + df['NDM']) / df['TR']


    # 删除中间过程数据
    del df['max_high']
    del df['max_low']
    del df['XPDM']
    del df['PDM']
    del df['XNDM']
    del df['NDM']
    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['TR_sum']



    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coDing: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Adx 指标
    """
    N1=14
    MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
    MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
    XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
    PDM=SUM(XPDM,N1)
    XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
    NDM=SUM(XNDM,N1)
    TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
    TR=SUM(TR,N1)
    Di+=PDM/TR
    Di-=NDM/TR
    Adx 指标计算过程中的 Di+与 Di-指标用相邻两天的最高价之差与最
    低价之差来反映价格的变化趋势。当 Di+上穿 Di-时，产生买入信号；
    当 Di+下穿 Di-时，产生卖出信号。
    """
    max_high = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    max_low = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    xpdm = np.where(pd.Series(max_high) > pd.Series(max_low), pd.Series(max_high) - pd.Series(max_high).shift(1), 0)
    xndm = np.where(pd.Series(max_low) > pd.Series(max_high), pd.Series(max_low).shift(1) - pd.Series(max_low), 0)
    tr = np.max(np.array([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close']).abs(),
        (df['low'] - df['close']).abs()
    ]), axis=0)  # 三个数列取其大值
    pdm = pd.Series(xpdm).rolling(n, min_periods=1).sum()
    ndm = pd.Series(xndm).rolling(n, min_periods=1).sum()

    di_pos = pd.Series(pdm / pd.Series(tr).rolling(n, min_periods=1).sum())
    di_neg = pd.Series(ndm / pd.Series(tr).rolling(n, min_periods=1).sum())

    adxr_pos = 0.5 * pd.Series(di_pos) + 0.5 * pd.Series(di_pos).shift(n)
    adxr_neg = 0.5 * pd.Series(di_neg) + 0.5 * pd.Series(di_neg).shift(n)

    df[factor_name] = adxr_pos - adxr_neg

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




#!/usr/bin/python3
# -*- coDing: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # Adx 指标
    """
    N1=14
    MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
    MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
    XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
    PDM=SUM(XPDM,N1)
    XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
    NDM=SUM(XNDM,N1)
    TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
    TR=SUM(TR,N1)
    Di+=PDM/TR
    Di-=NDM/TR
    Adx 指标计算过程中的 Di+与 Di-指标用相邻两天的最高价之差与最
    低价之差来反映价格的变化趋势。当 Di+上穿 Di-时，产生买入信号；
    当 Di+下穿 Di-时，产生卖出信号。
    """
    max_high = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    max_low = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    # xpdm = np.where(pd.Series(max_high) > pd.Series(max_low), pd.Series(max_high) - pd.Series(max_high).shift(1), 0)
    xndm = np.where(pd.Series(max_low) > pd.Series(max_high), pd.Series(max_low).shift(1) - pd.Series(max_low), 0)
    tr = np.max(np.array([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close']).abs(),
        (df['low'] - df['close']).abs()
    ]), axis=0)  # 三个数列取其大值
    # pdm = pd.Series(xpdm).rolling(n, min_periods=1).sum()
    ndm = pd.Series(xndm).rolling(n, min_periods=1).sum()

    # di_pos = pd.Series(pdm / pd.Series(tr).rolling(n, min_periods=1).sum())
    di_neg = pd.Series(ndm / pd.Series(tr).rolling(n, min_periods=1).sum())

    signal = 0.5 * pd.Series(di_neg) + 0.5 * pd.Series(di_neg).shift(n)
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




#!/usr/bin/python3
# -*- coDing: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # Adx 指标
    """
    N1=14
    MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
    MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
    XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
    PDM=SUM(XPDM,N1)
    XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
    NDM=SUM(XNDM,N1)
    TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
    TR=SUM(TR,N1)
    Di+=PDM/TR
    Di-=NDM/TR
    Adx 指标计算过程中的 Di+与 Di-指标用相邻两天的最高价之差与最
    低价之差来反映价格的变化趋势。当 Di+上穿 Di-时，产生买入信号；
    当 Di+下穿 Di-时，产生卖出信号。
    """
    max_high = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    max_low = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    xpdm = np.where(pd.Series(max_high) > pd.Series(max_low), pd.Series(max_high) - pd.Series(max_high).shift(1), 0)
    # xndm = np.where(pd.Series(max_low) > pd.Series(max_high), pd.Series(max_low).shift(1) - pd.Series(max_low), 0)
    tr = np.max(np.array([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close']).abs(),
        (df['low'] - df['close']).abs()
    ]), axis=0)  # 三个数列取其大值
    pdm = pd.Series(xpdm).rolling(n, min_periods=1).sum()
    # ndm = pd.Series(xndm).rolling(n, min_periods=1).sum()

    di_pos = pd.Series(pdm / pd.Series(tr).rolling(n, min_periods=1).sum())
    # di_neg = pd.Series(ndm / pd.Series(tr).rolling(n, min_periods=1).sum())

    df[factor_name] = 0.5 * pd.Series(di_pos) + 0.5 * pd.Series(di_pos).shift(n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




#!/usr/bin/python3
# -*- coDing: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # Adx 指标
    """
    N1=14
    MAX_HIGH=IF(HIGH>REF(HIGH,1),HIGH-REF(HIGH,1),0)
    MAX_LOW=IF(REF(LOW,1)>LOW,REF(LOW,1)-LOW,0)
    XPDM=IF(MAX_HIGH>MAX_LOW,HIGH-REF(HIGH,1),0)
    PDM=SUM(XPDM,N1)
    XNDM=IF(MAX_LOW>MAX_HIGH,REF(LOW,1)-LOW,0)
    NDM=SUM(XNDM,N1)
    TR=MAX([ABS(HIGH-LOW),ABS(HIGH-CLOSE),ABS(LOW-CLOSE)])
    TR=SUM(TR,N1)
    Di+=PDM/TR
    Di-=NDM/TR
    Adx 指标计算过程中的 Di+与 Di-指标用相邻两天的最高价之差与最
    低价之差来反映价格的变化趋势。当 Di+上穿 Di-时，产生买入信号；
    当 Di+下穿 Di-时，产生卖出信号。
    """
    max_high = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    max_low = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    xpdm = np.where(pd.Series(max_high) > pd.Series(max_low), pd.Series(max_high) - pd.Series(max_high).shift(1), 0)
    xndm = np.where(pd.Series(max_low) > pd.Series(max_high), pd.Series(max_low).shift(1) - pd.Series(max_low), 0)
    tr = np.max(np.array([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close']).abs(),
        (df['low'] - df['close']).abs()
    ]), axis=0)  # 三个数列取其大值
    pdm = pd.Series(xpdm).rolling(n, min_periods=1).sum()
    ndm = pd.Series(xndm).rolling(n, min_periods=1).sum()

    di_pos = pd.Series(pdm / pd.Series(tr).rolling(n, min_periods=1).sum())
    di_neg = pd.Series(ndm / pd.Series(tr).rolling(n, min_periods=1).sum())

    adxr = di_pos - di_neg
    df[factor_name] = 0.5 * pd.Series(adxr) + 0.5 * pd.Series(adxr).shift(n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # AMV 指标
    """
    N1=13
    N2=34
    AMOV=VOLUME*(OPEN+CLOSE)/2
    AMV1=SUM(AMOV,N1)/SUM(VOLUME,N1)
    AMV2=SUM(AMOV,N2)/SUM(VOLUME,N2)
    AMV 指标用成交量作为权重对开盘价和收盘价的均值进行加权移动
    平均。成交量越大的价格对移动平均结果的影响越大，AMV 指标减
    小了成交量小的价格波动的影响。当短期 AMV 线上穿/下穿长期 AMV
    线时，产生买入/卖出信号。
    """
    df['AMOV'] = df['volume'] * (df['open'] + df['close']) / 2
    df['AMV1'] = df['AMOV'].rolling(n).sum() / df['volume'].rolling(n).sum()
    # df['AMV2'] = df['AMOV'].rolling(n * 3).sum() / df['volume'].rolling(n * 3).sum()
    # 去量纲
    df[factor_name] = (df['AMV1'] - df['AMV1'].rolling(n).min()) / (df['AMV1'].rolling(n).max() - df['AMV1'].rolling(n).min()) # 标准化
    
    del df['AMOV']
    del df['AMV1']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps
import talib as ta


def signal(*args):
    # AvgPrice
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = ta.LINEARREG_ANGLE(df['close'], timeperiod=n)

    # 删除多余列


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    """
    N=10
    M=20
    PARAM=2
    VOL=EMA(EMA(HIGH-LOW,N),N)
    UPPER=EMA(EMA(CLOSE,M),M)+PARAM*VOL
    LOWER= EMA(EMA(CLOSE,M),M)-PARAM*VOL
    APZ（Adaptive Price Zone 自适应性价格区间）与布林线 Bollinger 
    Band 和肯通纳通道 Keltner Channel 很相似，都是根据价格波动性围
    绕均线而制成的价格通道。只是在这三个指标中计算价格波动性的方
    法不同。在布林线中用了收盘价的标准差，在肯通纳通道中用了真波
    幅 ATR，而在 APZ 中运用了最高价与最低价差值的 N 日双重指数平
    均来反映价格的波动幅度。
    """
    df['hl'] = df['high'] - df['low']
    df['ema_hl'] = df['hl'].ewm(n, adjust=False).mean()
    df['vol'] = df['ema_hl'].ewm(n, adjust=False).mean()

    # 计算通道 可以作为CTA策略 作为因子的时候进行改造
    df['ema_close'] = df['close'].ewm(2 * n, adjust=False).mean()
    df['ema_ema_close'] = df['ema_close'].ewm(2 * n, adjust=False).mean()
    # EMA去量纲
    df[factor_name] = df['vol'] / df['ema_ema_close']

    del df['hl']
    del df['ema_hl']
    del df['vol']
    del df['ema_close']
    del df['ema_ema_close']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df





    
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]    
    diff_num = args[2]
    factor_name = args[3]
# ApzLower 指标
    """
    N=10
    M=20
    PARAM=2
    VOL=EMA(EMA(HIGH-LOW,N),N)
    UPPER=EMA(EMA(CLOSE,M),M)+PARAM*VOL
    LOWER= EMA(EMA(CLOSE,M),M)-PARAM*VOL
    ApzLower（Adaptive Price Zone 自适应性价格区间）与布林线 Bollinger 
    Band 和肯通纳通道 Keltner Channel 很相似，都是根据价格波动性围
    绕均线而制成的价格通道。只是在这三个指标中计算价格波动性的方
    法不同。在布林线中用了收盘价的标准差，在肯通纳通道中用了真波
    幅 ATR，而在 ApzLower 中运用了最高价与最低价差值的 N 日双重指数平
    均来反映价格的波动幅度。
    """
    vol = (df['high'] - df['low']).ewm(span=n, adjust=False, min_periods=1).mean().ewm(
        span=n, adjust=False, min_periods=1).mean()
    upper = df['close'].ewm(span=int(2 * n), adjust=False, min_periods=1).mean().ewm(
        span=int(2 * n), adjust=False, min_periods=1).mean() + 2 * vol

    signal = upper - 4 * vol
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




    

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]    

# ApzUpper 指标
    """
    N=10
    M=20
    PARAM=2
    VOL=EMA(EMA(HIGH-LOW,N),N)
    UPPER=EMA(EMA(CLOSE,M),M)+PARAM*VOL
    LOWER= EMA(EMA(CLOSE,M),M)-PARAM*VOL
    ApzUpper（Adaptive Price Zone 自适应性价格区间）与布林线 Bollinger 
    Band 和肯通纳通道 Keltner Channel 很相似，都是根据价格波动性围
    绕均线而制成的价格通道。只是在这三个指标中计算价格波动性的方
    法不同。在布林线中用了收盘价的标准差，在肯通纳通道中用了真波
    幅 ATR，而在 ApzUpper 中运用了最高价与最低价差值的 N 日双重指数平
    均来反映价格的波动幅度。
    """
    vol = (df['high'] - df['low']).ewm(span=n, adjust=False, min_periods=1).mean().ewm(
        span=n, adjust=False, min_periods=1).mean()
    upper = df['close'].ewm(span=int(2 * n), adjust=False, min_periods=1).mean().ewm(
        span=int(2 * n), adjust=False, min_periods=1).mean() + 2 * vol
    signal = upper
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




    

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    v1 = (df['high'] - df['open']).rolling(n, min_periods=1).sum()
    v2 = (df['open'] - df['low']).rolling(n, min_periods=1).sum()
    _ar = 100 * v1 / v2
    df[factor_name] = pd.Series(_ar)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # ARBR指标
    """
    AR=SUM((HIGH-OPEN),N)/SUM((OPEN-LOW),N)*100
    # BR=SUM((HIGH-REF(CLOSE,1)),N)/SUM((REF(CLOSE,1)-LOW),N)*100
    AR 衡量开盘价在最高价、最低价之间的位置；BR 衡量昨日收盘价在
    今日最高价、最低价之间的位置。AR 为人气指标，用来计算多空双
    方的力量对比。当 AR 值偏低（低于 50）时表示人气非常低迷，股价
    很低，若从 50 下方上穿 50，则说明股价未来可能要上升，低点买入。
    当 AR 值下穿 200 时卖出。
    """
    df['HO'] = df['high'] - df['open']
    df['OL'] = df['open'] - df['low']
    df[factor_name] = df['HO'].rolling(n).sum() / df['OL'].rolling(n).sum() * 100

    del df['HO']
    del df['OL']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df






#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # ARBR指标
    """
    # AR=SUM((HIGH-OPEN),N)/SUM((OPEN-LOW),N)*100
    BR=SUM((HIGH-REF(CLOSE,1)),N)/SUM((REF(CLOSE,1)-LOW),N)*100
    AR 衡量开盘价在最高价、最低价之间的位置；BR 衡量昨日收盘价在
    今日最高价、最低价之间的位置。AR 为人气指标，用来计算多空双
    方的力量对比。当 AR 值偏低（低于 50）时表示人气非常低迷，股价
    很低，若从 50 下方上穿 50，则说明股价未来可能要上升，低点买入。
    当 AR 值下穿 200 时卖出。
    """

    df['HC'] = df['high'] - df['close'].shift(1)
    df['CL'] = df['close'].shift(1) - df['low']
    df[factor_name] = df['HC'].rolling(n).sum() / df['CL'].rolling(n).sum() * 100

    del df['HC']
    del df['CL']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import numba as nb
from utils.diff import add_diff


# =====函数  zscore归一化
def scale_zscore(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).mean()
          ) / pd.Series(_s).rolling(_n, min_periods=1).std()
    return pd.Series(_s)


@nb.njit(nb.int32[:](nb.float64[:], nb.int32), cache=True)
def rolling_argmin_queue(arr, n):
    results = np.empty(len(arr), dtype=np.int32)

    head = 0
    tail = 0
    que_idx = np.empty(len(arr), dtype=np.int32)
    for i, x in enumerate(arr[:n]):
        while tail > 0 and arr[que_idx[tail - 1]] > x:
            tail -= 1
        que_idx[tail] = i
        tail += 1
        results[i] = que_idx[0]

    for i, x in enumerate(arr[n:], n):
        if que_idx[head] <= i - n:
            head += 1
        while tail > head and arr[que_idx[tail - 1]] > x:
            tail -= 1
        que_idx[tail] = i
        tail += 1
        results[i] = que_idx[head]
    return results


def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** Arron ********************
    # ArronUp = (N - HIGH_LEN) / N * 100
    # ArronDown = (N - LOW_LEN) / N * 100
    # ArronOs = ArronUp - ArronDown
    # 其中 HIGH_LEN，LOW_LEN 分别为过去N天最高/最低价距离当前日的天数
    # ArronUp、ArronDown指标分别为考虑的时间段内最高价、最低价出现时间与当前时间的距离占时间段长度的百分比。
    # 如果价格当天创新高，则ArronUp等于100；创新低，则ArronDown等于100。Aroon指标为两者之差，
    # 变化范围为-100到100。Arron指标大于0表示股价呈上升趋势，Arron指标小于0表示股价呈下降趋势。
    # 距离0点越远则趋势越强。我们这里以20/-20为阈值构造交易信号。如果ArronOs上穿20/下穿-20则产生买入/卖出信号。
    low_len = (rolling_argmin_queue(df['low'].values, n))
    high_len = (rolling_argmin_queue(-df['high'].values, n))
    signal =  pd.Series((high_len - low_len) * 100 / n)
    df[factor_name] = scale_zscore(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    N=20
    TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    ATR=MA(TR,N)
    MIDDLE=MA(CLOSE,N)
    """
    df['c1'] = df['high'] - df['low']  # HIGH-LOW
    df['c2'] = abs(df['high'] - df['close'].shift(1))  # ABS(HIGH-REF(CLOSE,1)
    df['c3'] = abs(df['low'] - df['close'].shift(1))  # ABS(LOW-REF(CLOSE,1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(
        axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    df['_ATR'] = df['TR'].rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
    df['middle'] = df['close'].rolling(n, min_periods=1).mean()  # MIDDLE=MA(CLOSE,N)
    # ATR指标去量纲
    df[factor_name] = df['_ATR'] / (df['middle'] + eps)


    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['_ATR']
    del df['middle']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # AtrLower
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    tr = np.max(np.array([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ]), axis=0)
    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    _ma = df['close'].rolling(n, min_periods=1).mean()

    dn = _ma - atr * 0.2 * n
    df[factor_name] = dn / (_ma + eps)
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['close_1'] = df['close'].shift()
    tr = df[['high', 'low', 'close_1']].max(
        axis=1) - df[['high', 'low', 'close_1']].min(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    df[factor_name] = atr.pct_change(n)

    del df['close_1']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # AtrUpper
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    tr = np.max(np.array([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ]), axis=0)  # 三个数列取其大值
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().shift(1)
    _low = df['low'].rolling(int(n / 2), min_periods=1).min()
    _ma = df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = (_low + 3 * atr) / (_ma + eps)
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    N=20
    TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    ATR=MA(TR,N)
    MIDDLE=MA(CLOSE,N)
    """
    df['c1'] = df['high'] - df['low']  # HIGH-LOW
    df['c2'] = abs(df['high'] - df['close'].shift(1))  # ABS(HIGH-REF(CLOSE,1)
    df['c3'] = abs(df['low'] - df['close'].shift(1))  # ABS(LOW-REF(CLOSE,1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(
        axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    df['_ATR'] = df['TR'].rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
    df['middle'] = df['close'].rolling(n, min_periods=1).mean()  # MIDDLE=MA(CLOSE,N)
    df['upper'] = df['middle'] + 2 * df['_ATR']
    df['lower'] = df['middle'] - 2 * df['_ATR']

    df['count'] = 0
    df.loc[df['close'] > df['upper'], 'count'] = 1
    df.loc[df['close'] < df['lower'], 'count'] = -1
    df[factor_name] = df['count'].rolling(n).sum()
    # del df['mean']
    # del df['std']
    del df['upper']
    del df['lower']
    del df['count']
    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['_ATR']
    del df['middle']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # AvgPrice
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['price'] = df['quote_volume'].rolling(n, min_periods=1).sum() / df['volume'].rolling(n, min_periods=1).sum()
    df[factor_name] = (df['price'] - df['price'].rolling(n, min_periods=1).min()) / (
        df['price'].rolling(n, min_periods=1).max() - df['price'].rolling(n, min_periods=1).min() + eps)

    # 删除多余列
    del df['price']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

import numpy  as np
import pandas as pd
from utils.diff import add_diff


def signal(*args):
    # AvgPriceToHigh
    # https://bbs.quantclass.cn/thread/9454

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['price'] = df['quote_volume'].rolling(n, min_periods=1).sum() / df['volume'].rolling(n, min_periods=1).sum()
    df[factor_name] = df['price'] / df['high'] - 1

    # 删除多余列
    del df['price']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
import numpy  as np
import pandas as pd
from utils.diff import add_diff


def signal(*args):
    # AvgPriceToLow
    # https://bbs.quantclass.cn/thread/9454

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['price'] = df['quote_volume'].rolling(n, min_periods=1).sum() / df['volume'].rolling(n, min_periods=1).sum()
    df[factor_name] = df['price']/df['low'] - 1

    # 删除多余列
    del df['price']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18266

def signal(args):
 
     df, n, diff_num, factor_name = args
 
     n1 = int(n)
 
     # ==============================================================
     ts = df[['high', 'low']].sum(axis=1) / 2
 
     close_ma = ts.rolling(n, min_periods=1).mean()
     tma = close_ma.rolling(n, min_periods=1).mean()
     df['mtm'] = df['close'] / (tma+eps) - 1
 
     df['mtm_mean'] = df['mtm'].rolling(window=n1, min_periods=1).mean()
 
     # 基于价格atr，计算波动率因子wd_atr
     df['c1'] = df['high'] - df['low']
     df['c2'] = abs(df['high'] - df['close'].shift(1))
     df['c3'] = abs(df['low'] - df['close'].shift(1))
     df['tr'] = df[['c1', 'c2', 'c3']].max(axis=1)
     df['atr'] = df['tr'].rolling(window=n1, min_periods=1).mean()
     df['avg_price'] = df['close'].rolling(window=n1, min_periods=1).mean()
     df['wd_atr'] = df['atr'] / df['avg_price']
 
     # 平均主动买入
     df['vma'] = df['quote_volume'].rolling(n, min_periods=1).mean()
     df['taker_buy_ma'] = (df['taker_buy_quote_asset_volume'] / df['vma'])  100
     df['taker_buy_mean'] = df['taker_buy_ma'].rolling(window=n).mean()
 
     indicator = 'mtm_mean'
 
     # mtm_mean指标分别乘以三个波动率因子
     df[indicator] = df[indicator]  df['wd_atr']  df['taker_buy_mean']
     df[factor_name] = df[indicator] * 100000000
 
     drop_col = [
         'mtm', 'mtm_mean', 'c1', 'c2', 'c3', 'tr', 'atr', 'wd_atr','avg_price_',
         'vma' ,'taker_buy_ma','taker_buy_mean'
     ]
     df.drop(columns=drop_col, inplace=True)
 
     if diff_num > 0:
         return add_diff(df, diff_num, factor_name)
     else:
         return df
 
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # Bbi
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    """
    BBI=(MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,12)+MA(CLOSE,24))/4
    BBI 是对不同时间长度的移动平均线取平均，能够综合不同移动平均
    线的平滑性和滞后性。如果收盘价上穿/下穿 BBI 则产生买入/卖出信
    号。
    """
    # 将BBI指标计算出来求bias
    ma1 = df['close'].rolling(n, min_periods=1).mean()
    ma2 = df['close'].rolling(2 * n, min_periods=1).mean()
    ma3 = df['close'].rolling(4 * n, min_periods=1).mean()
    ma4 = df['close'].rolling(8 * n, min_periods=1).mean()
    # BBI=(MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,12)+MA(CLOSE,24))/4
    bbi = (ma1 + ma2 + ma3 + ma4) / 4
    df[factor_name] = bbi / (df['close'] + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # BbiBias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    """
    BBI=(MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,12)+MA(CLOSE,24))/4
    BBI 是对不同时间长度的移动平均线取平均，能够综合不同移动平均
    线的平滑性和滞后性。如果收盘价上穿/下穿 BBI 则产生买入/卖出信
    号。
    """
    # 将BBI指标计算出来求bias
    ma1 = df['close'].rolling(n, min_periods=1).mean()
    ma2 = df['close'].rolling(2 * n, min_periods=1).mean()
    ma3 = df['close'].rolling(4 * n, min_periods=1).mean()
    ma4 = df['close'].rolling(8 * n, min_periods=1).mean()
    # BBI=(MA(CLOSE,3)+MA(CLOSE,6)+MA(CLOSE,12)+MA(CLOSE,24))/4
    bbi = (ma1 + ma2 + ma3 + ma4) / 4
    df[factor_name] = df['close'] / (bbi + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Bbw
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    close_dif = df['close'].diff()
    df['up'] = np.where(close_dif > 0, close_dif, 0)
    df['down'] = np.where(close_dif < 0, abs(close_dif), 0)
    a = df['up'].rolling(n).sum()
    b = df['down'].rolling(n).sum()
    df['rsi'] = (a / (a + b)) * 100
    df['median'] = df['close'].rolling(n, min_periods=1).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['bbw'] = (df['std'] / df['median']).diff(n)
    df[factor_name] = (df['bbw']) * (df['close'] / df['close'].shift(n) - 1 + eps) * df['rsi']

    del df['up'], df['down'],  df['rsi'], df['median']
    del df['std'], df['bbw']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff


def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['close'] / df['ma'] - 1)

    del df['ma']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    bias36 = df['close'].rolling(3, min_periods=1).mean() - df['close'].rolling(6, min_periods=1).mean()
    bias36_ma = bias36.rolling(n, min_periods=1).mean()

    signal = bias36 - bias36_ma
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    bias36 = df['close'].rolling(3, min_periods=1).mean() - df['close'].rolling(6, min_periods=1).mean()
    df[factor_name] = bias36.rolling(n, min_periods=1).mean()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18643

def signal(*args):
    '''
    BiasCubic指标    三个bias相乘
    要把变量尽可能减少  back_hour_list  = [3, 4, 6, 8, 9, 12, 24, 30, 36, 48, 60, 72, 96]
    int(n / 2) 取个整数，除以1.5，乘以1.5来进行三个区分也是可以的    实现1个变量当三个用。
    :param args:
    :return:
    '''

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma_1'] = df['close'].rolling(int(n / 2), min_periods=1).mean()
    df['ma_2'] = df['close'].rolling(n, min_periods=1).mean()
    df['ma_3'] = df['close'].rolling(n * 2, min_periods=1).mean()
    df['bias_1'] = (df['close'] / df['ma_1'] - 1)
    df['bias_2'] = (df['close'] / df['ma_2'] - 1)
    df['bias_3'] = (df['close'] / df['ma_3'] - 1)


    df['mtm'] = (df['bias_1'] * df['bias_2'] *df['bias_3'])* df['quote_volume']/df['quote_volume'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['mtm'].rolling(n, min_periods=1).mean()

    del df['ma_1'], df['ma_2'], df['ma_3'], df['bias_1'], df['bias_2'], df['bias_3'],df['mtm']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # BIASVOL 指标
    """
    N=6，12，24
    BIASVOL(N)=(VOLUME-MA(VOLUME,N))/MA(VOLUME,N)
    BIASVOL 是乖离率 BIAS 指标的成交量版本。如果 BIASVOL6 大于
    5 且 BIASVOL12 大于 7 且 BIASVOL24 大于 11，则产生买入信号；
    如果 BIASVOL6 小于-5 且 BIASVOL12 小于-7 且 BIASVOL24 小于
    -11，则产生卖出信号。
    """
    df['ma_volume'] = df['volume'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['volume'] - df['ma_volume']) / df['ma_volume']

    del df['ma_volume']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18160

def signal(*args):
    # Bias_ema
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['close'] / df['ma'] - 1).ewm(n, adjust=False).mean()

    del df['ma']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/17947

def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['close'] / df['ma'] - 1).rolling(n, min_periods=1).mean()

    del df['ma']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff


def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['mtm'] = (df['close'] / df['ma'] - 1) * df['quote_volume']/df['quote_volume'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['mtm'].rolling(n, min_periods=1).mean()

    del df['ma'], df['mtm']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff



def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算移动平均值和交易量的归一化值
    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['min_volume'] = df['quote_volume'].rolling(n, min_periods=1).min()
    df['max_volume'] = df['quote_volume'].rolling(n, min_periods=1).max()
    df['norm_volume'] = (df['quote_volume'] - df['min_volume']) / (df['max_volume'] - df['min_volume'] + diff_num)

    # 计算mtm
    df['mtm'] = (df['close'] / df['ma'] - 1) * df['norm_volume']
    df[factor_name] = df['mtm'].rolling(n, min_periods=1).mean()

    # 删除多余的列
    del df['ma'], df['mtm'], df['min_volume'], df['max_volume'], df['norm_volume']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff



def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算移动平均值和移动平均交易量
    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['ma_volume'] = df['quote_volume'].rolling(n, min_periods=1).mean()

    # 计算交易量的z-score
    df['volume_zscore'] = (df['quote_volume'] - df['ma_volume']) / df['quote_volume'].rolling(n, min_periods=1).std()

    # 计算mtm
    df['mtm'] = (df['close'] / df['ma'] - 1) * df['volume_zscore']
    df[factor_name] = df['mtm'].rolling(n, min_periods=1).mean()

    # 删除多余的列
    del df['ma'], df['mtm'], df['ma_volume'], df['volume_zscore']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff


def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['mtm'] = (df['close'] / df['ma'] - 1) * df['quote_volume']/df['quote_volume'].rolling(n, min_periods=1).mean()
    # EMA
    df[factor_name] = df['mtm'].ewm(n, adjust=False).mean()

    del df['ma'], df['mtm']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18506

def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['mafast'] = df['close'].rolling(int(n/2), min_periods=1).mean()
    df[factor_name] = (df['mafast'] / df['ma'] - 1).rolling(n, min_periods=1).mean()

    del df['ma'],df['mafast']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18506

def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['ma2'] = df['close'].rolling(int(n/2), min_periods=1).mean()
    df['mtm'] = (df['ma2'] / df['ma'] - 1) * df['quote_volume']/df['quote_volume'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['mtm'].rolling(n, min_periods=1).mean()

    del df['ma'],df['ma2'],df['mtm']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Bias_v2
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算线性回归
    df['new_close'] = ta.LINEARREG(df['close'], timeperiod=n)
    # EMA再次平滑曲线
    df['new_close'] = ta.EMA(df['new_close'], timeperiod=n)
    # 以新的收盘价计算中轨
    ma = df['new_close'].rolling(n, min_periods=1).mean()
    # 修改收盘价的定义为 最高和最低价的平均值 * 成交量
    # df['close'] =   (df['high'] + df['low']) / 2 * df['volume']
    close = (df['high'] + df['low']) / 2 * df['volume']
    # 计算bias
    df[factor_name] = close / (ma + eps) - 1

    del df['new_close']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # Bias_v3
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ma = df['close'].rolling(n, min_periods=1).mean()
    # will output nan, / 0.03 to normalize data
    df[factor_name] = np.log((df['close'] / (ma + eps))) / 0.03

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # Bias_v4   
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ts = df[['high', 'low', 'close']].sum(axis=1) / 3.
    ma = ts.rolling(n, min_periods=1).mean()
    df[factor_name] = ts / (ma + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff


# https://bbs.quantclass.cn/thread/18610

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df["p"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    df["p_max"] = df["p"].rolling(n, min_periods=1).max()
    df["p_min"] = df["p"].rolling(n, min_periods=1).min()
    short_period = max(n//3, 1)
    df["up"] = np.where(df["p"] > df["p_max"].shift(short_period), df["p"], df["p_max"].shift(short_period))
    df["up"] = (df["up"] - df["p_max"].shift(short_period)) / df["p_max"].shift(short_period)
    df["down"] = np.where(df["p"] < df["p_min"].shift(short_period), df["p"], df["p_min"].shift(short_period))
    df["down"] = (df["down"] - df["p_min"].shift(short_period)) / df["p_min"].shift(short_period)
    df[factor_name] = (df["up"] + df["down"]).rolling(short_period, min_periods=1).mean()

    del df["p"], df["p_max"], df["p_min"], df["up"], df["down"]

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps

# https://bbs.quantclass.cn/thread/18989

def signal(*args):
    # Boll_Count
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    #
    df['Demax'] = df['high'].diff()  # Demax=HIGH-REF(HIGH,1)；
    df['Demax'] = np.where(df['Demax'] > 0, df['Demax'], 0)  # Demax=IF(Demax>0,Demax,0)
    df['Demin'] = df['low'].shift(1) - df['low']  # Demin=REF(LOW,1)-LOW
    df['Demin'] = np.where(df['Demin'] > 0, df['Demin'], 0)  # Demin=IF(Demin>0,Demin,0)
    df['Ma_Demax'] = df['Demax'].rolling(n, min_periods=1).mean()  # MA(Demax, N)
    df['Ma_Demin'] = df['Demin'].rolling(n, min_periods=1).mean()  # MA(Demin, N)
    df['Demaker'] = df['Ma_Demax'] / (
                df['Ma_Demax'] + df['Ma_Demin'])  # Demaker = MA(Demax, N) / (MA(Demax, N) + MA(Demin, N))
    # df['Demaker_chg'] = df['Demaker']/df

    df['count'] = 0
    df.loc[df['Demaker'] > 0.7, 'count'] = 1
    df.loc[df['Demaker'] < 0.3, 'count'] = -1
    df[factor_name] = df['count'].rolling(n).sum()

    del df['Demax']
    del df['Demin']
    del df['Ma_Demax']
    del df['Ma_Demin']
    del df['Demaker']
    del df['count']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Bolling 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算布林上下轨
    df['std'] = df['close'].rolling(n, min_periods=1).std()
    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['upper'] = df['ma'] + 1.0 * df['std']
    df['lower'] = df['ma'] - 1.0 * df['std']
    df['distance'] = 0
    condition_1 = df['close'] > df['upper']
    condition_2 = df['close'] < df['lower']
    df.loc[condition_1, 'distance'] = df['close'] - df['upper']
    df.loc[condition_2, 'distance'] = df['close'] - df['lower']
    df[factor_name] = df['distance'] / (df['std'] + eps)

    # 删除多余列
    del df['std'], df['ma'], df['upper'], df['lower']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Bolling_width 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['median'] = df['close'].rolling(window=n).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['z_score'] = abs(df['close'] - df['median']) / df['std']
    df['m'] = df['z_score'].rolling(window=n).mean()
    df['upper'] = df['median'] + df['std'] * df['m']
    df['lower'] = df['median'] - df['std'] * df['m']
    df[factor_name] = df['std'] * df['m'] * 2 / (df['median'] + eps)

    # 删除多余列
    del df['median'], df['std'], df['z_score'], df['m']
    del df['upper'], df['lower']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 布林线变种
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = (df['close'] - df['close'].rolling(n, min_periods=1).mean()) / df['close'].rolling(n, min_periods=1).std()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Bolling_v2 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['median'] = df['close'].rolling(n, min_periods=1).mean()
    df['std'] = df['close'].rolling(n).std(ddof=0)
    df['upper'] = df['median'] + 0.5 * df['std']
    df['lower'] = df['median'] - 0.5 * df['std']
    df[factor_name] = (df['upper'] - df['lower']) / (df['median'] + eps)

    # 删除多余列
    del df['median'], df['std'], df['upper'], df['lower']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Bolling_v3 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['median'] = df['close'].rolling(n, min_periods=1).mean()
    df['std'] = df['close'].rolling(n).std(ddof=0)
    df['upper'] = df['median'] + 0.5 * df['std']
    df[factor_name] = (df['upper'] - df['upper'].shift(1)) / (df['median'] + eps)

    # 删除多余列
    del df['median'], df['std'], df['upper']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Bolling_v4 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算布林上下轨
    df['std'] = df['close'].rolling(n, min_periods=1).std()
    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['upper'] = df['ma'] + 1.0 * df['std']
    df['lower'] = df['ma'] - 1.0 * df['std']
    # 将上下轨中间的部分设为0
    condition_0 = (df['close'] <= df['upper']) & (df['close'] >= df['lower'])
    condition_1 = df['close'] > df['upper']
    condition_2 = df['close'] < df['lower']
    df.loc[condition_0, 'distance'] = 0
    df.loc[condition_1, 'distance'] = df['close'] - df['upper']
    df.loc[condition_2, 'distance'] = df['close'] - df['lower']

    df[factor_name] = df['distance'] / (eps + df['std'])

    # 删除多余列
    del df['distance'], df['std'], df['upper'], df['lower'], df['ma']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

def signal(*args):
    # Boll_Count
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    #
    df[f'mean'] = df['close'].rolling(n).mean()
    df['std'] = df['close'].rolling(n).std(ddof=0)
    df['upper'] = df['mean'] + 2 * df['std']
    df['lower'] = df['mean'] - 2 * df['std']
    df['count'] = 0
    df.loc[df['close'] > df['upper'], 'count'] = 1
    df.loc[df['close'] < df['lower'], 'count'] = -1
    df[factor_name] = df['count'].rolling(n).sum()
    del df['mean']
    del df['std']
    del df['upper']
    del df['lower']
    del df['count']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

def signal(*args):
    # Boll_Count
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df['mtm'] = df['mtm']*df['quote_volume']/df['quote_volume'].rolling(window=n, min_periods=1).mean()

    #
    df[f'mean'] = df['mtm'].rolling(n).mean()
    df['std'] = df['mtm'].rolling(n).std(ddof=0)
    df['upper'] = df['mean'] + 2 * df['std']
    df['lower'] = df['mean'] - 2 * df['std']
    df['count'] = 0
    df.loc[df['mtm'] > df['upper'], 'count'] = 1
    df.loc[df['mtm'] < df['lower'], 'count'] = -1
    df[factor_name] = df['count'].rolling(n).sum()
    del df['mean']
    del df['std']
    del df['upper']
    del df['lower']
    del df['count']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

def signal(*args):
    # Boll_Count
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]


    df['upper'] = df['close'].rolling(n, min_periods=1).max().shift(1)
    df['lower'] = df['close'].rolling(n, min_periods=1).min().shift(1)

    df['count'] = 0
    df.loc[df['close'] > df['upper'], 'count'] = 1
    df.loc[df['close'] < df['lower'], 'count'] = -1
    df[factor_name] = df['count'].rolling(n).sum()
    # del df['mean']
    # del df['std']
    del df['upper']
    del df['lower']
    del df['count']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # BOP 指标
    """
    N=20
    BOP=MA((CLOSE-OPEN)/(HIGH-LOW),N)
    BOP 的变化范围为-1 到 1，用来衡量收盘价与开盘价的距离（正、负
    距离）占最高价与最低价的距离的比例，反映了市场的多空力量对比。
    如果 BOP>0，则多头更占优势；BOP<0 则说明空头更占优势。BOP
    越大，则说明价格被往最高价的方向推动得越多；BOP 越小，则说
    明价格被往最低价的方向推动得越多。我们可以用 BOP 上穿/下穿 0
    线来产生买入/卖出信号。
    """
    df['co'] = df['close'] - df['open']
    df['hl'] = df['high'] - df['low']
    df[factor_name] = (df['co'] / df['hl']).rolling(n, min_periods=1).mean()

    del df['co']
    del df['hl']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    v1 = (df['high'] - df['close'].shift(1)).rolling(n, min_periods=1).sum()
    v2 = (df['close'].shift(1) - df['low']).rolling(n, min_periods=1).sum()
    _br = 100 * v1 / v2
    df[factor_name] = pd.Series(_br)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Burr
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    # 只在上涨跌时关注回落幅度
    df['scores_high'] = (1 - df['close'] / df['high'].rolling(
        window=n, min_periods=1).max()).where(df['close'] - df['open'].shift(n) > 0)
    # 只在下跌时关注回升幅度
    df['scores_low'] = (1 - df['close'] / df['low'].rolling(
        window=n, min_periods=1).min()).where(df['close'] - df['open'].shift(n) < 0)
    df[factor_name] = df['scores_high'].fillna(
        0) + df['scores_low'].fillna(0)  # [-1, 1]
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 过去N分钟的主买比例
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = df['taker_buy_quote_asset_volume'].rolling(n, min_periods=1).sum() / df['quote_volume'].rolling(n, min_periods=1).sum()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 主买的vwap与当前vwap的比例.
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['vwap'] = df['quote_volume'].rolling(n, min_periods=1).sum() / df['volume'].rolling(n, min_periods=1).sum()
    df['buy_vwap'] = df['taker_buy_quote_asset_volume'].rolling(n, min_periods=1).sum() / df['taker_buy_base_asset_volume'].rolling(n, min_periods=1).sum()
    df[factor_name] = df['buy_vwap'] / df['vwap']

    del df['vwap'], df['buy_vwap']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18908

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # Copp
    df['RC'] = 100 * ((df['close'] - df['close'].shift(n)) / df['close'].shift(n) + (df['close'] - df['close'].shift(2 * n)) / df['close'].shift(2 * n))
    df['RC'] = df['RC'].rolling(n, min_periods=1).mean()

    # bbw
    df['median'] = df['close'].rolling(n, min_periods=1).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['bbw'] = (df['std'] / df['median'])

    # corr
    df['corr'] = ta.CORREL(df['close'], df['volume'], n) + 1
    df['corr'] = df['corr'].rolling(n, min_periods=1).mean()

    df[factor_name] = df['RC'] * df['bbw'] * df['corr']

    del df['RC'], df['median'],  df['std'], df['bbw'], df['corr']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
from utils.diff import add_diff, eps


def signal(*args):
    # CCI 最常用的T指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    '''
    N=14 
    TP=(HIGH+LOW+CLOSE)/3 
    MA=MA(TP,N) 
    MD=MA(ABS(TP-MA),N) 
    CCI=(TP-MA)/(0.015MD)
    CCI 指标用来衡量典型价格(最高价、最低价和收盘价的均值)与其一段时间的移动平均的偏离程度。
    CCI 可以用来反映市场的超买超卖状态。
    一般认为，CCI 超过 100 则市场处于超买状态；CCI 低于 -100 则市场处于超卖状态。
    当 CCI 下穿 100/上穿-100 时，说明股价可能要开始发生反转，可以考虑卖出/买入。
    '''

    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['ma'] = df['tp'].rolling(window=n, min_periods=1).mean()
    df['md'] = abs(df['close'] - df['ma']).rolling(window=n, min_periods=1).mean()
    df[factor_name] = (df['tp'] - df['ma']) / (df['md'] * 0.015 + eps)

    del df['tp']
    del df['ma']
    del df['md']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算魔改CCI指标
    open_ma = df['open'].rolling(n, min_periods=1).mean()
    high_ma = df['high'].rolling(n, min_periods=1).mean()
    low_ma = df['low'].rolling(n, min_periods=1).mean()
    close_ma = df['close'].rolling(n, min_periods=1).mean()
    tp = (high_ma + low_ma + close_ma) / 3
    ma = tp.rolling(n, min_periods=1).mean()
    md = abs(ma - close_ma).rolling(n, min_periods=1).mean()
    df[factor_name] = ((tp - ma) / md / 0.015)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Cci_v2 最常用的T指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    oma = ta.WMA(df['open'], timeperiod=n)
    hma = ta.WMA(df['high'], timeperiod=n)
    lma = ta.WMA(df['low'], timeperiod=n)
    cma = ta.WMA(df['close'], timeperiod=n)

    tp = (hma + lma + cma + oma) / 4
    ma = ta.WMA(tp, n)
    md = abs(ma - cma).rolling(n, min_periods=1).mean()  # MD=MA(ABS(TP-MA),N)
    df[factor_name] = (tp - ma) / (md + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Cci_v3 最常用的T指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    oma = df['open'].ewm(span=n, adjust=False).mean()
    hma = df['high'].ewm(span=n, adjust=False).mean()
    lma = df['low'].ewm(span=n, adjust=False).mean()
    cma = df['close'].ewm(span=n, adjust=False).mean()
    tp = (oma + hma + lma + cma) / 4
    ma = tp.ewm(span=n, adjust=False).mean()
    md = (cma - ma).abs().ewm(span=n, adjust=False).mean()
    df[factor_name] = (tp - ma) / (md + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # ChangeStd
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    rtn = df['close'].pct_change()
    df[factor_name] = df['close'].pct_change(n) * rtn.rolling(n).std(ddof=0)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['divnum'] = df['high'] - df['low']
    df['divnum'] = df['divnum'].replace(0, np.nan)
    df['temp'] = (2 * df['close'] - df['high'] - df['low']) / df['divnum'] * df['quote_volume']
    df[factor_name] = df['temp'].rolling(n, min_periods=1).sum()

    del df['divnum'], df['temp']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Clv 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # CLV=(2*CLOSE-LOW-HIGH)/(HIGH-LOW)
    df['CLV'] = (2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low'])
    df[factor_name] = df['CLV'].rolling(n, min_periods=1).mean()  # CLVMA=MA(CLV,N)

    # 删除多余列
    del df['CLV']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # CMF 指标
    """
    N=60
    CMF=SUM(((CLOSE-LOW)-(HIGH-CLOSE))*VOLUME/(HIGH-LOW),N)/SUM(VOLUME,N)
    CMF 用 CLV 对成交量进行加权，如果收盘价在高低价的中点之上，
    则为正的成交量（买方力量占优势）；若收盘价在高低价的中点之下，
    则为负的成交量（卖方力量占优势）。
    如果 CMF 上穿 0，则产生买入信号；
    如果 CMF 下穿 0，则产生卖出信号。
    """
    A = ((df['close'] - df['low']) - (df['high'] - df['close']) )* df['volume'] / (df['high'] - df['low'])
    df[factor_name] = A.rolling(n).sum() / df['volume'].rolling(n).sum()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Cmo
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # MAX(CLOSE-REF(CLOSE,1), 0
    df['max_su'] = np.where(df['close'] > df['close'].shift(
        1), df['close'] - df['close'].shift(1), 0)
    # SU=SUM(MAX(CLOSE-REF(CLOSE,1),0),N)
    df['sum_su'] = df['max_su'].rolling(n, min_periods=1).sum()
    # MAX(REF(CLOSE,1)-CLOSE,0)
    df['max_sd'] = np.where(df['close'].shift(
        1) > df['close'], df['close'].shift(1) - df['close'], 0)
    # SD=SUM(MAX(REF(CLOSE,1)-CLOSE,0),N)
    df['sum_sd'] = df['max_sd'].rolling(n, min_periods=1).sum()
    # CMO=(SU-SD)/(SU+SD)*100
    df[factor_name] = (df['sum_su'] - df['sum_sd']) / \
        (df['sum_su'] + df['sum_sd'] + eps) * 100

    # 删除多余列
    del df['max_su'], df['sum_su'], df['max_sd'], df['sum_sd']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Cmo_v2
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['momentum'] = df['close'] - df['close'].shift(1)
    df['up'] = np.where(df['momentum'] > 0, df['momentum'], 0)
    df['dn'] = np.where(df['momentum'] < 0, abs(df['momentum']), 0)
    df['up_sum'] = df['up'].rolling(window=n, min_periods=1).max()
    df['dn_sum'] = df['dn'].rolling(window=n, min_periods=1).max()
    df[factor_name] = (
        df['up_sum'] - df['dn_sum']) / (df['up_sum'] + df['dn_sum'] + eps)

    # 删除多余列
    del df['momentum'], df['up'], df['dn'], df['up_sum'], df['dn_sum']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Cmo_v3
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['momentum'] = df['close'] - df['close'].shift(1)
    df['up'] = np.where(df['momentum'] > 0, df['momentum'], 0)
    df['dn'] = np.where(df['momentum'] < 0, abs(df['momentum']), 0)
    df['up_sum'] = df['up'].rolling(window=n, min_periods=1).sum()
    df['dn_sum'] = df['dn'].rolling(window=n, min_periods=1).sum()
    df['cmo'] = (
        df['up_sum'] - df['dn_sum']) / (df['up_sum'] + df['dn_sum'] + eps) * 100
    df[factor_name] = df['cmo'].rolling(window=n, min_periods=1).mean()

    # 删除多余列
    del df['momentum'], df['up'], df['dn'], df['up_sum'], df['dn_sum'], df['cmo']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # COPP 指标
    """
    RC=100*((CLOSE-REF(CLOSE,N1))/REF(CLOSE,N1)+(CLOSE-REF(CLOSE,N2))/REF(CLOSE,N2))
    COPP=WMA(RC,M)
    COPP 指标用不同时间长度的价格变化率的加权移动平均值来衡量
    动量。如果 COPP 上穿/下穿 0 则产生买入/卖出信号。
    """
    df['RC'] = 100 * ((df['close'] - df['close'].shift(n)) / df['close'].shift(n) + (df['close'] - df['close'].shift(2 * n)) / df['close'].shift(2 * n))
    df[factor_name] = df['RC'].rolling(n, min_periods=1).mean()

    del df['RC']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/17821

def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # COPP
    # RC=100*((CLOSE-REF(CLOSE,N1))/REF(CLOSE,N1)+(CLOSE-REF(CLOSE,N2))/REF(CLOSE,N2))
    df['RC'] = 100 * ((df['close'] - df['close'].shift(n)) / df['close'].shift(n) + (df['close'] - df['close'].shift(2 * n)) / df['close'].shift(2 * n))
    df['RC_mean'] = df['RC'].rolling(n, min_periods=1).mean()

    # ATR
    df['median'] = df['close'].rolling(window=n).mean()
    df['c1'] = df['high'] - df['low']  # HIGH-LOW
    df['c2'] = abs(df['high'] - df['close'].shift(1))  # ABS(HIGH-REF(CLOSE,1)
    df['c3'] = abs(df['low'] - df['close'].shift(1))  # ABS(LOW-REF(CLOSE,1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    df['_ATR'] = df['TR'].rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
    # ATR指标去量纲
    df['ATR'] = df['_ATR'] / df['median']

    # 平均主动买入
    df['vma'] = df['quote_volume'].rolling(n, min_periods=1).mean()
    df['taker_buy_ma'] = (df['taker_buy_quote_asset_volume'] / df['vma']) * 100
    df['taker_buy_mean'] = df['taker_buy_ma'].rolling(window=n).mean()

    # 组合指标
    df[factor_name] = df['RC_mean'] * df['ATR'] * df['taker_buy_mean']
    # 删除多余列
    del df['RC'], df['RC_mean']
    del df['median'], df['c1'], df['c2'], df['c3'], df['TR'], df['_ATR'], df['ATR']
    del df['vma'], df['taker_buy_ma'], df['taker_buy_mean']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # Bias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]


    df['route_1'] = (df['high'] - df['open']) + ( df['high'] - df['low']) + ( df['close'] - df['low'] )
    df['route_2'] = (df['open'] - df['low']) + ( df['high'] - df['low']) + (df['high'] - df['close'])
    df['min_route']  = df[['route_1','route_2']].min(axis=1)/df['open'] #  最短路径归一化

    df['RC'] = 100 * (df['close'] / df['close'].shift(n) - 1 + df['close'] / df['close'].shift(2 * n) - 1)
    df['RC'] = df['RC'].ewm(n, adjust=False).mean()
    df['min_route'] = df['min_route'].ewm(n, adjust=False).mean()
    df[factor_name] = df['RC'] / (df['min_route']+eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18707

#Copp_v3 by zhengk
def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # COPP 指标
    """
    RC=100*((CLOSE-REF(CLOSE,N1))/REF(CLOSE,N1)+(CLOSE-REF(CLOSE,N2))/REF(CLOSE,N2))
    COPP=WMA(RC,M)
    COPP 指标用不同时间长度的价格变化率的加权移动平均值来衡量
    动量。如果 COPP 上穿/下穿 0 则产生买入/卖出信号。
    """
    df['RC'] = 100 * ((df['close'] - df['close'].shift(n)) / df['close'].shift(n) + ( df['close'] - df['close'].shift(int(1.618 * n))) / df['close'].shift(int(1.618 * n)))
    df[factor_name] = df['RC'].rolling(n, min_periods=1).mean()

    del df['RC']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # Cr
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    _typ = (df['high'] + df['low'] + df['close']) / 3
    _h = np.maximum(df['high'] - pd.Series(_typ).shift(1), 0)  # 两个数列取大值
    _l = np.maximum(pd.Series(_typ).shift(1) - df['low'], 0)

    signal = 100 * pd.Series(_h).rolling(n, min_periods=1).sum() / (
        1e-9 + pd.Series(_l).rolling(n, min_periods=1).sum())
    df[factor_name] = pd.Series(signal)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18979

def signal(*args):
    df, n, diff_num, factor_name = args

    # 时序标准化
    cr = df['close'].rolling(n, min_periods=1)
    close_standard = (df['close'] - cr.min()) / (cr.max() - cr.min())
    # 指数平均
    df[factor_name] = close_standard.ewm(span=n - 1, min_periods=1, adjust=False).mean()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/18743

def signal(*args):
    # Cs_mtm_v2 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # 收盘价动量
    df['c_mtm'] = df['close'] / df['close'].shift(n) - 1
    df['c_mtm'] = df['c_mtm'].rolling(n, min_periods=1).mean()
    # 标准差动量
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['s_mtm'] = df['std'] / df['std'].shift(n)
    df['s_mtm'] = df['s_mtm'].rolling(n, min_periods=1).mean()
    # 成交量变化
    df['v_mtm'] = df['quote_volume'] / df['quote_volume'].shift(n)
    df['v_mtm'] = df['v_mtm'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['c_mtm'] * df['s_mtm'] * df['v_mtm']

    del df['c_mtm'], df['std'], df['s_mtm'], df['v_mtm']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/17641


def signal(*args):
    # Cs_mtm_v3 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # 收盘价动量
    df['c_mtm'] = df['close'] / df['close'].shift(n) - 1
    df['c_mtm'] = df['c_mtm'].rolling(n, min_periods=1).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    # 标准差动量
    df['s_mtm'] = df['std'] / df['std'].shift(n) - 1
    df['s_mtm'] = df['s_mtm'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['c_mtm'] * df['s_mtm']

    del df['c_mtm'], df['std'], df['s_mtm']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
from utils.diff import add_diff, eps


def signal(*args):
    # Cv 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    N=10
    H_L_EMA=EMA(HIGH-LOW,N)
    CV=(H_L_EMA-REF(H_L_EMA,N))/REF(H_L_EMA,N)*100
    CV 指标用来衡量股价的波动，反映一段时间内最高价与最低价之差
    （价格变化幅度）的变化率。如果 CV 的绝对值下穿 30，买入；
    如果 CV 的绝对值上穿 70，卖出。
    """
    # H_L_EMA=EMA(HIGH-LOW,N)
    df['H_L_ema'] = (df['high'] - df['low']).ewm(n, adjust=False).mean()  
    df[factor_name] = (df['H_L_ema'] - df['H_L_ema'].shift(n)) / \
        (df['H_L_ema'].shift(n) + eps) * 100

    del df['H_L_ema']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/18772

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['pc'] = df['close'].pct_change()
    df['vol'] = df['pc'].rolling(n).std()
    df['ret'] = df['pc'].rolling(n).sum()
    df['cvr'] = (df['ret']/(df['vol'] + eps)) * (df['quote_volume']/df['quote_volume'].rolling(n, min_periods=1).mean())
    df[factor_name] = df['cvr'].rolling(n, min_periods=1).mean()
    df.drop(columns = ['pc', 'vol', 'ret', 'cvr'], inplace=True)
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/17682


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # COPP
    # RC=100*((CLOSE-REF(CLOSE,N1))/REF(CLOSE,N1)+(CLOSE-REF(CLOSE,N2))/REF(CLOSE,N2))
    df['RC'] = 100 * ((df['close'] - df['close'].shift(n)) / df['close'].shift(n) + (df['close'] - df['close'].shift(2 * n)) / df['close'].shift(2 * n))
    df['RC_mean'] = df['RC'].rolling(n, min_periods=1).mean()
    # BBW
    df['median'] = df['close'].rolling(window=n).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['z_score'] = abs(df['close'] - df['median']) / df['std']
    df['m'] = df['z_score'].rolling(window=n).mean()
    df['BBW'] = df['std'] * df['m'] * 2 / (df['median'] + 1e-8)
    df['BBW_mean'] = df['BBW'].rolling(n, min_periods=1).mean()
    # ATR
    df['c1'] = df['high'] - df['low']  # HIGH-LOW
    df['c2'] = abs(df['high'] - df['close'].shift(1))  # ABS(HIGH-REF(CLOSE,1)
    df['c3'] = abs(df['low'] - df['close'].shift(1))  # ABS(LOW-REF(CLOSE,1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(
        axis=1)  # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    df['_ATR'] = df['TR'].rolling(n, min_periods=1).mean()  # ATR=MA(TR,N)
    # ATR指标去量纲
    df['ATR'] = df['_ATR'] / df['median']

    df[factor_name] = df['RC_mean'] * df['BBW_mean'] * df['ATR']
    # 删除多余列
    del df['RC'], df['RC_mean'], df['median']
    del df['std'], df['z_score'], df['m']
    del df['BBW'], df['BBW_mean'], df['c1']
    del df['c2'], df['c3'], df['TR'], df['_ATR']
    del df['ATR']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # Dbcd 指标
    '''
    N=5
    M=16
    T=17 
    BIAS=(CLOSE-MA(CLOSE,N)/MA(CLOSE,N))*100
    BIAS_DIF=BIAS-REF(BIAS,M) 
    DBCD=SMA(BIAS_DIFF,T,1)
    DBCD(异同离差乖离率)为乖离率离差的移动平均。
    我们用 DBCD 上穿 5%/下穿-5%来产生买入/卖出信号。
    '''
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['BIAS'] = (df['close'] - df['ma']) / df['ma'] * 100
    df['BIAS_DIF'] = df['BIAS'] - df['BIAS'].shift(3 * n)
    df[factor_name] = df['BIAS_DIF'].rolling(3 * n + 2, min_periods=1).mean()

    del df['ma']
    del df['BIAS']
    del df['BIAS_DIF']


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df






#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18458

def signal(*args):
    # PMO 指标
    """

    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['Bias'] = (df['close'] - df['ma']) / df['ma'] * 100
    df['Bias_DIF'] = df['Bias'] - df['Bias'].shift(3 * n)

    volume = df['quote_volume'].rolling(n, min_periods=1).sum()
    buy_volume = df['taker_buy_quote_asset_volume'].rolling(
        n, min_periods=1).sum()

    df[factor_name] = df['Bias_DIF'].rolling(3 * n + 2, min_periods=1).mean() * (buy_volume / volume)

    del df['ma']
    del df['Bias']
    del df['Bias_DIF']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df






#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # Dbcd_v2 指标
    """

    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    close_s = df['close']
    ma = close_s.rolling(n, min_periods=1).mean()
    bias = 100 * (close_s - ma) / ma
    bias_dif = bias - bias.shift(int(3 * n + 1))
    _dbcd = bias_dif.ewm(alpha=1 / (3 * n + 2), adjust=False).mean()
    df[factor_name] = pd.Series(_dbcd)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df






#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff
from utils.tools import sma


def signal(*args):
    # Dbcd_v3 指标
    '''
    N=5
    M=16
    T=17 
    BIAS=(CLOSE-MA(CLOSE,N)/MA(CLOSE,N))*100
    BIAS_DIF=BIAS-REF(BIAS,M) 
    DBCD=SMA(BIAS_DIFF,T,1)
    DBCD(异同离差乖离率)为乖离率离差的移动平均。
    我们用 DBCD 上穿 5%/下穿-5%来产生买入/卖出信号。
    '''
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['BIAS'] = (df['close'] - df['ma']) / df['ma'] * 100
    df['BIAS_DIF'] = df['BIAS'] - df['BIAS'].shift(3 * n)
    t = 3 * n + 2
    df[factor_name] = sma(df['BIAS_DIF'], t, 1)

    del df['ma']
    del df['BIAS']
    del df['BIAS_DIF']


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # DC 指标
    """
    N=20
    UPPER=MAX(HIGH,N)
    LOWER=MIN(LOW,N)
    MIDDLE=(UPPER+LOWER)/2
    DC 指标用 N 天最高价和 N 天最低价来构造价格变化的上轨和下轨，
    再取其均值作为中轨。当收盘价上穿/下穿中轨时产生买入/卖出信号。
    """
    upper = df['high'].rolling(n, min_periods=1).max()
    lower = df['low'].rolling(n, min_periods=1).min()
    middle = (upper + lower) / 2
    width = upper - lower
    # 进行无量纲处理
    df[factor_name] = width / middle

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Dc 指标
    """
    N=20
    UPPER=MAX(HIGH,N)
    LOWER=MIN(LOW,N)
    MIDDLE=(UPPER+LOWER)/2
    Dc 指标用 N 天最高价和 N 天最低价来构造价格变化的上轨和下轨，
    再取其均值作为中轨。当收盘价上穿/下穿中轨时产生买入/卖出信号。
    """
    dc = (df['high'].rolling(n, min_periods=1).max() + df['low'].rolling(n, min_periods=1).min()) / 2.
    df[factor_name] = df['close'] - dc

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff, eps


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Dc 指标
    """
    N=20
    UPPER=MAX(HIGH,N)
    LOWER=MIN(LOW,N)
    MIDDLE=(UPPER+LOWER)/2
    Dc 指标用 N 天最高价和 N 天最低价来构造价格变化的上轨和下轨，
    再取其均值作为中轨。当收盘价上穿/下穿中轨时产生买入/卖出信号。
    """
    upper = df['high'].rolling(n, min_periods=1).max()
    lower = df['low'].rolling(n, min_periods=1).min()
    middle = (upper + lower) / 2
    # 进行无量纲处理
    df[factor_name] = (df['close'] - middle) / (middle + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Dema指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    """
    N=60
    EMA=EMA(CLOSE,N)
    DEMA=2*EMA-EMA(EMA,N)
    DEMA 结合了单重 EMA 和双重 EMA，在保证平滑性的同时减少滞后
    性。
    """
    ema = df['close'].ewm(n, adjust=False).mean()  # EMA=EMA(CLOSE,N)
    ema_ema = ema.ewm(n, adjust=False).mean()  # EMA(EMA,N)
    dema = 2 * ema - ema_ema  # DEMA=2*EMA-EMA(EMA,N)
    # dema 去量纲
    df[factor_name] = dema / (ema + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # Demakder 指标
    """
    N=20
    Demax=HIGH-REF(HIGH,1)
    Demax=IF(Demax>0,Demax,0)
    Demin=REF(LOW,1)-LOW
    Demin=IF(Demin>0,Demin,0)
    Demaker=MA(Demax,N)/(MA(Demax,N)+MA(Demin,N))
    当 Demaker>0.7 时上升趋势强烈，当 Demaker<0.3 时下跌趋势强烈。
    当 Demaker 上穿 0.7/下穿 0.3 时产生买入/卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['Demax'] = df['high'] - df['high'].shift(1)
    df['Demax'] = np.where(df['Demax'] > 0, df['Demax'], 0)
    df['Demin'] = df['low'].shift(1) - df['low']
    df['Demin'] = np.where(df['Demin'] > 0, df['Demin'], 0)
    df['Demax_ma'] = df['Demax'].rolling(n, min_periods=1).mean()
    df['Demin_ma'] = df['Demin'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['Demax_ma'] / (df['Demax_ma'] + df['Demin_ma'])

    del df['Demax']
    del df['Demin']
    del df['Demax_ma']
    del df['Demin_ma']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    short_windows = n
    long_windows = 3 * n
    df['ema_short'] = df['close'].ewm(span=short_windows, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=long_windows, adjust=False).mean()
    df['diff_ema'] = df['ema_short'] - df['ema_long']

    df['diff_ema_mean'] = df['diff_ema'].ewm(span=n, adjust=False).mean()

    df[factor_name] = df['diff_ema'] / df['diff_ema_mean'] - 1
    
    del df['ema_short']
    del df['ema_long']
    del df['diff_ema']
    del df['diff_ema_mean']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Dma 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=n)
    atr_x = atr / atr.rolling(n, min_periods=1).sum()

    ma_short = df['close'].rolling(n, min_periods=1).mean()
    ma_long = df['close'].rolling(2 * n, min_periods=1).mean()
    ma_dif = ma_short - ma_long
    dma = (ma_dif / abs(ma_dif).rolling(2 * n, min_periods=1).sum()) + 1
    df[factor_name] = dma * (1 + atr_x)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # DO 指标
    """
    DO=EMA(EMA(RSI,N),M)
    DO 是平滑处理（双重移动平均）后的 RSI 指标。DO 大于 0 则说明
    市场处于上涨趋势，小于 0 说明市场处于下跌趋势。我们用 DO 上穿
    /下穿其移动平均线来产生买入/卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    diff = df['close'].diff()
    df['up'] = np.where(diff > 0, diff, 0)
    df['down'] = np.where(diff < 0, abs(diff), 0)
    A = df['up'].rolling(n).sum()
    B = df['down'].rolling(n).sum()
    df['rsi'] = A / (A + B)
    df['ema_rsi'] = df['rsi'].ewm(n, adjust=False).mean()
    df[factor_name] = df['ema_rsi'].ewm(n, adjust=False).mean()

    del df['up']
    del df['down']
    del df['rsi']
    del df['ema_rsi']
 
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff, eps


def signal(*args):
    # Dpo
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    '''
    N=20
    DPO=CLOSE-REF(MA(CLOSE,N),N/2+1)
    DPO 是当前价格与延迟的移动平均线的差值，
    通过去除前一段时间的移动平均价格来减少长期的趋势对短期价格波动的影响。
    DPO>0 表示目前处于多头市场;
    DPO<0 表示当前处于空头市场。
    我们通过 DPO 上穿/下穿 0 线来产生买入/卖出信号。
    '''

    df['median'] = df['close'].rolling(
        window=n, min_periods=1).mean()  # 计算中轨
    df[factor_name] = (df['close'] - df['median'].shift(int(n / 2) + 1)) / (df['median'] + eps)

    del df['median']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # DzcciLower
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    tp = (df['high'] + df['low'] + df['close']) / 3.
    _ma = tp.rolling(n, min_periods=1).mean()
    md = (tp - _ma).abs().rolling(n, min_periods=1).mean()
    _cci = (tp - _ma) / (1e-9 + 0.015 * md)
    cci_middle = pd.Series(_cci).rolling(n, min_periods=1).mean()
    cci_lower = cci_middle - 2 * \
        pd.Series(_cci).rolling(n, min_periods=1).std()
    cci_ma = pd.Series(_cci).rolling(max(1, int(n/4)), min_periods=1).mean()

    signal = cci_lower - cci_ma
    df[factor_name] = pd.Series(signal)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
factors = ['DzcciLowerSignal', ]
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  zscore归一化
def scale_zscore(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).mean()
          ) / pd.Series(_s).rolling(_n, min_periods=1).std()
    return pd.Series(_s)

def signal(*args):
    # DzcciLowerSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    tp = df[['high', 'low', 'close']].sum(axis=1) / 3.
    ma = tp.rolling(n, min_periods=1).mean()
    md = (tp - ma).abs().rolling(n, min_periods=1).mean()
    cci = (tp - ma) / (1e-9 + 0.015 * md)
    cci_middle = pd.Series(cci).rolling(n, min_periods=1).mean()
    cci_lower = cci_middle - 2 * pd.Series(cci).rolling(n, min_periods=1).std()

    signal = cci_lower - df['close']
    df[factor_name] = scale_zscore(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # DzcciLowerSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    tp = df[['high', 'low', 'close']].sum(axis=1) / 3.
    ma = tp.rolling(n, min_periods=1).mean()
    md = (tp - ma).abs().rolling(n, min_periods=1).mean()
    cci = (tp - ma) / (1e-9 + 0.015 * md)
    cci_middle = pd.Series(cci).rolling(n, min_periods=1).mean()
    cci_lower = cci_middle - 2 * pd.Series(cci).rolling(n, min_periods=1).std()
    cci_ma = pd.Series(cci).rolling(max(1, int(n / 4)), min_periods=1).mean()

    signal = cci_lower - cci_ma
    df[factor_name] = scale_01(signal, n)


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # DzcciUpper
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    tp = df[['high', 'low', 'close']].sum(axis=1) / 3.
    ma = tp.rolling(n, min_periods=1).mean()
    md = (tp - ma).abs().rolling(n, min_periods=1).mean()
    cci = (tp - ma) / (1e-9 + 0.015 * md)
    cci_middle = pd.Series(cci).rolling(n, min_periods=1).mean()
    cci_upper = cci_middle + 2 * pd.Series(cci).rolling(n, min_periods=1).std()

    signal = cci_upper
    df[factor_name] = scale_01(signal, n)
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # DzcciUpperSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    tp = df[['high', 'low', 'close']].sum(axis=1) / 3.
    ma = tp.rolling(n, min_periods=1).mean()
    md = (tp - ma).abs().rolling(n, min_periods=1).mean()
    cci = (tp - ma) / (1e-9 + 0.015 * md)
    cci_middle = pd.Series(cci).rolling(n, min_periods=1).mean()
    cci_upper = cci_middle + 2 * pd.Series(cci).rolling(n, min_periods=1).std()

    signal = df['close'] - cci_upper
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # DzcciUpperSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    tp = df[['high', 'low', 'close']].sum(axis=1) / 3.
    ma = tp.rolling(n, min_periods=1).mean()
    md = (tp - ma).abs().rolling(n, min_periods=1).mean()
    cci = (tp - ma) / (1e-9 + 0.015 * md)
    cci_middle = pd.Series(cci).rolling(n, min_periods=1).mean()
    cci_upper = cci_middle + 2 * pd.Series(cci).rolling(n, min_periods=1).std()
    cci_ma = pd.Series(cci).rolling(max(1, int(n / 4)), min_periods=1).mean()

    signal = cci_ma - cci_upper
    df[factor_name] = scale_01(signal, n)
   
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  zscore归一化
def scale_zscore(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).mean()
          ) / pd.Series(_s).rolling(_n, min_periods=1).std()
    return pd.Series(_s)

def signal(*args):
    # DzrsiLowerSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    rtn = df['close'].diff()
    up = np.where(rtn > 0, rtn, 0)
    dn = np.where(rtn < 0, rtn.abs(), 0)
    a = pd.Series(up).rolling(n, min_periods=1).sum()
    b = pd.Series(dn).rolling(n, min_periods=1).sum()

    a *= 1e3
    b *= 1e3

    rsi = a / (1e-9 + a + b)

    rsi_middle = rsi.rolling(n, min_periods=1).mean()
    # rsi_upper = rsi_middle + 2 * rsi.rolling(n, min_periods=1).std()
    rsi_lower = rsi_middle - 2 * rsi.rolling(n, min_periods=1).std()
    rsi_ma = rsi.rolling(int(n / 2), min_periods=1).mean()

    signal = rsi_lower - rsi_ma
    df[factor_name] = scale_zscore(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # DzrsiUpperSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    rtn = df['close'].diff()
    up = np.where(rtn > 0, rtn, 0)
    dn = np.where(rtn < 0, rtn.abs(), 0)
    a = pd.Series(up).rolling(n, min_periods=1).sum()
    b = pd.Series(dn).rolling(n, min_periods=1).sum()

    a *= 1e3
    b *= 1e3

    rsi = a / (1e-9 + a + b)

    rsi_middle = rsi.rolling(n, min_periods=1).mean()
    rsi_upper = rsi_middle + 2 * rsi.rolling(n, min_periods=1).std()
    # rsi_lower = rsi_middle - 2 * rsi.rolling(n, min_periods=1).std()
    rsi_ma = rsi.rolling(int(n / 2), min_periods=1).mean()

    signal = rsi_ma - rsi_upper
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # Emv
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    mpm = (df['high'] + df['low']) / 2. - \
        (df['high'].shift(1) + df['low'].shift(1)) / 2.
    v_divisor = df['volume'].rolling(n, min_periods=1).mean()
    _br = df['volume'] / v_divisor / (1e-9 + df['high'] - df['low'])

    signal = mpm / (1e-9 + _br)
    df[factor_name] = pd.Series(signal)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)


def signal(*args):
    # EnvLower
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    N=25
    PARAM=0.05 
    MAC=MA(CLOSE,N) 
    UPPER=MAC*(1+PARAM) 
    LOWER=MAC*(1-PARAM)
    ENV(Envolope 包络线)指标是由移动平均线上下平移一定的幅度 (百分比)所得。
    我们知道，价格与移动平均线的交叉可以产生交易信号。
    但是因为市场本身波动性比较强，可能产生很多虚假的交易信号。
    所以我们把移动平均线往上往下平移。
    当价格突破上轨时再产生买入信号或者当价格突破下轨再产生卖出信号。
    这样的方式可以去掉很多假信号
    '''

    lower = (1 - 0.05) * df['close'].rolling(n, min_periods=1).mean()

    signal = lower
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  zscore归一化
def scale_zscore(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).mean()
          ) / pd.Series(_s).rolling(_n, min_periods=1).std()
    return pd.Series(_s)

def signal(*args):
    # EnvLowerSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    N=25
    PARAM=0.05 
    MAC=MA(CLOSE,N) 
    UPPER=MAC*(1+PARAM) 
    LOWER=MAC*(1-PARAM)
    ENV(Envolope 包络线)指标是由移动平均线上下平移一定的幅度 (百分比)所得。
    我们知道，价格与移动平均线的交叉可以产生交易信号。
    但是因为市场本身波动性比较强，可能产生很多虚假的交易信号。
    所以我们把移动平均线往上往下平移。
    当价格突破上轨时再产生买入信号或者当价格突破下轨再产生卖出信号。
    这样的方式可以去掉很多假信号
    '''

    lower = (1 - 0.05) * df['close'].rolling(n, min_periods=1).mean()

    signal = lower - df['close']
    df[factor_name] = scale_zscore(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # EnvSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    lower = (1 - 0.05) * df['close'].rolling(n, min_periods=1).mean()

    df[factor_name] = (df['close'] - lower) / (0.1 * df['close'].rolling(n, min_periods=1).mean())
   

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # EnvUpper
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    upper = (1 + 0.05) * df['close'].rolling(n, min_periods=1).mean()

    signal = upper
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  zscore归一化
def scale_zscore(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).mean()
          ) / pd.Series(_s).rolling(_n, min_periods=1).std()
    return pd.Series(_s)


def signal(*args):
    # EnvUpperSignal
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    N=25
    PARAM=0.05 
    MAC=MA(CLOSE,N) 
    UPPER=MAC*(1+PARAM) 
    LOWER=MAC*(1-PARAM)
    ENV(Envolope 包络线)指标是由移动平均线上下平移一定的幅度 (百分比)所得。
    我们知道，价格与移动平均线的交叉可以产生交易信号。
    但是因为市场本身波动性比较强，可能产生很多虚假的交易信号。
    所以我们把移动平均线往上往下平移。
    当价格突破上轨时再产生买入信号或者当价格突破下轨再产生卖出信号。
    这样的方式可以去掉很多假信号
    '''

    upper = (1 + 0.05) * df['close'].rolling(n, min_periods=1).mean()

    signal = df['close'] - upper
    df[factor_name] = scale_zscore(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Er 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    N=20 
    BullPower=HIGH-EMA(CLOSE,N) 
    BearPower=LOW-EMA(CLOSE,N)
    ER 为动量指标, 用来衡量市场的多空力量对比。
    在多头市场，人们会更贪婪地在接近高价的地方买入，BullPower 越高则当前多头力量越强;
    在空头市场，人们可能因为恐惧而在接近低价的地方卖出, BearPower 越低则当前空头力量越强。
    当两者都大于 0 时，反映当前多头力量占据主导地位;
    两者都小于 0 则反映空头力量占据主导地位。 
    如果 BearPower 上穿 0，则产生买入信号; 
    如果 BullPower 下穿 0，则产生卖出信号。
    '''

    a = 2 / (n + 1)
    df['ema'] = df['close'].ewm(alpha=a, adjust=False).mean()
    df['BullPower'] = (df['high'] - df['ema']) / df['ema']
    df['BearPower'] = (df['low'] - df['ema']) / df['ema']
    df[factor_name] = df['BullPower'] + df['BearPower']

    # 删除多余列
    del df['ema'], df['BullPower'], df['BearPower']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff,eps

def signal(*args):
    """
    N=20
    BullPower=HIGH-EMA(CLOSE,N)
    BearPower=LOW-EMA(CLOSE,N)
    ER 为动量指标。用来衡量市场的多空力量对比。在多头市场，人们
    会更贪婪地在接近高价的地方买入，BullPower 越高则当前多头力量
    越强；而在空头市场，人们可能因为恐惧而在接近低价的地方卖出。
    BearPower 越低则当前空头力量越强。当两者都大于 0 时，反映当前
    多头力量占据主导地位；两者都小于0则反映空头力量占据主导地位。
    如果 BearPower 上穿 0，则产生买入信号；
    如果 BullPower 下穿 0，则产生卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ema = df['close'].ewm(n, adjust=False).mean()  # EMA(CLOSE,N)
    bull_power = df['high'] - ema  # 越高表示上涨 牛市 BullPower=HIGH-EMA(CLOSE,N)
    bear_power = df['low'] - ema  # 越低表示下降越厉害  熊市 BearPower=LOW-EMA(CLOSE,N)
    df[factor_name] = bear_power / (ema + eps)  # 去量纲

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff

eps = 1e-8


def signal(*args):
    """
    N=20
    BullPower=HIGH-EMA(CLOSE,N)
    BearPower=LOW-EMA(CLOSE,N)
    ER 为动量指标。用来衡量市场的多空力量对比。在多头市场，人们
    会更贪婪地在接近高价的地方买入，BullPower 越高则当前多头力量
    越强；而在空头市场，人们可能因为恐惧而在接近低价的地方卖出。
    BearPower 越低则当前空头力量越强。当两者都大于 0 时，反映当前
    多头力量占据主导地位；两者都小于0则反映空头力量占据主导地位。
    如果 BearPower 上穿 0，则产生买入信号；
    如果 BullPower 下穿 0，则产生卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ema = df['close'].ewm(n, adjust=False).mean()  # EMA(CLOSE,N)
    bull_power = df['high'] - ema  # 越高表示上涨 牛市 BullPower=HIGH-EMA(CLOSE,N)
    bear_power = df['low'] - ema  # 越低表示下降越厉害  熊市 BearPower=LOW-EMA(CLOSE,N)
    df[factor_name] = bull_power / (ema + eps)  # 去量纲

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** Expma ********************
    # N1=12
    # N2=50
    # EMA1=EMA(CLOSE,N1)
    # EMA2=EMA(CLOSE,N2)
    # 指数移动平均是简单移动平均的改进版，用于改善简单移动平均的滞后性问题。
    ema1 = df['close'].ewm(span=n, min_periods=1).mean()
    ema2 = df['close'].ewm(span=(4 * n), min_periods=1).mean()

    signal = ema1 - ema2
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbLower 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle - 1.618 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbLowerSignal 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # Upper1=MIDDLE+1.618*ATR
    # Upper2=MIDDLE+2.618*ATR
    # Upper3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle - 1.618 * atr - df['close']
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbLowerSignal 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # Upper1=MIDDLE+1.618*ATR
    # Upper2=MIDDLE+2.618*ATR
    # Upper3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle - 2.618 * atr - df['close']
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbLowerSignal_v3 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # Upper1=MIDDLE+1.618*ATR
    # Upper2=MIDDLE+2.618*ATR
    # Upper3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle - 4.236 * atr - df['close']
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # ******************** FbLower 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle - 2.618 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbLower 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle - 4.236 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18373

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

  
    params = [5, 8, 13, 21, 34, 55, 89]
    df['Fbnq_mean'] = 0
    df['BbwOri'] = 0
    for pn in params:
        # 动量
        df['Fbnq_mean'] += df['close'].ewm(span=pn, adjust=False).mean()
        # 波动率
        df['BbwOri'] += df['close'].rolling(n).std(ddof=0) / df['close'].rolling(n, min_periods=1).mean()
    # 动量
    df['Fbnq_mean'] = df['Fbnq_mean'] / len(params)
    df['Fbnq_mean'] = df['Fbnq_mean'].pct_change(n)

    # 波动率
    df['BbwOri'] = df['BbwOri'] / len(params)

    # 动量 * 波动率
    df[factor_name] = df['Fbnq_mean'] * df['BbwOri']
    del df['Fbnq_mean'], df['BbwOri']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbUpper 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle + 1.618 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbUpperSignal 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = df['close'] - middle - 1.618 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbUpperSignal 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = df['close'] - middle - 2.618 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbUpperSignal_v3 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = df['close'] - middle - 4.236 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbUpper_v2 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle + 2.618 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** FbUpper_v3 指标 ********************
    # N=20
    # TR=MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1)),ABS(LOW-REF(CLOSE,1)))
    # ATR=MA(TR,N)
    # MIDDLE=MA(CLOSE,N)
    # UPPER1=MIDDLE+1.618*ATR
    # UPPER2=MIDDLE+2.618*ATR
    # UPPER3=MIDDLE+4.236*ATR
    # LOWER1=MIDDLE-1.618*ATR
    # LOWER2=MIDDLE-2.618*ATR
    # LOWER3=MIDDLE-4.236*ATR
    # FB指标类似于布林带，都以价格的移动平均线为中轨，在中线上下浮动一定数值构造上下轨。
    # 不同的是，Fibonacci Bands有三条上轨和三条下轨，且分别为中轨加减ATR乘Fibonacci因子所得。
    # 当收盘价突破较高的两个上轨的其中之一时，产生买入信号；收盘价突破较低的两个下轨的其中之一时，产生卖出信号。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()

    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].rolling(n, min_periods=1).mean()

    signal = middle + 4.236 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


#wma(加权移动平均线)
def wma(df, column='close', k=10):
    weights = np.arange(1, k + 1)
    wmas = df[column].rolling(k).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
    return wmas

#sma(简单移动平均线)
def sma(df, column='close', k=10):
    smas = df[column].rolling(k, min_periods=1).mean()
    return smas

#ema（指數平滑移動平均線）备用
def ema(df, column='close', k=10):
    emas = df[column].ewm(k, adjust=False).mean()
    return emas

# 指标名 版本： FearGreed_Yidai_v1
# https://bbs.quantclass.cn/thread/9458
def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    #计算TR 真实振幅 并作平滑 标准化（后续计算采用标准化参数）
    df['c1'] = df['high'] - df['low']  # HIGH-LOW
    df['c2'] = abs(df['high'] - df['close'].shift(1))  # ABS(HIGH-REF(CLOSE,1)
    df['c3'] = abs(df['low'] - df['close'].shift(1))  # ABS(LOW-REF(CLOSE,1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)
    df['sma'] = sma(df, column='close', k=n)
    df['STR'] = df['TR']/df['sma']

    # 多空振幅分离
    df['trUp'] = np.where(df['close'] > df['close'].shift(1), df['STR'], 0)
    df['trDn'] = np.where(df['close'] < df['close'].shift(1), df['STR'], 0)

    # 多空振幅平滑 快慢均线
    df['wmatrUp1'] = wma(df, column='trUp', k=n)
    df['wmatrDn1'] = wma(df, column='trDn', k=n)
    df['wmatrUp2'] = wma(df, column='trUp', k=2*n)
    df['wmatrDn2'] = wma(df, column='trDn', k=2*n)

    # 多空振幅比较 1阶导 描绘速度 并作平滑
    df['fastDiff'] = df['wmatrUp1'] - df['wmatrDn1']
    df['slowDiff'] = df['wmatrUp2'] - df['wmatrDn2']

    # 快慢均线比较描绘 2阶导 描绘加速度
    df['FastMinusSlow'] = df['fastDiff'] - df['slowDiff']
    df['fgi'] = wma(df, column='FastMinusSlow', k=n)

    # 返回df
    df[factor_name] = df['fgi']

    # 删除多余列
    del df['c1'], df['c2'], df['c3'], df['TR'], df['STR'], df['sma']
    del df['trUp'], df['trDn'], df['fastDiff'], df['slowDiff'], df['FastMinusSlow'], df['fgi']
    del df['wmatrUp1'], df['wmatrDn1'], df['wmatrUp2'], df['wmatrDn2']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Fi
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    _fi = df['volume'] * (df['close'] - df['close'].shift(1))
    _fi_zscore = (_fi - _fi.rolling(n, min_periods=1).mean()) / \
                 (_fi.rolling(n, min_periods=1).std() + eps)
    signal = _fi_zscore.ewm(span=n, adjust=False, min_periods=1).mean()
    df[factor_name] = pd.Series(signal)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18739

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['_fi'] = df['volume'] * (df['close'] - df['close'].shift(1))

    diff = df['_fi'].diff()
    df['up'] = np.where(diff > 0, diff, 0)
    df['down'] = np.where(diff < 0, abs(diff), 0)
    A = df['up'].rolling(n,min_periods=1).sum()
    B = df['down'].rolling(n,min_periods=1).sum()
    RSI = A / (A + B + eps)

    signal = RSI.ewm(span=n, adjust=False, min_periods=1).mean()
    df[factor_name] = pd.Series(signal)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # FISHER指标
    """
    N=20
    PARAM=0.3
    PRICE=(HIGH+LOW)/2
    PRICE_CH=2*(PRICE-MIN(LOW,N)/(MAX(HIGH,N)-MIN(LOW,N))-
    0.5)
    PRICE_CHANGE=0.999 IF PRICE_CHANGE>0.99 
    PRICE_CHANGE=-0.999 IF PRICE_CHANGE<-0.99
    PRICE_CHANGE=PARAM*PRICE_CH+(1-PARAM)*REF(PRICE_CHANGE,1)
    FISHER=0.5*REF(FISHER,1)+0.5*log((1+PRICE_CHANGE)/(1-PRICE_CHANGE))
    PRICE_CH 用来衡量当前价位于过去 N 天的最高价和最低价之间的
    位置。Fisher Transformation 是一个可以把股价数据变为类似于正态
    分布的方法。Fisher 指标的优点是减少了普通技术指标的滞后性。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    PARAM = 1/ n
    df['price'] = (df['high'] + df['low']) / 2
    df['min_low'] = df['low'].rolling(n).min()
    df['max_high'] = df['high'].rolling(n).max()
    df['price_ch'] = 2 * (df['price'] - df['min_low']) / (df['max_high'] - df['low']) - 0.5
    df['price_change'] = PARAM * df['price_ch'] + (1 - PARAM) * df['price_ch'].shift(1)
    df['price_change'] = np.where(df['price_change'] > 0.99, 0.999, df['price_change'])
    df['price_change'] = np.where(df['price_change'] < -0.99, -0.999, df['price_change'])
    df[factor_name] = 0.5 * np.log((1+df['price_change']) / (1 - df['price_change']))

    del df['price']
    del df['min_low']
    del df['max_high']
    del df['price_ch']
    del df['price_change']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df








#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import talib
import pandas as pd
from utils.diff import add_diff


# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
            1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

# FISHER_v2
def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # FISHER_v2指标
    """
    N=20
    PARAM=0.3
    PRICE=(HIGH+LOW)/2
    PRICE_CH=2*(PRICE-MIN(LOW,N)/(MAX(HIGH,N)-MIN(LOW,N))-
    0.5)
    PRICE_CHANGE=0.999 IF PRICE_CHANGE>0.99 
    PRICE_CHANGE=-0.999 IF PRICE_CHANGE<-0.99
    PRICE_CHANGE=PARAM*PRICE_CH+(1-PARAM)*REF(PRICE_CHANGE,1)
    FISHER=0.5*REF(FISHER,1)+0.5*log((1+PRICE_CHANGE)/(1-PRICE_CHANGE))
    PRICE_CH 用来衡量当前价位于过去 N 天的最高价和最低价之间的
    位置。
    Fisher Transformation 是一个可以把股价数据变为类似于正态
    分布的方法。
    Fisher 指标的优点是减少了普通技术指标的滞后性。
    """
    PARAM = 0.5  # 0.33
    df['price'] = (df['high'] + df['low']) / 2
    df['min_low'] = df['low'].rolling(n).min()
    df['max_high'] = df['high'].rolling(n).max()
    df['price_ch'] = PARAM * 2 * ((df['price'] - df['min_low']) / (df['max_high'] - df['min_low']) - 0.5)
    df['price_change'] = df['price_ch'] + (1 - PARAM) * df['price_ch'].shift(1)
    df['price_change'] = np.where(df['price_change'] > 0.99, 0.999, df['price_change'])
    df['price_change'] = np.where(df['price_change'] < -0.99, -0.999, df['price_change'])

    df[factor_name] = 0.3 * df['price_change'] + 0.7 * df['price_change'].shift(1)

    # price = (df['high'] + df['low']) / 2.
    # low_min = df['low'].rolling(n, min_periods=1).min()
    # high_max = df['high'].rolling(n, min_periods=1).max()
    # price_ch = 2 * (price - 0.5 - low_min / (1e-9 + high_max - low_min))
    # price_ch = np.where(price_ch > 0.99, 0.99, price_ch)
    # price_ch = np.where(price_ch < -0.99, -0.99, price_ch)
    # price_ch = 0.3 * pd.Series(price_ch) + 0.7 * pd.Series(price_ch).shift(1)

    # signal = fisher
    # df[factor_name] = scale_01(signal, n)

    del df['price']
    del df['min_low']
    del df['max_high']
    del df['price_ch']
    del df['price_change']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import talib
import pandas as pd
from utils.diff import add_diff


# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
            1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

# FISHER_v3
def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # FISHER_v3指标
    """
    N=20
    PARAM=0.3
    PRICE=(HIGH+LOW)/2
    PRICE_CH=2*(PRICE-MIN(LOW,N)/(MAX(HIGH,N)-MIN(LOW,N))-
    0.5)
    PRICE_CHANGE=0.999 IF PRICE_CHANGE>0.99 
    PRICE_CHANGE=-0.999 IF PRICE_CHANGE<-0.99
    PRICE_CHANGE=PARAM*PRICE_CH+(1-PARAM)*REF(PRICE_CHANGE,1)
    FISHER=0.5*REF(FISHER,1)+0.5*log((1+PRICE_CHANGE)/(1-PRICE_CHANGE))
    PRICE_CH 用来衡量当前价位于过去 N 天的最高价和最低价之间的
    位置。
    Fisher Transformation 是一个可以把股价数据变为类似于正态
    分布的方法。
    Fisher 指标的优点是减少了普通技术指标的滞后性。
    """
    PARAM = 0.33  # 0.33
    df['price'] = (df['high'] + df['low']) / 2
    df['min_low'] = df['low'].rolling(n).min()
    df['max_high'] = df['high'].rolling(n).max()
    df['price_ch'] = PARAM * 2 * ((df['price'] - df['min_low']) / (df['max_high'] - df['min_low']) - 0.5)
    df['price_change'] = df['price_ch'] + (1 - PARAM) * df['price_ch'].shift(1)
    df['price_change'] = np.where(df['price_change'] > 0.99, 0.999, df['price_change'])
    df['price_change'] = np.where(df['price_change'] < -0.99, -0.999, df['price_change'])

    df['price_change'] = 0.3 * df['price_change'] + 0.7 * df['price_change'].shift(1)
    df[factor_name] = 0.5 * df['price_change'].shift(1) + 0.5 * np.log(
        ((1 + df['price_change']) / (1 - df['price_change'])))

    # price = (df['high'] + df['low']) / 2.
    # low_min = df['low'].rolling(n, min_periods=1).min()
    # high_max = df['high'].rolling(n, min_periods=1).max()
    # price_ch = 2 * (price - 0.5 - low_min / (high_max - low_min))
    # price_ch = np.where(price_ch > 0.99, 0.99, price_ch)
    # price_ch = np.where(price_ch < -0.99, -0.99, price_ch)
    # price_ch = 0.3 * pd.Series(price_ch) + 0.7 * pd.Series(price_ch).shift(1)
    # fisher = 0.5 * pd.Series(price_ch).shift(1) + 0.5 * pd.Series(np.log((1 + price_ch) / (1 - price_ch)))
    #
    # signal = fisher
    # df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    #该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['force'] = df['quote_volume'] * (df['close'] - df['close'].shift(1))
    df[factor_name] = df['force']/ df['force'].rolling(n, min_periods=1).mean()

    # ref = ma.shift(n)  # MADisplaced=REF(MA_CLOSE,M)


    del df['force']


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # GAP 常用的T指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['_ma'] = df['close'].rolling(window=n, min_periods=1).mean()
    df['_wma'] = ta.WMA(df['close'], n)
    df['_gap'] = df['_wma'] - df['_ma']
    df[factor_name] = (df['_gap'] / abs(df['_gap']).rolling(window=n).sum())

    del df['_ma']
    del df['_wma']
    del df['_gap']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # Grid
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df['median'] = df['close'].rolling(n, min_periods=1).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['grid'] = (df['close'] - df['median']) / df['std']
    df['grid'] = df['grid'].replace([np.inf, -np.inf], np.nan)
    df['grid'].fillna(value=0, inplace=True)
    df['grid'] = df['grid'].rolling(window=n).mean()
    df[factor_name] = df['grid'].pct_change(n)
    # df['gridInt'] = df['grid'].astype("int")
    # df[factor_name] = df['gridInt'].pct_change(n)

    del df['median'], df['std'], df['grid']  # , df['gridInt']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff


def signal(*args):
 # HLMA 指标
    """
    N1=20
    N2=20
    HMA=MA(HIGH,N1)
    LMA=MA(LOW,N2)
    HLMA 指标是把普通的移动平均中的收盘价换为最高价和最低价分
    别得到 HMA 和 LMA。当收盘价上穿 HMA/下穿 LMA 时产生买入/卖
    出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    hma = df['high'].rolling(n, min_periods=1).mean()
    lma = df['low'].rolling(n, min_periods=1).mean()
    df['HLMA'] = hma - lma
    df['HLMA_mean'] = df['HLMA'].rolling(n, min_periods=1).mean()

    # 去量纲
    df[factor_name] = df['HLMA'] / df['HLMA_mean'] - 1

    
    del df['HLMA']
    del df['HLMA_mean']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Hma
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    N=20 
    HMA=MA(HIGH,N)
    HMA 指标为简单移动平均线把收盘价替换为最高价。
    当最高价上穿/下穿 HMA 时产生买入/卖出信号
    '''
    hma = df['high'].rolling(n, min_periods=1).mean()
    # 剔除量纲
    df[factor_name] = (df['high'] - hma) / (hma + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # HmaSignal 指标
    """
    N=20
    HmaSignal=MA(HIGH,N)
    HmaSignal 指标为简单移动平均线把收盘价替换为最高价。当最高价上穿/
    下穿 HmaSignal 时产生买入/卖出信号。
    """
    hma = df['high'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['high'] - hma

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # HULLMA 指标
    """
    N=20,80
    X=2*EMA(CLOSE,[N/2])-EMA(CLOSE,N)
    HULLMA=EMA(X,[√𝑁])
    HULLMA 也是均线的一种，相比于普通均线有着更低的延迟性。我们
    用短期均线上/下穿长期均线来产生买入/卖出信号。
    """
    ema1 = df['close'].ewm(n, adjust=False).mean()
    ema2 = df['close'].ewm(n * 2, adjust=False).mean()
    df['X'] = 2 * ema1 - ema2
    df['HULLMA'] = df['X'].ewm(int(np.sqrt(2 * n)), adjust=False).mean()

    df[factor_name] = df['X'] / df['HULLMA']
    
    del df['X']
    del df['HULLMA']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Hullma 指标
    """
    N=20,80
    X=2*EMA(CLOSE,[N/2])-EMA(CLOSE,N)
    Hullma=EMA(X,[√𝑁])
    Hullma 也是均线的一种，相比于普通均线有着更低的延迟性。我们
    用短期均线上/下穿长期均线来产生买入/卖出信号。
    """
    _x = 2 * df['close'].ewm(span=int(n / 2), adjust=False, min_periods=1).mean() - df['close'].ewm(
        span=n, adjust=False, min_periods=1).mean()
    hullma = _x.ewm(span=int(np.sqrt(n)), adjust=False, min_periods=1).mean()

    signal = _x - hullma
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # IC 指标
    """
    N1=9
    N2=26
    N3=52
    TS=(MAX(HIGH,N1)+MIN(LOW,N1))/2
    KS=(MAX(HIGH,N2)+MIN(LOW,N2))/2
    SPAN_A=(TS+KS)/2
    SPAN_B=(MAX(HIGH,N3)+MIN(LOW,N3))/2
    在 IC 指标中，SPAN_A 与 SPAN_B 之间的部分称为云。如果价格在
    云上，则说明是上涨趋势（如果 SPAN_A>SPAN_B，则上涨趋势强
    烈；否则上涨趋势较弱）；如果价格在云下，则为下跌趋势（如果
    SPAN_A<SPAN_B，则下跌趋势强烈；否则下跌趋势较弱）。该指
    标的使用方式与移动平均线有许多相似之处，比如较快的线（TS）突
    破较慢的线（KS），价格突破 KS,价格突破云，SPAN_A 突破 SPAN_B
    等。我们产生信号的方式是：如果价格在云上方 SPAN_A>SPAN_B，
    则当价格上穿 KS 时买入；如果价格在云下方且 SPAN_A<SPAN_B，
    则当价格下穿 KS 时卖出。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    n2 = 3 * n
    n3 = 2 * n2
    df['max_high_1'] = df['high'].rolling(n, min_periods=1).max()
    df['min_low_1'] = df['low'].rolling(n, min_periods=1).min()
    df['TS'] = (df['max_high_1'] + df['min_low_1']) / 2
    df['max_high_2'] = df['high'].rolling(n2, min_periods=1).max()
    df['min_low_2'] = df['low'].rolling(n2, min_periods=1).min()
    df['KS'] = (df['max_high_2'] + df['min_low_2']) / 2
    df['span_A'] = (df['TS'] + df['KS']) / 2
    df['max_high_3'] = df['high'].rolling(n3, min_periods=1).max()
    df['min_low_3'] = df['low'].rolling(n3, min_periods=1).min()
    df['span_B'] = (df['max_high_3'] + df['min_low_3']) / 2

    # 去量纲
    df[factor_name] = df['span_A'] / df['span_B']

    del df['max_high_1']
    del df['max_high_2']
    del df['max_high_3']
    del df['min_low_1']
    del df['min_low_2']
    del df['min_low_3']
    del df['TS']
    del df['KS']
    del df['span_A']
    del df['span_B']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # Ic
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high_max1 = df['high'].rolling(n, min_periods=1).max()
    high_max2 = df['high'].rolling(int(2 * n), min_periods=1).max()
    high_max3 = df['high'].rolling(int(3 * n), min_periods=1).max()
    low_min1 = df['low'].rolling(n, min_periods=1).min()
    low_min2 = df['low'].rolling(int(2 * n), min_periods=1).min()
    low_min3 = df['low'].rolling(int(3 * n), min_periods=1).min()
    ts = (high_max1 + low_min1) / 2.
    ks = (high_max2 + low_min2) / 2.
    span_a = (ts + ks) / 2.
    span_b = (high_max3 + low_min3) / 2.
    signal = (df['close'] - span_b) / (1e-9 + span_a - span_b)

    df[factor_name] = pd.Series(signal)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # Ic
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high_max1 = df['high'].rolling(n, min_periods=1).max()
    high_max2 = df['high'].rolling(int(2 * n), min_periods=1).max()
    high_max3 = df['high'].rolling(int(3 * n), min_periods=1).max()
    low_min1 = df['low'].rolling(n, min_periods=1).min()
    low_min2 = df['low'].rolling(int(2 * n), min_periods=1).min()
    low_min3 = df['low'].rolling(int(3 * n), min_periods=1).min()
    ts = (high_max1 + low_min1) / 2.
    ks = (high_max2 + low_min2) / 2.
    span_a = (ts + ks) / 2.
    span_b = (high_max3 + low_min3) / 2.

    signal = span_a - span_b
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    # Ic
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high_max1 = df['high'].rolling(n, min_periods=1).max()
    high_max2 = df['high'].rolling(int(2 * n), min_periods=1).max()
    high_max3 = df['high'].rolling(int(3 * n), min_periods=1).max()
    low_min1 = df['low'].rolling(n, min_periods=1).min()
    low_min2 = df['low'].rolling(int(2 * n), min_periods=1).min()
    low_min3 = df['low'].rolling(int(3 * n), min_periods=1).min()
    ts = (high_max1 + low_min1) / 2.
    ks = (high_max2 + low_min2) / 2.
    span_a = (ts + ks) / 2.
    span_b = (high_max3 + low_min3) / 2.

    signal = (df['close'] - span_b) / (1e-9 + span_a - span_b)
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # IMI 指标
    """
    N=14
    INC=SUM(IF(CLOSE>OPEN,CLOSE-OPEN,0),N)
    DEC=SUM(IF(OPEN>CLOSE,OPEN-CLOSE,0),N)
    IMI=INC/(INC+DEC)
    IMI 的计算方法与 RSI 很相似。其区别在于，在 IMI 计算过程中使用
    的是收盘价和开盘价，而 RSI 使用的是收盘价和前一天的收盘价。所
    以，RSI 做的是前后两天的比较，而 IMI 做的是同一个交易日内的比
    较。如果 IMI 上穿 80，则产生买入信号；如果 IMI 下穿 20，则产生
    卖出信号。
    """
    df['INC'] = np.where(df['close'] > df['open'], df['close'] - df['open'], 0)
    df['INC_sum'] = df['INC'].rolling(n).sum()
    df['DEC'] = np.where(df['open'] > df['close'], df['open'] - df['close'], 0)
    df['DEC_sum'] = df['DEC'].rolling(n).sum()
    df[factor_name] = df['INC_sum'] / (df['INC_sum'] + df['DEC_sum'])

    
    del df['INC']
    del df['INC_sum']
    del df['DEC']
    del df['DEC_sum']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy  as np
import talib as ta
import pandas as pd
from utils.diff import add_diff, eps


def signal(*args):
    # ******************** KAMA ********************
    # N=10
    # N1=2
    # N2=30
    # DIRECTION=CLOSE-REF(CLOSE,N)
    # VOLATILITY=SUM(ABS(CLOSE-REF(CLOSE,1)),N)
    # ER=DIRETION/VOLATILITY
    # FAST=2/(N1+1)
    # SLOW=2/(N2+1)
    # SMOOTH=ER*(FAST-SLOW)+SLOW
    # COF=SMOOTH*SMOOTH
    # KAMA=COF*CLOSE+(1-COF)*REF(KAMA,1)
    # KAMA指标与VIDYA指标类似，都是把ER(EfficiencyRatio)指标加入到移动平均的权重中，
    # 其用法与其他移动平均线类似。在当前趋势较强时，ER值较大，KAMA会赋予当前价格更大的权重，
    # 使得KAMA紧随价格变动，减小其滞后性；在当前趋势较弱（比如振荡市中）,ER值较小，
    # KAMA会赋予当前价格较小的权重，增大KAMA的滞后性，使其更加平滑，避免产生过多的交易信号。
    # 与VIDYA指标不同的是，KAMA指标可以设置权值的上界FAST和下界SLOW。
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    direction = df['close'] - df['close'].shift(1)
    volatility = df['close'].diff(1).abs().rolling(int(10 * n), min_periods=1).sum()
    fast = 2 / (n / 5 + 1)
    slow = 2 / (3 * n + 1)

    _l = []
    # 计算kama
    for i, (c, d, v) in enumerate(zip(df['close'], direction, volatility)):
        if i < n:
            _l.append(0)
        else:
            er = np.divide(d, (v + eps))
            smooth = er * (fast - slow) + slow
            cof = smooth * smooth
            _l.append(cof * c + (1-cof) * _l[-1])

    df[factor_name] = _l

    # df[factor_name] = ta.KAMA(df['close'], timeperiod=n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** KC ********************
    # N=14
    # TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-REF(LOW,1)))
    # ATR=MA(TR,N)
    # Middle=EMA(CLOSE,20)
    # UPPER=MIDDLE+2*ATR
    # LOWER=MIDDLE-2*ATR
    # KC指标（KeltnerChannel）与布林带类似，都是用价格的移动平均构造中轨，不同的是表示波幅的方法，
    # 这里用ATR来作为波幅构造上下轨。价格突破上轨，可看成新的上升趋势，买入；价格突破下轨，可看成新的下降趋势，卖出。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()
    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].ewm(span=n, adjust=False, min_periods=1).mean()

    signal = middle - 2 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** KC ********************
    # N=14
    # TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-REF(LOW,1)))
    # ATR=MA(TR,N)
    # Middle=EMA(CLOSE,20)
    # UPPER=MIDDLE+2*ATR
    # LOWER=MIDDLE-2*ATR
    # KC指标（KeltnerChannel）与布林带类似，都是用价格的移动平均构造中轨，不同的是表示波幅的方法，
    # 这里用ATR来作为波幅构造上下轨。价格突破上轨，可看成新的上升趋势，买入；价格突破下轨，可看成新的下降趋势，卖出。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()
    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].ewm(span=n, adjust=False, min_periods=1).mean()

    signal = middle - 2 * atr - df['close']
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** KC ********************
    # N=14
    # TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-REF(LOW,1)))
    # ATR=MA(TR,N)
    # Middle=EMA(CLOSE,20)
    # UPPER=MIDDLE+2*ATR
    # LOWER=MIDDLE-2*ATR
    # KC指标（KeltnerChannel）与布林带类似，都是用价格的移动平均构造中轨，不同的是表示波幅的方法，
    # 这里用ATR来作为波幅构造上下轨。价格突破上轨，可看成新的上升趋势，买入；价格突破下轨，可看成新的下降趋势，卖出。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()
    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].ewm(span=n, adjust=False, min_periods=1).mean()

    df[factor_name] = (df['close'] - middle + 2 * atr) / (4 * atr)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** KC ********************
    # N=14
    # TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-REF(LOW,1)))
    # ATR=MA(TR,N)
    # Middle=EMA(CLOSE,20)
    # UPPER=MIDDLE+2*ATR
    # LOWER=MIDDLE-2*ATR
    # KC指标（KeltnerChannel）与布林带类似，都是用价格的移动平均构造中轨，不同的是表示波幅的方法，
    # 这里用ATR来作为波幅构造上下轨。价格突破上轨，可看成新的上升趋势，买入；价格突破下轨，可看成新的下降趋势，卖出。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()
    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].ewm(span=n, adjust=False, min_periods=1).mean()

    signal = middle + 2 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** KC ********************
    # N=14
    # TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(CLOSE,1)-REF(LOW,1)))
    # ATR=MA(TR,N)
    # Middle=EMA(CLOSE,20)
    # UPPER=MIDDLE+2*ATR
    # LOWER=MIDDLE-2*ATR
    # KC指标（KeltnerChannel）与布林带类似，都是用价格的移动平均构造中轨，不同的是表示波幅的方法，
    # 这里用ATR来作为波幅构造上下轨。价格突破上轨，可看成新的上升趋势，买入；价格突破下轨，可看成新的下降趋势，卖出。
    tmp1_s = df['high'] - df['low']
    tmp2_s = (df['high'] - df['close'].shift(1)).abs()
    tmp3_s = (df['low'] - df['close'].shift(1)).abs()
    tr = np.max(np.array([tmp1_s, tmp2_s, tmp3_s]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    middle = df['close'].ewm(span=n, adjust=False, min_periods=1).mean()

    signal = df['close'] - middle - 2 * atr
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # D
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    low_list = df['low'].rolling(n, min_periods=1).min()  # MIN(LOW,N) 求周期内low的最小值
    high_list = df['high'].rolling(
        n, min_periods=1).max()  # MAX(HIGH,N) 求周期内high 的最大值
    # Stochastics=(CLOSE-LOW_N)/(HIGH_N-LOW_N)*100 计算一个随机值
    rsv = (df['close'] - low_list) / (high_list - low_list + eps) * 100
    # K D J的值在固定的范围内
    df['K'] = rsv.ewm(com=2).mean()  # K=SMA(Stochastics,3,1) 计算k
    df[factor_name] = df['K'].ewm(com=2).mean()  # D=SMA(K,3,1)  计算D

    # 删除多余列
    del df['K']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # KDJD 指标
    """
    N=20
    M=60
    LOW_N=MIN(LOW,N)
    HIGH_N=MAX(HIGH,N)
    Stochastics=(CLOSE-LOW_N)/(HIGH_N-LOW_N)*100
    Stochastics_LOW=MIN(Stochastics,M)
    Stochastics_HIGH=MAX(Stochastics,M)
    Stochastics_DOUBLE=(Stochastics-Stochastics_LOW)/(Stochastics_HIGH-Stochastics_LOW)*100
    K=SMA(Stochastics_DOUBLE,3,1)
    D=SMA(K,3,1)
    KDJD 可以看作 KDJ 的变形。KDJ 计算过程中的变量 Stochastics 用
    来衡量收盘价位于最近 N 天最高价和最低价之间的位置。而 KDJD 计
    算过程中的 Stochastics_DOUBLE 可以用来衡量 Stochastics 在最近
    N 天的 Stochastics 最大值与最小值之间的位置。我们这里将其用作
    动量指标。当 D 上穿 70/下穿 30 时，产生买入/卖出信号。
    """
    min_low = df['low'].rolling(n).min()
    max_high = df['high'].rolling(n).max()
    Stochastics = (df['close'] - min_low) / (max_high - min_low) * 100
    Stochastics_LOW = Stochastics.rolling(n*3).min()
    Stochastics_HIGH = Stochastics.rolling(n*3).max()
    Stochastics_DOUBLE = (Stochastics - Stochastics_LOW) / (Stochastics_HIGH - Stochastics_LOW)
    K = Stochastics_DOUBLE.ewm(com=2).mean() #K
    df[factor_name] = K.ewm(com=2).mean() #D


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df



#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # KDJD 指标
    """
    N=20
    M=60
    LOW_N=MIN(LOW,N)
    HIGH_N=MAX(HIGH,N)
    Stochastics=(CLOSE-LOW_N)/(HIGH_N-LOW_N)*100
    Stochastics_LOW=MIN(Stochastics,M)
    Stochastics_HIGH=MAX(Stochastics,M)
    Stochastics_DOUBLE=(Stochastics-Stochastics_LOW)/(Stochastics_HIGH-Stochastics_LOW)*100
    K=SMA(Stochastics_DOUBLE,3,1)
    D=SMA(K,3,1)
    KDJD 可以看作 KDJ 的变形。KDJ 计算过程中的变量 Stochastics 用
    来衡量收盘价位于最近 N 天最高价和最低价之间的位置。而 KDJD 计
    算过程中的 Stochastics_DOUBLE 可以用来衡量 Stochastics 在最近
    N 天的 Stochastics 最大值与最小值之间的位置。我们这里将其用作
    动量指标。当 D 上穿 70/下穿 30 时，产生买入/卖出信号。
    """
    min_low = df['low'].rolling(n).min()
    max_high = df['high'].rolling(n).max()
    Stochastics = (df['close'] - min_low) / (max_high - min_low) * 100
    Stochastics_LOW = Stochastics.rolling(n*3).min()
    Stochastics_HIGH = Stochastics.rolling(n*3).max()
    Stochastics_DOUBLE = (Stochastics - Stochastics_LOW) / (Stochastics_HIGH - Stochastics_LOW)
    df[factor_name] = Stochastics_DOUBLE.ewm(com=2).mean() #K
    # df[factor_name] = K.ewm(com=2).mean() #D


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df



#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # J
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    low_list = df['low'].rolling(
        n, min_periods=1).min()  # MIN(LOW,N) 求周期内low的最小值
    high_list = df['high'].rolling(
        n, min_periods=1).max()  # MAX(HIGH,N) 求周期内high 的最大值
    # Stochastics=(CLOSE-LOW_N)/(HIGH_N-LOW_N)*100 计算一个随机值
    rsv = (df['close'] - low_list) / (high_list - low_list + eps) * 100
    # K D J的值在固定的范围内
    df['K'] = rsv.ewm(com=2).mean()  # K=SMA(Stochastics,3,1) 计算k
    df['D'] = df['K'].ewm(com=2).mean()  # D=SMA(K,3,1)  计算D
    df[factor_name] = 3 * df['K'] - 2 * df['D']  # 计算J

    # 删除多余列
    del df['K'], df['D']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # K
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    low_list = df['low'].rolling(n, min_periods=1).min()  # MIN(LOW,N) 求周期内low的最小值
    high_list = df['high'].rolling(n, min_periods=1).max()  # MAX(HIGH,N) 求周期内high 的最大值
    # Stochastics=(CLOSE-LOW_N)/(HIGH_N-LOW_N)*100 计算一个随机值
    rsv = (df['close'] - low_list) / (high_list - low_list + eps) * 100
    # K D J的值在固定的范围内
    df[factor_name] = rsv.ewm(com=2).mean()  # K=SMA(Stochastics,3,1) 计算k

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Ke指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    volume_avg = df['volume'].rolling(n).mean()
    volume_stander = df['volume'] / volume_avg
    price_change = df['close'].pct_change(n)
    df[factor_name] = (price_change / (abs(price_change) + eps)) * \
        volume_stander * price_change ** 2

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Ko指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['price'] = (df['high'] + df['low'] + df['close']) / 3
    df['V'] = np.where(df['price'] > df['price'].shift(1), df['volume'], -df['volume'])
    df['V_ema1'] = df['V'].ewm(n, adjust=False).mean()
    df['V_ema2'] = df['V'].ewm(int(n * 1.618), adjust=False).mean()
    df['KO'] = df['V_ema1'] - df['V_ema2']
    # 标准化
    df[factor_name] = (df['KO'] - df['KO'].rolling(n).min()) / (
        df['KO'].rolling(n).max() - df['KO'].rolling(n).min() + eps)

    # 删除多余列
    del df['price'], df['V'], df['V_ema1'], df['V_ema2'], df['KO']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Kpower 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['k_power'] = (df['close'] - df['open'])/df['avg_price'] * 0.6 + 0.2 * (
        df[['close', 'open']].min(axis=1) - df['low'])/df['avg_price'] - 0.2 * (
        df['high'] - df[['close', 'open']].max(axis=1))/df['avg_price']
    df[factor_name] = df['k_power'].rolling(window=n).sum()

    # 删除多余列
    del df['k_power']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # KST 指标
    """
    ROC_MA1=MA(CLOSE-REF(CLOSE,10),10)
    ROC_MA2=MA(CLOSE -REF(CLOSE,15),10)
    ROC_MA3=MA(CLOSE -REF(CLOSE,20),10)
    ROC_MA4=MA(CLOSE -REF(CLOSE,30),10)
    KST_IND=ROC_MA1+ROC_MA2*2+ROC_MA3*3+ROC_MA4*4
    KST=MA(KST_IND,9)
    KST 结合了不同时间长度的 ROC 指标。如果 KST 上穿/下穿 0 则产
    生买入/卖出信号。
    """
    df['ROC1'] = df['close'] - df['close'].shift(n)
    df['ROC_MA1'] = df['ROC1'].rolling(n, min_periods=1).mean()
    df['ROC2'] = df['close'] - df['close'].shift(int(n * 1.5))
    df['ROC_MA2'] = df['ROC2'].rolling(n, min_periods=1).mean()
    df['ROC3'] = df['close'] - df['close'].shift(int(n * 2))
    df['ROC_MA3'] = df['ROC3'].rolling(n, min_periods=1).mean()
    df['ROC4'] = df['close'] - df['close'].shift(int(n * 3))
    df['ROC_MA4'] = df['ROC4'].rolling(n, min_periods=1).mean()
    df['KST_IND'] = df['ROC_MA1'] + df['ROC_MA2'] * 2 + df['ROC_MA3'] * 3 + df['ROC_MA4'] * 4
    df['KST'] = df['KST_IND'].rolling(n, min_periods=1).mean()
    # 去量纲
    df[factor_name] = df['KST_IND'] / df['KST']

    del df['ROC1']
    del df['ROC_MA1']
    del df['ROC2']
    del df['ROC_MA2']
    del df['ROC3']
    del df['ROC_MA3']
    del df['ROC4']
    del df['ROC_MA4']
    del df['KST_IND']
    del df['KST']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 当前价格与过去N分钟的最高价最低价之比，看上涨还是下跌的动力更强
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = -1 * df['low'].rolling(n, min_periods=1).min() / df['close'] - df['high'].rolling(n, min_periods=1).max() / df['close']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Lcsd 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['median'] = df['close'].rolling(n).mean()
    df[factor_name] = (df['low'] - df['median']) / (df['low'] + eps)

    # 删除多余列
    del df['median']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # LMA 指标
    """
    N=20
    LMA=MA(LOW,N)
    LMA 为简单移动平均把收盘价替换为最低价。如果最低价上穿/下穿
    LMA 则产生买入/卖出信号。
    """
    df['low_ma'] = df['low'].rolling(n, min_periods=1).mean()
    # 进行去量纲
    df[factor_name] = df['low'] / df['low_ma'] - 1

    del df['low_ma']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # LongMoment
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    def range_plus(x, np_tmp, rolling_window, lam):
        # 计算滚动到的index
        li = x.index.to_list()
        # 从整块array中截取对应的index的array块
        np_tmp2 = np_tmp[li, :]
        # 按照振幅排序
        np_tmp2 = np_tmp2[np.argsort(np_tmp2[:, 0])]
        # 计算需要切分的个数
        t = int(rolling_window * lam)
        # 计算低价涨跌幅因子
        np_tmp2 = np_tmp2[:t, :]
        s = np_tmp2[:, 1].sum()
        return s

    df['涨跌幅'] = df['close'].pct_change(n)
    # 计算窗口20-180的切割动量与反转因子
    df['振幅'] = (df['high'] / df['low']) - 1
    # 先把需要滚动的两列数据变成array
    np_tmp = df[['振幅', '涨跌幅']].values
    # 计算因子
    df[factor_name] = df['涨跌幅'].rolling(
        n * 10).apply(range_plus, args=(np_tmp, n * 10, 0.7), raw=False)

    del df['振幅'], df['涨跌幅']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils.diff import add_diff, eps

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ma 指标

    signal = df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # MAAMT 指标
    """
    N=40
    MAAMT=MA(AMOUNT,N)
    MAAMT 是成交额的移动平均线。当成交额上穿/下穿移动平均线时产
    生买入/卖出信号。
    """
    MAAMT = df['volume'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['volume'] - MAAMT) / MAAMT  # 避免量纲问题

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)


def signal(*args):
    # ********************均线收缩********************
    # MAC

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ma_short = df['close'].rolling(n, min_periods=1).mean()
    ma_long = df['close'].rolling(2 * n, min_periods=1).mean()

    _mac = 10 * (ma_short - ma_long)
    df[factor_name] = scale_01(_mac, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # 计算macd指标
    '''
    N1=20
    N2=40
    N3=5 
    MACD=EMA(CLOSE,N1)-EMA(CLOSE,N2) 
    MACD_SIGNAL=EMA(MACD,N3) 
    MACD_HISTOGRAM=MACD-MACD_SIGNAL
    MACD 指标衡量快速均线与慢速均线的差值。
    由于慢速均线反映的是之前较长时间的价格的走向，而快速均线反映的是较短时间的价格的走向，
    所以在上涨趋势中快速均线会比慢速均线涨的快，而在下跌趋势中快速均线会比慢速均线跌得快。
    所以 MACD 上穿/下穿 0 可以作为一种构造交易信号的方式。
    另外一种构造交易信号的方式是求 MACD 与其移动平均(信号线)的差值得到 MACD 柱，
    利用 MACD 柱上穿/下穿 0(即 MACD 上穿/下穿其信号线)来构造交易信号。
    这种方式在其他指标的使用中也可以借鉴
    '''
    short_windows = n
    long_windows = 3 * n
    macd_windows = int(1.618 * n)

    df['ema_short'] = df['close'].ewm(span=short_windows, adjust=False).mean()
    df['ema_long']  = df['close'].ewm(span=long_windows, adjust=False).mean()
    df['dif']  = df['ema_short'] - df['ema_long']
    df['dea']  = df['dif'].ewm(span=macd_windows, adjust=False).mean()
    df['macd'] = 2 * (df['dif'] - df['dea'])

    df[factor_name] = df['macd'] / df['macd'].rolling(macd_windows, min_periods=1).mean() - 1

    del df['ema_short']
    del df['ema_long']
    del df['dif']
    del df['dea']
    del df['macd']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # MACDVOL 指标
    """
    N1=20
    N2=40
    N3=10
    MACDVOL=EMA(VOLUME,N1)-EMA(VOLUME,N2)
    SIGNAL=MA(MACDVOL,N3)
    MACDVOL 是 MACD 的成交量版本。如果 MACDVOL 上穿 SIGNAL，
    则买入；下穿 SIGNAL 则卖出。
    """
    N1 = 2 * n
    N2 = 4 * n
    N3 = n
    df['ema_volume_1'] = df['volume'].ewm(N1, adjust=False).mean()
    df['ema_volume_2'] = df['volume'].ewm(N2, adjust=False).mean()
    df['MACDV'] = df['ema_volume_1'] - df['ema_volume_2']
    df['SIGNAL'] = df['MACDV'].rolling(N3, min_periods=1).mean()
    # 去量纲
    df[factor_name] = df['MACDV'] / df['SIGNAL'] - 1
    
    del df['ema_volume_1']
    del df['ema_volume_2']
    del df['MACDV']
    del df['SIGNAL']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # 计算Macd指标
    ema1 = (0.5 * df['high'] + 0.5 * df['low']).ewm(span=n, adjust=False).mean()
    ema2 = (0.5 * df['high'] + 0.5 * df['low']).ewm(span=2 * n, adjust=False).mean()

    dif = ema1 - ema2
    dea = dif.ewm(span=int(n / 2.), adjust=False).mean()

    df[factor_name] = 10 * (2 * dif - dea)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
from utils.diff import add_diff

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算均线
    df['ma_short'] = df['close'].rolling(int(n * 0.618), min_periods=1).mean()
    df['ma_long'] = df['close'].rolling(n, min_periods=1).mean()

    df['gap'] = df['ma_short'] - df['ma_long']
    # 连续k个gap变大才做多， 连续k个gap变小才做空



    df['count'] = 0
    df.loc[df['gap'] > df['gap'].shift(), 'count'] = 1
    df.loc[df['gap'] < df['gap'].shift(), 'count'] = -1
    df[factor_name] = df['count'].rolling(n).sum()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

    
def signal(*args):
    # ********************均线收缩********************

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ma_short = (0.5 * df['high'] + 0.5 * df['low']).rolling(n, min_periods=1).mean()
    ma_long = (0.5 * df['high'] + 0.5 * df['low']).rolling(2 * n, min_periods=1).mean()

    _mac = 10 * (ma_short - ma_long)
    df[factor_name] = scale_01(_mac, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

    
def signal(*args):
    # ********************均线收缩********************

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high = df['high'].rolling(n, min_periods=1).max()
    low = df['low'].rolling(n, min_periods=1).min()

    ma_short = (0.5 * high + 0.5 * low).rolling(n, min_periods=1).mean()
    ma_long = (0.5 * high + 0.5 * low).rolling(2 * n, min_periods=1).mean()

    _mac = 10 * (ma_short - ma_long)
    df[factor_name] = scale_01(_mac, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

    
def signal(*args):
    # ********************均线收缩********************

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high = df['high'].rolling(n, min_periods=1).max()
    low = df['low'].rolling(n, min_periods=1).min()
    close = df['close']

    ma_short = ((high + low + close) / 3.).rolling(n, min_periods=1).mean()
    ma_long = ((high + low + close) / 3.).rolling(2 * n, min_periods=1).mean()

    _mac = 10 * (ma_short - ma_long)
    df[factor_name] = scale_01(_mac, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)


def signal(*args):
    # ********************均线收缩********************

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high = df['high'].rolling(n, min_periods=1).max()
    low = df['low'].rolling(n, min_periods=1).min()
    _open = df['open']

    ma_short = ((high + low + _open) / 3.).rolling(n, min_periods=1).mean()
    ma_long = ((high + low + _open) / 3.).rolling(2 * n, min_periods=1).mean()

    _mac = 10 * (ma_short - ma_long)
    df[factor_name] = scale_01(_mac, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    #该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）
    """
    N=20
    M=10
    MA_CLOSE=MA(CLOSE,N)
    MADisplaced=REF(MA_CLOSE,M)
    MADisplaced 指标把简单移动平均线向前移动了 M 个交易日，用法
    与一般的移动平均线一样。如果收盘价上穿/下穿 MADisplaced 则产
    生买入/卖出信号。
    有点变种bias
    """
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ma = df['close'].rolling(
        2 * n, min_periods=1).mean()  # MA(CLOSE,N) 固定俩个参数之间的关系  减少参数
    ref = ma.shift(n)  # MADisplaced=REF(MA_CLOSE,M)

    df[factor_name] = df['close'] / ref - 1  # 去量纲



    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    #该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['oma'] = df['open'].ewm(span=n, adjust=False).mean()
    df['hma'] = df['high'].ewm(span=n, adjust=False).mean()
    df['lma'] = df['low'].ewm(span=n, adjust=False).mean()
    df['cma'] = df['close'].ewm(span=n, adjust=False).mean()
    df['tp'] = (df['oma'] + df['hma'] + df['lma'] + df['cma']) / 4
    df['ma'] = df['tp'].ewm(span=n, adjust=False).mean()
    df['abs_diff_close'] = abs(df['tp'] - df['ma'])
    df['md'] = df['abs_diff_close'].ewm(span=n, adjust=False).mean()

    df[factor_name] = (df['tp'] - df['ma']) / (df['md'] + eps)

    # # 删除中间数据
    del df['oma']
    del df['hma']
    del df['lma']
    del df['cma']
    del df['tp']
    del df['ma']
    del df['abs_diff_close']
    del df['md']







    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


# https://bbs.quantclass.cn/thread/18187

def signal(*args):
    #该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    #df['oma'] = df['open'].ewm(span=n, adjust=False).mean()
    df['hma'] = df['high'].ewm(span=n, adjust=False).mean()
    df['lma'] = df['low'].ewm(span=n, adjust=False).mean()
    df['cma'] = df['close'].ewm(span=n, adjust=False).mean()
    df['tp'] = (df['hma'] + df['lma'] + df['cma']) / 3
    df['ma'] = df['tp'].ewm(span=n, adjust=False).mean()
    df['abs_diff_close'] = abs(df['tp'] - df['ma'])
    df['md'] = df['abs_diff_close'].ewm(span=n, adjust=False).mean()

    df[factor_name] = (df['tp'] - df['ma']) / df['md']

    # # 删除中间数据
    #del df['oma']
    del df['hma']
    del df['lma']
    del df['cma']
    del df['tp']
    del df['ma']
    del df['abs_diff_close']
    del df['md']






    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

import numpy  as np
import pandas as pd
from utils.diff import add_diff


def signal(*args):
    # Mak
    # https://bbs.quantclass.cn/thread/9446

    df = args[0]
    n = 15
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['ma'] / df['ma'].shift(1) - 1) * 1000  # 原涨跌幅值太小，所以乘以1000放大一下。

    del df['ma']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
import numpy  as np
import pandas as pd
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18782
# https://bbs.quantclass.cn/thread/9446

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['Mak'] = (df['ma'] / df['ma'].shift(1) - 1) * 1000
    df[factor_name] = df['Mak'].rolling(n, min_periods=1).mean()

    del df['ma'], df['Mak']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # MarketPl指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    quote_volume_ema = df['quote_volume'].ewm(span=n, adjust=False).mean()
    volume_ema = df['volume'].ewm(span=n, adjust=False).mean()
    df['平均持仓成本'] = quote_volume_ema / volume_ema
    df[factor_name] = df['close'] / (df['平均持仓成本'] + eps) - 1

    del df['平均持仓成本']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # MarketPl_v2指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    quote_volume_ema = df['quote_volume'].ewm(span=n, adjust=False).mean()
    volume_ema = df['volume'].ewm(span=n, adjust=False).mean()
    cost = (df['open'] + df['low'] + df['close']) / 3
    cost_ema = cost.ewm(span=n, adjust=False).mean()
    condition = df['quote_volume'] > 0
    df.loc[condition, 'avg_p'] = df['quote_volume'] / df['volume']
    condition = df['quote_volume'] == 0
    df.loc[condition, 'avg_p'] = df['close'].shift(1)
    condition1 = df['avg_p'] <= df['high']
    condition2 = df['avg_p'] >= df['low']
    df.loc[condition1 & condition2, '平均持仓成本'] = quote_volume_ema / volume_ema
    condition1 = df['avg_p'] > df['high']
    condition2 = df['avg_p'] < df['low']
    df.loc[condition1 & condition2, '平均持仓成本'] = cost_ema
    df[factor_name] = df['close'] / (df['平均持仓成本'] + eps) - 1

    del df['平均持仓成本']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
            1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # MaSignal 指标

    signal = df['close'] - df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # 该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    N=14
    TYPICAL_PRICE=(HIGH+LOW+CLOSE)/3
    MF=TYPICAL_PRICE*VOLUME
    MF_POS=SUM(IF(TYPICAL_PRICE>=REF(TYPICAL_PRICE,1),M
    F,0),N)
    MF_NEG=SUM(IF(TYPICAL_PRICE<=REF(TYPICAL_PRICE,1),
    MF,0),N)
    MFI=100-100/(1+MF_POS/MF_NEG)
    MFI 指标的计算与 RSI 指标类似，不同的是，其在上升和下跌的条件
    判断用的是典型价格而不是收盘价，且其是对 MF 求和而不是收盘价
    的变化值。MFI 同样可以用来判断市场的超买超卖状态。
    如果 MFI 上穿 80，则产生买入信号；
    如果 MFI 下穿 20，则产生卖出信号。
    """
    df['price'] = (df['high'] + df['low'] + df['close']) / 3  # TYPICAL_PRICE=(HIGH+LOW+CLOSE)/3
    df['MF'] = df['price'] * df['volume']  # MF=TYPICAL_PRICE*VOLUME
    df['pos'] = np.where(df['price'] >= df['price'].shift(1), df['MF'],
                         0)  # IF(TYPICAL_PRICE>=REF(TYPICAL_PRICE,1),MF,0)MF,0),N)
    df['MF_POS'] = df['pos'].rolling(n).sum()
    df['neg'] = np.where(df['price'] <= df['price'].shift(1), df['MF'],
                         0)  # IF(TYPICAL_PRICE<=REF(TYPICAL_PRICE,1),MF,0)
    df['MF_NEG'] = df['neg'].rolling(n).sum()  # MF_NEG=SUM(IF(TYPICAL_PRICE<=REF(TYPICAL_PRICE,1),MF,0),N)

    df[factor_name] = 100 - 100 / (1 + df['MF_POS'] / df['MF_NEG'])  # MFI=100-100/(1+MF_POS/MF_NEG)


    # 删除中间数据
    del df['price']
    del df['MF']
    del df['pos']
    del df['MF_POS']
    del df['neg']
    del df['MF_NEG']







    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # MICD 指标
    """
    N=20
    N1=10
    N2=20
    M=10
    MI=CLOSE-REF(CLOSE,1)
    MTMMA=SMA(MI,N,1)
    DIF=MA(REF(MTMMA,1),N1)-MA(REF(MTMMA,1),N2)
    MICD=SMA(DIF,M,1)
    如果 MICD 上穿 0，则产生买入信号；
    如果 MICD 下穿 0，则产生卖出信号。
    """
    df['MI'] = df['close'] - df['close'].shift(1)
    df['MIMMA'] = df['MI'].rolling(n, min_periods=1).mean()
    df['MIMMA_MA1'] = df['MIMMA'].shift(1).rolling(n, min_periods=1).mean()
    df['MIMMA_MA2'] = df['MIMMA'].shift(1).rolling(2 *n, min_periods=1).mean()
    df['DIF'] = df['MIMMA_MA1'] - df['MIMMA_MA2']
    df['MICD'] = df['DIF'].rolling(n, min_periods=1).mean()
    # 去量纲
    df[factor_name] = df['DIF'] / df['MICD']

    del df['MI']
    del df['MIMMA']
    del df['MIMMA_MA1']
    del df['MIMMA_MA2']
    del df['DIF']
    del df['MICD']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df





#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff, eps


def signal(*args):
    # Mm指标
    
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ma_fast = df['close'].rolling(n, min_periods=1).mean()
    ma_slow = df['close'].rolling(5*n, min_periods=1).mean()
    df[factor_name] = ma_fast / (ma_slow + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # Mreg
    # https://bbs.quantclass.cn/thread/9840

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]



    df['reg_close'] = ta.LINEARREG(df['close'], timeperiod=n)  # 该部分为talib内置求线性回归
    df['mreg'] = df['close'] / df['reg_close'] - 1
    df[factor_name] = df['mreg'].rolling(n, min_periods=1).mean()



    # 删除多余列
    del df['reg_close'], df['mreg']



    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/17753

def signal(*args):

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(window=n).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)

    # 收盘价动量
    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df['mtm'] = df['mtm'].rolling(n, min_periods=1).mean()
  
    # 标准差动量
    df['s_mtm'] = df['std'] / df['std'].shift(n) - 1
    df['s_mtm'] = df['s_mtm'].rolling(n, min_periods=1).mean()

    # bbw波动率
    df['bbw'] = df['std'] / df['ma']
    df['bbw_mean'] = df['bbw'].rolling(window=n).mean()
  
    # taker_buy_quote_asset_volume波动率
    df['volatility'] = df['taker_buy_quote_asset_volume'].rolling(window=n, min_periods=1).sum() / \
                   df['taker_buy_quote_asset_volume'].rolling(window=int(0.5 * n), min_periods=1).sum()

    df[factor_name] = df['mtm'] * df['s_mtm'] * df['bbw_mean'] * df['volatility']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

import numpy  as np
import pandas as pd
from utils.diff import add_diff


def signal(*args):
    """
    https://bbs.quantclass.cn/thread/9501
    取一段时间内的平均最大回撤和平均最大反向回撤中的最大值构成市场情绪平稳度指数
    Market Sentiment Stability Index
    指标越小，说明趋势性越强
    """

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['max2here'] = df['high'].rolling(n, min_periods=1).max()
    df['dd1here'] = abs(df['close']/df['max2here'] - 1)
    df['avg_max_drawdown'] = df['dd1here'].rolling(n, min_periods=1).mean()

    df['min2here'] = df['low'].rolling(n, min_periods=1).min()
    df['dd2here'] = abs(df['close'] / df['min2here'] - 1)
    df['avg_reverse_drawdown'] = df['dd2here'].rolling(n, min_periods=1).mean()

    df[factor_name] = df[['avg_max_drawdown', 'avg_reverse_drawdown']].max(axis=1)

    del df['max2here']
    del df['dd1here']
    del df['avg_max_drawdown']
    del df['min2here']
    del df['dd2here']
    del df['avg_reverse_drawdown']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/18410

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算动量
    df['mtm'] = df['close'] / df['close'].shift(n) - 1

    # 主动成交占比
    volume = df['quote_volume'].rolling(n, min_periods=1).sum()
    buy_volume = df['taker_buy_quote_asset_volume'].rolling(
        n, min_periods=1).sum()
    df['taker_by_ratio'] = buy_volume / volume

    # 波动率因子
    df['c1'] = df['high'] - df['low']
    df['c2'] = abs(df['high'] - df['close'].shift(1))
    df['c3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['c1', 'c2', 'c3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=n, min_periods=1).mean()
    df['avg_price_'] = df['close'].rolling(window=n, min_periods=1).mean()
    df['wd_atr'] = df['atr'] / df['avg_price_']

    # 动量 * 主动成交占比 * 波动率
    df['mtm'] = df['mtm'] * df['taker_by_ratio'] * df['wd_atr']
    df[factor_name] = df['mtm'].rolling(window=n, min_periods=1).mean()

    drop_col = [
        'mtm', 'taker_by_ratio', 'c1', 'c2', 'c3', 'tr', 'atr', 'wd_atr', 'avg_price_'
    ]
    df.drop(columns=drop_col, inplace=True)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Mtm 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = (df['close'] / df['close'].shift(n) - 1) * 100

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/17771


def signal(*args):
    # MtmTBear
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 动量
    df['ma'] = df['close'].rolling(window=n).mean()
    df['mtm'] = (df['close'] / df['ma'].shift(n) - 1) * 100
    df['mtm_mean'] = df['mtm'].rolling(window=n).mean()

    # 平均波幅
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR_abs'] = df['tr'].rolling(window=n, min_periods=1).mean()
    df['ATR'] = df['ATR_abs'] / df['ma'] * 100

    # 平均主动卖出
    df['taker_sell_quote_asset_volume'] = df['quote_volume'] - df['taker_buy_quote_asset_volume']
    df['vma'] = df['quote_volume'].rolling(n, min_periods=1).mean()
    df['taker_sell_ma'] = (df['taker_sell_quote_asset_volume'] / df['vma']) * 100
    df['taker_sell_mean'] = df['taker_sell_ma'].rolling(window=n).mean()


    # 组合指标
    df[factor_name] = df['mtm_mean'] * df['ATR'] * df['taker_sell_mean']

    drop_col = [
        'ma', 'mtm', 'mtm_mean', 'tr1', 'tr2', 'tr3', 'tr', 'ATR_abs', 'ATR',
        'taker_sell_quote_asset_volume', 'vma', 'taker_sell_ma', 'taker_sell_mean'
    ]
    df.drop(columns=drop_col, inplace=True)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/17771

def signal(*args):
    # MtmTBull
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 动量
    df['ma'] = df['close'].rolling(window=n).mean()
    df['mtm'] = (df['close'] / df['ma'].shift(n) - 1) * 100
    df['mtm_mean'] = df['mtm'].rolling(window=n).mean()

    # 平均波幅
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR_abs'] = df['tr'].rolling(window=n, min_periods=1).mean()
    df['ATR'] = df['ATR_abs'] / df['ma'] * 100

    # 平均主动买入
    df['vma'] = df['quote_volume'].rolling(n, min_periods=1).mean()
    df['taker_buy_ma'] = (df['taker_buy_quote_asset_volume'] / df['vma']) * 100
    df['taker_buy_mean'] = df['taker_buy_ma'].rolling(window=n).mean()

    # 组合指标
    df[factor_name] = df['mtm_mean'] * df['ATR'] * df['taker_buy_mean']

    drop_col = [
        'ma', 'mtm', 'mtm_mean', 'tr1', 'tr2', 'tr3', 'tr', 'ATR_abs', 'ATR',
        'vma', 'taker_buy_ma', 'taker_buy_mean',
    ]
    df.drop(columns=drop_col, inplace=True)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/17962

def signal(*args):
    '''
    以hint构建作为选B因子
    '''
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ==============================================================



    df['mtm'] = df['high'] / df['high'].shift(n) - 1
    df['mtm_mean'] = df['mtm'].rolling(window=n, min_periods=1).mean()
    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['cm'] = df['close'] / df['ma']
    df[factor_name] = (df['mtm_mean'] - df['cm']) / df['cm']
  
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # MtmMax 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df['up'] = df['mtm'].rolling(window=n).max().shift(1)
    df[factor_name] = df['mtm'] - df['up']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # MtmMean 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = (df['close'] / df['close'].shift(n) - 1).rolling(
        window=n, min_periods=1).mean()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/18651

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['mtm'] = df['close'] / df['close'].shift(n) - 1

    df['_g']  = 1 - abs((df['close'] - df['open'])/(df['high'] - df['low'] + eps))
    df['gap'] = df['_g'].rolling(window=n, min_periods=1).mean()

    df[factor_name] = df['mtm'].rolling(window=n, min_periods=1).mean()/(df['gap'] + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/18435

def signal(*args):
    # MtmMean_v10
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df['波动'] =  df['high'].rolling(n, min_periods=1).max() / df['low'].rolling(
        n, min_periods=1).min() - 1
    df['每小时波动'] = df['high'] / df['low'] - 1
    df['每小时波动均值'] = df['每小时波动'].rolling(n,min_periods=1).mean()
    df[factor_name] = df['mtm'].rolling(window=n, min_periods=1).mean() * (df['波动'] + df['每小时波动均值'])

    del df['mtm'], df['波动'],df['每小时波动'] , df['每小时波动均值']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # MtmMean 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df['mtm'] = df['mtm']*df['quote_volume']/df['quote_volume'].rolling(window=n, min_periods=1).mean()
    df[factor_name] = df['mtm'].rolling(window=n, min_periods=1).mean()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/18957

def signal(*args):
    # MtmMean_v12 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df['mtm'] = df['mtm']*df['taker_buy_quote_asset_volume']/df['taker_buy_quote_asset_volume'].rolling(window=n, min_periods=1).mean()
    df[factor_name] = df['mtm'].rolling(window=n, min_periods=1).mean()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # MtmMean 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = (df['close'] / df['close'].shift(n) - 1).ewm(n, adjust=False).mean()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # MtmMean 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    reg_price = ta.LINEARREG(df['close'], timeperiod=n)

    df[factor_name] = (reg_price / df['close'].shift(n) - 1).rolling(
        window=n, min_periods=1).mean()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # MtmMean 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df[factor_name] = ta.LINEARREG(df['mtm'], timeperiod=n)
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/18387


def signal(*args):
    # Mtm乘波动率，波动率用最高值与最低值比值表示
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df['波动'] = df['high'].rolling(n, min_periods=1).max() / df['low'].rolling(
        n, min_periods=1).min() - 1
    df[factor_name] = df['mtm'].rolling(window=n, min_periods=1).mean() * df['波动']

    del df['mtm'], df['波动']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/18411

def signal(*args):  
    # MtnTb 指标  
    df = args[0]  
    n = args[1]  
    diff_num = args[2]  
    factor_name = args[3]  
  
    # df['tr_trix'] = df['close'].ewm(span=n, adjust=False).mean()  
    # df['tr_pct'] = df['tr_trix'].pct_change()    df['MtmMean'] = (df['close'] / df['close'].shift(n) - 1).ewm(n, adjust=False).mean()  
    # 平均主动买入  
    df['vma'] = df['quote_volume'].rolling(n, min_periods=1).mean()  
    df['taker_buy_ma'] = (df['taker_buy_quote_asset_volume'] / df['vma']) * 100  
    df['taker_buy_mean'] = df['taker_buy_ma'].rolling(window=n).mean()  
  
    df[factor_name] = df['MtmMean'] * df['taker_buy_mean']  
  
    del df['MtmMean'], df['vma'], df['taker_buy_ma'], df['taker_buy_mean']  
  
    if diff_num > 0:  
        return add_diff(df, diff_num, factor_name)  
    else:  
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/18140

def signal(*args):
    # MtmVolMean 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['close_change'] = (df['close'] / df['close'].shift(n) - 1).ewm(n, adjust=False).mean() * 100
    df['vol_change'] = (df['quote_volume'] / df['quote_volume'].shift(n) - 1).ewm(n, adjust=False).mean() * 100

    df[factor_name] = df['close_change'] * df['vol_change']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/18291

def signal(*args):
    # MtmVol_Resonance 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
  
    df['mtm'] = df['close'] / df['close'].shift(n) - 1
    df['mtm_mean'] = df['mtm'].rolling(window=n, min_periods=1).mean()
  
    df['quote_volume_mean'] = df['quote_volume'].rolling(n,min_periods=1).mean()
    df['quote_volume_change'] = (df['quote_volume'] / df['quote_volume_mean'])
    df['quote_volume_change_mean'] = df['quote_volume_change'].rolling(n,min_periods=1).mean()

    df[factor_name] = df['mtm_mean']*df['quote_volume_change_mean']

    drop_col = [
        'mtm', 'mtm_mean','quote_volume_mean', 'quote_volume_change','quote_volume_change_mean'
    ]
    df.drop(columns=drop_col, inplace=True)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 成交额使用收益率的sign进行加权求和，模拟资金净流出净流入的效果
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['_ret_sign'] = np.sign(df['close'].pct_change())
    df[factor_name] = (df['_ret_sign']*df['quote_volume']).rolling(n, min_periods=1).sum()

    del df['_ret_sign']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils.diff import add_diff, eps

def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算滚动窗口内的最大值和最小值
    df['min'] = df['close'].rolling(n).min()
    df['max'] = df['close'].rolling(n).max()

    # 计算归一化的位置，使用 diff_num 防止分母为零的情况
    df[factor_name] = (df['close'] - df['min']) / (df['max'] - df['min'] + eps)

    # 删除多余列
    del df['min'], df['max']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
            1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # ******************** Nvi ********************
    # --- NVI --- 099/125
    # N=144
    # NVI_INC=IF(VOLUME<REF(VOLUME,1),1+(CLOSE-REF(CLOSE,1))/CLOSE,1)
    # NVI_INC[0]=100
    # NVI=CUM_PROD(NVI_INC)
    # NVI_MA=MA(NVI,N)
    # NVI是成交量降低的交易日的价格变化百分比的累积。NVI相关理论认为，如果当前价涨量缩，
    # 则说明大户主导市场，NVI可以用来识别价涨量缩的市场（大户主导的市场）。
    # 如果NVI上穿NVI_MA，则产生买入信号；
    # 如果NVI下穿NVI_MA，则产生卖出信号。

    nvi_inc = np.where(df['volume'] < df['volume'].shift(1),
                       1 + (df['close'] - df['close'].shift(1)) / (1e-9 + df['close']), 1)
    nvi_inc[0] = 100
    nvi = pd.Series(nvi_inc).cumprod()
    nvi_ma = nvi.rolling(n, min_periods=1).mean()

    signal = nvi - nvi_ma
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # 该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['_va'] = (df['close'] - df['low'] - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['_obv'] = df['_va'].rolling(n).sum()

    # ref = ma.shift(n)  # MADisplaced=REF(MA_CLOSE,M)

    df[factor_name] = df['_obv'] / df['_obv'].rolling(n).mean()  # 去量纲


    del df['_va']
    del df['_obv']




    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['_va'] = (df['close'] - df['low'] - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['_obv'] = df['_va'].rolling(n).sum()

    # ref = ma.shift(n)  # MADisplaced=REF(MA_CLOSE,M)

    df[factor_name] = df['_obv'] / df['_obv'].rolling(n).mean()  # 去量纲


    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # OSC 指标
    """
    N=40
    M=20
    OSC=CLOSE-MA(CLOSE,N)
    OSCMA=MA(OSC,M)
    OSC 反映收盘价与收盘价移动平均相差的程度。如果 OSC 上穿/下 穿 OSCMA 则产生买入/卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ma'] = df['close'].rolling(2 * n, min_periods=1).mean()
    df['OSC'] = df['close'] - df['ma']
    df[factor_name] = df['OSC'].rolling(n, min_periods=1).mean()

    del df['ma']
    del df['OSC']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Pac 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    N1=20
    N2=20
    UPPER=SMA(HIGH,N1,1)
    LOWER=SMA(LOW,N2,1)
    用最高价和最低价的移动平均来构造价格变化的通道，如果价格突破
    上轨则做多，突破下轨则做空。
    """
    df['upper'] = df['high'].ewm(span=n).mean()  # UPPER=SMA(HIGH,N1,1)
    df['lower'] = df['low'].ewm(span=n).mean()  # LOWER=SMA(LOW,N2,1)
    df['width'] = df['upper'] - df['lower']  # 添加指标求宽度进行去量纲
    df['width_ma'] = df['width'].rolling(n, min_periods=1).mean()

    df[factor_name] = df['width'] / (df['width_ma'] + eps) - 1

    # 删除多余列
    del df['upper'], df['lower'], df['width'], df['width_ma']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** Pac ********************
    # N1=20
    # N2=20
    # UPPER=SMA(HIGH,N1,1)
    # LOWER=SMA(LOW,N2,1)
    # 用最高价和最低价的移动平均来构造价格变化的通道，如果价格突破上轨则做多，突破下轨则做空。
    lower = df['low'].ewm(alpha=1 / n, adjust=False).mean()
    df[factor_name] = scale_01(lower, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** Pac ********************
    # N1=20
    # N2=20
    # UPPER=SMA(HIGH,N1,1)
    # LOWER=SMA(LOW,N2,1)
    # 用最高价和最低价的移动平均来构造价格变化的通道，如果价格突破上轨则做多，突破下轨则做空。
    lower = df['low'].ewm(alpha=1 / n, adjust=False).mean()
    lower = lower - df['close']
    df[factor_name] = scale_01(lower, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** Pac ********************
    # N1=20
    # N2=20
    # UPPER=SMA(HIGH,N1,1)
    # LOWER=SMA(LOW,N2,1)
    # 用最高价和最低价的移动平均来构造价格变化的通道，如果价格突破上轨则做多，突破下轨则做空。
    upper = df['high'].ewm(alpha=1 / n, adjust=False).mean()
    df[factor_name] = scale_01(upper, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** Pac ********************
    # N1=20
    # N2=20
    # UPPER=SMA(HIGH,N1,1)
    # LOWER=SMA(LOW,N2,1)
    # 用最高价和最低价的移动平均来构造价格变化的通道，如果价格突破上轨则做多，突破下轨则做空。
    upper = df['high'].ewm(alpha=1 / n, adjust=False).mean()
    upper = df['close'] - upper

    df[factor_name] = scale_01(upper, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Pfe指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算首尾价格的直线距离
    totle_y = df['close'] - df['close'].shift(n - 1)
    direct_distance = (totle_y ** 2 + (n - 1) ** 2) ** 0.5
    # 计算相邻价格间的距离
    each_y = df['close'].diff()
    each_distance = (each_y ** 2 + 1) ** 0.5
    actual_distance = each_distance.rolling(n - 1).sum()
    # 计算PFE
    PFE = 100 * (direct_distance / actual_distance)
    pct_change = ((df['close'] - df['close'].shift(n - 1)) / df['close'].shift(n - 1))
    df[factor_name] = PFE * pct_change

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # PjcDistance指标
    
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算均线
    df['median'] = df['close'].rolling(n, min_periods=1).mean()
    # 计算每根k线收盘价和均线的差值，取绝对数
    df['cha'] = abs(df['close'] - df['median'])
    # 计算平均差
    df['ping_jun_cha'] = df['cha'].rolling(n, min_periods=1).mean()

    # 将收盘价小于平均差的偏移量设置为0
    condition_0 = df['close'] <= df['ping_jun_cha']
    condition_1 = df['close'] > df['ping_jun_cha']
    df.loc[condition_0, 'distance'] = 0
    df.loc[condition_1, 'distance'] = df['close'] - df['ping_jun_cha']

    # 计算收盘价相对平均差的偏移比例
    df[factor_name] = (df['distance'] / df['ping_jun_cha']) - 1

    del df['median'], df['cha'], df['ping_jun_cha'], df['distance']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


# 因子 指标名 版本： Pmarp_Yidai_v1
# https://bbs.quantclass.cn/thread/9501
def signal(*args):
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]


    # 计算sma
    df['sma'] = df['close'].rolling(n, min_periods=1).mean()

    # 当前价格与sma比较（百分比）：价格对移动均线的相对涨跌
    df['pmar'] = abs(df['close']/df['sma'])

    # 计算当前k线某一特征值超过了统计范围内多少根k线？返回百分比
    # 统计 当前k线的pmar 超过了统计周期内 多少根k线的pmar 返回值
    df['pmarpSum'] = 0

    k = n
    while k > 0:
        df['pmardiff'] = df['pmar'] - df['pmar'].shift(k)
        df['add'] = np.where(df['pmardiff'] > 0, 1, 0)
        df['pmarpSum'] = df['pmarpSum'] + df['add']
        k -= 1

    df[factor_name] = df['pmarpSum'] / n * 100


    # 删除多余列
    del df['sma'], df['pmar'], df['pmardiff'], df['add'], df['pmarpSum']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # PMO 指标
    """
    N1=10
    N2=40
    N3=20
    ROC=(CLOSE-REF(CLOSE,1))/REF(CLOSE,1)*100
    ROC_MA=DMA(ROC,2/N1)
    ROC_MA10=ROC_MA*10
    PMO=DMA(ROC_MA10,2/N2)
    PMO_SIGNAL=DMA(PMO,2/(N3+1))
    PMO 指标是 ROC 指标的双重平滑（移动平均）版本。与 SROC 不 同(SROC 是先对价格作平滑再求 ROC)，而 PMO 是先求 ROC 再对
    ROC 作平滑处理。PMO 越大（大于 0），则说明市场上涨趋势越强；
    PMO 越小（小于 0），则说明市场下跌趋势越强。如果 PMO 上穿/
    下穿其信号线，则产生买入/卖出指标。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ROC'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['ROC_MA'] = df['ROC'].rolling(n, min_periods=1).mean()
    df['ROC_MA10'] = df['ROC_MA'] * 10
    df['PMO'] = df['ROC_MA10'].rolling(4 * n, min_periods=1).mean()
    df[factor_name] = df['PMO'].rolling(2 * n, min_periods=1).mean()

    del df['ROC']
    del df['ROC_MA']
    del df['ROC_MA10']
    del df['PMO']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df





    
#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # PmoTema 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # TEMA均线
    df['ema'] = df['close'].ewm(n, adjust=False).mean()
    df['ema_ema'] = df['ema'].ewm(n, adjust=False).mean()
    df['ema_ema_ema'] = df['ema_ema'].ewm(n, adjust=False).mean()
    df['TEMA'] = 3 * df['ema'] - 3 * df['ema_ema'] + df['ema_ema_ema']

    # 计算PMO
    df['ROC'] = (df['TEMA'] - df['TEMA'].shift(1)) / \
        df['TEMA'].shift(1) * 100  # 用TEMA均线代替原CLOSE
    df['ROC_MA'] = df['ROC'].rolling(
        n, min_periods=1).mean()  # 均线代替动态移动平均
    df['ROC_MA10'] = df['ROC_MA'] * 10
    df['PMO'] = df['ROC_MA10'].rolling(4 * n, min_periods=1).mean()
    df[factor_name] = df['PMO'].rolling(2 * n, min_periods=1).mean()

    del df['ema'], df['ema_ema'], df['ema_ema_ema'], df['TEMA']
    del df['ROC'], df['ROC_MA']
    del df['ROC_MA10']
    del df['PMO']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df





    

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # PO指标
    """
    EMA_SHORT=EMA(CLOSE,9)
    EMA_LONG=EMA(CLOSE,26)
    PO=(EMA_SHORT-EMA_LONG)/EMA_LONG*100
    PO 指标求的是短期均线与长期均线之间的变化率。
    如果 PO 上穿 0，则产生买入信号；
    如果 PO 下穿 0，则产生卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ema_short = df['close'].ewm(n, adjust=False).mean()
    ema_long = df['close'].ewm(n * 3, adjust=False).mean()
    df[factor_name] = (ema_short - ema_long) / ema_long * 100

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # POS指标
    """
    N=100
    PRICE=(CLOSE-REF(CLOSE,N))/REF(CLOSE,N)
    POS=(PRICE-MIN(PRICE,N))/(MAX(PRICE,N)-MIN(PRICE,N))
    POS 指标衡量当前的 N 天收益率在过去 N 天的 N 天收益率最大值和
    最小值之间的位置。当 POS 上穿 80 时产生买入信号；当 POS 下穿
    20 时产生卖出信号。

    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ref = df['close'].shift(n)
    price = (df['close'] - ref) / ref
    min_price = price.rolling(n).min()
    max_price = price.rolling(n).max()
    df[factor_name] = (price - min_price) / (max_price - min_price)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # PPo 指标
    
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    N3 = n
    N1 = int(n * 1.382)
    N2 = 3 * n
    df['ema_1'] = df['close'].ewm(
        N1, adjust=False).mean()  # EMA(CLOSE,N1)
    df['ema_2'] = df['close'].ewm(
        N2, adjust=False).mean()  # EMA(CLOSE,N2)
    # PPO=(EMA(CLOSE,N1)-EMA(CLOSE,N2))/EMA(CLOSE,N2)
    df['PPO'] = (df['ema_1'] - df['ema_2']) / df['ema_2']
    df[factor_name] = df['PPO'].ewm(N3, adjust=False).mean()  # PPO_SIGNAL=EMA(PPO,N3)

    del df['ema_1']
    del df['ema_2']
    del df['PPO']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df










        

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

# https://bbs.quantclass.cn/thread/18152

def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    N1 = n
    N2 = 2 * n
    df['ema_1'] = df['close'].ewm(N1, adjust=False).mean()  # EMA(CLOSE,N1)
    df['ema_2'] = df['close'].ewm(N2, adjust=False).mean()  # EMA(CLOSE,N2)
    df['PPO'] = (df['ema_1'] / df['ema_1'].shift(N1) - 1) * abs(df['ema_2'] / df['ema_2'].shift(N2) - 1)

    df[factor_name] = df['PPO'].ewm(N1, adjust=False).mean()  

    del df['ema_1']
    del df['ema_2']
    del df['PPO']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df










        

import pandas as pd
import numpy as np
from utils.diff import add_diff


def signal(*args):
    # https://bbs.quantclass.cn/thread/14038
    # 一种衡量量价的指标 描述价格的突破难度
    # 一定的价格变动幅度是由于量的变动引起的，如果一定的价格变动需要更多的量，可以说明该标的更难受到控制
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df["close_shift"] = df["close"].shift(n)
    df["volume_shift"] = df["volume"].shift(n)
    df["close_ratio"] = abs((df["close"] - df["close_shift"].rolling(n).mean()) / df["close_shift"])
    df["volume_ratio"] = (df["volume"] - df["volume_shift"].rolling(n).mean()) / df["volume_shift"]

    df["angle"] = df["close_ratio"] * df["volume_ratio"]

    condition = df["angle"] < 0  # 量价方向不同,突破毫不费力，设定为inf
    df["direction"] = 1
    df["adj"] = 1
    df.loc[condition, 'direction'] = -1  # 用来把指标改为正数
    df.loc[condition, 'adj'] = np.inf

    df[factor_name] = df["close_ratio"] / df["volume_ratio"] * df["direction"] * df["adj"]
    df[factor_name] = df[factor_name] / n  # 时间窗口越长，拉平指标

    del df['close_shift']
    del df['volume_shift']
    del df['close_ratio']
    del df['volume_ratio']
    del df['angle']
    del df['direction']
    del df['adj']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Psy
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df['P'] = np.where(df['close'] > df['close'].shift(1), 1, 0)  # IF(CLOSE>REF(CLOSE,1),1,0)
    df[factor_name] = df['P'].rolling(n, min_periods=1).sum() / n * 100  # PSY=IF(CLOSE>REF(CLOSE,1),1,0)/N*100

    # 删除多余列
    del df['P']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff



def signal(*args):

    # PVI 指标
    """
    N=40
    PVI_INC=IF(VOLUME>REF(VOLUME,1),(CLOSE-REF(CLOSE))/
    CLOSE,0)
    PVI=CUM_SUM(PVI_INC)
    PVI_MA=MA(PVI,N)
    PVI 是成交量升高的交易日的价格变化百分比的累积。
    PVI 相关理论认为，如果当前价涨量增，则说明散户主导市场，PVI
    可以用来识别价涨量增的市场（散户主导的市场）。
    如果 PVI 上穿 PVI_MA，则产生买入信号；
    如果 PVI 下穿 PVI_MA，则产生卖出信号。
    """

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['ref_close'] = (df['close'] - df['close'].shift(1)) / df['close']
    df['PVI_INC'] = np.where(df['volume'] > df['volume'].shift(1), df['ref_close'], 0)
    df['PVI'] = df['PVI_INC'].cumsum()
    df[factor_name] = df['PVI'].rolling(n, min_periods=1).mean()

    del df['ref_close']
    del df['PVI_INC']
    del df['PVI']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # Pvo
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['emap_1'] = df['volume'].ewm(n, min_periods=1).mean()
    df['emap_2'] = df['volume'].ewm(n * 2, min_periods=1).mean()
    df[factor_name] = (df['emap_1'] - df['emap_2']) / df['emap_2']
    
    del df['emap_1']
    del df['emap_2']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Pvt 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    PVT=(CLOSE-REF(CLOSE,1))/REF(CLOSE,1)*VOLUME 
    PVT_MA1=MA(PVT,N1)
    PVT_MA2=MA(PVT,N2)
    PVT 指标用价格的变化率作为权重求成交量的移动平均。
    PVT 指标 与 OBV 指标的思想类似，但与 OBV 指标相比，
    PVT 考虑了价格不同涨跌幅的影响，而 OBV 只考虑了价格的变化方向。
    我们这里用 PVT 短期和长期均线的交叉来产生交易信号。
    如果 PVT_MA1 上穿 PVT_MA2，则产生买入信号; 
    如果 PVT_MA1 下穿 PVT_MA2，则产生卖出信号。
    '''

    df['PVT'] = (df['close'] - df['close'].shift(1)) / \
        df['close'].shift(1) * df['volume']
    df['PVT_MA'] = df['PVT'].rolling(n, min_periods=1).mean()

    # 去量纲
    df['PVT_SIGNAL'] = (df['PVT'] / (df['PVT_MA'] + eps) - 1)
    df[factor_name] = df['PVT_SIGNAL'].rolling(n, min_periods=1).sum()

    # 删除多余列
    del df['PVT'], df['PVT_MA'], df['PVT_SIGNAL']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Pvt_v2 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    PVT=(CLOSE-REF(CLOSE,1))/REF(CLOSE,1)*VOLUME 
    PVT_MA1=MA(PVT,N1)
    PVT_MA2=MA(PVT,N2)
    PVT 指标用价格的变化率作为权重求成交量的移动平均。
    PVT 指标 与 OBV 指标的思想类似，但与 OBV 指标相比，
    PVT 考虑了价格不同涨跌幅的影响，而 OBV 只考虑了价格的变化方向。
    我们这里用 PVT 短期和长期均线的交叉来产生交易信号。
    如果 PVT_MA1 上穿 PVT_MA2，则产生买入信号; 
    如果 PVT_MA1 下穿 PVT_MA2，则产生卖出信号。
    '''

    df['PVT'] = df['close'].pct_change() * df['volume']
    # 去量纲

    df['PVT_score'] = ((df['PVT'] - df['PVT'].rolling(n, min_periods=1).mean())
                       / (df['PVT'].rolling(n, min_periods=1).std() + eps))
    df[factor_name] = df['PVT_score'].rolling(n, min_periods=1).sum()

    # 删除无关列
    del df['PVT'], df['PVT_score']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Pvt_v3 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    PVT=(CLOSE-REF(CLOSE,1))/REF(CLOSE,1)*VOLUME 
    PVT_MA1=MA(PVT,N1)
    PVT_MA2=MA(PVT,N2)
    PVT 指标用价格的变化率作为权重求成交量的移动平均。
    PVT 指标 与 OBV 指标的思想类似，但与 OBV 指标相比，
    PVT 考虑了价格不同涨跌幅的影响，而 OBV 只考虑了价格的变化方向。
    我们这里用 PVT 短期和长期均线的交叉来产生交易信号。
    如果 PVT_MA1 上穿 PVT_MA2，则产生买入信号; 
    如果 PVT_MA1 下穿 PVT_MA2，则产生卖出信号。
    '''

    df['PVT'] = df['close'].pct_change() * df['volume']
    # 去量纲
    df['PVT_score'] = ((df['PVT'] - df['PVT'].rolling(n, min_periods=1).mean())
                       / (df['PVT'].rolling(n, min_periods=1).std() + eps))
    df['PVT_sum'] = df['PVT_score'].rolling(n, min_periods=1).sum()
    pvt_ma1 = df['PVT_sum'].rolling(n, min_periods=1).mean()
    pvt_ma2 = df['PVT_sum'].rolling(2 * n, min_periods=1).mean()
    df[factor_name] = pd.Series(pvt_ma1 - pvt_ma2)

    # 删除无关列
    del df['PVT'], df['PVT_score'], df['PVT_sum']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # PVT 指标
    """
    PVT=(CLOSE-REF(CLOSE,1))/REF(CLOSE,1)*VOLUME
    PVT_MA1=MA(PVT,N1)
    PVT_MA2=MA(PVT,N2)
    PVT 指标用价格的变化率作为权重求成交量的移动平均。PVT 指标
    与 OBV 指标的思想类似，但与 OBV 指标相比，PVT 考虑了价格不
    同涨跌幅的影响，而 OBV 只考虑了价格的变化方向。我们这里用 PVT
    短期和长期均线的交叉来产生交易信号。
    如果 PVT_MA1 上穿 PVT_MA2，则产生买入信号；
    如果 PVT_MA1 下穿 PVT_MA2，则产生卖出信号。
    """
    df['PVT'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']
    df['PVT_MA1'] = df['PVT'].rolling(n, min_periods=1).mean()
    df['PVT_MA2'] = df['PVT'].rolling(2 * n, min_periods=1).mean()
    df['Pvt_v2'] = df['PVT_MA1'] - df['PVT_MA2']

    # 去量纲
    df[factor_name] = df['PVT'] / df['Pvt_v2'] - 1
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # Qstick 指标
    """
    N=20
    Qstick=MA(CLOSE-OPEN,N)
    Qstick 通过比较收盘价与开盘价来反映股价趋势的方向和强度。如果
    Qstick 上穿/下穿 0 则产生买入/卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    cl = df['close'] - df['open']
    Qstick = cl.rolling(n, min_periods=1).mean()
    # 进行无量纲处理
    df[factor_name] = cl / Qstick - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff


def signal(*args):
    # QuanlityPriceCorr
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df[factor_name] = df['close'].rolling(
        n).corr(df['quote_volume'])
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff


def signal(*args):
    # QuoteVolumeRatio
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df[factor_name] = df['quote_volume'] / \
        df['quote_volume'].rolling(n, min_periods=1).mean()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Rbias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    ma = df['close'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['close'] / ma) / (df['close'].shift(1)/ma.shift(1)) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.tools import sma
from utils.diff import add_diff


def signal(*args):
    # RCCD 指标
    """
    M=40
    N1=20
    N2=40
    RC=CLOSE/REF(CLOSE,M)
    ARC1=SMA(REF(RC,1),M,1)
    DIF=MA(REF(ARC1,1),N1)-MA(REF(ARC1,1),N2)
    RCCD=SMA(DIF,M,1)
    RC 指标为当前价格与昨日价格的比值。当 RC 指标>1 时，说明价格在上升；当 RC 指标增大时，说明价格上升速度在增快。当 RC 指标
    <1 时，说明价格在下降；当 RC 指标减小时，说明价格下降速度在增
    快。RCCD 指标先对 RC 指标进行平滑处理，再取不同时间长度的移
    动平均的差值，再取移动平均。如 RCCD 上穿/下穿 0 则产生买入/
    卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['RC'] = df['close'] / df['close'].shift(2 * n)
    df['ARC1'] = df['RC'].rolling(2 * n, min_periods=1).mean()
    df['MA1'] = df['ARC1'].shift(1).rolling(n, min_periods=1).mean()
    df['MA2'] = df['ARC1'].shift(1).rolling(2 * n, min_periods=1).mean()
    df['DIF'] = df['MA1'] - df['MA2']
    df[factor_name] = df['DIF'].rolling(2 * n, min_periods=1).mean()

    del df['RC']
    del df['ARC1']
    del df['MA1']
    del df['MA2']
    del df['DIF']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.tools import sma
from utils.diff import add_diff


def signal(*args):
    # RCCD 指标
    """
    M=40
    N1=20
    N2=40
    RC=CLOSE/REF(CLOSE,M)
    ARC1=SMA(REF(RC,1),M,1)
    DIF=MA(REF(ARC1,1),N1)-MA(REF(ARC1,1),N2)
    RCCD=SMA(DIF,M,1)
    RC 指标为当前价格与昨日价格的比值。当 RC 指标>1 时，说明价格在上升；当 RC 指标增大时，说明价格上升速度在增快。当 RC 指标
    <1 时，说明价格在下降；当 RC 指标减小时，说明价格下降速度在增
    快。RCCD 指标先对 RC 指标进行平滑处理，再取不同时间长度的移
    动平均的差值，再取移动平均。如 RCCD 上穿/下穿 0 则产生买入/
    卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['RC'] = df['close'] / df['close'].shift(2 * n)
    df['ARC1'] = sma(df['RC'], n, 1)
    df['MA1'] = df['ARC1'].shift(1).rolling(n, min_periods=1).mean()
    df['MA2'] = df['ARC1'].shift(1).rolling(2 * n, min_periods=1).mean()
    df['DIF'] = df['MA1'] - df['MA2']
    df[factor_name] = sma(df['DIF'], n, 1)

    del df['RC']
    del df['ARC1']
    del df['MA1']
    del df['MA2']
    del df['DIF']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # Reg
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df['reg_close'] = ta.LINEARREG(df['close'], timeperiod=n)  # 该部分为talib内置求线性回归
    df[factor_name] = df['close'] / df['reg_close'] - 1

    # 删除多余列
    del df['reg_close']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # RegEma
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    ema = df['close'].ewm(span=n, adjust=False, min_periods=1).mean()
    reg_close = ta.LINEARREG(ema, timeperiod=n)
    df[factor_name] = df['close'] / (reg_close + eps) - 1
        
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # RegTema
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    ema = df['close'].ewm(span=n, adjust=False).mean()
    emax2 = ema.ewm(span=n, adjust=False).mean()
    emax3 = emax2.ewm(span=n, adjust=False).mean()
    tema = 3 * ema - 3 * emax2 + emax3

    # 计算reg
    reg_tema = ta.LINEARREG(tema, timeperiod=n)  # 该部分为talib内置求线性回归
    df[factor_name] = tema / (reg_tema + eps) - 1
        
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Reg_v2    
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df['LINEARREG'] = ta.LINEARREG(df['close'], timeperiod=2 * n)
    df[factor_name] = 100 * (df['close'] - df['LINEARREG']) / (df['LINEARREG'] + eps)

    # 删除多余列
    del df['LINEARREG']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff, eps
from sklearn.linear_model import LinearRegression


def signal(*args):
    # Reg_v3    
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    # sklearn 线性回归
    def reg_ols(_y):
        _x = np.arange(n) + 1
        model = LinearRegression().fit(_x.reshape(-1, 1), _y)  # 线性回归训练
        y_pred = model.coef_ * _x + model.intercept_  # y = ax + b
        return y_pred[-1]

    df['reg_close'] = df['close'].rolling(n).apply(
        lambda y: reg_ols(y), raw=False)
    df[factor_name] = df['close'] / (df['reg_close'] + eps) - 1
    
    # 删除多余列
    del df['reg_close']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 布林线应用到收益率上
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['_ret'] = df['close'].pct_change()
    df[factor_name] =(df['_ret'] - df['_ret'].rolling(n, min_periods=1).mean()) / df['_ret'].rolling(n, min_periods=1).std()

    del df['_ret']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # RMI 指标
    """
    N=7
    RMI=SMA(MAX(CLOSE-REF(CLOSE,4),0),N,1)/SMA(ABS(CLOSE-REF(CLOSE,1)),N,1)*100
    RMI 与 RSI 的计算方式类似，将 RSI 中的动量与前一天价格之差
    CLOSE-REF(CLOSE,1)项改为了与前四天价格之差 CLOSEREF(CLOSE,4)
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['max_close'] = np.where(df['close'] > df['close'].shift(4), df['close'] - df['close'].shift(4), 0)
    df['abs_close'] = df['close'] - df['close'].shift(1)
    df['sma_1'] = df['max_close'].rolling(n, min_periods=1).mean()
    df['sma_2'] = df['abs_close'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['sma_1'] / df['sma_2'] * 100

    del df['max_close']
    del df['abs_close']
    del df['sma_1']
    del df['sma_2']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Roc 指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = df['close'] / df['close'].shift(n) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # RocVol 指标
    """
    N = 80
    RocVol=(VOLUME-REF(VOLUME,N))/REF(VOLUME,N)
    RocVol 是 ROC 的成交量版本。如果 RocVol 上穿 0 则买入，下
    穿 0 则卖出。
    """
    df[factor_name] = df['volume'] / df['volume'].shift(n) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # Rsi
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    CLOSEUP=IF(CLOSE>REF(CLOSE,1),CLOSE-REF(CLOSE,1),0) 
    CLOSEDOWN=IF(CLOSE<REF(CLOSE,1),ABS(CLOSE-REF(CLOSE,1)),0)
    CLOSEUP_MA=SMA(CLOSEUP,N,1)
    CLOSEDOWN_MA=SMA(CLOSEDOWN,N,1) 
    RSI=100*CLOSEUP_MA/(CLOSEUP_MA+CLOSEDOWN_MA)
    RSI 反映一段时间内平均收益与平均亏损的对比。
    通常认为当 RSI 大于 70，市场处于强势上涨甚至达到超买的状态;
    当 RSI 小于 30，市 场处于强势下跌甚至达到超卖的状态。
    当 RSI 跌到 30 以下又上穿 30 时，通常认为股价要从超卖的状态反弹;
    当 RSI 超过 70 又下穿 70 时，通常认为市场要从超买的状态回落了。
    实际应用中，不一定要使 用 70/30 的阈值选取。这里我们用 60/40 作为信号产生的阈值。 
    RSI 上穿 40 则产生买入信号; RSI 下穿 60 则产生卖出信号。
    '''

    diff = df['close'].diff()  # CLOSE-REF(CLOSE,1) 计算当前close 与前一周期的close的差值
    # IF(CLOSE>REF(CLOSE,1),CLOSE-REF(CLOSE,1),0) 表示当前是上涨状态，记录上涨幅度
    df['up'] = np.where(diff > 0, diff, 0)
    # IF(CLOSE<REF(CLOSE,1),ABS(CLOSE-REF(CLOSE,1)),0) 表示当前为下降状态，记录下降幅度
    df['down'] = np.where(diff < 0, abs(diff), 0)
    A = df['up'].ewm(span=n).mean()  # SMA(CLOSEUP,N,1) 计算周期内的上涨幅度的sma
    B = df['down'].ewm(span=n).mean()  # SMA(CLOSEDOWN,N,1)计算周期内的下降幅度的sma
    # RSI=100*CLOSEUP_MA/(CLOSEUP_MA+CLOSEDOWN_MA)  没有乘以100   没有量纲即可
    df[factor_name] = A / (A + B + eps)

    # 删除多余列
    del df['up'], df['down']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff, eps


def signal(*args):
    # RsiBbw指标
    
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    close_dif = df['close'].diff()
    df['up'] = np.where(close_dif > 0, close_dif, 0)
    df['down'] = np.where(close_dif < 0, abs(close_dif), 0)
    a = df['up'].rolling(n).sum()
    b = df['down'].rolling(n).sum()
    df['rsi'] = (a / (a+b+eps)) * 100
    df['median'] = df['close'].rolling(n, min_periods=1).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['bbw'] = (df['std'] / df['median']).diff(n)
    df[factor_name] = (df['bbw']) * (df['close'] / df['close'].shift(n) - 1) * df['rsi']

    del df['up'], df['down'], df['rsi'], df['median'], df['std'], df['bbw']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # Rsih
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # CLOSE_DIFF_POS=IF(CLOSE>REF(CLOSE,1),CLOSE-REF(CLOSE,1),0)
    df['close_diff_pos'] = np.where(df['close'] > df['close'].shift(
        1), df['close'] - df['close'].shift(1), 0)
    # sma_diff_pos = df['close_diff_pos'].rolling(n, min_periods=1).mean()
    sma_diff_pos = df['close_diff_pos'].ewm(
        span=n).mean()  # SMA(CLOSE_DIFF_POS,N1,1)
    # abs_sma_diff_pos = abs(df['close'] - df['close'].shift(1)).rolling(n, min_periods=1).mean()
    # SMA(ABS(CLOSE-REF(CLOSE,1)),N1,1
    abs_sma_diff_pos = abs(
        df['close'] - df['close'].shift(1)).ewm(span=n).mean()
    # RSI=SMA(CLOSE_DIFF_POS,N1,1)/SMA(ABS(CLOSE-REF(CLOSE,1)),N1,1)*100
    df['RSI'] = sma_diff_pos / abs_sma_diff_pos * 100
    # RSI_SIGNAL=EMA(RSI,N2)
    df['RSI_ema'] = df['RSI'].ewm(4 * n, adjust=False).mean()
    # RSIH=RSI-RSI_SIGNAL
    df[factor_name] = df['RSI'] - df['RSI_ema']

    # 删除中间过程数据
    del df['close_diff_pos']
    del df['RSI']
    del df['RSI_ema']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/17926

def signal(*args):
  
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    close_dif = df['close'].diff()
    df['up'] = np.where(close_dif > 0, close_dif, 0)
    df['down'] = np.where(close_dif < 0, abs(close_dif), 0)
    a = df['up'].rolling(n).sum()
    b = df['down'].rolling(n).sum()
    df['rsi'] = a / (a + b + eps)
    df[factor_name] = df['rsi'].rolling(n, min_periods=1).mean()

    # 删除多余列
    del df['up'], df['down'], df['rsi']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # Rsis
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    close_diff_pos = np.where(df['close'] > df['close'].shift(
        1), df['close'] - df['close'].shift(1), 0)
    rsi_a = pd.Series(close_diff_pos).ewm(
        alpha=1 / (4 * n), adjust=False).mean()
    rsi_b = (df['close'] - df['close'].shift(1)
             ).abs().ewm(alpha=1 / (4 * n), adjust=False).mean()
    rsi = 100 * rsi_a / (eps + rsi_b)
    rsi_min = pd.Series(rsi).rolling(int(4 * n), min_periods=1).min()
    rsi_max = pd.Series(rsi).rolling(int(4 * n), min_periods=1).max()
    rsis = 100 * (rsi - rsi_min) / (eps + rsi_max - rsi_min)
    df[factor_name] = pd.Series(rsis)    

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # RSIS 指标
    """
    N=120
    M=20
    CLOSE_DIFF_POS=IF(CLOSE>REF(CLOSE,1),CLOSE-REF(CL
    OSE,1),0)
    RSI=SMA(CLOSE_DIFF_POS,N,1)/SMA(ABS(CLOSE-REF(CLOS
    E,1)),N,1)*100
    RSIS=(RSI-MIN(RSI,N))/(MAX(RSI,N)-MIN(RSI,N))*100
    RSISMA=EMA(RSIS,M)
    RSIS 反映当前 RSI 在最近 N 天的 RSI 最大值和最小值之间的位置，
    与 KDJ 指标的构造思想类似。由于 RSIS 波动性比较大，我们先取移
    动平均再用其产生信号。其用法与 RSI 指标的用法类似。
    RSISMA 上穿 40 则产生买入信号；
    RSISMA 下穿 60 则产生卖出信号。
    """
    close_diff_pos = np.where(df['close'] > df['close'].shift(1), df['close'] - df['close'].shift(1), 0)
    rsi_a = pd.Series(close_diff_pos).ewm(alpha=1/(4*n), adjust=False).mean()
    rsi_b = (df['close'] - df['close'].shift(1)).abs().ewm(alpha=1/(4*n), adjust=False).mean()
    rsi = 100 * rsi_a / (1e-9 + rsi_b)
    rsi_min = pd.Series(rsi).rolling(int(4*n), min_periods=1).min()
    rsi_max = pd.Series(rsi).rolling(int(4 * n), min_periods=1).max()
    df[factor_name] = 100 * (rsi - rsi_min) / (1e-9 + rsi_max - rsi_min)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df









        

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    # RSIV 指标
    """
    N=20
    VOLUP=IF(CLOSE>REF(CLOSE,1),VOLUME,0)
    VOLDOWN=IF(CLOSE<REF(CLOSE,1),VOLUME,0)
    SUMUP=SUM(VOLUP,N)
    SUMDOWN=SUM(VOLDOWN,N)
    RSIV=100*SUMUP/(SUMUP+SUMDOWN)
    RSIV 的计算方式与 RSI 相同，只是把其中的价格变化 CLOSEREF(CLOSE,1)替换成了成交量 VOLUME。用法与 RSI 类似。我们
    这里将其用作动量指标，上穿 60 买入，下穿 40 卖出。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['VOLUP'] = np.where(df['close'] > df['close'].shift(1), df['volume'], 0)
    df['VOLDOWN'] = np.where(df['close'] < df['close'].shift(1), df['volume'], 0)
    df['SUMUP'] = df['VOLUP'].rolling(n).sum()
    df['SUMDOWN'] = df['VOLDOWN'].rolling(n).sum()
    df[factor_name] = df['SUMUP'] / (df['SUMUP'] + df['SUMDOWN']) * 100

    del df['VOLUP']
    del df['VOLDOWN']
    del df['SUMUP']
    del df['SUMDOWN']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # Rsi_v2
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    close_dif = df['close'].diff()
    df['up'] = np.where(close_dif > 0, close_dif, 0)
    df['down'] = np.where(close_dif < 0, abs(close_dif), 0)
    a = df['up'].rolling(n).sum()
    b = df['down'].rolling(n).sum()
    df[factor_name] = a / (a + b + eps)

    # 删除多余列
    del df['up'], df['down']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # Rsi_v3
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    diff = df['close'].diff()
    df['up'] = np.where(diff > 0, diff, 0)
    df['down'] = np.where(diff < 0, abs(diff), 0)
    a = df['up'].ewm(span=n).mean()
    b = df['down'].ewm(span=n).mean()
    df[factor_name] = a / (a + b + eps)

    # 删除多余列
    del df['up'], df['down']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff, eps


def signal(*args):
    # Rsj 指标
    
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算收益率
    df['return'] = df['close'] / df['close'].shift(1) - 1

    # 计算RV
    df['pow_return'] = pow(df['return'], 2)
    df['rv'] = df['pow_return'].rolling(window=n, min_periods=1).sum()

    # 计算RV +/-
    df['positive_data'] = np.where(df['return'] > 0, df['return'], 0)
    df['negative_data'] = np.where(df['return'] < 0, df['return'], 0)
    df['pow_positive_data'] = pow(df['positive_data'], 2)
    df['pow_negetive_data'] = pow(df['negative_data'], 2)
    df['rv+'] = df['pow_positive_data'].rolling(window=n, min_periods=1).sum()
    df['rv-'] = df['pow_negetive_data'].rolling(window=n, min_periods=1).sum()

    # 计算RSJ
    df[factor_name] = (df['rv+'] - df['rv-']) / (df['rv'] + eps)

    # 删除多余列
    del df['return'], df['rv'], df['positive_data'], df['negative_data']
    del df['rv+'], df['rv-'], df['pow_positive_data'], df['pow_negetive_data']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # RVI 指标
    """
    N1=10
    N2=20
    STD=STD(CLOSE,N)
    USTD=SUM(IF(CLOSE>REF(CLOSE,1),STD,0),N2)
    DSTD=SUM(IF(CLOSE<REF(CLOSE,1),STD,0),N2)
    RVI=100*USTD/(USTD+DSTD)
    RVI 的计算方式与 RSI 一样，不同的是将 RSI 计算中的收盘价变化值
    替换为收盘价在过去一段时间的标准差，用来反映一段时间内上升
    的波动率和下降的波动率的对比。我们也可以像计算 RSI 指标时一样
    先对公式中的 USTD 和 DSTD 作移动平均得到 USTD_MA 和
    DSTD_MA 再求出 RVI=100*USTD_MV/(USTD_MV+DSTD_MV)。
    RVI 的用法与 RSI 一样。通常认为当 RVI 大于 70，市场处于强势上
    涨甚至达到超买的状态；当 RVI 小于 30，市场处于强势下跌甚至达
    到超卖的状态。当 RVI 跌到 30 以下又上穿 30 时，通常认为股价要
    从超卖的状态反弹；当 RVI 超过 70 又下穿 70 时，通常认为市场要
    从超买的状态回落了。
    如果 RVI 上穿 30，则产生买入信号；
    如果 RVI 下穿 70，则产生卖出信号。
    """
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['ustd'] = np.where(df['close'] > df['close'].shift(1), df['std'], 0)
    df['sum_ustd'] = df['ustd'].rolling(2 * n).sum()

    df['dstd'] = np.where(df['close'] < df['close'].shift(1), df['std'], 0)
    df['sum_dstd'] = df['dstd'].rolling(2 * n).sum()

    df[factor_name] = df['sum_ustd'] / (df['sum_ustd'] + df['sum_dstd']) * 100
    
    del df['std']
    del df['ustd']
    del df['sum_ustd']
    del df['dstd']
    del df['sum_dstd']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df






#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # Rwi
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    tr = np.max(np.array([
                (df['high'] - df['low']).abs(),
                (df['high'] - df['close'].shift(1)).abs(),
                (df['close'].shift(1) - df['low']).abs()
                ]), axis=0)  # 三个数列取其大值

    atr = pd.Series(tr).rolling(n, min_periods=1).mean()
    _rwih = (df['high'] - df['low'].shift(1)) / (np.sqrt(n) * atr)
    _rwil = (df['high'].shift(1) - df['low']) / (np.sqrt(n) * atr)

    _rwi = (df['close'] - _rwil) / (1e-9 + _rwih - _rwil)
    df[factor_name] = pd.Series(_rwi)
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Rwi 指标
    """
    N=14
    TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(
    CLOSE,1)-LOW))
    ATR=MA(TR,N)
    RwiH=(HIGH-REF(LOW,1))/(ATR*√N)
    RwiL=(REF(HIGH,1)-LOW)/(ATR*√N)
    Rwi（随机漫步指标）对一段时间股票的随机漫步区间与真实运动区
    间进行比较以判断股票价格的走势。
    如果 RwiH>1，说明股价长期是上涨趋势，则产生买入信号；
    如果 RwiL>1，说明股价长期是下跌趋势，则产生卖出信号。
    """
    df['c1'] = abs(df['high'] - df['low'])
    df['c2'] = abs(df['close'] - df['close'].shift(1))
    df['c3'] = abs(df['high'] - df['close'].shift(1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(n, min_periods=1).mean()
    df[factor_name] = (df['high'] - df['low'].shift(1)) / (df['ATR'] * np.sqrt(n))
    # df['RwiL'] = (df['high'].shift(1) - df['low']) / (df['ATR'] * np.sqrt(n))
    # df['Rwi'] = (df['close'] - df['RwiL']) / (1e-9 + df['RwiH'] - df['RwiL'])
    # df[f'RwiH_bh_{n}'] = df['RwiH'].shift(1)
    # df[f'RwiL_bh_{n}'] = df['RwiL'].shift(1)
    # df[f'Rwi_bh_{n}'] = df['Rwi'].shift(1)

    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['ATR']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Rwi 指标
    """
    N=14
    TR=MAX(ABS(HIGH-LOW),ABS(HIGH-REF(CLOSE,1)),ABS(REF(
    CLOSE,1)-LOW))
    ATR=MA(TR,N)
    RwiH=(HIGH-REF(LOW,1))/(ATR*√N)
    RwiL=(REF(HIGH,1)-LOW)/(ATR*√N)
    Rwi（随机漫步指标）对一段时间股票的随机漫步区间与真实运动区
    间进行比较以判断股票价格的走势。
    如果 RwiH>1，说明股价长期是上涨趋势，则产生买入信号；
    如果 RwiL>1，说明股价长期是下跌趋势，则产生卖出信号。
    """
    df['c1'] = abs(df['high'] - df['low'])
    df['c2'] = abs(df['close'] - df['close'].shift(1))
    df['c3'] = abs(df['high'] - df['close'].shift(1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)
    df['ATR'] = df['TR'].rolling(n, min_periods=1).mean()
    # df[factor_name] = (df['high'] - df['low'].shift(1)) / (df['ATR'] * np.sqrt(n))
    df[factor_name] = (df['high'].shift(1) - df['low']) / (df['ATR'] * np.sqrt(n))
    # df['Rwi'] = (df['close'] - df['RwiL']) / (1e-9 + df['RwiH'] - df['RwiL'])
    # df[f'RwiH_bh_{n}'] = df['RwiH'].shift(1)
    # df[f'RwiL_bh_{n}'] = df['RwiL'].shift(1)
    # df[f'Rwi_bh_{n}'] = df['Rwi'].shift(1)

    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['ATR']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # 收高差值 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high = df['high'].rolling(n, min_periods=1).mean()
    close = df['close']
    df[factor_name] = (close - high) / (high + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # ShortMoment
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    def range_plus(x, np_tmp, rolling_window, lam):
        # 计算滚动到的index
        li = x.index.to_list()
        # 从整块array中截取对应的index的array块
        np_tmp2 = np_tmp[li, :]
        # 按照振幅排序
        np_tmp2 = np_tmp2[np.argsort(np_tmp2[:, 0])]
        # 计算需要切分的个数
        t = int(rolling_window * lam)
        # 计算低价涨跌幅因子
        np_tmp2 = np_tmp2[:t, :]
        s = np_tmp2[:, 1].sum()
        return s

    df['涨跌幅'] = df['close'].pct_change(n)
    # 计算窗口20-180的切割动量与反转因子
    df['振幅'] = (df['high'] / df['low']) - 1
    # 先把需要滚动的两列数据变成array
    np_tmp = df[['振幅', '涨跌幅']].values
    # 计算因子
    df[factor_name] = df['涨跌幅'].rolling(n).apply(
        range_plus, args=(np_tmp, n, 0.7), raw=False)
    
    del df['振幅'], df['涨跌幅']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Si 指标
    """
    A=ABS(HIGH-REF(CLOSE,1))
    B=ABS(LOW-REF(CLOSE,1))
    C=ABS(HIGH-REF(LOW,1))
    D=ABS(REF(CLOSE,1)-REF(OPEN,1))
    N=20
    K=MAX(A,B)
    M=MAX(HIGH-LOW,N)
    R1=A+0.5*B+0.25*D
    R2=B+0.5*A+0.25*D
    R3=C+0.25*D
    R4=IF((A>=B) & (A>=C),R1,R2)
    R=IF((C>=A) & (C>=B),R3,R4)
    Si=50*(CLOSE-REF(CLOSE,1)+(REF(CLOSE,1)-REF(OPEN,1))+
    0.5*(CLOSE-OPEN))/R*K/M
    Si 用价格变化（即两天收盘价之差，昨日收盘与开盘价之差，今日收
    盘与开盘价之差）的加权平均来反映价格的变化。如果 Si 上穿/下穿
    0 则产生买入/卖出信号。
    """
    df['A'] = abs(df['high'] - df['close'].shift(1))
    df['B'] = abs(df['low'] - df['close'].shift(1))
    df['C'] = abs(df['high'] - df['low'].shift(1))
    df['D'] = abs(df['close'].shift(1) - df['open'].shift(1))
    df['K'] = df[['A', 'B']].max(axis=1)
    df['M'] = (df['high'] - df['low']).rolling(n).max()
    df['R1'] = df['A'] + 0.5 * df['B'] + 0.25 * df['D']
    df['R2'] = df['B'] + 0.5 * df['A'] + 0.25 * df['D']
    df['R3'] = df['C'] + 0.25 * df['D']
    df['R4'] = np.where((df['A'] >= df['B']) & (df['A'] >= df['C']), df['R1'], df['R2'])
    df['R'] = np.where((df['C'] >= df['A']) & (df['C'] >= df['B']), df['R3'], df['R4'])
    df[factor_name] = 50 * (df['close'] - df['close'].shift(1) + (df['close'].shift(1) - df['open'].shift(1)) +
                     0.5 * (df['close'] - df['open'])) / df['R'] * df['K'] / df['M']

    
    del df['A']
    del df['B']
    del df['C']
    del df['D']
    del df['K']
    del df['M']
    del df['R1']
    del df['R2']
    del df['R3']
    del df['R4']
    del df['R']
    # del df['Si']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df




#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # SKDJ 指标
    """
    N=60
    M=5
    RSV=(CLOSE-MIN(LOW,N))/(MAX(HIGH,N)-MIN(LOW,N))*100
    MARSV=SMA(RSV,3,1)
    K=SMA(MARSV,3,1)
    D=MA(K,3)
    SKDJ 为慢速随机波动（即慢速 KDJ）。SKDJ 中的 K 即 KDJ 中的 D，
    SKJ 中的 D 即 KDJ 中的 D 取移动平均。其用法与 KDJ 相同。
    当 D<40(处于超卖状态)且 K 上穿 D 时买入，当 D>60（处于超买状
    态）K 下穿 D 时卖出。
    """
    df['RSV'] = (df['close'] - df['low'].rolling(n, min_periods=1).min()) / (df['high'].rolling(n, min_periods=1).max() - df['low'].rolling(n, min_periods=1).min()) * 100
    df['MARSV'] = df['RSV'].ewm(com=2).mean()

    df['K'] = df['MARSV'].ewm(com=2).mean()
    df[factor_name] = df['K'].rolling(3, min_periods=1).mean()
    
    del df['RSV']
    del df['MARSV']
    del df['K']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # SMI 指标
    """
    N1=20
    N2=20
    N3=20
    M=(MAX(HIGH,N1)+MIN(LOW,N1))/2
    D=CLOSE-M
    DS=EMA(EMA(D,N2),N2)
    DHL=EMA(EMA(MAX(HIGH,N1)-MIN(LOW,N1),N2),N2)
    SMI=100*DS/DHL
    SMIMA=MA(SMI,N3)
    SMI 指标可以看作 KDJ 指标的变形。不同的是，KD 指标衡量的是当
    天收盘价位于最近 N 天的最高价和最低价之间的位置，而 SMI 指标
    是衡量当天收盘价与最近 N 天的最高价与最低价均值之间的距离。我
    们用 SMI 指标上穿/下穿其均线产生买入/卖出信号。
    """
    df['max_high'] = df['high'].rolling(n, min_periods=1).mean()
    df['min_low'] = df['low'].rolling(n, min_periods=1).mean()
    df['M'] = (df['max_high'] + df['min_low']) / 2
    df['D'] = df['close'] - df['M']
    df['ema'] = df['D'].ewm(n, adjust=False).mean()
    df['DS'] = df['ema'].ewm(n, adjust=False).mean()
    df['HL'] = df['max_high'] - df['min_low']
    df['ema_hl'] = df['HL'].ewm(n, adjust=False).mean()
    df['DHL'] = df['ema_hl'].ewm(n, adjust=False).mean()
    df['SMI'] = 100 * df['DS'] / df['DHL']
    df[factor_name] = df['SMI'].rolling(n, min_periods=1).mean()

    del df['max_high']
    del df['min_low']
    del df['M']
    del df['D']
    del df['ema']
    del df['DS']
    del df['HL']
    del df['ema_hl']
    del df['DHL']
    del df['SMI']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df





#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** smi ********************
    # --- SMI --- 073/125
    # N1=20
    # N2=20
    # N3=20
    # M=(Smi_v2X(HIGH,N1)+MIN(LOW,N1))/2
    # D=CLOSE-M
    # DS=ESmi_v2(ESmi_v2(D,N2),N2)
    # DHL=ESmi_v2(ESmi_v2(Smi_v2X(HIGH,N1)-MIN(LOW,N1),N2),N2)
    # SMI=100*DS/DHL
    # SMISmi_v2=Smi_v2(SMI,N3)
    # SMI指标可以看作KDJ指标的变形。不同的是，KD指标衡量的是当天收盘价位于最近N天的最高价和最低价之间的位置，
    # 而SMI指标是衡量当天收盘价与最近N天的最高价与最低价均值之间的距离。
    # 我们用SMI指标上穿/下穿其均线产生买入/卖出信号。

    m = 0.5 * df['high'].rolling(n, min_periods=1).max() + 0.5 * df['low'].rolling(n, min_periods=1).min()
    d = df['close'] - m
    ds = d.ewm(span=n, adjust=False, min_periods=1).mean()
    ds = ds.ewm(span=n, adjust=False, min_periods=1).mean()

    dhl = df['high'].rolling(n, min_periods=1).max() - df['low'].rolling(n, min_periods=1).min()
    dhl = dhl.ewm(span=n, adjust=False, min_periods=1).mean()
    dhl = dhl.ewm(span=n, adjust=False, min_periods=1).mean()

    smi = 100 * ds / dhl

    signal = smi.rolling(n, min_periods=1).mean()
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # Sroc
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    """
    N=13
    M=21
    EMAP=EMA(CLOSE,N)
    SROC=(EMAP-REF(EMAP,M))/REF(EMAP,M)
    SROC 与 ROC 类似，但是会对收盘价进行平滑处理后再求变化率。
    """
    ema = df['close'].ewm(n, adjust=False).mean()
    ref = ema.shift(2 * n)
    df[factor_name] = (ema - ref) / (ref + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # SrocVol 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # EMAP=EMA(VOLUME,N)
    df['emap'] = df['volume'].ewm(2 * n, adjust=False).mean()
    # SROCVOL=(EMAP-REF(EMAP,M))/REF(EMAP,M)
    df[factor_name] = (df['emap'] - df['emap'].shift(n)) / df['emap'].shift(n)

    del df['emap']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Sroc_v2
    # https://bbs.quantclass.cn/thread/9807

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    ema = ta.KAMA(df['close'], n)
    ref = ema.shift(2 * n)
    df[factor_name] = (ema - ref) / (ref + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # STC 指标
    """
    N1=23
    N2=50
    N=40
    MACDX=EMA(CLOSE,N1)-EMA(CLOSE,N2)
    V1=MIN(MACDX,N)
    V2=MAX(MACDX,N)-V1
    FK=IF(V2>0,(MACDX-V1)/V2*100,REF(FK,1))
    FD=SMA(FK,N,1)
    V3=MIN(FD,N)
    V4=MAX(FD,N)-V3
    SK=IF(V4>0,(FD-V3)/V4*100,REF(SK,1))
    STC=SD=SMA(SK,N,1) 
    STC 指标结合了 MACD 指标和 KDJ 指标的算法。首先用短期均线与
    长期均线之差算出 MACD，再求 MACD 的随机快速随机指标 FK 和
    FD，最后求 MACD 的慢速随机指标 SK 和 SD。其中慢速随机指标就
    是 STC 指标。STC 指标可以用来反映市场的超买超卖状态。一般认
    为 STC 指标超过 75 为超买，STC 指标低于 25 为超卖。
    如果 STC 上穿 25，则产生买入信号；
    如果 STC 下穿 75，则产生卖出信号。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    N1 = n
    N2 = int(N1 * 1.5) # 大约值
    N = 2 * n
    df['ema1'] = df['close'].ewm(N1, adjust=False).mean()
    df['ema2'] = df['close'].ewm(N, adjust=False).mean()
    df['MACDX'] = df['ema1'] - df['ema2']
    df['V1'] = df['MACDX'].rolling(N2, min_periods=1).min()
    df['V2'] = df['MACDX'].rolling(N2, min_periods=1).max()- df['V1']
    df['FK'] = (df['MACDX'] - df['V1']) / df['V2'] * 100
    df['FK'] = np.where(df['V2'] > 0, (df['MACDX'] - df['V1']) / df['V2'] * 100, df['FK'].shift(1))
    df['FD'] = df['FK'].rolling(N2, min_periods=1).mean()
    df['V3'] = df['FD'].rolling(N2, min_periods=1).min()
    df['V4'] = df['FD'].rolling(N2, min_periods=1).max() - df['V3']
    df['SK'] = (df['FD'] - df['V3']) / df['V4'] * 100
    df['SK'] = np.where(df['V4'] > 0, (df['FD'] - df['V3']) / df['V4'] * 100, df['SK'].shift(1))
    df[factor_name] = df['SK'].rolling(N1, min_periods=1).mean()

    del df['ema1']
    del df['ema2']
    del df['MACDX']
    del df['V1']
    del df['V2']
    del df['V3']
    del df['V4']
    del df['FK']
    del df['FD']
    del df['SK']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df










        
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # T3
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    va = 0.5
    ema = df['close'].ewm(n, adjust=False).mean()  # EMA(CLOSE,N)
    ema_ema = ema.ewm(n, adjust=False).mean()  # EMA(EMA(CLOSE,N),N)
    T1 = ema * (1 + va) - ema_ema * va
    T1_ema = T1.ewm(n, adjust=False).mean()  # EMA(T1,N)
    T1_ema_ema = T1_ema.ewm(n, adjust=False).mean()  # EMA(EMA(T1,N),N)
    T2 = T1_ema * (1 + va) - T1_ema_ema * va
    T2_ema = T2.ewm(n, adjust=False).mean()  # EMA(T2,N)
    T2_ema_ema = T2_ema.ewm(n, adjust=False).mean()  # EMA(EMA(T2,N),N)
    T3 = T2_ema * (1 + va) - T2_ema_ema * va
    df[factor_name] = df['close'] / (T3 + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff


def signal(*args):
    # TakerByRatio
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    volume = df['quote_volume'].rolling(n, min_periods=1).sum()
    buy_volume = df['taker_buy_quote_asset_volume'].rolling(
        n, min_periods=1).sum()
    df[factor_name] = buy_volume / volume

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** TDI ********************
    # RSI_PriceLine=EMA(RSI,N2)
    # RSI_SignalLine=EMA(RSI,N3)
    # RSI_MarketLine=EMA(RSI,N4)
    # DI是根据RSI指标构造得到的技术指标，包括RSI价格线，交易信号线，市场基线等。
    # RSI价格线同时上穿/下穿交易信号线、市场基线时产生买入/卖出信号。

    rtn = df['close'].diff()
    up = np.where(rtn > 0, rtn, 0)
    dn = np.where(rtn < 0, rtn.abs(), 0)
    a = pd.Series(up).rolling(n, min_periods=1).sum()
    b = pd.Series(dn).rolling(n, min_periods=1).sum()
    a *= 1e3
    b *= 1e3
    rsi = a / (1e-9 + a + b)
    rsi_price_line = pd.Series(rsi).ewm(span=n, adjust=False, min_periods=1).mean()
    rsi_signal_line = pd.Series(rsi).ewm(span=int(2 * n), adjust=False, min_periods=1).mean()

    signal = rsi_price_line - rsi_signal_line
    df[factor_name] = scale_01(signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Tema指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    N=20,40
    TEMA=3*EMA(CLOSE,N)-3*EMA(EMA(CLOSE,N),N)+EMA(EMA(EMA(CLOSE,N),N),N)
    TEMA 结合了单重、双重和三重的 EMA，相比于一般均线延迟性较
    低。我们用快、慢 TEMA 的交叉来产生交易信号。
    """
    df['ema'] = df['close'].ewm(n, adjust=False).mean()  # EMA(CLOSE,N)
    df['ema_ema'] = df['ema'].ewm(
        n, adjust=False).mean()  # EMA(EMA(CLOSE,N),N)
    df['ema_ema_ema'] = df['ema_ema'].ewm(
        n, adjust=False).mean()  # EMA(EMA(EMA(CLOSE,N),N),N)
    # TEMA=3*EMA(CLOSE,N)-3*EMA(EMA(CLOSE,N),N)+EMA(EMA(EMA(CLOSE,N),N),N)
    df['TEMA'] = 3 * df['ema'] - 3 * df['ema_ema'] + df['ema_ema_ema']
    # 去量纲
    df[factor_name] = df['ema'] / (df['TEMA'] + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Tema_v2指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['TEMA'] = ta.TEMA(df['close'], timeperiod=2 * n)
    df[factor_name] = 100 * (df['close'] - df['TEMA']) / (df['TEMA'] + eps)

    del df['TEMA']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)
    

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** TII ********************
    # N1=40
    # M=[N1/2]+1
    # N2=9
    # CLOSE_MA=MA(CLOSE,N1)
    # DEV=CLOSE-CLOSE_MA
    # DEVPOS=IF(DEV>0,DEV,0)
    # DEVNEG=IF(DEV<0,-DEV,0)
    # SUMPOS=SUM(DEVPOS,M)
    # SUMNEG=SUM(DEVNEG,M)
    # TII=100*SUMPOS/(SUMPOS+SUMNEG)
    # TII_SIGNAL=EMA(TII,N2)
    # TII的计算方式与RSI相同，只是把其中的前后两天价格变化替换为价格与均线的差值。
    # TII可以用来反映价格的趋势以及趋势的强度。一般认为TII>80(<20)时上涨（下跌）趋势强烈。
    # TII产生交易信号有几种不同的方法：上穿20买入，下穿80卖出（作为反转指标）；上穿50买入，下穿50卖出；
    # 上穿信号线买入，下穿信号线卖出。 如果TII上穿TII_SIGNAL，则产生买入信号； 如果TII下穿TII_SIGNAL，则产生卖出信号。
    close_ma = df['close'].rolling(n, min_periods=1).mean()
    dev = df['close'] - close_ma
    devpos = np.where(dev > 0, dev, 0)
    devneg = np.where(dev < 0, -dev, 0)
    sumpos = pd.Series(devpos).rolling(int(1 + n / 2), min_periods=1).sum()
    sumneg = pd.Series(devneg).rolling(int(1 + n / 2), min_periods=1).sum()

    tii = 100 * sumpos / (sumpos + sumneg)
    df[factor_name] = scale_01(tii, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  zscore归一化
def scale_zscore(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).mean()
          ) / pd.Series(_s).rolling(_n, min_periods=1).std()
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** TII ********************
    # N1=40
    # M=[N1/2]+1
    # N2=9
    # CLOSE_MA=MA(CLOSE,N1)
    # DEV=CLOSE-CLOSE_MA
    # DEVPOS=IF(DEV>0,DEV,0)
    # DEVNEG=IF(DEV<0,-DEV,0)
    # SUMPOS=SUM(DEVPOS,M)
    # SUMNEG=SUM(DEVNEG,M)
    # TII=100*SUMPOS/(SUMPOS+SUMNEG)
    # TII_SIGNAL=EMA(TII,N2)
    # TII的计算方式与RSI相同，只是把其中的前后两天价格变化替换为价格与均线的差值。
    # TII可以用来反映价格的趋势以及趋势的强度。一般认为TII>80(<20)时上涨（下跌）趋势强烈。
    # TII产生交易信号有几种不同的方法：上穿20买入，下穿80卖出（作为反转指标）；上穿50买入，下穿50卖出；
    # 上穿信号线买入，下穿信号线卖出。 如果TII上穿TII_SIGNAL，则产生买入信号； 如果TII下穿TII_SIGNAL，则产生卖出信号。
    close_ma = df['close'].rolling(n, min_periods=1).mean()
    dev = df['close'] - close_ma
    devpos = np.where(dev > 0, dev, 0)
    devneg = np.where(dev < 0, -dev, 0)
    sumpos = pd.Series(devpos).rolling(int(1 + n / 2), min_periods=1).sum()
    sumneg = pd.Series(devneg).rolling(int(1 + n / 2), min_periods=1).sum()

    tii = 100 * sumpos / (sumpos + sumneg)
    tii_signal = pd.Series(tii).ewm(span=int(n / 2), adjust=False, min_periods=1).mean()
    df[factor_name] = scale_zscore(tii_signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
import pandas as pd
from utils.diff import add_diff

# =====函数  01归一化
def scale_01(_s, _n):
    _s = (pd.Series(_s) - pd.Series(_s).rolling(_n, min_periods=1).min()) / (
        1e-9 + pd.Series(_s).rolling(_n, min_periods=1).max() - pd.Series(_s).rolling(_n, min_periods=1).min()
    )
    return pd.Series(_s)

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ******************** TII ********************
    # N1=40
    # M=[N1/2]+1
    # N2=9
    # CLOSE_MA=MA(CLOSE,N1)
    # DEV=CLOSE-CLOSE_MA
    # DEVPOS=IF(DEV>0,DEV,0)
    # DEVNEG=IF(DEV<0,-DEV,0)
    # SUMPOS=SUM(DEVPOS,M)
    # SUMNEG=SUM(DEVNEG,M)
    # TII=100*SUMPOS/(SUMPOS+SUMNEG)
    # TII_SIGNAL=EMA(TII,N2)
    # TII的计算方式与RSI相同，只是把其中的前后两天价格变化替换为价格与均线的差值。
    # TII可以用来反映价格的趋势以及趋势的强度。一般认为TII>80(<20)时上涨（下跌）趋势强烈。
    # TII产生交易信号有几种不同的方法：上穿20买入，下穿80卖出（作为反转指标）；上穿50买入，下穿50卖出；
    # 上穿信号线买入，下穿信号线卖出。 如果TII上穿TII_SIGNAL，则产生买入信号； 如果TII下穿TII_SIGNAL，则产生卖出信号。
    close_ma = df['close'].rolling(n, min_periods=1).mean()
    dev = df['close'] - close_ma
    devpos = np.where(dev > 0, dev, 0)
    devneg = np.where(dev < 0, -dev, 0)
    sumpos = pd.Series(devpos).rolling(int(1 + n / 2), min_periods=1).sum()
    sumneg = pd.Series(devneg).rolling(int(1 + n / 2), min_periods=1).sum()

    tii = 100 * sumpos / (sumpos + sumneg)
    tii_signal = pd.Series(tii).ewm(span=int(n / 2), adjust=False, min_periods=1).mean()
    tii_signal = tii - tii_signal
    df[factor_name] = scale_01(tii_signal, n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # Tma
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df['ma'] = df['close'].rolling(n, min_periods=1).mean()
    df['ma2'] = df['ma'].rolling(n, min_periods=1).mean()
    df[factor_name] = df['close'] / (df['ma2'] + eps) - 1

    # 删除多余列
    del df['ma'], df['ma2']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # TmaBias
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    """
    N=20
    CLOSE_MA=MA(CLOSE,N)
    TMA=MA(CLOSE_MA,N)
    TMA 均线与其他的均线类似，不同的是，像 EMA 这类的均线会赋予
    越靠近当天的价格越高的权重，而 TMA 则赋予考虑的时间段内时间
    靠中间的价格更高的权重。如果收盘价上穿/下穿 TMA 则产生买入/
    卖出信号。
    """
    ma = df['close'].rolling(n, min_periods=1).mean()  # CLOSE_MA=MA(CLOSE,N)
    tma = ma.rolling(n, min_periods=1).mean()  # TMA=MA(CLOSE_MA,N)
    df[factor_name] = df['close'] / (tma + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # Tma2
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    """
    N=20
    CLOSE_MA=MA(CLOSE,N)
    TMA=MA(CLOSE_MA,N)
    TMA 均线与其他的均线类似，不同的是，像 EMA 这类的均线会赋予
    越靠近当天的价格越高的权重，而 TMA 则赋予考虑的时间段内时间
    靠中间的价格更高的权重。如果收盘价上穿/下穿 TMA 则产生买入/
    卖出信号。
    """
    _ts = df[['high', 'low']].sum(axis=1) / 2

    close_ma = _ts.rolling(n, min_periods=1).mean()
    tma = close_ma.rolling(n, min_periods=1).mean()
    df[factor_name] = df['close'] / (tma+eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # TMA 指标
    """
    N=20
    CLOSE_MA=MA(CLOSE,N)
    TMA=MA(CLOSE_MA,N)
    TMA 均线与其他的均线类似，不同的是，像 EMA 这类的均线会赋予
    越靠近当天的价格越高的权重，而 TMA 则赋予考虑的时间段内时间
    靠中间的价格更高的权重。如果收盘价上穿/下穿 TMA 则产生买入/
    卖出信号。
    """
    _ts = (df['high'].rolling(n, min_periods=1).max() + df['low'].rolling(n, min_periods=1).min()) / 2.

    close_ma = _ts.rolling(n, min_periods=1).mean()
    tma = close_ma.rolling(n, min_periods=1).mean()
    df[factor_name] = df['close'] / tma - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # TMF 指标
    """
    N=80
    HIGH_TRUE=MAX(HIGH,REF(CLOSE,1))
    LOW_TRUE=MIN(LOW,REF(CLOSE,1))
    TMF=EMA(VOL*(2*CLOSE-HIGH_TRUE-LOW_TRUE)/(HIGH_TR
    UE-LOW_TRUE),N)/EMA(VOL,N)
    TMF 指标和 CMF 指标类似，都是用价格对成交量加权。但是 CMF
    指标用 CLV 做权重，而 TMF 指标用的是真实最低价和真实最高价，
    且取的是移动平均而不是求和。如果 TMF 上穿 0，则产生买入信号；
    如果 TMF 下穿 0，则产生卖出信号。
    """
    df['ref'] = df['close'].shift(1)
    df['max_high'] = df[['high', 'ref']].max(axis=1)
    df['min_low'] = df[['low', 'ref']].min(axis=1)

    T = df['volume'] * ( 2 * df['close'] - df['max_high'] - df['min_low']) / (df['max_high'] - df['min_low'])
    df[factor_name] = T.ewm(n, adjust=False).mean() / df['volume'].ewm(n, adjust=False).mean()
    
    del df['ref']
    del df['max_high']
    del df['min_low']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df






#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff


def signal(*args):
    # TradeNum
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df[factor_name] = df['trade_num'].rolling(n, min_periods=1).sum()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # TrendZhangDieFu
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['涨跌幅'] = df['close'].pct_change(n).shift(1)
    high = df['high'].rolling(n, min_periods=1).max()
    low = df['low'].rolling(n, min_periods=1).min()
    df['振幅'] = (high / low - 1).shift(1)

    # 涨跌幅 / 振幅  的计算结果在[-1,1]之间，越大说明也趋近于单边上涨，越小说明越趋近于单边下跌
    df['单边趋势'] = df['涨跌幅'] / (df['振幅'] + eps)
    df[factor_name] = df['涨跌幅'] * abs(df['单边趋势'])

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # Trix
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df['ema'] = df['close'].ewm(n, adjust=False).mean()  # EMA(CLOSE,N)
    df['ema_ema'] = df['ema'].ewm(n, adjust=False).mean()  # EMA(EMA(CLOSE,N),N)
    df['ema_ema_ema'] = df['ema_ema'].ewm(n, adjust=False).mean()  # EMA(EMA(EMA(CLOSE,N),N),N)
    # TRIX=(TRIPLE_EMA-REF(TRIPLE_EMA,1))/REF(TRIPLE_EMA,1)
    df[factor_name] = (df['ema_ema_ema'] - df['ema_ema_ema'].shift(1)) / (df['ema_ema_ema'].shift(1) + eps)

    # 删除多余列
    del df['ema'], df['ema_ema'], df['ema_ema_ema']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Trrq 指标
    
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['归一成交额'] = df['quote_volume'] / \
        df['quote_volume'].rolling(n, min_periods=1).mean()
    reg_price = ta.LINEARREG(df['tp'], timeperiod=n)
    df['tp_reg涨跌幅'] = reg_price.pct_change(n)
    df['tp_reg涨跌幅除以归一成交额'] = df['tp_reg涨跌幅'] / (eps + df['归一成交额'])
    df[factor_name] = df['tp_reg涨跌幅除以归一成交额'].rolling(n).sum()

    del df['tp'], df['归一成交额'], df['tp_reg涨跌幅'], df['tp_reg涨跌幅除以归一成交额']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df










        

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # TrTrix 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['tr_trix'] = df['close'].ewm(span=n, adjust=False).mean()
    df[factor_name] = df['tr_trix'].pct_change()

    # 删除多余列
    del df['tr_trix']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


# https://bbs.quantclass.cn/thread/18347


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算波动率因子
    df['ma'] = df['close'].rolling(window=n, min_periods=1).mean()
    df['trv'] = 100 * ((df['ma'] - df['ma'].shift(n)) / df['ma'].shift(n))
    df[factor_name] = df['trv'].rolling(n, min_periods=1).mean()

    drop_col = [
       'ma', 'trv'
    ]
    df.drop(columns=drop_col, inplace=True)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # TSI 指标
    """
    N1=25
    N2=13
    TSI=EMA(EMA(CLOSE-REF(CLOSE,1),N1),N2)/EMA(EMA(ABS(
    CLOSE-REF(CLOSE,1)),N1),N2)*100
    TSI 是一种双重移动平均指标。与常用的移动平均指标对收盘价取移
    动平均不同，TSI 对两天收盘价的差值取移动平均。如果 TSI 上穿 10/
    下穿-10 则产生买入/卖出指标。
    """
    n1 = 2 * n
    df['diff_close'] = df['close'] - df['close'].shift(1)
    df['ema'] = df['diff_close'].ewm(n1, adjust=False).mean()
    df['ema_ema'] = df['ema'].ewm(n, adjust=False).mean()

    df['abs_diff_close'] = abs(df['diff_close'])
    df['abs_ema'] = df['abs_diff_close'].ewm(n1, adjust=False).mean()
    df['abs_ema_ema'] = df['abs_ema'].ewm(n, adjust=False).mean()

    df[factor_name] = df['ema_ema'] / df['abs_ema_ema'] * 100

    
    del df['diff_close']
    del df['ema']
    del df['ema_ema']
    del df['abs_diff_close']
    del df['abs_ema']
    del df['abs_ema_ema']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df





#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Turtle 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 计算海龟
    df['open_close_high'] = df[['open', 'close']].max(axis=1)
    df['open_close_low'] = df[['open', 'close']].min(axis=1)
    # 计算atr
    df['c1'] = df['high'] - df['low']
    df['c2'] = abs(df['high'] - df['close'].shift(1))
    df['c3'] = abs(df['low'] - df['close'].shift(1))
    # 计算上下轨
    df['up'] = df['open_close_high'].rolling(
        window=n, min_periods=1).max().shift(1)
    df['dn'] = df['open_close_low'].rolling(
        window=n, min_periods=1).min().shift(1)
    # 计算std
    df['std'] = df['close'].rolling(n, min_periods=1).std()
    # 计算atr
    df['tr'] = df[['c1', 'c2', 'c3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=n).mean()
    # 将上下轨中间的部分设为0
    condition_0 = (df['close'] <= df['up']) & (df['close'] >= df['dn'])
    condition_1 = df['close'] > df['up']
    condition_2 = df['close'] < df['dn']
    df.loc[condition_0, 'd'] = 0
    df.loc[condition_1, 'd'] = df['close'] - df['up']
    df.loc[condition_2, 'd'] = df['close'] - df['dn']
    df[factor_name] = df['d'] / (df['up'] - df['dn'] + eps)

    del df['up'], df['dn'], df['std'], df['tr'], df['atr'], df['d']
    del df['open_close_high'], df['open_close_low']
    del df['c1'], df['c2'], df['c3']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # TYP 指标
    """
    N1=10
    N2=30
    TYP=(CLOSE+HIGH+LOW)/3
    TYPMA1=EMA(TYP,N1)
    TYPMA2=EMA(TYP,N2)
    在技术分析中，典型价格（最高价+最低价+收盘价）/3 经常被用来代
    替收盘价。比如我们在利用均线交叉产生交易信号时，就可以用典型
    价格的均线。
    TYPMA1 上穿/下穿 TYPMA2 时产生买入/卖出信号。
    """
    TYP = (df['close'] + df['high'] + df['low']) / 3
    TYPMA1 = TYP.ewm(n, adjust=False).mean()
    TYPMA2 = TYP.ewm(n * 3, adjust=False).mean()
    diff_TYP = TYPMA1 - TYPMA2
    diff_TYP_mean = diff_TYP.rolling(n, min_periods=1).mean()
    diff_TYP_std = diff_TYP.rolling(n, min_periods=1).std()

    # 无量纲
    df[factor_name] = diff_TYP - diff_TYP_mean / diff_TYP_std

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # Uos指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    M = n
    N = 2 * n
    O = 4 * n
    df['ref_close'] = df['close'].shift(1)
    df['TH'] = df[['high', 'ref_close']].max(axis=1)
    df['TL'] = df[['low', 'ref_close']].min(axis=1)
    df['TR'] = df['TH'] - df['TL']
    df['XR'] = df['close'] - df['TL']
    df['XRM'] = df['XR'].rolling(M).sum() / df['TR'].rolling(M).sum()
    df['XRN'] = df['XR'].rolling(N).sum() / df['TR'].rolling(N).sum()
    df['XRO'] = df['XR'].rolling(O).sum() / df['TR'].rolling(O).sum()
    df[factor_name] = 100 * (df['XRM'] * N * O + df['XRN'] * M * O + df['XRO'] * M * N) / (M * N + M * O + N * O)

    # 删除多余列
    del df['ref_close'], df['TH'], df['TL'], df['TR'], df['XR']
    del df['XRM'], df['XRN'], df['XRO']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 过去n分钟有多少分钟是上涨的
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['_ret_sign'] = df['close'].pct_change() > 0
    df[factor_name] = df['_ret_sign'].rolling(n, min_periods=1).sum()

    del df['_ret_sign']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # V1 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    n1 = n
    # 计算动量因子
    mtm = df['close'] / df['close'].shift(n1) - 1
    mtm_mean = mtm.rolling(window=n1, min_periods=1).mean()

    # 基于价格atr，计算波动率因子wd_atr
    c1 = df['high'] - df['low']
    c2 = abs(df['high'] - df['close'].shift(1))
    c3 = abs(df['low'] - df['close'].shift(1))
    tr = np.max(np.array([c1, c2, c3]), axis=0)  # 三个数列取其大值
    atr = pd.Series(tr).rolling(window=n1, min_periods=1).mean()
    avg_price = df['close'].rolling(window=n1, min_periods=1).mean()
    wd_atr = atr / avg_price  # === 波动率因子

    # 参考ATR，对MTM指标，计算波动率因子
    mtm_l = df['low'] / df['low'].shift(n1) - 1
    mtm_h = df['high'] / df['high'].shift(n1) - 1
    mtm_c = df['close'] / df['close'].shift(n1) - 1
    mtm_c1 = mtm_h - mtm_l
    mtm_c2 = abs(mtm_h - mtm_c.shift(1))
    mtm_c3 = abs(mtm_l - mtm_c.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm 波动率因子

    # 参考ATR，对MTM mean指标，计算波动率因子
    mtm_l_mean = mtm_l.rolling(window=n1, min_periods=1).mean()
    mtm_h_mean = mtm_h.rolling(window=n1, min_periods=1).mean()
    mtm_c_mean = mtm_c.rolling(window=n1, min_periods=1).mean()
    mtm_c1 = mtm_h_mean - mtm_l_mean
    mtm_c2 = abs(mtm_h_mean - mtm_c_mean.shift(1))
    mtm_c3 = abs(mtm_l_mean - mtm_c_mean.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr_mean = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm_mean 波动率因子

    indicator = mtm_mean
    # mtm_mean指标分别乘以三个波动率因子
    indicator *= wd_atr * mtm_atr * mtm_atr_mean
    df[factor_name] = pd.Series(indicator)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # V1Dn 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    n1 = n

    # 计算动量因子
    mtm = df['close'] / df['close'].shift(n1) - 1
    mtm_mean = mtm.rolling(window=n1, min_periods=1).mean()

    # 基于价格atr，计算波动率因子wd_atr
    c1 = df['high'] - df['low']
    c2 = abs(df['high'] - df['close'].shift(1))
    c3 = abs(df['low'] - df['close'].shift(1))
    tr = np.max(np.array([c1, c2, c3]), axis=0)  # 三个数列取其大值
    atr = pd.Series(tr).rolling(window=n1, min_periods=1).mean()
    avg_price = df['close'].rolling(window=n1, min_periods=1).mean()
    wd_atr = atr / avg_price  # === 波动率因子

    # 参考ATR，对MTM指标，计算波动率因子
    mtm_l = df['low'] / df['low'].shift(n1) - 1
    mtm_h = df['high'] / df['high'].shift(n1) - 1
    mtm_c = df['close'] / df['close'].shift(n1) - 1
    mtm_c1 = mtm_h - mtm_l
    mtm_c2 = abs(mtm_h - mtm_c.shift(1))
    mtm_c3 = abs(mtm_l - mtm_c.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm 波动率因子

    # 参考ATR，对MTM mean指标，计算波动率因子
    mtm_l_mean = mtm_l.rolling(window=n1, min_periods=1).mean()
    mtm_h_mean = mtm_h.rolling(window=n1, min_periods=1).mean()
    mtm_c_mean = mtm_c.rolling(window=n1, min_periods=1).mean()
    mtm_c1 = mtm_h_mean - mtm_l_mean
    mtm_c2 = abs(mtm_h_mean - mtm_c_mean.shift(1))
    mtm_c3 = abs(mtm_l_mean - mtm_c_mean.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr_mean = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm_mean 波动率因子

    indicator = mtm_mean
    # mtm_mean指标分别乘以三个波动率因子
    indicator *= wd_atr * mtm_atr * mtm_atr_mean
    indicator = pd.Series(indicator)

    # 对新策略因子计算自适应布林
    median = indicator.rolling(window=n1).mean()
    std = indicator.rolling(n1, min_periods=1).std(
        ddof=0)  # ddof代表标准差自由度
    z_score = abs(indicator - median) / std
    m1 = pd.Series(z_score).rolling(window=n1).max().shift(1)
    dn1 = median - std * m1
    indicator *= 1e8
    dn1 *= 1e8
    df[factor_name] = dn1 - indicator

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # V1Dn_v2 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    n1 = n

    # 计算动量因子
    mtm = df['close'] / df['close'].shift(n1) - 1
    mtm_mean = mtm.rolling(window=n1, min_periods=1).mean()

    # 基于价格atr，计算波动率因子wd_atr
    c1 = df['high'] - df['low']
    c2 = abs(df['high'] - df['close'].shift(1))
    c3 = abs(df['low'] - df['close'].shift(1))
    tr = np.max(np.array([c1, c2, c3]), axis=0)  # 三个数列取其大值
    atr = pd.Series(tr).rolling(window=n1, min_periods=1).mean()
    avg_price = df['close'].rolling(window=n1, min_periods=1).mean()
    wd_atr = atr / avg_price  # === 波动率因子

    # 参考ATR，对MTM指标，计算波动率因子
    mtm_l = df['low'] / df['low'].shift(n1) - 1
    mtm_h = df['high'] / df['high'].shift(n1) - 1
    mtm_c = df['close'] / df['close'].shift(n1) - 1
    mtm_c1 = mtm_h - mtm_l
    mtm_c2 = abs(mtm_h - mtm_c.shift(1))
    mtm_c3 = abs(mtm_l - mtm_c.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm 波动率因子

    # 参考ATR，对MTM mean指标，计算波动率因子
    mtm_l_mean = mtm_l.rolling(window=n1, min_periods=1).mean()
    mtm_h_mean = mtm_h.rolling(window=n1, min_periods=1).mean()
    mtm_c_mean = mtm_c.rolling(window=n1, min_periods=1).mean()
    mtm_c1 = mtm_h_mean - mtm_l_mean
    mtm_c2 = abs(mtm_h_mean - mtm_c_mean.shift(1))
    mtm_c3 = abs(mtm_l_mean - mtm_c_mean.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr_mean = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm_mean 波动率因子

    indicator = mtm_mean
    # mtm_mean指标分别乘以三个波动率因子
    indicator *= wd_atr * mtm_atr * mtm_atr_mean
    indicator = pd.Series(indicator)

    # 对新策略因子计算自适应布林
    median = indicator.rolling(window=n1).mean()
    std = indicator.rolling(n1, min_periods=1).std(
        ddof=0)  # ddof代表标准差自由度
    z_score = abs(indicator - median) / std
    m1 = pd.Series(z_score).rolling(window=n1).mean()
    dn1 = median - std * m1
    indicator *= 1e8
    dn1 *= 1e8
    df[factor_name] = dn1 - indicator

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # V1Up 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    n1 = n

    # 计算动量因子
    mtm = df['close'] / df['close'].shift(n1) - 1
    mtm_mean = mtm.rolling(window=n1, min_periods=1).mean()

    # 基于价格atr，计算波动率因子wd_atr
    c1 = df['high'] - df['low']
    c2 = abs(df['high'] - df['close'].shift(1))
    c3 = abs(df['low'] - df['close'].shift(1))
    tr = np.max(np.array([c1, c2, c3]), axis=0)  # 三个数列取其大值
    atr = pd.Series(tr).rolling(window=n1, min_periods=1).mean()
    avg_price = df['close'].rolling(window=n1, min_periods=1).mean()
    wd_atr = atr / avg_price  # === 波动率因子

    # 参考ATR，对MTM指标，计算波动率因子
    mtm_l = df['low'] / df['low'].shift(n1) - 1
    mtm_h = df['high'] / df['high'].shift(n1) - 1
    mtm_c = df['close'] / df['close'].shift(n1) - 1
    mtm_c1 = mtm_h - mtm_l
    mtm_c2 = abs(mtm_h - mtm_c.shift(1))
    mtm_c3 = abs(mtm_l - mtm_c.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm 波动率因子

    # 参考ATR，对MTM mean指标，计算波动率因子
    mtm_l_mean = mtm_l.rolling(window=n1, min_periods=1).mean()
    mtm_h_mean = mtm_h.rolling(window=n1, min_periods=1).mean()
    mtm_c_mean = mtm_c.rolling(window=n1, min_periods=1).mean()
    mtm_c1 = mtm_h_mean - mtm_l_mean
    mtm_c2 = abs(mtm_h_mean - mtm_c_mean.shift(1))
    mtm_c3 = abs(mtm_l_mean - mtm_c_mean.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr_mean = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm_mean 波动率因子

    indicator = mtm_mean
    # mtm_mean指标分别乘以三个波动率因子
    indicator *= wd_atr * mtm_atr * mtm_atr_mean
    indicator = pd.Series(indicator)

    # 对新策略因子计算自适应布林
    median = indicator.rolling(window=n1).mean()
    std = indicator.rolling(n1, min_periods=1).std(
        ddof=0)  # ddof代表标准差自由度
    z_score = abs(indicator - median) / std
    m1 = pd.Series(z_score).rolling(window=n1).max().shift(1)
    up1 = median + std * m1
    indicator *= 1e8
    up1 *= 1e8
    df[factor_name] = up1 - indicator

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # V1Up_v2 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    n1 = n

    # 计算动量因子
    mtm = df['close'] / df['close'].shift(n1) - 1
    mtm_mean = mtm.rolling(window=n1, min_periods=1).mean()

    # 基于价格atr，计算波动率因子wd_atr
    c1 = df['high'] - df['low']
    c2 = abs(df['high'] - df['close'].shift(1))
    c3 = abs(df['low'] - df['close'].shift(1))
    tr = np.max(np.array([c1, c2, c3]), axis=0)  # 三个数列取其大值
    atr = pd.Series(tr).rolling(window=n1, min_periods=1).mean()
    avg_price = df['close'].rolling(window=n1, min_periods=1).mean()
    wd_atr = atr / avg_price  # === 波动率因子

    # 参考ATR，对MTM指标，计算波动率因子
    mtm_l = df['low'] / df['low'].shift(n1) - 1
    mtm_h = df['high'] / df['high'].shift(n1) - 1
    mtm_c = df['close'] / df['close'].shift(n1) - 1
    mtm_c1 = mtm_h - mtm_l
    mtm_c2 = abs(mtm_h - mtm_c.shift(1))
    mtm_c3 = abs(mtm_l - mtm_c.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm 波动率因子

    # 参考ATR，对MTM mean指标，计算波动率因子
    mtm_l_mean = mtm_l.rolling(window=n1, min_periods=1).mean()
    mtm_h_mean = mtm_h.rolling(window=n1, min_periods=1).mean()
    mtm_c_mean = mtm_c.rolling(window=n1, min_periods=1).mean()
    mtm_c1 = mtm_h_mean - mtm_l_mean
    mtm_c2 = abs(mtm_h_mean - mtm_c_mean.shift(1))
    mtm_c3 = abs(mtm_l_mean - mtm_c_mean.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr_mean = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm_mean 波动率因子

    indicator = mtm_mean
    # mtm_mean指标分别乘以三个波动率因子
    indicator *= wd_atr * mtm_atr * mtm_atr_mean
    indicator = pd.Series(indicator)

    # 对新策略因子计算自适应布林
    median = indicator.rolling(window=n1).mean()
    std = indicator.rolling(n1, min_periods=1).std(
        ddof=0)  # ddof代表标准差自由度
    z_score = abs(indicator - median) / std
    m1 = pd.Series(z_score).rolling(window=n1).mean()
    up1 = median + std * m1
    indicator *= 1e8
    up1 *= 1e8
    df[factor_name] = up1 - indicator

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff,eps


def signal(*args):
    # V1 指标
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    n1 = n
    # 计算动量因子
    mtm = df['close'] / df['close'].shift(n1) - 1
    mtm_mean = mtm.rolling(window=n1, min_periods=1).mean()

    # 基于价格atr，计算波动率因子wd_atr
    c1 = df['high'] - df['low']
    c2 = abs(df['high'] - df['close'].shift(1))
    c3 = abs(df['low'] - df['close'].shift(1))
    tr = np.max(np.array([c1, c2, c3]), axis=0)  # 三个数列取其大值
    atr = pd.Series(tr).rolling(window=n1, min_periods=1).mean()
    avg_price = df['close'].rolling(window=n1, min_periods=1).mean()
    wd_atr = atr / avg_price  # === 波动率因子

    # 参考ATR，对MTM指标，计算波动率因子
    mtm_l = df['low'] / df['low'].shift(n1) - 1
    mtm_h = df['high'] / df['high'].shift(n1) - 1
    mtm_c = df['close'] / df['close'].shift(n1) - 1
    mtm_c1 = mtm_h - mtm_l
    mtm_c2 = abs(mtm_h - mtm_c.shift(1))
    mtm_c3 = abs(mtm_l - mtm_c.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm 波动率因子

    # 参考ATR，对MTM mean指标，计算波动率因子
    mtm_l_mean = mtm_l.rolling(window=n1, min_periods=1).mean()
    mtm_h_mean = mtm_h.rolling(window=n1, min_periods=1).mean()
    mtm_c_mean = mtm_c.rolling(window=n1, min_periods=1).mean()
    mtm_c1 = mtm_h_mean - mtm_l_mean
    mtm_c2 = abs(mtm_h_mean - mtm_c_mean.shift(1))
    mtm_c3 = abs(mtm_l_mean - mtm_c_mean.shift(1))
    mtm_tr = np.max(
        np.array([mtm_c1, mtm_c2, mtm_c3]), axis=0)  # 三个数列取其大值
    mtm_atr_mean = pd.Series(mtm_tr).rolling(
        window=n1, min_periods=1).mean()  # === mtm_mean 波动率因子

    indicator = mtm_mean
    # mtm_mean指标分别乘以三个波动率因子
    indicator *= wd_atr * mtm_atr * mtm_atr_mean
    df[factor_name] = pd.Series(indicator)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff, eps

eps = 1e-8


def signal(*args):
    # Vao
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    wv = df['volume'] * (df['close'] - 0.5 * df['high'] - 0.5 * df['low'])
    _vao = wv + wv.shift(1)
    vao_ma1 = _vao.rolling(n, min_periods=1).mean()
    vao_ma2 = _vao.rolling(int(3*n), min_periods=1).mean()

    df[factor_name] = pd.Series(vao_ma1 - vao_ma2)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # PVT 指标
    """
    PVT=(CLOSE-REF(CLOSE,1))/REF(CLOSE,1)*VOLUME
    PVT_MA1=MA(PVT,N1)
    PVT_MA2=MA(PVT,N2)
    PVT 指标用价格的变化率作为权重求成交量的移动平均。PVT 指标
    与 OBV 指标的思想类似，但与 OBV 指标相比，PVT 考虑了价格不
    同涨跌幅的影响，而 OBV 只考虑了价格的变化方向。我们这里用 PVT
    短期和长期均线的交叉来产生交易信号。
    如果 PVT_MA1 上穿 PVT_MA2，则产生买入信号；
    如果 PVT_MA1 下穿 PVT_MA2，则产生卖出信号。
    """
    df['WV'] = df['volume'] * (df['close'] - 0.5 * df['high'] - 0.5 * df['low'])
    df['VAO'] = df['WV']+df['WV'].shift(1)
    df['VAO_MA1'] = df['VAO'].rolling(n, min_periods=1).mean()
    df['VAO_MA2'] = df['VAO'].rolling(3 * n, min_periods=1).mean()
    df['Vao_v2'] = df['VAO_MA1'] - df['VAO_MA2']

    # 去量纲
    df[factor_name] = df['VAO'] / df['Vao_v2'] - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # VI 指标
    """
    TR=MAX([ABS(HIGH-LOW),ABS(LOW-REF(CLOSE,1)),ABS(HIG
    H-REF(CLOSE,1))])
    VMPOS=ABS(HIGH-REF(LOW,1))
    VMNEG=ABS(LOW-REF(HIGH,1))
    N=40
    SUMPOS=SUM(VMPOS,N)
    SUMNEG=SUM(VMNEG,N)
    TRSUM=SUM(TR,N)
    VI+=SUMPOS/TRSUM
    VI-=SUMNEG/TRSUM
    VI 指标可看成 ADX 指标的变形。VI 指标中的 VI+与 VI-与 ADX 中的
    DI+与 DI-类似。不同的是 ADX 中用当前高价与前一天高价的差和当
    前低价与前一天低价的差来衡量价格变化，而 VI 指标用当前当前高
    价与前一天低价和当前低价与前一天高价的差来衡量价格变化。当
    VI+上穿/下穿 VI-时，多/空的信号更强，产生买入/卖出信号。
    """
    df['c1'] = abs(df['high'] - df['low'])
    df['c2'] = abs(df['close'] - df['close'].shift(1))
    df['c3'] = abs(df['high'] - df['close'].shift(1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)

    df['VMPOS'] = abs(df['high'] - df['low'].shift(1))
    df['VMNEG'] = abs(df['low'] - df['high'].shift(1))
    df['sum_pos'] = df['VMPOS'].rolling(n).sum()
    df['sum_neg'] = df['VMNEG'].rolling(n).sum()

    df['sum_tr'] = df['TR'].rolling(n).sum()
    df[factor_name] = df['sum_pos'] / df['sum_tr'] #Vi+
    # df['VI-'] = df['sum_neg'] / df['sum_tr'] #Vi-

    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['VMPOS']
    del df['VMNEG']
    del df['sum_pos']
    del df['sum_neg']
    del df['sum_tr']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df







#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # VI 指标
    """
    TR=MAX([ABS(HIGH-LOW),ABS(LOW-REF(CLOSE,1)),ABS(HIG
    H-REF(CLOSE,1))])
    VMPOS=ABS(HIGH-REF(LOW,1))
    VMNEG=ABS(LOW-REF(HIGH,1))
    N=40
    SUMPOS=SUM(VMPOS,N)
    SUMNEG=SUM(VMNEG,N)
    TRSUM=SUM(TR,N)
    VI+=SUMPOS/TRSUM
    VI-=SUMNEG/TRSUM
    VI 指标可看成 ADX 指标的变形。VI 指标中的 VI+与 VI-与 ADX 中的
    DI+与 DI-类似。不同的是 ADX 中用当前高价与前一天高价的差和当
    前低价与前一天低价的差来衡量价格变化，而 VI 指标用当前当前高
    价与前一天低价和当前低价与前一天高价的差来衡量价格变化。当
    VI+上穿/下穿 VI-时，多/空的信号更强，产生买入/卖出信号。
    """
    df['c1'] = abs(df['high'] - df['low'])
    df['c2'] = abs(df['close'] - df['close'].shift(1))
    df['c3'] = abs(df['high'] - df['close'].shift(1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)

    df['VMPOS'] = abs(df['high'] - df['low'].shift(1))
    df['VMNEG'] = abs(df['low'] - df['high'].shift(1))
    df['sum_pos'] = df['VMPOS'].rolling(n).sum()
    df['sum_neg'] = df['VMNEG'].rolling(n).sum()

    df['sum_tr'] = df['TR'].rolling(n).sum()
    # df[factor_name] = df['sum_pos'] / df['sum_tr'] #Vi+
    df[factor_name] = df['sum_neg'] / df['sum_tr'] #Vi-

    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['VMPOS']
    del df['VMNEG']
    del df['sum_pos']
    del df['sum_neg']
    del df['sum_tr']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df







#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # Vi 指标
    """
    TR=MAX([ABS(HIGH-LOW),ABS(LOW-REF(CLOSE,1)),ABS(HIG
    H-REF(CLOSE,1))])
    VMPOS=ABS(HIGH-REF(LOW,1))
    VMNEG=ABS(LOW-REF(HIGH,1))
    N=40
    SUMPOS=SUM(VMPOS,N)
    SUMNEG=SUM(VMNEG,N)
    TRSUM=SUM(TR,N)
    Vi+=SUMPOS/TRSUM
    Vi-=SUMNEG/TRSUM
    Vi 指标可看成 ADX 指标的变形。Vi 指标中的 Vi+与 Vi-与 ADX 中的
    DI+与 DI-类似。不同的是 ADX 中用当前高价与前一天高价的差和当
    前低价与前一天低价的差来衡量价格变化，而 Vi 指标用当前当前高
    价与前一天低价和当前低价与前一天高价的差来衡量价格变化。当
    Vi+上穿/下穿 Vi-时，多/空的信号更强，产生买入/卖出信号。
    """
    df['c1'] = abs(df['high'] - df['low'])
    df['c2'] = abs(df['close'] - df['close'].shift(1))
    df['c3'] = abs(df['high'] - df['close'].shift(1))
    df['TR'] = df[['c1', 'c2', 'c3']].max(axis=1)

    df['VMPOS'] = abs(df['high'] - df['low'].shift(1))
    df['VMNEG'] = abs(df['low'] - df['high'].shift(1))
    df['sum_pos'] = df['VMPOS'].rolling(n).sum()
    df['sum_neg'] = df['VMNEG'].rolling(n).sum()

    df['sum_tr'] = df['TR'].rolling(n).sum()
    df['Vi+'] = df['sum_pos'] / df['sum_tr']
    df['Vi-'] = df['sum_neg'] / df['sum_tr']
    df[factor_name] = df['Vi+'] - df['Vi-']
    # df[f'Vi+_bh_{n}'] = df['Vi+'].shift(1)
    # df[f'Vi-_bh_{n}'] = df['Vi-'].shift(1)
    # df[f'Vi_bh_{n}'] = df['Vi'].shift(1)

    del df['c1']
    del df['c2']
    del df['c3']
    del df['TR']
    del df['VMPOS']
    del df['VMNEG']
    del df['sum_pos']
    del df['sum_neg']
    del df['sum_tr']
    del df['Vi+']
    del df['Vi-']
    # del df['Vi']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df







#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # 该指标使用时注意n不能大于过滤K线数量的一半（不是获取K线数据的一半）
    """
    N=10
    VI=ABS(CLOSE-REF(CLOSE,N))/SUM(ABS(CLOSE-REF(CLOSE,1)),N)
    VIDYA=VI*CLOSE+(1-VI)*REF(CLOSE,1)
    VIDYA 也属于均线的一种，不同的是，VIDYA 的权值加入了 ER
    （EfficiencyRatio）指标。在当前趋势较强时，ER 值较大，VIDYA
    会赋予当前价格更大的权重，使得 VIDYA 紧随价格变动，减小其滞
    后性；在当前趋势较弱（比如振荡市中）,ER 值较小，VIDYA 会赋予
    当前价格较小的权重，增大 VIDYA 的滞后性，使其更加平滑，避免
    产生过多的交易信号。
    当收盘价上穿/下穿 VIDYA 时产生买入/卖出信号。
    """
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['abs_diff_close'] = abs(
        df['close'] - df['close'].shift(1))  # ABS(CLOSE-REF(CLOSE,N))
    df['abs_diff_close_n'] = abs(
        df['close'] - df['close'].shift(n))  # ABS(CLOSE-REF(CLOSE,N))
    df['abs_diff_close_sum'] = df['abs_diff_close'].rolling(
        n).sum()  # SUM(ABS(CLOSE-REF(CLOSE,1))
    # VI=ABS(CLOSE-REF(CLOSE,N))/SUM(ABS(CLOSE-REF(CLOSE,1)),N)
    VI = df['abs_diff_close_n'] / df['abs_diff_close_sum']
    # VIDYA=VI*CLOSE+(1-VI)*REF(CLOSE,1)
    VIDYA = VI * df['close'] + (1 - VI) * df['close'].shift(1)
    # 进行无量纲处理
    df[factor_name] = VIDYA / (df['close'].rolling(n, min_periods=1).mean()) - 1

    del df['abs_diff_close']
    del df['abs_diff_close_n']
    del df['abs_diff_close_sum']




    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # Vidya_v2
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    '''
    N=10 
    VI=ABS(CLOSE-REF(CLOSE,N))/SUM(ABS(CLOSE-REF(CLOSE ,1)),N)
    VIDYA=VI*CLOSE+(1-VI)*REF(CLOSE,1)
    VIDYA 也属于均线的一种，不同的是，VIDYA 的权值加入了 ER (EfficiencyRatio)指标。
    在当前趋势较强时，ER 值较大，VIDYA 会赋予当前价格更大的权重，
    使得 VIDYA 紧随价格变动，减小其滞后性;
    在当前趋势较弱(比如振荡市中)，ER 值较小，VIDYA 会赋予当前价格较小的权重，
    增大 VIDYA 的滞后性，使其更加平滑，避免产生过多的交易信号。
    当收盘价上穿/下穿 VIDYA 时产生买入/卖出信号。
    '''

    _ts = (df['open'] + df['close']) / 2.

    _vi = (_ts - _ts.shift(n)).abs() / (
        _ts - _ts.shift(1)).abs().rolling(n, min_periods=1).sum()
    _vidya = _vi * _ts + (1 - _vi) * _ts.shift(1)

    df[factor_name] = pd.Series(_vidya)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # VIDYA
    """
    N=10
    VI=ABS(CLOSE-REF(CLOSE,N))/SUM(ABS(CLOSE-REF(CLOSE,1)),N)
    VIDYA=VI*CLOSE+(1-VI)*REF(CLOSE,1)
    VIDYA 也属于均线的一种，不同的是，VIDYA 的权值加入了 ER
    （EfficiencyRatio）指标。在当前趋势较强时，ER 值较大，VIDYA
    会赋予当前价格更大的权重，使得 VIDYA 紧随价格变动，减小其滞
    后性；在当前趋势较弱（比如振荡市中）,ER 值较小，VIDYA 会赋予
    当前价格较小的权重，增大 VIDYA 的滞后性，使其更加平滑，避免
    产生过多的交易信号。
    当收盘价上穿/下穿 VIDYA 时产生买入/卖出信号。
    """
    _ts = df[['high', 'low']].sum(axis=1) / 2.

    vi = (_ts - _ts.shift(n)).abs() / (
            _ts - _ts.shift(1)).abs().rolling(n, min_periods=1).sum()
    vidya = vi * _ts + (1 - vi) * _ts.shift(1)
    # 进行无量纲处理
    df[factor_name] = vidya / df['close']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # VIDYA
    """
    N=10
    VI=ABS(CLOSE-REF(CLOSE,N))/SUM(ABS(CLOSE-REF(CLOSE,1)),N)
    VIDYA=VI*CLOSE+(1-VI)*REF(CLOSE,1)
    VIDYA 也属于均线的一种，不同的是，VIDYA 的权值加入了 ER
    （EfficiencyRatio）指标。在当前趋势较强时，ER 值较大，VIDYA
    会赋予当前价格更大的权重，使得 VIDYA 紧随价格变动，减小其滞
    后性；在当前趋势较弱（比如振荡市中）,ER 值较小，VIDYA 会赋予
    当前价格较小的权重，增大 VIDYA 的滞后性，使其更加平滑，避免
    产生过多的交易信号。
    当收盘价上穿/下穿 VIDYA 时产生买入/卖出信号。
    """
    _ts = df[['open', 'close']].sum(axis=1) / 2.

    vi = (_ts - _ts.shift(n)).abs() / (
            _ts - _ts.shift(1)).abs().rolling(n, min_periods=1).sum()
    vidya = vi * _ts + (1 - vi) * _ts.shift(1)

    # 进行无量纲处理
    df[factor_name] = vidya / df['close']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # VIDYA
    """
    N=10
    VI=ABS(CLOSE-REF(CLOSE,N))/SUM(ABS(CLOSE-REF(CLOSE,1)),N)
    VIDYA=VI*CLOSE+(1-VI)*REF(CLOSE,1)
    VIDYA 也属于均线的一种，不同的是，VIDYA 的权值加入了 ER
    （EfficiencyRatio）指标。在当前趋势较强时，ER 值较大，VIDYA
    会赋予当前价格更大的权重，使得 VIDYA 紧随价格变动，减小其滞
    后性；在当前趋势较弱（比如振荡市中）,ER 值较小，VIDYA 会赋予
    当前价格较小的权重，增大 VIDYA 的滞后性，使其更加平滑，避免
    产生过多的交易信号。
    当收盘价上穿/下穿 VIDYA 时产生买入/卖出信号。
    """
    _ts = df[['high', 'low', 'close']].sum(axis=1) / 3.

    vi = (_ts - _ts.shift(n)).abs() / (
            _ts - _ts.shift(1)).abs().rolling(n, min_periods=1).sum()
    vidya = vi * _ts + (1 - vi) * _ts.shift(1)

    # 进行无量纲处理
    df[factor_name] = vidya / df['close']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """

    """
    df['vix'] = df['close'] / df['close'].shift(n) - 1
    df['vix_median'] = df['vix'].rolling(
        window=n, min_periods=1).mean()
    df['vix_std'] = df['vix'].rolling(n, min_periods=1).std()
    df['vix_score'] = abs(
        df['vix'] - df['vix_median']) / df['vix_std']
    df['max'] = df['vix_score'].rolling(
        window=n, min_periods=1).mean().shift(1)
    df['min'] = df['vix_score'].rolling(
        window=n, min_periods=1).min().shift(1)
    df['vix_upper'] = df['vix_median'] + df['max'] * df['vix_std']
    df['vix_lower'] = df['vix_median'] - df['max'] * df['vix_std']
    df[factor_name] = (df['vix_upper'] - df['vix_lower']) * np.sign(df['vix_median'].diff(n))
    condition1 = np.sign(df['vix_median'].diff(
        n)) != np.sign(df['vix_median'].diff(1))
    condition2 = np.sign(df['vix_median'].diff(
        n)) != np.sign(df['vix_median'].diff(1).shift(1))
    df.loc[condition1, factor_name] = 0
    df.loc[condition2, factor_name] = 0
    # ATR指标去量纲



    del df['vix']
    del df['vix_median']
    del df['vix_std']
    del df['max']
    del df['min']
    del df['vix_upper'],df['vix_lower']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff, eps


def signal(*args):
    # Vma
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    N=20
    PRICE=(HIGH+LOW+OPEN+CLOSE)/4
    VMA=MA(PRICE,N)
    VMA 就是简单移动平均把收盘价替换为最高价、最低价、开盘价和
    收盘价的平均值。当 PRICE 上穿/下穿 VMA 时产生买入/卖出信号。
    """
    price = (df['high'] + df['low'] + df['open'] + df['close']) / 4  # PRICE=(HIGH+LOW+OPEN+CLOSE)/4
    vma = price.rolling(n, min_periods=1).mean()  # VMA=MA(PRICE,N)
    df[factor_name] = price / (vma + eps) - 1  # 去量纲

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df


#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff


def signal(*args):
    # 平均每笔成交，看此分钟是否有大单出现
    # https://bbs.quantclass.cn/thread/14374

    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = df['quote_volume'].rolling(n, min_periods=1).sum() / df['trade_num'].rolling(n, min_periods=1).sum()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff


def signal(*args):
    # Volume
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df[factor_name] = df['quote_volume'].rolling(n, min_periods=1).sum()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff


def signal(*args):
    # VolumeStd
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    df[factor_name] = df['quote_volume'].rolling(
        n, min_periods=2).std()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff


def signal(*args):
    # Vr
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    av = np.where(df['close'] > df['close'].shift(1), df['volume'], 0)
    bv = np.where(df['close'] < df['close'].shift(1), df['volume'], 0)
    _cv = np.where(df['close'] == df['close'].shift(1), df['volume'], 0)

    avs = pd.Series(av).rolling(n, min_periods=1).sum()
    bvs = pd.Series(bv).rolling(n, min_periods=1).sum()
    cvs = pd.Series(_cv).rolling(n, min_periods=1).sum()

    signal = (avs + 0.5 * cvs) / (1e-9 + bvs + 0.5 * cvs)

    df[factor_name] = pd.Series(signal)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib as ta
from utils.diff import add_diff, eps

# https://bbs.quantclass.cn/thread/17716

def signal(*args):

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['n日均价'] = df['close'].rolling(n, min_periods=1).mean()
    df['n日标准差'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['n日波动率'] = df['n日标准差'] / df['n日均价']*100
    # 计算上轨、下轨道
    df['RC'] = 100 * ((df['high'] - df['high'].shift(n)) / df['close'].shift(n) + (df['close'] - df['close'].shift(2 * n)) / df['low'].shift(2 * n))
    df['RC_mean'] = df['RC'].rolling(n, min_periods=1).mean()

    # 组合指标
    df[factor_name] = df['RC_mean']*df['n日波动率']

    del  df['n日均价'], df['n日标准差'], df['RC']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df












#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff, eps

eps = 1e-8


def signal(*args):
    """
    N=40
    AV=IF(CLOSE>REF(CLOSE,1),AMOUNT,0)
    BV=IF(CLOSE<REF(CLOSE,1),AMOUNT,0)
    CV=IF(CLOSE=REF(CLOSE,1),AMOUNT,0)
    AVS=SUM(AV,N)
    BVS=SUM(BV,N)
    CVS=SUM(CV,N)
    VRAMT=(AVS+CVS/2)/(BVS+CVS/2)
    VRAMT 的计算与 VR 指标（Volume Ratio）一样，只是把其中的成
    交量替换成了成交额。
    如果 VRAMT 上穿 180，则产生买入信号；
    如果 VRAMT 下穿 70，则产生卖出信号。
    """

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['AV'] = np.where(df['close'] > df['close'].shift(1), df['volume'], 0)  # AV=IF(CLOSE>REF(CLOSE,1),AMOUNT,0)
    df['BV'] = np.where(df['close'] < df['close'].shift(1), df['volume'], 0)  # BV=IF(CLOSE<REF(CLOSE,1),AMOUNT,0)
    df['CV'] = np.where(df['close'] == df['close'].shift(1), df['volume'], 0)  # CV=IF(CLOSE=REF(CLOSE,1),AMOUNT,0)
    df['AVS'] = df['AV'].rolling(n, min_periods=1).sum()  # AVS=SUM(AV,N)
    df['BVS'] = df['BV'].rolling(n, min_periods=1).sum()  # BVS=SUM(BV,N)
    df['CVS'] = df['CV'].rolling(n, min_periods=1).sum()  # CVS=SUM(CV,N)
    df[factor_name] = (df['AVS'] + df['CVS'] / 2) / (df['BVS'] + df['CVS'] / 2 + eps)  # VRAMT=(AVS+CVS/2)/(BVS+CVS/2)

    del df['AV']
    del df['BV']
    del df['CV']
    del df['AVS']
    del df['BVS']
    del df['CVS']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # WVAD 指标
    """
    N=20
    WVAD=SUM(((CLOSE-OPEN)/(HIGH-LOW)*VOLUME),N)
    WVAD 是用价格信息对成交量加权的价量指标，用来比较开盘到收盘
    期间多空双方的力量。WVAD 的构造与 CMF 类似，但是 CMF 的权
    值用的是 CLV(反映收盘价在最高价、最低价之间的位置)，而 WVAD
    用的是收盘价与开盘价的距离（即蜡烛图的实体部分的长度）占最高
    价与最低价的距离的比例，且没有再除以成交量之和。
    WVAD 上穿 0 线，代表买方力量强；
    WVAD 下穿 0 线，代表卖方力量强。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['VAD'] = (df['close'] - df['open']) / (df['high'] - df['low']) * df['volume']
    df['WVAD'] = df['VAD'].rolling(n).sum()

    # 标准化
    df[factor_name] = (df['WVAD'] - df['WVAD'].rolling(n).min()) / (df['WVAD'].rolling(n).max() - df['WVAD'].rolling(n).min())

    del df['VAD']
    del df['WVAD']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


# https://bbs.quantclass.cn/thread/18129

def signal(*args):
    # VwapBbw
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
  
    vwap = df['quote_volume'] / df['volume']
    vwap_chg = vwap.pct_change(n)
    # 计算宽度的变化率
    width = df['close'].rolling(n, min_periods=1).std(ddof=0) * 2
    avg = df['close'].rolling(n, min_periods=1).mean()
    top = avg + width
    bot = avg - width
    bbw = top / bot
    bbw_chg = bbw.pct_change(n)

    df['成交额归一'] = df['quote_volume'] / df['quote_volume'].rolling(n, min_periods=1).mean()

    feature = (vwap_chg * bbw_chg) / df['成交额归一']
    df[factor_name] = feature.rolling(n, min_periods=1).sum()

    del df['成交额归一']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import talib
from utils.diff import add_diff, eps


def signal(*args):
    # WVAD 指标
    """
    将bias 的close替换成vwap


    N=20
    WVAD=SUM(((CLOSE-OPEN)/(HIGH-LOW)*VOLUME),N)
    WVAD 是用价格信息对成交量加权的价量指标，用来比较开盘到收盘
    期间多空双方的力量。WVAD 的构造与 CMF 类似，但是 CMF 的权
    值用的是 CLV(反映收盘价在最高价、最低价之间的位置)，而 WVAD
    用的是收盘价与开盘价的距离（即蜡烛图的实体部分的长度）占最高
    价与最低价的距离的比例，且没有再除以成交量之和。
    WVAD 上穿 0 线，代表买方力量强；
    WVAD 下穿 0 线，代表卖方力量强。

    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['vwap'] = df['quote_volume'] / df['volume']  # 在周期内成交额除以成交量等于成交均价
    ma = df['vwap'].rolling(n, min_periods=1).mean()  # 求移动平均线
    df[factor_name] = df['vwap'] / (ma + eps) - 1  # 去量纲

    del df['vwap']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # VwapSignal指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    # N=20
    # Typical=(HIGH+LOW+CLOSE)/3
    # MF=VOLUME*Typical
    # VOLUME_SUM=SUM(VOLUME,N)
    # MF_SUM=SUM(MF,N)
    # VWAP=MF_SUM/VOLUME_SUM
    # VWAP以成交量为权重计算价格的加权平均。如果当前价格上穿VWAP，则买入；如果当前价格下穿VWAP，则卖出。
    """
    df['tp'] = df[['high', 'low', 'close']].sum(axis=1) / 3
    df['mf'] = df['volume'] * df['tp']
    df['vol_sum'] = df['volume'].rolling(n, min_periods=1).sum()
    df['mf_sum'] = df['mf'].rolling(n, min_periods=1).sum()
    df['vwap'] = df['mf_sum'] / (eps + df['vol_sum'])
    df[factor_name] = df['tp'] / (df['vwap'] + eps) - 1

    # 删除多余列
    del df['tp'], df['mf'], df['vol_sum'], df['mf_sum'], df['vwap']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff, eps

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    #  WAD 指标
    """
    TRH=MAX(HIGH,REF(CLOSE,1))
    TRL=MIN(LOW,REF(CLOSE,1))
    AD=IF(CLOSE>REF(CLOSE,1),CLOSE-TRL,CLOSE-TRH) 
    AD=IF(CLOSE==REF(CLOSE,1),0,AD)  
    WAD=CUMSUM(AD)
    N=20
    WADMA=MA(WAD,N)
    参考：https://zhidao.baidu.com/question/19720557.html
    如果今天收盘价大于昨天收盘价，A/D=收盘价减去昨日收盘价和今日最低价较小者；
    如果今日收盘价小于昨日收盘价，A/D=收盘价减去昨日收盘价和今日最高价较大者；
    如果今日收盘价等于昨日收盘价，A/D=0；
    WAD=从第一个交易日起累加A/D；
    """
    df['ref_close'] = df['close'].shift(1)
    df['TRH'] = df[['high', 'ref_close']].max(axis=1)
    df['TRL'] = df[['low', 'ref_close']].min(axis=1)
    df['AD'] = np.where(df['close'] > df['close'].shift(1), df['close'] - df['TRL'], df['close'] - df['TRH'])
    df['AD'] = np.where(df['close'] == df['close'].shift(1), 0, df['AD'])
    df['WAD'] = df['AD'].cumsum()
    df['WADMA'] = df['WAD'].rolling(n, min_periods=1).mean()
    # 去量纲
    df[factor_name] = df['WAD'] / (df['WADMA'] + eps)
    
    del df['ref_close']
    del df['TRH'], df['TRL']
    del df['AD']
    del df['WAD']
    del df['WADMA'] 

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df










#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # Uos指标
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    """
    WC=(HIGH+LOW+2*CLOSE)/4
    N1=20
    N2=40
    EMA1=EMA(WC,N1)
    EMA2=EMA(WC,N2)
    WC 也可以用来代替收盘价构造一些技术指标（不过相对比较少用
    到）。我们这里用 WC 的短期均线和长期均线的交叉来产生交易信号。
    """
    WC = (df['high'] + df['low'] + 2 * df['close']) / 4  # WC=(HIGH+LOW+2*CLOSE)/4
    df['ema1'] = WC.ewm(n, adjust=False).mean()  # EMA1=EMA(WC,N1)
    df['ema2'] = WC.ewm(2 * n, adjust=False).mean()  # EMA2=EMA(WC,N2)
    # 去量纲
    df[factor_name] = df['ema1'] / (df['ema2'] + eps) - 1

    # 删除多余列
    del df['ema1'], df['ema2']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # WR 指标
    """
    HIGH(N)=MAX(HIGH,N)
    LOW(N)=MIN(LOW,N)
    WR=100*(HIGH(N)-CLOSE)/(HIGH(N)-LOW(N))
    WR 指标事实上就是 100-KDJ 指标计算过程中的 Stochastics。WR
    指标用来衡量市场的强弱和超买超卖状态。一般认为，当 WR 小于
    20 时，市场处于超买状态；当 WR 大于 80 时，市场处于超卖状态；
    当 WR 处于 20 到 80 之间时，多空较为平衡。
    如果 WR 上穿 80，则产生买入信号；
    如果 WR 下穿 20，则产生卖出信号。
    """
    df['max_high'] = df['high'].rolling(n, min_periods=1).max()
    df['min_low'] = df['low'].rolling(n, min_periods=1).min()
    df[factor_name] = (df['max_high'] - df['close']) / (df['max_high'] - df['min_low']) * 100
    
    del df['max_high']
    del df['min_low']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff


def signal(*args):

    # WVAD 指标
    """
    N=20
    WVAD=SUM(((CLOSE-OPEN)/(HIGH-LOW)*VOLUME),N)
    WVAD 是用价格信息对成交量加权的价量指标，用来比较开盘到收盘
    期间多空双方的力量。WVAD 的构造与 CMF 类似，但是 CMF 的权
    值用的是 CLV(反映收盘价在最高价、最低价之间的位置)，而 WVAD
    用的是收盘价与开盘价的距离（即蜡烛图的实体部分的长度）占最高
    价与最低价的距离的比例，且没有再除以成交量之和。
    WVAD 上穿 0 线，代表买方力量强；
    WVAD 下穿 0 线，代表卖方力量强。
    """
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['VAD'] = (df['close'] - df['open']) / (df['high'] - df['low']) * df['volume']
    df['WVAD'] = df['VAD'].rolling(n).sum()

    # 标准化
    df[factor_name] = (df['WVAD'] - df['WVAD'].rolling(n).min()) / (df['WVAD'].rolling(n).max() - df['WVAD'].rolling(n).min())

    del df['VAD']
    del df['WVAD']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff


def signal(*args):
    # ZhangDieFu
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df[factor_name] = df['close'].pct_change(n)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff, eps


def signal(*args):
    # ZhangDieFuAllHour
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    zhangdiefu_hour_list = [2, 3, 5]
    #  --- 涨跌幅_all_hour ---
    for m in zhangdiefu_hour_list:
        df[f'涨跌幅_bh_{m}'] = df['close'].pct_change(m)
        if m == zhangdiefu_hour_list[0]:
            df[f'涨跌幅_all_hour'] = df[f'涨跌幅_bh_{m}']
        else:
            df[f'涨跌幅_all_hour'] = df[f'涨跌幅_all_hour'] + df[f'涨跌幅_bh_{m}']
        del df[f'涨跌幅_bh_{m}']

    df[factor_name] = df[f'涨跌幅_all_hour'] / len(zhangdiefu_hour_list)

    del df[f'涨跌幅_all_hour']
    
    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from utils.diff import add_diff


def signal(*args):
    # ZhangDieFuSkew
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]
    
    # 涨跌幅偏度：在商品期货市场有效
    change = df['close'].pct_change()
    df[factor_name] = pd.Series(change).rolling(n).skew()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff


def signal(*args):
    # ZhangDieFuStd
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 涨跌幅std，振幅的另外一种形式
    change = df['close'].pct_change()
    df[factor_name] = pd.Series(change).rolling(n).std()

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # ZhenFu
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high = df['high'].rolling(n, min_periods=1).max()
    low = df['low'].rolling(n, min_periods=1).min()
    df[factor_name] = high / (low + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # ZhenFuBear
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high = df[['close', 'open']].max(axis=1)
    low = df[['close', 'open']].min(axis=1)
    high = high.rolling(n, min_periods=1).max()
    high = high.shift(1)
    low = low.rolling(n, min_periods=1).min()
    low = low.shift(1)
    df[factor_name] = (low - df['close']) / (df['close'] + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # ZhenFuBull
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    high = df[['close', 'open']].max(axis=1)
    low = df[['close', 'open']].min(axis=1)
    high = high.rolling(n, min_periods=1).max()
    high = high.shift(1)
    low = low.rolling(n, min_periods=1).min()
    low = low.shift(1)
    df[factor_name] = (df['close'] - high) / (df['close'] + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff


def signal(*args):
    # ZhenFu
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    df['振幅'] = (df['high'] - df['low']) / df['open'] - 1
    df[factor_name] = df['振幅'].rolling(n).std(ddof=0)
    del df['振幅']

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np
from utils.diff import add_diff, eps


def signal(*args):
    # ZhenFu_v2
    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 振幅：收盘价、开盘价
    high = df[['close', 'open']].max(axis=1)
    low = df[['close', 'open']].min(axis=1)
    high = pd.Series(high).rolling(n, min_periods=1).max()
    low = pd.Series(low).rolling(n, min_periods=1).min()
    df[factor_name] = high / (low + eps) - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df

import numpy  as np
import pandas as pd
import talib as ta
from utils.diff import add_diff, eps


def signal(*args):
    # ZhenZhangRatio 振幅 涨幅的比率
    # https://bbs.quantclass.cn/thread/9454

    df = args[0]
    n = args[1]
    diff_num = args[2]
    factor_name = args[3]

    # 涨跌幅std，振幅的另外一种形式
    df['Zhang'] = df['close'].pct_change()
    df['zhen'] = df['high']/df['low'] - 1
    df[factor_name] = df['zhen']/(df['Zhang'] + eps)

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import talib
from utils.diff import add_diff

def signal(*args):
    df = args[0]
    n  = args[1]
    diff_num = args[2]
    factor_name = args[3]
    # ZLMACD 指标
    """
    N1=20
    N2=100
    ZLMACD=(2*EMA(CLOSE,N1)-EMA(EMA(CLOSE,N1),N1))-(2*EM
    A(CLOSE,N2)-EMA(EMA(CLOSE,N2),N2))
    ZLMACD 指标是对 MACD 指标的改进，它在计算中使用 DEMA 而不
    是 EMA，可以克服 MACD 指标的滞后性问题。如果 ZLMACD 上穿/
    下穿 0，则产生买入/卖出信号。
    """
    ema1 = df['close'].ewm(n, adjust=False).mean()
    ema_ema_1 = ema1.ewm(n, adjust=False).mean()
    n2 = 5 * n
    ema2 = df['close'].ewm(n2, adjust=False).mean()
    ema_ema_2 = ema2.ewm(n2, adjust=False).mean()
    ZLMACD = (2 * ema1 - ema_ema_1) - (2 * ema2 - ema_ema_2)
    df[factor_name] = df['close'] / ZLMACD - 1

    if diff_num > 0:
        return add_diff(df, diff_num, factor_name)
    else:
        return df
