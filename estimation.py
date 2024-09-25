import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas_datareader as pdr

warnings.filterwarnings("ignore")

# Стиль для Seaborn
sns.set(style="whitegrid", context="talk")

def drawdown(returns):
    """
    Функция для вычисления просадки на основе доходности (returns_predict).
    
    Параметры:
    returns (pd.Series): Временной ряд доходности,  домноженной 
    на предсказания модели (returns_predict).
    
    Возвращает:
    pd.DataFrame: Временной ряд просадки.
    """
    portfolio = pd.DataFrame(returns)
    cum_rets = (portfolio + 1).cumprod()
    running_max = np.maximum.accumulate(cum_rets.dropna())
    return ((cum_rets / running_max) - 1)

def get_data_join(returns, index='RTSI', start='2010-01-01'):
    """
    Функция для получения объединенных данных портфеля и бенчмарка.
    """
    portfolio = pd.DataFrame(returns)
    returns_benchmark = pdr.get_data_moex(index, start)['CLOSE'].pct_change(1).dropna()
    returns_benchmark.name = 'rts'
    return pd.concat((portfolio, returns_benchmark), axis=1).dropna()

def sharpe_ratio(returns, spread=0.00011):
    """
    Функция для вычисления коэффициента Шарпа.
    """
    return np.sqrt(260) * (returns.mean() - (spread)) / returns.std()

def plot_returns(returns, join=None):
    """
    Функция для построения графика доходности на основе объединенного df с бенчмарком - 'join'.
    """
    if join is None:
        join = get_data_join(returns)

    plt.figure(figsize=(15, 6))

    # Построение графика стратегии
    plt.plot(join.index, join.iloc[:, 0].cumsum() * 100, label='Стратегия', color='DarkSlateBlue', linewidth=2.5)

    # Построение графика РТС
    plt.plot(join.index, join.iloc[:, 1].cumsum() * 100, label='РТС', color='Maroon', linewidth=2.5)

    # Настройки графика
    plt.title('Доходность', fontsize=15)
    plt.ylabel('Доходность %', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.show()

def plot_drawdown(returns, drawdown_portfolio=None):
    """
    Функция для построения графика просадки.
    """
    if drawdown_portfolio is None:
        drawdown_portfolio = drawdown(returns)

    plt.figure(figsize=(15, 6))

    # Построение графика просадки
    plt.fill_between(drawdown_portfolio.index, drawdown_portfolio.iloc[:, 0] * 100, color='DarkSeaGreen', alpha=0.5, label='Просадка')
    plt.plot(drawdown_portfolio.index, drawdown_portfolio.iloc[:, 0] * 100, color='SeaGreen', linewidth=2.5, label='Drawdown')

    # Настройки графика
    plt.title('Просадка', fontsize=15)
    plt.ylabel('Просадка %', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.show()

def get_estimate(returns, spread=0.00011):
    """
    Функция для оценки доходности и просадки портфеля.
    """
    drawdown_portfolio = drawdown(returns)
    sharpe = sharpe_ratio(returns, spread=spread)

    print(f"""
    ----------------------------------------------------------------------------- 
    Максимальная просадка: {round((-drawdown_portfolio.min())[0] * 100)}% 
    Коэффициент Шарпа: {round(sharpe, 3)} 
    -----------------------------------------------------------------------------
    """)

    plot_drawdown(returns)
    plot_returns(returns)
