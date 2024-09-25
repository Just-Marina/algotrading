import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import plotly.graph_objects as go
import warnings
import pandas_datareader as pdr

warnings.filterwarnings("ignore")

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
    # Подсчет накопленной доходности
    cum_rets = (portfolio + 1).cumprod()

    # Вычисление скользящего максимума
    running_max = np.maximum.accumulate(cum_rets.dropna())

    # Просадка
    return ((cum_rets / running_max) - 1)

def get_data_join(returns, index='RTSI', start='2010-01-01'):
    """
    Функция для получения объединенных данных портфеля и бенчмарка.
    
    Параметры:
    returns (pd.Series): Временной ряд доходности портфеля (returns_predict).
    index (str): Тикер бенчмарка. По умолчанию 'RTSI'.
    start (str): Начальная дата загрузки данных. По умолчанию '2010-01-01'.
    
    Возвращает:
    pd.DataFrame: Объединенные данные доходности портфеля (returns_predict)
    и бенчмарка (returns_benchmark).
    """
    portfolio = pd.DataFrame(returns)

    # Импорт данных бенчмарка
    returns_benchmark= pdr.get_data_moex(index, start)['CLOSE'].pct_change(1).dropna()
    returns_benchmark.name = 'rts'

    # Объединение данных портфеля и бенчмарка
    return pd.concat((portfolio, returns_benchmark), axis=1).dropna()

def sharpe_ratio(returns, spread=0.00011):
    """
    Функция для вычисления коэффициента Шарпа.
    
    Параметры:
    returns (pd.Series): Временной ряд доходности.
    spread (float): Усреднённый спред. 
    По умолчанию 0.00011 (доля от bid для EURUSD).
    
    Возвращает:
    float: Коэффициент Шарпа.
    """
    return np.sqrt(260) * (returns.mean() - (spread)) / returns.std()

def plot_returns(returns, join=None):
    """
    Функция для построения графика доходности на основе объединенного df с бенчмарком - 'join'.
    
    Параметры:
    returns (pd.Series): Временной ряд доходности.
    join (pd.DataFrame): df с данными, где строки представляют даты,
                         а столбцы - доходность стратегии и доходность индекса РТС. 
                         Если None, данные будут получены с помощью get_data_join.
    
    Возвращает:
    None
    """
    if join is None:
        join = get_data_join(returns)

    fig = go.Figure()

    # Первая линия графика
    fig.add_trace(go.Scatter(
        x=join.index, 
        y=join.iloc[:, 0].cumsum() * 100, 
        line=dict(color='DarkSlateBlue', width=2.5), 
        name='Стратегия'
    ))

    # Вторая линия графика
    fig.add_trace(go.Scatter(
        x=join.index, 
        y=join.iloc[:, 1].cumsum() * 100, 
        line=dict(color='Maroon', width=2.5), 
        name='РТС'
    ))

    # Настройки графика
    fig.update_layout(
        title='ДОХОДНОСТЬ',
        title_font_size=15,
        yaxis_title='Доходность %',
        yaxis_title_font_size=15,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        width=1200,  
        height=600
    )

    fig.show()

def plot_drawdown(returns, drawdown_portfolio=None):
    """
    Функция для построения графика просадки.
    
    Параметры:
    returns (pd.Series): Временной ряд доходности.
    drawdown_portfolio (pd.DataFrame): Временной ряд просадки. 
    Если None, просадка будет рассчитана с помощью функции drawdown.
    
    Возвращает:
    None
    """ 
    if drawdown_portfolio is None:
        drawdown_portfolio = drawdown(returns)

    fig = go.Figure()

    # Заполнение области между графиком и осью x
    fig.add_trace(go.Scatter(
        x=drawdown_portfolio.index, 
        y=drawdown_portfolio.iloc[:, 0] * 100, 
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(143, 188, 143, 0.5)',  # Цвет 'DarkSeaGreen' с прозрачностью
        name='Просадка'
    ))

    # Линия графика
    fig.add_trace(go.Scatter(
        x=drawdown_portfolio.index, 
        y=drawdown_portfolio.iloc[:, 0] * 100, 
        line=dict(color='SeaGreen'),
        name='Drawdown'
    ))

    # Настройки графика
    fig.update_layout(
        title='Просадка',
        title_font_size=15,
        yaxis_title='Просадка %',
        yaxis_title_font_size=15,
        showlegend=False,
        width=1200,  # Примерный размер фигуры
        height=600
    )

    fig.show()

def get_estimate(returns, spread=0.00011):
    """
    Функция для оценки доходности и просадки портфеля.
    
    Параметры:
    returns (pd.Series): Временной ряд доходности.
    
    Возвращает:
    None
    """

    drawdown_portfolio = drawdown(returns)
    sharpe = sharpe_ratio(returns, spread=spread)

    print(f"""
    -----------------------------------------------------------------------------
    Максимальная просадка: {round((-drawdown_portfolio.min())[0] * 100)}% \t
    Коэффициент Шарпа: {round(sharpe, 3)} \t 
    -----------------------------------------------------------------------------
    """)

    plot_drawdown(returns)
    plot_returns(returns)


