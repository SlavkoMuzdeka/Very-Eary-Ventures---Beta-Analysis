import os
import pytz

from alpha import Alpha
from typing import Optional
from datetime import datetime
from data_models.YFinanceFetcher import YFinanceFetcher


def create_alpha(
    period_start: Optional[int] = None,
    period_end: Optional[int] = None,
    use_rolling: bool = True,
    window: int = 60,
) -> Alpha:
    """
    Creates an Alpha object that computes alpha and beta values over a specified period.

    Args:
        period_start (Optional[int]): The starting year of the period. If None, defaults to 2023.
        period_end (Optional[int]): The ending year of the period. If None, defaults to the current year.
        use_rolling (bool): Whether to use rolling calculations for alpha and beta.
        window (int): The window size for rolling calculations.

    Returns:
        Alpha: An Alpha object containing tickers, dataframes, and the rolling window configuration.
    """
    if period_start is None:
        start = datetime(2023, 1, 1, tzinfo=pytz.utc)
    else:
        start = datetime(period_start, 1, 1, tzinfo=pytz.utc)

    if period_end is None:
        end = datetime.now(pytz.utc)
    else:
        end = datetime(period_end, 1, 1, tzinfo=pytz.utc)

    if use_rolling:
        start_year = "with_rolling"
    else:
        start_year = start.year

    obj_path = os.path.join(os.getcwd(), "Data", f"yfinance_dataset_{start_year}.obj")
    data_fetcher = YFinanceFetcher()  # Fetching data from Yahoo Finance

    tickers, ticker_dfs = data_fetcher.get_ticker_dfs(
        start=start, end=end, obj_path=obj_path
    )

    return Alpha(
        insts=tickers,
        dfs=ticker_dfs,
        window=window,
        use_rolling=use_rolling,
    )
