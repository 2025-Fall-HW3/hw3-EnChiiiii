"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # 先全部設 0（等於一開始都持有現金）
        self.portfolio_weights.loc[:, :] = 0.0

        # 基礎長期配置：選幾個歷史 Sharpe 較高、波動較低的 sector
        base_weights = {
            "XLK": 0.35,  # Tech
            "XLV": 0.25,  # Health Care
            "XLP": 0.20,  # Consumer Staples
            "XLU": 0.20,  # Utilities
        }

        # 只保留在目前資產池中的 ticker（排除 SPY 已在 assets 裡處理）
        base_weights = {k: v for k, v in base_weights.items() if k in assets}
        total_base = sum(base_weights.values())

        if total_base > 0:
            # 使用 200 日均線當作 trend filter
            ma_window = 200
            ma = self.price[list(base_weights.keys())].rolling(window=ma_window).mean()

            dates = self.price.index

            # 用「前一天」的資訊決定「今天」的部位，避免 look-ahead
            for i in range(1, len(dates)):
                date = dates[i]          # 今天要設定的權重
                prev_date = dates[i - 1] # 只能看前一天的價格與均線

                eligible = []

                # 判斷前一天哪些 sector 在多頭（前一日收盤價 > 前一日 200 日均線）
                for ticker in base_weights.keys():
                    px_prev = self.price.loc[prev_date, ticker]
                    mav_prev = ma.loc[prev_date, ticker]

                    # 前 ma_window 天沒有均線資料時，視為不過濾（當作 eligible）
                    if pd.isna(mav_prev) or (px_prev > mav_prev):
                        eligible.append(ticker)

                if len(eligible) == 0:
                    # 若前一天全部跌破均線 -> 今天持有現金（維持 0）
                    continue

                # 對仍在多頭的 sector，依 base_weights 比例重新 normalize 成加總 1
                total = sum(base_weights[t] for t in eligible)
                for ticker in eligible:
                    self.portfolio_weights.loc[date, ticker] = (
                        base_weights[ticker] / total
                    )

        # 確保被排除的 SPY 權重永遠是 0
        self.portfolio_weights[self.exclude] = 0.0
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
