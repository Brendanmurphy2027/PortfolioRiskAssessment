import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# -----------------------------
# USER PORTFOLIO INPUT
# -----------------------------
def get_user_portfolio():
    tickers = []
    shares = []
    dca = []
    current_prices = []

    n = int(input("How many stocks do you own? "))

    for i in range(n):
        while True:
            t = input(f"\nEnter ticker {i+1}: ").strip().upper()
            try:
                price_data = yf.Ticker(t).history(period="1d")
                if price_data.empty:
                    raise ValueError("No price data found")
                current_price = price_data["Close"].iloc[-1]
                print(f"Current price for {t}: ${current_price:.2f}")
                break
            except Exception as e:
                print(f"⚠ Could not fetch price for {t}: {e}. Try again.")

        tickers.append(t)
        current_prices.append(current_price)

        s = float(input(f"How many shares of {t} do you own? "))
        shares.append(s)

        c = float(input(f"What is your average cost (DCA) for {t}? "))
        dca.append(c)

    projection_years = int(input("\nHow many years do you want projected? "))

    return tickers, shares, dca, current_prices, projection_years


# -----------------------------
# DOWNLOAD HISTORICAL DATA
# -----------------------------
def get_historical_prices(tickers, years=5):
    data = yf.download([t.strip().upper() for t in tickers], period=f"{years}y", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    valid_tickers = [t for t in tickers if t in data.columns]
    invalid_tickers = [t for t in tickers if t not in data.columns]

    if invalid_tickers:
        print("\n⚠ WARNING — Failed tickers removed:", invalid_tickers)

    data = data[valid_tickers].dropna()
    return data, valid_tickers


# -----------------------------
# RISK METRICS
# -----------------------------
def calculate_risk_metrics(prices, weights):
    returns = prices.pct_change().dropna()
    portfolio_returns = returns.dot(weights)

    daily_return = portfolio_returns.mean()
    daily_volatility = portfolio_returns.std()
    annual_return = (1 + daily_return) ** 252 - 1
    annual_volatility = daily_volatility * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

    # Max drawdown
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # VaR and CVaR
    historical_var = portfolio_returns.quantile(0.05)
    parametric_var = portfolio_returns.mean() - 1.65 * portfolio_returns.std()
    cvar = portfolio_returns[portfolio_returns <= historical_var].mean()

    metrics = {
        "daily_return": daily_return,
        "daily_volatility": daily_volatility,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "historical_var": historical_var,
        "parametric_var": parametric_var,
        "cvar": cvar,
        "portfolio_returns": portfolio_returns
    }

    return metrics


# -----------------------------
# MONTE CARLO SIMULATION
# -----------------------------
def monte_carlo_simulation(initial_value, mean, std, years, sims=500):
    days = years * 252
    results = np.zeros((sims, days))
    for i in range(sims):
        daily_returns = np.random.normal(mean, std, days)
        results[i] = initial_value * np.cumprod(1 + daily_returns)
    return results


# -----------------------------
# REPORT GENERATOR
# -----------------------------
def save_report(filename, tickers, weights, shares, dca, metrics, initial_value, final_values):
    with open(filename, "w") as f:
        f.write("PORTFOLIO RISK REPORT\n")
        f.write("=======================\n\n")
        f.write(f"Starting Portfolio Value: ${initial_value:,.2f}\n\n")

        f.write("Tickers & Weights:\n")
        for t, w in zip(tickers, weights):
            f.write(f" - {t}: {w*100:.2f}%\n")

        f.write("\nPositions:\n")
        for t, s, c in zip(tickers, shares, dca):
            f.write(f" - {t}: {s} shares, DCA ${c:.2f}\n")

        f.write("\nRISK METRICS\n")
        for k in ["annual_return","annual_volatility","sharpe_ratio","max_drawdown","historical_var","parametric_var","cvar"]:
            f.write(f"{k.replace('_',' ').title()}: {metrics[k]:.4f}\n")

        f.write("\nMONTE CARLO PROJECTION\n")
        f.write(f"Worst-case (5th percentile): ${np.percentile(final_values,5):,.2f}\n")
        f.write(f"Median (50th percentile): ${np.percentile(final_values,50):,.2f}\n")
        f.write(f"Best-case (95th percentile): ${np.percentile(final_values,95):,.2f}\n")

    print(f"\n📄 Report saved to {filename}")


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    print("\n====== PORTFOLIO RISK ANALYSIS ======\n")
    tickers, shares, dca, current_prices, projection_years = get_user_portfolio()

    prices, valid_tickers = get_historical_prices(tickers)
    if len(valid_tickers) == 0:
        print("No valid tickers downloaded. Exiting.")
        return

    idx = [tickers.index(t) for t in valid_tickers]
    shares = [shares[i] for i in idx]
    dca = [dca[i] for i in idx]
    current_prices = [current_prices[i] for i in idx]

    market_values = np.array(current_prices) * np.array(shares)
    initial_value = market_values.sum()
    weights = market_values / initial_value

    metrics = calculate_risk_metrics(prices, weights)

    mc_results = monte_carlo_simulation(initial_value, metrics["daily_return"], metrics["daily_volatility"], projection_years)
    final_values = mc_results[:, -1]

    today = datetime.date.today().strftime("%Y-%m-%d")
    save_report(f"portfolio_risk_report_{today}.txt", valid_tickers, weights, shares, dca, metrics, initial_value, final_values)

    # -----------------------------
    # THREE-LINE GRAPH (Worst, Median, Best)
    # -----------------------------
    weeks = np.arange(mc_results.shape[1]) / 5  # approx 5 trading days per week
    worst = np.percentile(mc_results, 5, axis=0)
    median = np.percentile(mc_results, 50, axis=0)
    best = np.percentile(mc_results, 95, axis=0)

    plt.figure(figsize=(10,6))
    plt.plot(weeks, worst, label="5th Percentile (Worst Case)", color='red')
    plt.plot(weeks, median, label="50th Percentile (Median)", color='blue')
    plt.plot(weeks, best, label="95th Percentile (Best Case)", color='green')
    plt.title(f"Portfolio Dollar Growth Over {projection_years} Years")
    plt.xlabel("Weeks")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.legend()
    plt.savefig("portfolio_projection.png", dpi=300)
    plt.show()
    print("📈 Graph saved as portfolio_projection.png")


if __name__ == "__main__":
    main()


