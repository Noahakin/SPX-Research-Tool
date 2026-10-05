# SPX Research Interactive

Open **index.html** in current Chrome or Edge. No server or internet connection is needed. Keep the entire folder together.

The website opens with only the SPX benchmark. Load a saved setup explicitly to restore a comparison. The chart’s upper-left panel shows CAGR, annualized volatility, maximum drawdown and Sharpe for the visible portfolios; with-SPX lines also show differences versus SPX over the same dates.

Select any number of option strategies, choose Options only / With SPX / Both for each, and adjust the date range. Option-expiry filters are separate from the historical date window. All lines are independent portfolios.

The site includes 7,350 original strategies plus 22,050 profit-taking variants: 29,400 strategies and 58,800 options-only/SPX combinations, plus SPX as a benchmark. All 8,820 existing chart curves are represented. Data through September 18, 2026.

SPX overlays use fixed quantities between option rolls and resize the complete portfolio at the next roll. SPX remains invested during option cash cycles. These are 100% SPX price exposure plus 100% options notional, not a 50/50 mix or a sum of compounded equity curves. No dividends, financing, taxes or interest are modeled.

Time-window metrics include returns from the close before the first selected session; the original first session uses initial capital. Existing positions are not restarted when you change the date range. Buffers retain entry-date affordability selection; long-put targets remain retrospective research winners.

Rebuild: `python -B spx_option_research/scripts/build_interactive_spx.py`. Refresh only the interface: add `--assets-only`.

Select a **Strategy folder** for hold to expiration or profit taking at **25%, 50%, or 75%**. All option legs close together at the net-profit target, followed by a fresh entry at the same daily snapshot. Put buying measures profit against the initial debit; Put selling and Both use retained net credit. SPX remains invested. Entry and early exit both include costs. See [Protocol.json](profit-data/Protocol.json) for the full rules.
