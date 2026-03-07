# CryptoAgent v2 — Free Serverless Trading Agent

Ultra-aggressive serverless crypto trading agent running 100% free on GitHub Actions.

## Architecture

**Completely Serverless** — No servers, no costs, runs forever free on GitHub Actions:
- Public repository = Unlimited GitHub Actions minutes
- Workflows execute every 10 minutes (144 runs/day)
- Intel sub-agents gather multi-source market data every 3 hours
- State persisted to `state.json` (committed by GitHub Actions)
- Secrets stored safely in GitHub Secrets (encrypted)

## Features

### Multi-Strategy Trading
- **Technical Analysis**: RSI, MACD, Bollinger Bands, momentum, volume
- **TradingView Integration**: Real-time consensus signals
- **AI Decision Making**: Dual-brain (Claude + Grok) for trade validation
- **Risk Management**: ATR-based stops, trailing stops, position sizing

### Intel Sub-Agents
Five specialized agents gather market intelligence:
1. **Liquidations** (every 10 min) — Cascade risk detection across exchanges
2. **Whale Tracker** (every 30 min) — Large holder movements (BTC/ETH)
3. **Sentiment** (every 30 min) — Social sentiment + Fear & Greed Index
4. **News** (every 30 min) — Alpha events from crypto news feeds
5. **On-Chain** (every 2 hours) — Stablecoin flows, TVL, DeFi metrics

### Multi-Exchange Data
- **Derivatives Intel**: Bybit + Binance + OKX funding rates, OI, liquidations
- **Spot Execution**: Coinbase Advanced Trade (limit orders only)
- **DeFi Data**: DEX volumes, TVL, stablecoin dominance

## Setup

### 1. Fork This Repository
Make it **public** for unlimited free GitHub Actions minutes.

### 2. Configure GitHub Secrets
Go to Settings → Secrets and variables → Actions → New repository secret:

```
COINBASE_KEY_JSON       - Coinbase API key JSON (from CDP portal)
ANTHROPIC_API_KEY       - Claude API key (for AI analysis)
XAI_API_KEY            - xAI API key (optional - Grok for second opinion)
CRYPTOPANIC_API_KEY    - CryptoPanic key (optional - news sentiment)
GROQ_API_KEY           - Groq API key (optional - fast inference)
```

### 3. Enable GitHub Actions
- Go to Actions tab → Enable workflows
- The agent starts automatically on next schedule

### 4. Configure Trading Parameters
Edit `config.py` to adjust:
- Trading pairs (default: BTC, ETH, SOL, DOGE, etc.)
- Risk limits (position size, max drawdown, daily loss)
- Stop loss / take profit levels
- Minimum confidence threshold

## Workflows

### Main Trading Agent (`trade.yml`)
- **Schedule**: Every 10 minutes
- **Actions**: Fetch prices → Analyze signals → Execute trades → Manage positions
- **Runtime**: ~45 seconds per run
- **Monthly cost**: $0 (unlimited on public repos)

### Intel Sub-Agents (`intel-*.yml`)
- **Combined workflow** (`intel-combined.yml`): Every 3 hours
- Runs all 5 intel agents sequentially
- Commits results to `intel/data/*.json`
- Trading agent reads latest intel on each run

## Cost Breakdown

**Total Monthly Cost: $0**
- GitHub Actions: Free unlimited (public repo)
- Anthropic Claude Haiku: ~$0.50/month (150K tokens/day)
- Coinbase trading: Pay per trade (not hosting)

## Risk Management

### Fee Optimization
- **Detected fees**: 0.4% maker / 0.8% taker
- **Strategy**: Limit orders only (never market) to minimize fees
- **Minimum move**: 4% to overcome round-trip fees
- **Higher conviction**: 0.65 confidence threshold

### Position Limits
- Max 2 positions open simultaneously
- 20% of portfolio per trade
- 2.5x ATR stop loss (wider to avoid fee churn)
- 10x ATR take profit (bigger wins vs fees)
- 72-hour time stop (reduce turnover)

### Safety Features
- 40% max drawdown halt
- 10% daily loss limit
- 5 consecutive losses → 1 hour pause
- Auto fee detection (overrides config if different)

## Monitoring

### Check Status
- **GitHub Actions** tab shows all workflow runs
- **state.json** shows current positions and P&L
- **intel/data/** shows latest market intelligence

### Portfolio Value
```bash
cat state.json | grep -E '(account_balance|last_portfolio_value)'
```

### Recent Trades
```bash
cat state.json | python -m json.tool | grep -A 20 '"trades"'
```

## Deployment

The agent is **already deployed** and running. To redeploy after changes:

1. Commit your changes to the repository
2. Push to GitHub
3. Workflows auto-update on next scheduled run

Or trigger manually:
- Go to Actions → Select workflow → Run workflow

## Performance

### Current Status
- **Starting Value**: $3,000
- **Current Balance**: Check `state.json`
- **Total Trades**: 6 completed
- **Win Rate**: Monitor in state file

### Optimization Notes
- Recent trades showed net losses due to fees
- Config optimized for high fee environment (0.4-0.8%)
- Increased minimum move threshold from 3% to 4%
- Reduced position frequency with higher confidence bar

## Architecture Details

### State Management
- `state.json` persisted via git commits after each run
- Contains positions, trade history, P&L, detected fees
- Survives workflow crashes (committed on every run)

### Concurrency Control
- Single trading group → no overlapping trades
- Intel workflows can run in parallel
- Aggregator merges all intel sources before trading

### Error Handling
- API failures → graceful skip (no trades)
- Missing intel → continues with available data
- Order failures → logged, state preserved

## Security

- API keys never committed to code
- Secrets encrypted by GitHub
- Coinbase key written to file only during execution
- Cleaned up immediately after run (even on failure)

## Free Forever

This setup costs **$0/month** because:
1. Public repo = unlimited GitHub Actions
2. All intel sources are free public APIs
3. AI inference costs < $1/month
4. No server hosting fees

## Contributing

This is an open-source trading agent. Improvements welcome:
- Better indicators or strategies
- Additional intel sources
- Risk management enhancements
- Bug fixes

## Disclaimer

**Use at your own risk.** This bot trades real money. Past performance ≠ future results. Crypto is volatile. Only trade what you can afford to lose.

## License

MIT License - Use freely, no warranty provided.
