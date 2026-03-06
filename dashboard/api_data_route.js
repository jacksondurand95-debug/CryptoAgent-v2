import { readFile } from "fs/promises";
import { execSync } from "child_process";

const STATE_PATH = "/tmp/CryptoAgent-v2/state.json";
const COINS = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "PEPE", "SUI", "NEAR", "INJ", "SEI", "WIF"];

async function getCoinbasePrices() {
  const prices = {};
  try {
    const results = await Promise.allSettled(
      COINS.map((c) =>
        fetch(`https://api.coinbase.com/v2/prices/${c}-USD/spot`, { cache: "no-store" })
          .then((r) => r.json())
      )
    );
    results.forEach((r, i) => {
      if (r.status === "fulfilled" && r.value?.data?.amount) {
        prices[COINS[i]] = parseFloat(r.value.data.amount);
      }
    });
  } catch {}
  return prices;
}

export async function GET() {
  try {
    // Pull latest state
    try { execSync("cd /tmp/CryptoAgent-v2 && git pull --ff-only 2>/dev/null", { timeout: 8000 }); } catch {}

    const state = JSON.parse(await readFile(STATE_PATH, "utf8"));
    const prices = await getCoinbasePrices();

    const positions = (state.positions || []).map((p) => {
      const coin = p.pair.replace("-USD", "");
      const currentPrice = prices[coin] || p.entry_price;
      const unrealizedPct = ((currentPrice - p.entry_price) / p.entry_price) * 100 * (p.side === "long" ? 1 : -1);
      const unrealizedUsd = (p.usd_amount * unrealizedPct) / 100;
      return { ...p, currentPrice, unrealizedPct, unrealizedUsd };
    });

    const trades = state.trades || [];
    const totalPnl = trades.reduce((s, t) => s + (t.pnl_usd || 0), 0);
    const totalFees = trades.reduce((s, t) => s + (t.fees_usd || 0), 0);
    const wins = trades.filter((t) => t.pnl_usd > 0).length;
    const startingValue = state.starting_value || 0;

    // Full Coinbase account balance from transaction_summary endpoint
    const portfolioValue = state.account_balance || state.last_portfolio_value || (startingValue + totalPnl - totalFees);

    return Response.json({
      timestamp: Date.now(),
      portfolio: {
        value: portfolioValue,
        startingValue,
        totalPnl,
        totalFees,
        netPnl: portfolioValue - startingValue,
        pnlPct: ((portfolioValue - startingValue) / startingValue) * 100,
        lastUpdate: state.last_portfolio_update || state.last_updated,
        source: state.account_balance ? "coinbase_full" : state.last_portfolio_value ? "coinbase_bot" : "estimated",
      },
      positions,
      trades,
      prices,
      stats: {
        totalTrades: trades.length,
        wins,
        losses: trades.length - wins,
        winRate: trades.length > 0 ? (wins / trades.length) * 100 : 0,
        avgPnl: trades.length > 0 ? totalPnl / trades.length : 0,
        avgFees: trades.length > 0 ? totalFees / trades.length : 0,
      },
    });
  } catch (err) {
    return Response.json({ error: err.message }, { status: 500 });
  }
}
