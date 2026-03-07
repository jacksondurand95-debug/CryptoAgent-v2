import { readFile } from "fs/promises";
import { resolve } from "path";

const COINS = [
  "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK",
  "PEPE", "SHIB", "SUI", "NEAR", "RENDER", "FET",
  "INJ", "TIA", "SEI", "WIF",
];

async function getCoinbasePrices(): Promise<Record<string, number>> {
  const prices: Record<string, number> = {};
  try {
    const results = await Promise.allSettled(
      COINS.map((c) =>
        fetch(`https://api.coinbase.com/v2/prices/${c}-USD/spot`, {
          cache: "no-store",
        }).then((r) => r.json())
      )
    );
    results.forEach((r, i) => {
      if (r.status === "fulfilled" && r.value?.data?.amount) {
        prices[COINS[i]] = parseFloat(r.value.data.amount);
      }
    });
  } catch {
    // Prices will be empty, positions fall back to entry price
  }
  return prices;
}

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // Read state.json from the repo root (one level up from dashboard/)
    const statePath = resolve(process.cwd(), "..", "state.json");
    const state = JSON.parse(await readFile(statePath, "utf8"));
    const prices = await getCoinbasePrices();

    // Read intel aggregate if available
    let intel = null;
    try {
      const intelPath = resolve(process.cwd(), "..", "intel", "data", "aggregate.json");
      intel = JSON.parse(await readFile(intelPath, "utf8"));
    } catch {
      // Intel data may not exist
    }

    const positions = (state.positions || []).map(
      (p: Record<string, unknown>) => {
        const coin = (p.pair as string).replace("-USD", "");
        const currentPrice =
          prices[coin] || (p.entry_price as number);
        const unrealizedPct =
          ((currentPrice - (p.entry_price as number)) /
            (p.entry_price as number)) *
          100 *
          (p.side === "long" ? 1 : -1);
        const unrealizedUsd =
          ((p.usd_amount as number) * unrealizedPct) / 100;
        return { ...p, currentPrice, unrealizedPct, unrealizedUsd };
      }
    );

    const trades = state.trades || [];
    const totalPnl = trades.reduce(
      (s: number, t: Record<string, unknown>) =>
        s + ((t.pnl_usd as number) || 0),
      0
    );
    const totalFees = trades.reduce(
      (s: number, t: Record<string, unknown>) =>
        s + ((t.fees_usd as number) || 0),
      0
    );
    const wins = trades.filter(
      (t: Record<string, unknown>) => (t.pnl_usd as number) > 0
    ).length;
    const startingValue = state.starting_value || 0;

    const portfolioValue =
      state.account_balance ||
      state.last_portfolio_value ||
      startingValue + totalPnl - totalFees;

    return Response.json({
      timestamp: Date.now(),
      portfolio: {
        value: portfolioValue,
        startingValue,
        totalPnl,
        totalFees,
        netPnl: portfolioValue - startingValue,
        pnlPct:
          ((portfolioValue - startingValue) / startingValue) * 100,
        lastUpdate:
          state.last_portfolio_update || state.last_updated,
        source: state.account_balance
          ? "coinbase_full"
          : state.last_portfolio_value
            ? "coinbase_bot"
            : "estimated",
      },
      positions,
      trades,
      prices,
      intel,
      stats: {
        totalTrades: trades.length,
        wins,
        losses: trades.length - wins,
        winRate:
          trades.length > 0 ? (wins / trades.length) * 100 : 0,
        avgPnl: trades.length > 0 ? totalPnl / trades.length : 0,
        avgFees: trades.length > 0 ? totalFees / trades.length : 0,
        detectedFees: state.detected_fees || null,
        dailyLosses: state.daily_losses || [],
      },
      pendingOrders: state.pending_orders || [],
    });
  } catch (err) {
    return Response.json(
      { error: (err as Error).message },
      { status: 500 }
    );
  }
}
