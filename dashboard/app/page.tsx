"use client";

import { useEffect, useState, useCallback } from "react";
import PortfolioOverview from "./components/PortfolioOverview";
import PnlChart from "./components/PnlChart";
import Positions from "./components/Positions";
import TradeHistory from "./components/TradeHistory";
import PriceGrid from "./components/PriceGrid";
import AgentStatus from "./components/AgentStatus";

interface DashboardData {
  timestamp: number;
  portfolio: {
    value: number;
    startingValue: number;
    totalPnl: number;
    totalFees: number;
    netPnl: number;
    pnlPct: number;
    lastUpdate: string | number;
    source: string;
  };
  positions: Array<{
    pair: string;
    side: string;
    entry_price: number;
    qty: number;
    usd_amount: number;
    currentPrice: number;
    unrealizedPct: number;
    unrealizedUsd: number;
    opened_at: number;
    highest_price_seen: number;
    reason: string;
  }>;
  trades: Array<{
    pair: string;
    side: string;
    entry_price: number;
    exit_price: number;
    qty: number;
    usd_amount: number;
    pnl_pct: number;
    pnl_usd: number;
    fees_usd: number;
    reason: string;
    closed_at_str: string;
    duration_min: number;
    trailing_was_active: boolean;
  }>;
  prices: Record<string, number>;
  intel: {
    composite_score: number;
    bias: string;
    summary: string;
    sources: Record<string, unknown>;
    timestamp: string;
  } | null;
  stats: {
    totalTrades: number;
    wins: number;
    losses: number;
    winRate: number;
    avgPnl: number;
    avgFees: number;
    detectedFees: { maker: number; taker: number } | null;
    dailyLosses: Array<{ usd: number; time: number }>;
  };
  pendingOrders: Array<Record<string, unknown>>;
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch("/api/data", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      setData(json);
      setError(null);
      setLastRefresh(new Date());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60_000); // refresh every 60s
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading && !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="pulse-dot mx-auto mb-4" />
          <p style={{ color: "var(--text-muted)" }}>Loading agent data...</p>
        </div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="card text-center" style={{ maxWidth: 400 }}>
          <p style={{ color: "var(--red)", marginBottom: 8 }}>Error</p>
          <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
            {error}
          </p>
          <button className="refresh-btn mt-4" onClick={fetchData}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "1.5rem" }}>
      {/* Header */}
      <div
        className="flex items-center justify-between mb-6"
        style={{ borderBottom: "1px solid var(--border)", paddingBottom: "1rem" }}
      >
        <div>
          <h1
            style={{
              fontSize: "1.25rem",
              fontWeight: 700,
              letterSpacing: "-0.02em",
            }}
          >
            CryptoAgent
          </h1>
          <p style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
            v3.1 — Quant + LLM Trading Engine
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            style={{ color: "var(--text-muted)", fontSize: "0.7rem" }}
          >
            {lastRefresh.toLocaleTimeString()}
          </span>
          <button className="refresh-btn" onClick={fetchData}>
            {loading ? "..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Agent Status Bar */}
      <AgentStatus
        lastUpdate={data.portfolio.lastUpdate}
        source={data.portfolio.source}
        pendingOrders={data.pendingOrders.length}
        openPositions={data.positions.length}
        intel={data.intel}
      />

      {/* Portfolio Overview Cards */}
      <PortfolioOverview
        portfolio={data.portfolio}
        stats={data.stats}
      />

      {/* P&L Chart */}
      <PnlChart trades={data.trades} startingValue={data.portfolio.startingValue} />

      {/* Open Positions */}
      {data.positions.length > 0 && (
        <Positions positions={data.positions} />
      )}

      {/* Trade History */}
      <TradeHistory trades={data.trades} />

      {/* Market Prices */}
      <PriceGrid prices={data.prices} />

      {/* Footer */}
      <div
        className="mt-8 text-center"
        style={{
          color: "var(--text-muted)",
          fontSize: "0.7rem",
          padding: "1rem 0",
        }}
      >
        Data refreshes every 60s — Agent runs every 10min via GitHub Actions
        {data.stats.detectedFees && (
          <span>
            {" "}
            — Fees: {(data.stats.detectedFees.maker * 100).toFixed(1)}% maker /{" "}
            {(data.stats.detectedFees.taker * 100).toFixed(1)}% taker
          </span>
        )}
      </div>
    </div>
  );
}
