"use client";

interface Props {
  portfolio: {
    value: number;
    startingValue: number;
    totalPnl: number;
    totalFees: number;
    netPnl: number;
    pnlPct: number;
  };
  stats: {
    totalTrades: number;
    wins: number;
    losses: number;
    winRate: number;
    avgPnl: number;
    avgFees: number;
  };
}

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtUsd(n: number): string {
  return `$${fmt(Math.abs(n))}`;
}

export default function PortfolioOverview({ portfolio, stats }: Props) {
  const isPositive = portfolio.netPnl >= 0;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
      {/* Account Value */}
      <div className="card">
        <div className="stat-label">Account Value</div>
        <div className="stat-value">${fmt(portfolio.value)}</div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          Started at ${fmt(portfolio.startingValue)}
        </div>
      </div>

      {/* Net P&L */}
      <div className="card">
        <div className="stat-label">Net P&L</div>
        <div className={`stat-value ${isPositive ? "positive" : "negative"}`}>
          {isPositive ? "+" : "-"}{fmtUsd(portfolio.netPnl)}
        </div>
        <div
          className={isPositive ? "positive" : "negative"}
          style={{ fontSize: "0.8rem" }}
        >
          {isPositive ? "+" : ""}{fmt(portfolio.pnlPct)}%
        </div>
      </div>

      {/* Win Rate */}
      <div className="card">
        <div className="stat-label">Win Rate</div>
        <div className="stat-value">{fmt(stats.winRate, 0)}%</div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          {stats.wins}W / {stats.losses}L — {stats.totalTrades} trades
        </div>
      </div>

      {/* Fees */}
      <div className="card">
        <div className="stat-label">Total Fees Paid</div>
        <div className="stat-value negative">${fmt(portfolio.totalFees)}</div>
        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
          Avg ${fmt(stats.avgFees)} / trade
        </div>
      </div>
    </div>
  );
}
