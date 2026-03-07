"use client";

interface Trade {
  pair: string;
  side: string;
  entry_price: number;
  exit_price: number;
  usd_amount: number;
  pnl_pct: number;
  pnl_usd: number;
  fees_usd: number;
  reason: string;
  closed_at_str: string;
  duration_min: number;
  trailing_was_active: boolean;
}

interface Props {
  trades: Trade[];
}

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtDuration(mins: number): string {
  if (mins < 60) return `${Math.round(mins)}m`;
  if (mins < 1440) return `${(mins / 60).toFixed(1)}h`;
  return `${(mins / 1440).toFixed(1)}d`;
}

function parseReason(reason: string): string {
  if (reason.startsWith("claude:")) {
    return reason.slice(7).slice(0, 80) + (reason.length > 87 ? "..." : "");
  }
  return reason;
}

export default function TradeHistory({ trades }: Props) {
  if (trades.length === 0) {
    return (
      <div className="card mb-4">
        <div className="stat-label mb-3">Trade History</div>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
          No trades yet.
        </p>
      </div>
    );
  }

  // Show newest first
  const sorted = [...trades].reverse();

  return (
    <div className="card mb-4">
      <div className="stat-label mb-3">Trade History ({trades.length})</div>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Pair</th>
              <th>Side</th>
              <th>Entry</th>
              <th>Exit</th>
              <th>Size</th>
              <th>P&L</th>
              <th>Fees</th>
              <th>Duration</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((t, i) => {
              const isWin = t.pnl_usd > 0;
              const netPnl = t.pnl_usd - t.fees_usd;
              return (
                <tr key={i}>
                  <td style={{ color: "var(--text-muted)", whiteSpace: "nowrap" }}>
                    {t.closed_at_str}
                  </td>
                  <td style={{ fontWeight: 600 }}>
                    {t.pair.replace("-USD", "")}
                  </td>
                  <td>
                    <span
                      className={`badge ${t.side === "long" ? "badge-long" : "badge-short"}`}
                    >
                      {t.side}
                    </span>
                  </td>
                  <td>${fmt(t.entry_price)}</td>
                  <td>${fmt(t.exit_price)}</td>
                  <td>${fmt(t.usd_amount)}</td>
                  <td className={isWin ? "positive" : "negative"}>
                    {netPnl >= 0 ? "+" : ""}${fmt(netPnl)}
                    <span
                      style={{
                        fontSize: "0.7rem",
                        color: "var(--text-muted)",
                        marginLeft: 4,
                      }}
                    >
                      ({t.pnl_pct >= 0 ? "+" : ""}{fmt(t.pnl_pct)}%)
                    </span>
                  </td>
                  <td style={{ color: "var(--red)" }}>${fmt(t.fees_usd)}</td>
                  <td style={{ color: "var(--text-muted)" }}>
                    {fmtDuration(t.duration_min)}
                    {t.trailing_was_active && (
                      <span
                        style={{
                          color: "var(--yellow)",
                          marginLeft: 4,
                          fontSize: "0.7rem",
                        }}
                        title="Trailing stop was active"
                      >
                        TS
                      </span>
                    )}
                  </td>
                  <td
                    style={{
                      color: "var(--text-muted)",
                      fontSize: "0.75rem",
                      maxWidth: 200,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={t.reason}
                  >
                    {parseReason(t.reason)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
