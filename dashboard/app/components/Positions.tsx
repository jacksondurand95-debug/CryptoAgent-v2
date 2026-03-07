"use client";

interface Position {
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
}

interface Props {
  positions: Position[];
}

function fmt(n: number, decimals = 2): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function timeAgo(ts: number): string {
  const diff = (Date.now() / 1000 - ts) / 3600;
  if (diff < 1) return `${Math.round(diff * 60)}m ago`;
  if (diff < 24) return `${Math.round(diff)}h ago`;
  return `${Math.round(diff / 24)}d ago`;
}

export default function Positions({ positions }: Props) {
  return (
    <div className="card mb-4">
      <div className="stat-label mb-3">
        Open Positions ({positions.length})
      </div>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Pair</th>
              <th>Side</th>
              <th>Entry</th>
              <th>Current</th>
              <th>Size</th>
              <th>Unrealized P&L</th>
              <th>Opened</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p, i) => {
              const isUp = p.unrealizedPct >= 0;
              return (
                <tr key={i}>
                  <td style={{ fontWeight: 600 }}>
                    {p.pair.replace("-USD", "")}
                  </td>
                  <td>
                    <span
                      className={`badge ${p.side === "long" ? "badge-long" : "badge-short"}`}
                    >
                      {p.side}
                    </span>
                  </td>
                  <td>${fmt(p.entry_price)}</td>
                  <td>${fmt(p.currentPrice)}</td>
                  <td>${fmt(p.usd_amount)}</td>
                  <td className={isUp ? "positive" : "negative"}>
                    {isUp ? "+" : ""}${fmt(p.unrealizedUsd)} (
                    {isUp ? "+" : ""}
                    {fmt(p.unrealizedPct)}%)
                  </td>
                  <td style={{ color: "var(--text-muted)" }}>
                    {timeAgo(p.opened_at)}
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
