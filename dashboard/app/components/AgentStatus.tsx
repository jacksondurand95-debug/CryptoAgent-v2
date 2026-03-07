"use client";

interface Props {
  lastUpdate: string | number;
  source: string;
  pendingOrders: number;
  openPositions: number;
  intel: {
    composite_score: number;
    bias: string;
    summary: string;
    timestamp: string;
  } | null;
}

function formatLastUpdate(v: string | number): string {
  if (typeof v === "string") {
    return new Date(v).toLocaleString();
  }
  return new Date(v * 1000).toLocaleString();
}

function sourceLabel(source: string): string {
  switch (source) {
    case "coinbase_full":
      return "Coinbase (full account)";
    case "coinbase_bot":
      return "Coinbase (bot-managed)";
    default:
      return "Estimated";
  }
}

function biasColor(bias: string): string {
  if (bias === "bullish") return "var(--green)";
  if (bias === "bearish") return "var(--red)";
  return "var(--yellow)";
}

export default function AgentStatus({
  lastUpdate,
  source,
  pendingOrders,
  openPositions,
  intel,
}: Props) {
  return (
    <div
      className="card mb-4 flex flex-wrap items-center gap-4"
      style={{ padding: "0.75rem 1.25rem" }}
    >
      <div className="flex items-center gap-2">
        <div className="pulse-dot" />
        <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
          Agent Active
        </span>
      </div>

      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
        Last update: {formatLastUpdate(lastUpdate)}
      </div>

      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
        Source: {sourceLabel(source)}
      </div>

      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
        {openPositions} open position{openPositions !== 1 ? "s" : ""}
        {pendingOrders > 0 && ` / ${pendingOrders} pending`}
      </div>

      {intel && (
        <div
          style={{
            fontSize: "0.75rem",
            marginLeft: "auto",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span style={{ color: "var(--text-muted)" }}>Intel:</span>
          <span
            style={{
              color: biasColor(intel.bias),
              fontWeight: 600,
            }}
          >
            {intel.bias.toUpperCase()} ({intel.composite_score > 0 ? "+" : ""}
            {intel.composite_score})
          </span>
        </div>
      )}
    </div>
  );
}
