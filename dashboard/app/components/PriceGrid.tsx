"use client";

interface Props {
  prices: Record<string, number>;
}

function fmt(n: number): string {
  if (n >= 1000) return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (n >= 1) return n.toFixed(2);
  if (n >= 0.001) return n.toFixed(4);
  return n.toFixed(8);
}

const COIN_ORDER = [
  "BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK",
  "PEPE", "SHIB", "SUI", "NEAR", "RENDER", "FET",
  "INJ", "TIA", "SEI", "WIF",
];

export default function PriceGrid({ prices }: Props) {
  const coins = COIN_ORDER.filter((c) => c in prices);
  if (coins.length === 0) return null;

  return (
    <div className="card mb-4">
      <div className="stat-label mb-3">Market Prices (Coinbase Spot)</div>
      <div className="price-grid">
        {coins.map((coin) => (
          <div key={coin} className="price-tile">
            <div
              style={{
                fontSize: "0.7rem",
                fontWeight: 700,
                color: "var(--text-muted)",
                marginBottom: 2,
              }}
            >
              {coin}
            </div>
            <div style={{ fontSize: "1rem", fontWeight: 600 }}>
              ${fmt(prices[coin])}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
