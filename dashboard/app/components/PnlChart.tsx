"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface Trade {
  pnl_usd: number;
  fees_usd: number;
  closed_at_str: string;
}

interface Props {
  trades: Trade[];
  startingValue: number;
}

export default function PnlChart({ trades, startingValue }: Props) {
  if (trades.length === 0) return null;

  // Build cumulative P&L curve
  let cumulative = startingValue;
  const chartData = [
    { label: "Start", value: startingValue, pnl: 0 },
    ...trades.map((t, i) => {
      cumulative += t.pnl_usd - t.fees_usd;
      return {
        label: t.closed_at_str || `Trade ${i + 1}`,
        value: parseFloat(cumulative.toFixed(2)),
        pnl: parseFloat((cumulative - startingValue).toFixed(2)),
      };
    }),
  ];

  const minVal = Math.min(...chartData.map((d) => d.value));
  const maxVal = Math.max(...chartData.map((d) => d.value));
  const padding = (maxVal - minVal) * 0.1 || 50;
  const currentValue = chartData[chartData.length - 1].value;
  const isPositive = currentValue >= startingValue;

  return (
    <div className="card mb-4">
      <div className="stat-label mb-3">Portfolio Value Over Time</div>
      <div style={{ width: "100%", height: 220 }}>
        <ResponsiveContainer>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={isPositive ? "#22c55e" : "#ef4444"}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={isPositive ? "#22c55e" : "#ef4444"}
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fill: "#71717a" }}
              axisLine={{ stroke: "#2a2a3e" }}
              tickLine={false}
            />
            <YAxis
              domain={[minVal - padding, maxVal + padding]}
              tick={{ fontSize: 10, fill: "#71717a" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{
                background: "#1a1a2e",
                border: "1px solid #2a2a3e",
                borderRadius: 8,
                fontSize: "0.8rem",
                color: "#e4e4e7",
              }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, "Value"]}
              labelStyle={{ color: "#71717a" }}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke={isPositive ? "#22c55e" : "#ef4444"}
              strokeWidth={2}
              fill="url(#pnlGrad)"
              dot={{ r: 3, fill: isPositive ? "#22c55e" : "#ef4444" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
