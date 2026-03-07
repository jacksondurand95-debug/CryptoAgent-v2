import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CryptoAgent Dashboard",
  description: "Live trading dashboard for CryptoAgent v3",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
