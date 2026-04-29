import type { Metadata } from "next";
import { Poppins, Roboto } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
// DataProvider removed, aether-specific, not needed for GalamseyWatch

const poppins = Poppins({
  weight: ["300", "400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-poppins",
});

const roboto = Roboto({
  weight: ["300", "400", "500", "700"],
  subsets: ["latin"],
  variable: "--font-roboto",
});

export const metadata: Metadata = {
  title: "GalamseyWatch, AI-Powered Illegal Mining Detection",
  description: "Detect illegal gold mining in Ghana from Sentinel-2 satellite imagery using a vision-language model running in your browser.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // suppressHydrationWarning: some browser extensions (Brave Shields,
    // Dashlane, etc.) inject attributes like webcrx="" on <html> before React
    // hydrates, causing spurious mismatches. This tag only silences attribute
    // diffs one level deep, actual hydration bugs in our components still surface.
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${poppins.variable} ${roboto.variable} font-sans antialiased`}
      >
        {children}
        <Analytics />
      </body>
    </html>
  );
}
