import type { Metadata } from "next";
import { Shell } from "@/components/shell";
import "./globals.css";

export const metadata: Metadata = {title: "ImpactAI — Performance Intelligence", description: "Explainable performance analysis of historical professional VALORANT matches, 2020–2022."};
export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body><Shell>{children}</Shell></body></html>;
}
