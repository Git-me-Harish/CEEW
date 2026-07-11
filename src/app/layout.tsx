import type { Metadata } from "next";
import { Inter, Montserrat } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const montserrat = Montserrat({
  variable: "--font-montserrat",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "PashuMitra Indian Bovine Intelligence Platform",
  description:
    "One-stop platform for Indian cattle & bovine management: AI breed classification, breed encyclopedia, health & vaccination, nutrition, milk tracking, mandi prices, and government schemes for farmers.",
  keywords: [
    "Indian bovine",
    "cattle classification",
    "Gir cow",
    "Sahiwal",
    "indigenous breeds",
    "dairy farming India",
    "bovine health",
    "cattle vaccination",
    "PashuMitra",
  ],
  authors: [{ name: "PashuMitra Team" }],
  openGraph: {
    title: "PashuMitra — Indian Bovine Intelligence Platform",
    description:
      "AI-powered end-to-end platform for Indian cattle rearers, dairy farmers and veterinarians.",
    siteName: "PashuMitra",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${montserrat.variable} antialiased bg-background text-foreground font-sans`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
