"use client";

import { useState } from "react";
import { TrendingUp, TrendingDown, Minus, IndianRupee, ExternalLink, Landmark, Shield, Building2, FileText, CheckCircle2 } from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { govtSchemes, marketPrices, type MarketPrice } from "@/data/schemes";

const categoryIcons: Record<string, React.ElementType> = {
  subsidy: IndianRupee,
  insurance: Shield,
  infrastructure: Building2,
  breeding: FileText,
  health: Shield,
  market: TrendingUp,
};

const categoryColors: Record<string, string> = {
  subsidy: "pill-green",
  insurance: "pill-blue",
  infrastructure: "pill-navy",
  breeding: "pill-amber",
  health: "pill-clay",
  market: "pill-outline",
};

export function MarketAndSchemes() {
  const [activeTab, setActiveTab] = useState<"prices" | "schemes">("prices");

  return (
    <Section bg="white">
      <SectionHeading
        eyebrow="Economics & support"
        title="Market Prices & Government Schemes"
        subtitle="Live mandi prices for milk, cattle, and fodder across India. Plus a complete directory of central government schemes, subsidies, insurance, and infrastructure funds for bovine farmers."
      />

      {/* Tabs */}
      <div className="flex items-center gap-1 mb-6 border-b border-brand-line">
        {[
          { id: "prices" as const, label: "Mandi Prices", count: marketPrices.length },
          { id: "schemes" as const, label: "Government Schemes", count: govtSchemes.length },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`inline-flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === t.id
                ? "border-brand-navy text-brand-navy"
                : "border-transparent text-slate-500 hover:text-brand-navy"
            }`}
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            {t.label}
            <span className="pill-outline text-[10px] ml-1">{t.count}</span>
          </button>
        ))}
      </div>

      {/* Prices tab */}
      {activeTab === "prices" && (
        <div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {marketPrices.map((p, i) => (
              <PriceCard key={i} price={p} />
            ))}
          </div>
          <div className="mt-5 p-4 rounded-md bg-brand-blue-50 border border-brand-blue-100 text-xs text-brand-navy">
            <strong>Disclaimer:</strong> Prices are indicative based on cooperative averages and major
            mandi reports. Actual rates vary by district, season, quality, and buyer. Verify with your
            local milk cooperative or livestock market before any transaction.
          </div>
        </div>
      )}

      {/* Schemes tab */}
      {activeTab === "schemes" && (
        <div className="grid md:grid-cols-2 gap-4">
          {govtSchemes.map((s) => {
            const Icon = categoryIcons[s.category] || FileText;
            return (
              <div key={s.id} className="card-soft p-5">
                <div className="flex items-start justify-between gap-3 mb-3">
                  <div className="flex items-start gap-3">
                    <div className="h-10 w-10 rounded-md bg-brand-blue-50 flex items-center justify-center shrink-0">
                      <Icon className="h-5 w-5 text-brand-blue" />
                    </div>
                    <div>
                      <h3
                        className="text-base font-bold text-brand-navy leading-tight"
                        style={{ fontFamily: "var(--font-montserrat)" }}
                      >
                        {s.name}
                      </h3>
                      <div className="text-xs text-slate-500 mt-0.5">{s.ministry}</div>
                    </div>
                  </div>
                  <span className={`${categoryColors[s.category]} text-[10px] capitalize shrink-0`}>
                    {s.category}
                  </span>
                </div>

                <p className="text-xs text-slate-600 leading-relaxed mb-3">{s.summary}</p>

                <div className="space-y-2 text-xs mb-3">
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mt-0.5 shrink-0 w-20">
                      Eligibility
                    </span>
                    <span className="text-slate-700">{s.eligibility}</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mt-0.5 shrink-0 w-20">
                      Benefit
                    </span>
                    <span className="text-slate-700">{s.benefit}</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mt-0.5 shrink-0 w-20">
                      Subsidy
                    </span>
                    <span className="text-slate-700 font-medium">{s.subsidyPct}</span>
                  </div>
                </div>

                <div className="pt-3 border-t border-brand-line flex items-center justify-between gap-2">
                  <div className="text-[10px] text-slate-500">
                    <Landmark className="h-3 w-3 inline mr-1" />
                    Apply via State AH Dept / DAHD portal
                  </div>
                  <a
                    href={s.website}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-xs text-brand-blue font-semibold hover:underline"
                  >
                    Official site <ExternalLink className="h-3 w-3" />
                  </a>
                </div>

                <details className="mt-3 group">
                  <summary className="text-xs text-brand-blue font-semibold cursor-pointer hover:underline flex items-center gap-1">
                    <CheckCircle2 className="h-3 w-3" /> Required documents &amp; application process
                  </summary>
                  <div className="mt-2 p-3 rounded-md bg-brand-mist text-xs space-y-2">
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">
                        Documents needed
                      </div>
                      <ul className="space-y-0.5">
                        {s.documents.map((d, i) => (
                          <li key={i} className="text-slate-700">• {d}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">
                        Application process
                      </div>
                      <p className="text-slate-700 leading-relaxed">{s.applicationProcess}</p>
                    </div>
                  </div>
                </details>
              </div>
            );
          })}
        </div>
      )}
    </Section>
  );
}

function PriceCard({ price }: { price: MarketPrice }) {
  const TrendIcon = price.trend === "up" ? TrendingUp : price.trend === "down" ? TrendingDown : Minus;
  const trendColor =
    price.trend === "up" ? "text-brand-green" : price.trend === "down" ? "text-red-500" : "text-slate-500";

  const formatPrice = (n: number) => {
    if (n >= 100000) return `₹${(n / 100000).toFixed(2)} L`;
    if (n >= 1000) return `₹${(n / 1000).toFixed(1)}k`;
    return `₹${n}`;
  };

  return (
    <div className="card-soft p-4">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div>
          <h4
            className="text-sm font-bold text-brand-navy leading-tight"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            {price.commodity}
          </h4>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">
            {price.market}
          </div>
        </div>
        <div className={`flex items-center gap-1 text-xs font-semibold ${trendColor}`}>
          <TrendIcon className="h-3.5 w-3.5" />
          {price.changePct > 0 ? "+" : ""}
          {price.changePct}%
        </div>
      </div>
      <div className="flex items-baseline gap-2 mb-2">
        <span className="text-2xl font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
          {formatPrice(price.modalPrice)}
        </span>
        <span className="text-xs text-slate-500">{price.unit}</span>
      </div>
      <div className="flex items-center justify-between text-[10px] text-slate-500 pt-2 border-t border-brand-line">
        <span>Min: {formatPrice(price.minPrice)}</span>
        <span>Modal: {formatPrice(price.modalPrice)}</span>
        <span>Max: {formatPrice(price.maxPrice)}</span>
      </div>
      <div className="mt-1 text-[10px] text-slate-400">{price.date}</div>
    </div>
  );
}
