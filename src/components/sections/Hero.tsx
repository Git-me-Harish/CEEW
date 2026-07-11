"use client";

import { ArrowRight, Sparkles, ShieldCheck, TrendingUp, Leaf } from "lucide-react";
import { breedStats } from "@/data/breeds";
import { schemeStats } from "@/data/schemes";
import { diseases } from "@/data/health";

interface HeroProps {
  onNavigate: (tab: "classifier" | "encyclopedia" | "dashboard") => void;
}

export function Hero({ onNavigate }: HeroProps) {
  return (
    <section className="relative bg-brand-navy overflow-hidden">
      {/* Hero background image placeholder — replace with /hero/hero.jpg */}
      <div className="absolute inset-0">
        <div
          className="absolute inset-0 image-placeholder text-white/60"
          style={{ background: "linear-gradient(120deg, #1A2332 0%, #2D5A87 60%, #4CAF50 100%)" }}
        >
          {/* Decorative placeholder pattern indicating hero image area */}
          <div className="absolute inset-0 flex items-center justify-center opacity-20">
            <Sparkles className="h-40 w-40" />
          </div>
          <div className="absolute bottom-4 right-4 text-[11px] uppercase tracking-widest text-white/40">
            Hero image · drop /public/hero/hero.jpg
          </div>
        </div>
        {/* Dark overlay for text legibility */}
        <div className="absolute inset-0 bg-gradient-to-r from-brand-navy via-brand-navy/85 to-brand-navy/40" />
      </div>

      <div className="container-page relative py-16 md:py-24 lg:py-28">
        <div className="max-w-3xl">
          {/* Eyebrow */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 backdrop-blur border border-white/15 text-brand-amber text-xs font-semibold tracking-wider uppercase mb-5"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            <span className="h-1.5 w-1.5 bg-brand-amber rounded-full" />
            India's first end-to-end bovine intelligence platform
          </div>

          {/* Headline */}
          <h1
            className="text-4xl md:text-5xl lg:text-6xl font-bold text-white leading-[1.05] tracking-tight"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            One platform for every Indian <span className="text-brand-amber">cattle &amp; buffalo</span> need.
          </h1>

          {/* Sub-headline */}
          <p className="mt-5 text-base md:text-lg text-white/80 leading-relaxed max-w-2xl">
            AI breed identification, full breed library, health &amp; vaccination tracking,
            nutrition science, milk logs, mandi prices, and government schemes — all in one place.
            Built for India's farmers, dairies, and veterinarians.
          </p>

          {/* CTAs */}
          <div className="mt-8 flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <button
              onClick={() => onNavigate("classifier")}
              className="inline-flex items-center gap-2 px-5 py-3 bg-brand-amber text-brand-navy text-sm font-semibold rounded-md hover:bg-white transition-colors"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Identify a breed with AI <ArrowRight className="h-4 w-4" />
            </button>
            <button
              onClick={() => onNavigate("encyclopedia")}
              className="inline-flex items-center gap-2 px-5 py-3 bg-white/10 backdrop-blur border border-white/20 text-white text-sm font-semibold rounded-md hover:bg-white/20 transition-colors"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Browse {breedStats.totalBreeds} bovine breeds
            </button>
          </div>

          {/* Trust badges */}
          <div className="mt-10 grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { icon: Sparkles, label: "AI Breed Classifier", sub: "Photo → breed in seconds" },
              { icon: ShieldCheck, label: `${diseases.length} Major Diseases`, sub: "Symptoms & vaccines" },
              { icon: TrendingUp, label: "Daily Mandi Prices", sub: "Milk, cattle, fodder" },
              { icon: Leaf, label: `${schemeStats.total} Govt Schemes`, sub: "Subsidies & insurance" },
            ].map((b, i) => (
              <div
                key={i}
                className="px-4 py-3 rounded-md bg-white/10 backdrop-blur border border-white/15"
              >
                <b.icon className="h-4 w-4 text-brand-amber mb-2" />
                <div
                  className="text-sm font-semibold text-white"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  {b.label}
                </div>
                <div className="text-[11px] text-white/70 mt-0.5">{b.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Decorative bottom strip */}
      <div className="bg-brand-amber h-1.5 w-full" />
    </section>
  );
}
