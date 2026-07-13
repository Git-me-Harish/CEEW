"use client";

import { Sparkles, ShieldCheck, Microscope, TrendingUp, Database, Users, Globe } from "lucide-react";
import { Section, SectionHeading } from "./Section";

interface FeaturesProps {
  onNavigate: (tab: "classifier" | "encyclopedia" | "health" | "market") => void;
}

export function Features({ onNavigate }: FeaturesProps) {
  const features = [
    {
      icon: Sparkles,
      title: "AI Breed Identification",
      description:
        "Upload any cattle or buffalo photo and our vision model identifies the breed against 32+ Indian bovine breeds in seconds, with confidence scoring and distinguishing features.",
      number: "01",
      tab: "classifier" as const,
    },
    {
      icon: Database,
      title: "Comprehensive Breed Library",
      description:
        "Full profiles of 26 indigenous cattle, 4 buffalo breeds, and 6 exotic breeds used in Indian crossbreeding including origin, milk yield, fat content, and conservation status.",
      number: "02",
      tab: "encyclopedia" as const,
    },
    {
      icon: ShieldCheck,
      title: "Health & Vaccination Hub",
      description:
        "13 major bovine diseases with symptoms, causes, treatment, and prevention. Complete NDBB-aligned vaccination schedule and a national veterinary directory.",
      number: "03",
      tab: "health" as const,
    },
    {
      icon: Microscope,
      title: "ICAR-Based Nutrition Science",
      description:
        "Daily ration calculator based on ICAR-NIANP feeding standards. Get green fodder, dry roughage, concentrate, mineral, and water recommendations for any animal.",
      number: "04",
      tab: null,
    },
    {
      icon: TrendingUp,
      title: "Production & Market Analytics",
      description:
        "Log daily milk yields, track 14-day trends, view per-animal performance. Plus live mandi prices for milk, cattle, and fodder across India.",
      number: "05",
      tab: "market" as const,
    },
    {
      icon: Users,
      title: "Government Schemes Directory",
      description:
        "10 central government schemes Rashtriya Gokul Mission, Pashu KCC, Pashu Bima, AHIDF, DEDS with eligibility, subsidies, documents, and application process.",
      number: "06",
      tab: null,
    },
  ];

  return (
    <Section bg="white">
      <SectionHeading
        eyebrow="Why PashuMitra"
        title="Everything Indian cattle rearers need in one place"
        subtitle="Built specifically for Indian bovine species, breeds, and farming conditions. No more juggling 10 different apps, WhatsApp groups, and PDFs."
        align="center"
      />

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
        {features.map((f, i) => (
          <div
            key={i}
            className={`p-6 rounded-lg border border-brand-line bg-white hover:border-brand-blue hover:shadow-md transition-all ${
              f.tab ? "cursor-pointer" : ""
            }`}
            onClick={() => f.tab && onNavigate(f.tab)}
          >
            <div className="flex items-start justify-between mb-4">
              <div className="h-11 w-11 rounded-md bg-brand-blue-50 flex items-center justify-center">
                <f.icon className="h-5 w-5 text-brand-blue" />
              </div>
              <span
                className="text-3xl font-bold text-brand-line"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                {f.number}
              </span>
            </div>
            <h3
              className="text-base font-bold text-brand-navy mb-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              {f.title}
            </h3>
            <p className="text-sm text-slate-600 leading-relaxed">{f.description}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

// Stats strip section
export function StatsStrip() {
  const stats = [
    { value: "303M+", label: "Bovines in India", sub: "Largest global population", icon: Globe },
    { value: "230M+", label: "Tonnes milk/year", sub: "World's #1 milk producer", icon: TrendingUp },
    { value: "32+", label: "Bovine breeds", sub: "Curated in our library", icon: Database },
    { value: "100%", label: "Free for farmers", sub: "Advisory & knowledge base", icon: Users },
  ];

  return (
    <section className="bg-brand-navy py-12">
      <div className="container-page">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          {stats.map((s, i) => (
            <div key={i} className="text-center">
              <s.icon className="h-6 w-6 mx-auto text-brand-amber mb-2" />
              <div
                className="text-3xl md:text-4xl font-bold text-white"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                {s.value}
              </div>
              <div className="text-xs font-semibold text-brand-amber mt-1 uppercase tracking-wider">
                {s.label}
              </div>
              <div className="text-[11px] text-white/70 mt-0.5">{s.sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
