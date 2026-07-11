import { Mail, Phone, MapPin, ArrowUpRight } from "lucide-react";

interface FooterProps {
  onNavigate: (tab: "home" | "dashboard" | "classifier" | "encyclopedia" | "health" | "nutrition" | "milk" | "market" | "forum") => void;
}

export function Footer({ onNavigate }: FooterProps) {
  const quickLinks: { label: string; tab: FooterProps["onNavigate"] extends (tab: infer T) => void ? T : never }[] = [
    { label: "Breed Classifier", tab: "classifier" },
    { label: "Breed Encyclopedia", tab: "encyclopedia" },
    { label: "Health & Vaccination Hub", tab: "health" },
    { label: "Nutrition Calculator", tab: "nutrition" },
    { label: "Milk Production Tracker", tab: "milk" },
    { label: "Market & Schemes", tab: "market" },
    { label: "Farmer Forum", tab: "forum" },
  ];

  return (
    <footer className="mt-auto bg-brand-navy text-white">
      {/* CTA strip */}
      <div className="bg-brand-blue border-y border-white/10">
        <div className="container-page py-7 flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <h3
              className="text-xl md:text-2xl font-bold"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Ready to modernise your bovine operation?
            </h3>
            <p className="text-white/80 text-sm mt-1">
              Join 2,400+ farmers using PashuMitra to manage their cattle and buffalo scientifically.
            </p>
          </div>
          <button
            onClick={() => onNavigate("dashboard")}
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-brand-amber text-brand-navy text-sm font-semibold rounded-md hover:bg-white transition-colors"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            Open Farmer Dashboard <ArrowUpRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Main footer */}
      <div className="container-page py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {/* Brand column */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="h-9 w-9 rounded-md bg-brand-amber flex items-center justify-center">
                <span
                  className="text-brand-navy font-extrabold text-lg"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  P
                </span>
              </div>
              <div className="leading-tight">
                <div
                  className="text-base font-bold"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  PashuMitra
                </div>
                <div className="text-[10px] uppercase tracking-[0.18em] text-brand-amber">
                  Indian Bovine Platform
                </div>
              </div>
            </div>
            <p className="text-sm text-white/70 leading-relaxed">
              One-stop intelligence platform for Indian cattle and buffalo management.
              Built for farmers, dairies, and veterinarians with breed AI, health records,
              nutrition science, and government scheme support.
            </p>
            <div className="mt-4 text-[11px] text-white/50">
              © {new Date().getFullYear()} PashuMitra. Made in India for Indian farmers.
            </div>
          </div>

          {/* Quick links */}
          <div>
            <h4
              className="text-sm font-semibold uppercase tracking-wider text-brand-amber mb-4"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Platform
            </h4>
            <ul className="space-y-2">
              {quickLinks.map((q) => (
                <li key={q.label}>
                  <button
                    onClick={() => onNavigate(q.tab)}
                    className="text-sm text-white/75 hover:text-white text-left transition-colors"
                  >
                    {q.label}
                  </button>
                </li>
              ))}
            </ul>
          </div>

          {/* Knowledge */}
          <div>
            <h4
              className="text-sm font-semibold uppercase tracking-wider text-brand-amber mb-4"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Knowledge Base
            </h4>
            <ul className="space-y-2 text-sm text-white/75">
              <li>32 Indian bovine breeds</li>
              <li>13 major diseases &amp; vaccines</li>
              <li>14 common feedstuffs database</li>
              <li>10 central govt schemes</li>
              <li>Veterinary directory</li>
              <li>Daily mandi prices</li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4
              className="text-sm font-semibold uppercase tracking-wider text-brand-amber mb-4"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Helplines
            </h4>
            <ul className="space-y-3 text-sm text-white/80">
              <li className="flex items-start gap-2.5">
                <Phone className="h-4 w-4 mt-0.5 text-brand-amber shrink-0" />
                <div>
                  <div className="font-medium text-white">Veterinary Helpline</div>
                  <div className="text-white/70">1962 (Toll-free)</div>
                </div>
              </li>
              <li className="flex items-start gap-2.5">
                <Mail className="h-4 w-4 mt-0.5 text-brand-amber shrink-0" />
                <div>
                  <div className="font-medium text-white">DAHD Helpdesk</div>
                  <div className="text-white/70">dahd-help@gov.in</div>
                </div>
              </li>
              <li className="flex items-start gap-2.5">
                <MapPin className="h-4 w-4 mt-0.5 text-brand-amber shrink-0" />
                <div>
                  <div className="font-medium text-white">DAHD, GoI</div>
                  <div className="text-white/70">Krishi Bhavan, New Delhi 110001</div>
                </div>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom strip */}
        <div className="mt-10 pt-6 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px] text-white/55">
          <div>
            Always consult a registered veterinarian for diagnosis. PashuMitra provides advisory information only.
          </div>
          <div className="flex items-center gap-4">
            <span>Data aligned with ICAR, NDDB, DAHD GoI, NIANP standards</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
