"use client";

import { useState, useEffect } from "react";
import { Menu, X, Phone, ChevronDown } from "lucide-react";

export type TabId =
  | "home"
  | "dashboard"
  | "classifier"
  | "encyclopedia"
  | "health"
  | "nutrition"
  | "milk"
  | "market"
  | "forum";

interface HeaderProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

const navItems: { id: TabId; label: string }[] = [
  { id: "home", label: "Home" },
  { id: "dashboard", label: "Dashboard" },
  { id: "classifier", label: "Breed Classifier" },
  { id: "encyclopedia", label: "Breed Library" },
  { id: "health", label: "Health Hub" },
  { id: "nutrition", label: "Nutrition" },
  { id: "milk", label: "Milk Tracker" },
  { id: "market", label: "Market & Schemes" },
  { id: "forum", label: "Farmer Forum" },
];

export function Header({ activeTab, onTabChange }: HeaderProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const handleNav = (id: TabId) => {
    onTabChange(id);
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <header
      className={`sticky top-0 z-40 transition-all ${
        scrolled
          ? "bg-brand-navy shadow-lg"
          : "bg-brand-navy"
      }`}
    >
      {/* Top utility bar */}
      <div className="hidden md:block border-b border-white/10">
        <div className="container-page flex items-center justify-between py-1.5 text-[12px] text-white/80">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <Phone className="h-3.5 w-3.5" /> Veterinary Helpline: 1962 / +91-11-23384194
            </span>
            <span className="text-white/30">|</span>
            <span>Free FMD Vaccination under National Programme</span>
          </div>
          <div className="flex items-center gap-3">
            <span>Government of India · DAHD Aligned</span>
          </div>
        </div>
      </div>

      {/* Main navigation */}
      <div className="container-page">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <button
            onClick={() => handleNav("home")}
            className="flex items-center gap-2.5 text-white"
            aria-label="PashuMitra Home"
          >
            <div className="h-9 w-9 rounded-md bg-brand-amber flex items-center justify-center">
              <span
                className="text-brand-navy font-extrabold text-lg"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                P
              </span>
            </div>
            <div className="leading-tight text-left">
              <div
                className="text-base font-bold tracking-tight"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                PashuMitra
              </div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-brand-amber">
                Indian Bovine Platform
              </div>
            </div>
          </button>

          {/* Desktop nav */}
          <nav className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleNav(item.id)}
                className={`px-3 py-2 text-[13px] font-medium rounded-md transition-colors ${
                  activeTab === item.id
                    ? "bg-brand-amber text-brand-navy"
                    : "text-white/85 hover:bg-white/10 hover:text-white"
                }`}
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                {item.label}
              </button>
            ))}
          </nav>

          {/* CTA + mobile toggle */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleNav("classifier")}
              className="hidden sm:inline-flex items-center gap-2 px-4 py-2 bg-white text-brand-navy text-sm font-semibold rounded-md hover:bg-brand-cream transition-colors"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Identify a Breed
            </button>
            <button
              onClick={() => setMobileOpen((v) => !v)}
              className="lg:hidden text-white p-2 rounded-md hover:bg-white/10"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="lg:hidden border-t border-white/10 bg-brand-navy">
          <div className="container-page py-3 space-y-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleNav(item.id)}
                className={`block w-full text-left px-3 py-2.5 text-sm font-medium rounded-md ${
                  activeTab === item.id
                    ? "bg-brand-amber text-brand-navy"
                    : "text-white/85 hover:bg-white/10"
                }`}
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </header>
  );
}
