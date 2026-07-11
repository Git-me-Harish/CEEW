"use client";

import { useState } from "react";
import { Header, type TabId } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { AIAssistant } from "@/components/layout/AIAssistant";
import { Hero } from "@/components/sections/Hero";
import { Features, StatsStrip } from "@/components/sections/Features";
import { Dashboard } from "@/components/sections/Dashboard";
import { BreedClassifier } from "@/components/sections/BreedClassifier";
import { BreedEncyclopedia } from "@/components/sections/BreedEncyclopedia";
import { HealthHub } from "@/components/sections/HealthHub";
import { NutritionCalculator } from "@/components/sections/NutritionCalculator";
import { MilkTracker } from "@/components/sections/MilkTracker";
import { MarketAndSchemes } from "@/components/sections/MarketAndSchemes";
import { FarmerForum } from "@/components/sections/FarmerForum";

export default function Home() {
  const [tab, setTab] = useState<TabId>("home");

  return (
    <div className="min-h-screen flex flex-col bg-white">
      <Header activeTab={tab} onTabChange={setTab} />

      <main className="flex-1">
        {tab === "home" && (
          <>
            <Hero
              onNavigate={(t) =>
                setTab(t === "classifier" ? "classifier" : t === "encyclopedia" ? "encyclopedia" : "dashboard")
              }
            />
            <StatsStrip />
            <Features
              onNavigate={(t) =>
                setTab(
                  t === "classifier"
                    ? "classifier"
                    : t === "encyclopedia"
                    ? "encyclopedia"
                    : t === "health"
                    ? "health"
                    : "market"
                )
              }
            />
            <Dashboard
              onNavigate={(t) =>
                setTab(
                  t === "classifier"
                    ? "classifier"
                    : t === "encyclopedia"
                    ? "encyclopedia"
                    : t === "health"
                    ? "health"
                    : t === "nutrition"
                    ? "nutrition"
                    : t === "milk"
                    ? "milk"
                    : t === "market"
                    ? "market"
                    : "forum"
                )
              }
            />
          </>
        )}

        {tab === "dashboard" && (
          <Dashboard
            onNavigate={(t) =>
              setTab(
                t === "classifier"
                  ? "classifier"
                  : t === "encyclopedia"
                  ? "encyclopedia"
                  : t === "health"
                  ? "health"
                  : t === "nutrition"
                  ? "nutrition"
                  : t === "milk"
                  ? "milk"
                  : t === "market"
                  ? "market"
                  : "forum"
              )
            }
          />
        )}

        {tab === "classifier" && <BreedClassifier />}
        {tab === "encyclopedia" && <BreedEncyclopedia />}
        {tab === "health" && <HealthHub />}
        {tab === "nutrition" && <NutritionCalculator />}
        {tab === "milk" && <MilkTracker />}
        {tab === "market" && <MarketAndSchemes />}
        {tab === "forum" && <FarmerForum />}
      </main>

      <Footer onNavigate={setTab} />
      <AIAssistant />
    </div>
  );
}
