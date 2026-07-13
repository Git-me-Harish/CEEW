"use client";

import { useState, useEffect, useRef } from "react";
import { Menu, X, Phone, LogIn, LogOut, Bell, Stethoscope, Shield, ChevronDown, User as UserIcon, Settings, LayoutDashboard, FolderOpen, Syringe, Sparkles, BookOpen, HeartPulse, Calculator, TrendingUp, MessageSquare, Users } from "lucide-react";
import { signOut } from "next-auth/react";
import { useCurrentUser } from "@/hooks/use-current-user";

export type TabId =
  | "home"
  | "dashboard"
  | "classifier"
  | "encyclopedia"
  | "health"
  | "nutrition"
  | "milk"
  | "market"
  | "forum"
  | "tickets"
  | "vaccination"
  | "management"
  | "profile"
  | "settings";

interface HeaderProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

interface NavItem {
  id: TabId;
  label: string;
  icon: React.ElementType;
}

interface NavGroup {
  label: string;
  icon: React.ElementType;
  items: NavItem[];
}

// Groups for authenticated users
const authedGroups: NavGroup[] = [
  {
    label: "My Herd",
    icon: Users,
    items: [
      { id: "milk", label: "Cattle & Milk", icon: FolderOpen },
      { id: "vaccination", label: "Vaccinations", icon: Syringe },
      { id: "tickets", label: "Problems", icon: MessageSquare },
    ],
  },
  {
    label: "AI & Knowledge",
    icon: Sparkles,
    items: [
      { id: "classifier", label: "Breed Classifier", icon: Sparkles },
      { id: "encyclopedia", label: "Breed Library", icon: BookOpen },
      { id: "health", label: "Health Hub", icon: HeartPulse },
      { id: "nutrition", label: "Nutrition", icon: Calculator },
    ],
  },
  {
    label: "Market & Community",
    icon: TrendingUp,
    items: [
      { id: "market", label: "Market & Schemes", icon: TrendingUp },
      { id: "forum", label: "Farmer Forum", icon: MessageSquare },
    ],
  },
];

// Groups for public (not signed in) users
const publicGroups: NavGroup[] = [
  {
    label: "AI & Knowledge",
    icon: Sparkles,
    items: [
      { id: "classifier", label: "Breed Classifier", icon: Sparkles },
      { id: "encyclopedia", label: "Breed Library", icon: BookOpen },
      { id: "health", label: "Health Hub", icon: HeartPulse },
      { id: "nutrition", label: "Nutrition", icon: Calculator },
    ],
  },
  {
    label: "Market & Community",
    icon: TrendingUp,
    items: [
      { id: "market", label: "Market & Schemes", icon: TrendingUp },
      { id: "forum", label: "Farmer Forum", icon: MessageSquare },
    ],
  },
];

export function Header({ activeTab, onTabChange }: HeaderProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const dropdownTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { user, loading, canManage } = useCurrentUser();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const handleNav = (id: TabId) => {
    onTabChange(id);
    setMobileOpen(false);
    setOpenDropdown(null);
    setProfileOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const enterDropdown = (label: string) => {
    if (dropdownTimeout.current) clearTimeout(dropdownTimeout.current);
    setOpenDropdown(label);
  };

  const leaveDropdown = () => {
    dropdownTimeout.current = setTimeout(() => setOpenDropdown(null), 150);
  };

  const enterProfile = () => {
    if (dropdownTimeout.current) clearTimeout(dropdownTimeout.current);
    setProfileOpen(true);
  };

  const leaveProfile = () => {
    dropdownTimeout.current = setTimeout(() => setProfileOpen(false), 150);
  };

  // Determine which groups to show
  const groups = user ? authedGroups : publicGroups;

  // Check if a group contains the active tab (for highlighting)
  const groupContainsActive = (group: NavGroup) =>
    group.items.some((item) => item.id === activeTab);

  const roleBadge = user
    ? {
        FARMER: { label: "Farmer", color: "bg-brand-green", icon: UserIcon },
        VET: { label: "Vet", color: "bg-brand-blue", icon: Stethoscope },
        ADMIN: { label: "Admin", color: "bg-brand-amber text-brand-navy", icon: Shield },
      }[user.role]
    : null;

  return (
    <header className={`sticky top-0 z-40 transition-all ${scrolled ? "bg-brand-navy shadow-lg" : "bg-brand-navy"}`}>
      {/* Top utility bar */}
      <div className="hidden md:block border-b border-white/10">
        <div className="container-page flex items-center justify-between py-1.5 text-[12px] text-white/80">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <Phone className="h-3.5 w-3.5" /> Veterinary Helpline: 1962
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
        <div className="flex items-center justify-between h-16 gap-4">
          {/* Logo */}
          <button
            onClick={() => handleNav(user ? "dashboard" : "home")}
            className="flex items-center gap-2.5 text-white shrink-0"
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
            <div className="leading-tight text-left hidden sm:block">
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

          {/* Desktop nav — grouped dropdowns */}
          <nav className="hidden lg:flex items-center gap-1 flex-1 justify-center">
            {/* Dashboard (authed only) — direct link */}
            {user && (
              <button
                onClick={() => handleNav("dashboard")}
                className={`inline-flex items-center gap-1.5 px-3 py-2 text-[13px] font-medium rounded-md transition-colors ${
                  activeTab === "dashboard"
                    ? "bg-brand-amber text-brand-navy"
                    : "text-white/85 hover:bg-white/10 hover:text-white"
                }`}
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                <LayoutDashboard className="h-3.5 w-3.5" />
                Dashboard
              </button>
            )}

            {/* Grouped dropdowns */}
            {groups.map((group) => {
              const isActive = groupContainsActive(group);
              const isOpen = openDropdown === group.label;
              return (
                <div
                  key={group.label}
                  className="relative"
                  onMouseEnter={() => enterDropdown(group.label)}
                  onMouseLeave={leaveDropdown}
                >
                  <button
                    className={`inline-flex items-center gap-1.5 px-3 py-2 text-[13px] font-medium rounded-md transition-colors ${
                      isActive
                        ? "bg-brand-amber text-brand-navy"
                        : "text-white/85 hover:bg-white/10 hover:text-white"
                    }`}
                    style={{ fontFamily: "var(--font-montserrat)" }}
                  >
                    <group.icon className="h-3.5 w-3.5" />
                    {group.label}
                    <ChevronDown
                      className={`h-3 w-3 transition-transform ${isOpen ? "rotate-180" : ""}`}
                    />
                  </button>

                  {/* Dropdown panel */}
                  {isOpen && (
                    <div className="absolute top-full left-0 pt-2 min-w-[220px]">
                      <div className="bg-white rounded-lg shadow-2xl border border-brand-line overflow-hidden py-1.5">
                        {group.items.map((item) => {
                          const itemActive = activeTab === item.id;
                          return (
                            <button
                              key={item.id}
                              onClick={() => handleNav(item.id)}
                              className={`w-full flex items-center gap-2.5 px-4 py-2 text-sm transition-colors text-left ${
                                itemActive
                                  ? "bg-brand-blue-50 text-brand-blue font-semibold"
                                  : "text-slate-700 hover:bg-brand-mist"
                              }`}
                              style={{ fontFamily: "var(--font-montserrat)" }}
                            >
                              <item.icon
                                className={`h-4 w-4 shrink-0 ${
                                  itemActive ? "text-brand-blue" : "text-slate-400"
                                }`}
                              />
                              {item.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            {/* Management (vet/admin only) — direct link */}
            {user && canManage && (
              <button
                onClick={() => handleNav("management")}
                className={`inline-flex items-center gap-1.5 px-3 py-2 text-[13px] font-medium rounded-md transition-colors ${
                  activeTab === "management"
                    ? "bg-brand-amber text-brand-navy"
                    : "text-white/85 hover:bg-white/10 hover:text-white"
                }`}
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                <Shield className="h-3.5 w-3.5" />
                Management
              </button>
            )}
          </nav>

          {/* Right side: auth + mobile toggle */}
          <div className="flex items-center gap-2 shrink-0">
            {loading ? (
              <div className="h-8 w-8 rounded-full bg-white/10 animate-pulse" />
            ) : user ? (
              <div className="flex items-center gap-1.5">
                {/* Notification bell */}
                <button
                  onClick={() => handleNav("profile")}
                  className="relative h-9 w-9 rounded-md text-white hover:bg-white/10 flex items-center justify-center"
                  aria-label="Notifications"
                >
                  <Bell className="h-4 w-4" />
                  {user._count.notifications > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 h-4 min-w-4 px-1 rounded-full bg-brand-amber text-brand-navy text-[10px] font-bold flex items-center justify-center">
                      {user._count.notifications > 9 ? "9+" : user._count.notifications}
                    </span>
                  )}
                </button>

                {/* Profile dropdown */}
                <div
                  className="relative"
                  onMouseEnter={enterProfile}
                  onMouseLeave={leaveProfile}
                >
                  <button
                    onClick={() => handleNav("profile")}
                    className="flex items-center gap-2 pl-1.5 pr-2 py-1 rounded-md bg-white/10 hover:bg-white/20 transition-colors"
                  >
                    <div
                      className={`h-7 w-7 rounded-full ${roleBadge?.color} flex items-center justify-center shrink-0`}
                    >
                      {roleBadge && <roleBadge.icon className="h-3.5 w-3.5 text-white" />}
                    </div>
                    <div className="text-left leading-tight hidden sm:block">
                      <div className="text-xs font-semibold text-white max-w-[100px] truncate">
                        {user.name.split(" ")[0]}
                      </div>
                      <div className="text-[9px] text-white/70">{roleBadge?.label}</div>
                    </div>
                    <ChevronDown
                      className={`h-3 w-3 text-white/60 transition-transform hidden sm:block ${
                        profileOpen ? "rotate-180" : ""
                      }`}
                    />
                  </button>

                  {/* Profile dropdown panel */}
                  {profileOpen && (
                    <div className="absolute top-full right-0 pt-2 min-w-[240px]">
                      <div className="bg-white rounded-lg shadow-2xl border border-brand-line overflow-hidden">
                        {/* User info header */}
                        <div className="px-4 py-3 bg-brand-mist border-b border-brand-line">
                          <div className="text-sm font-bold text-brand-navy truncate" style={{ fontFamily: "var(--font-montserrat)" }}>
                            {user.name}
                          </div>
                          <div className="text-xs text-slate-500 truncate">{user.email}</div>
                          <div className="mt-1.5 flex items-center gap-2 text-[10px] text-slate-500">
                            {user.location && <span>{user.location}</span>}
                            <span className={`pill ${roleBadge?.color} !text-[9px] !px-2 !py-0.5 text-white`}>
                              {roleBadge?.label}
                            </span>
                          </div>
                        </div>
                        {/* Menu items */}
                        <div className="py-1">
                          <button
                            onClick={() => handleNav("profile")}
                            className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-slate-700 hover:bg-brand-mist transition-colors text-left"
                            style={{ fontFamily: "var(--font-montserrat)" }}
                          >
                            <UserIcon className="h-4 w-4 text-slate-400" />
                            My Profile
                          </button>
                          <button
                            onClick={() => handleNav("dashboard")}
                            className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-slate-700 hover:bg-brand-mist transition-colors text-left"
                            style={{ fontFamily: "var(--font-montserrat)" }}
                          >
                            <LayoutDashboard className="h-4 w-4 text-slate-400" />
                            Dashboard
                          </button>
                          <button
                            onClick={() => handleNav("settings")}
                            className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-slate-700 hover:bg-brand-mist transition-colors text-left"
                            style={{ fontFamily: "var(--font-montserrat)" }}
                          >
                            <Settings className="h-4 w-4 text-slate-400" />
                            Settings
                          </button>
                        </div>
                        {/* Sign out */}
                        <div className="border-t border-brand-line py-1">
                          <button
                            onClick={() => signOut({ callbackUrl: "/" })}
                            className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors text-left"
                            style={{ fontFamily: "var(--font-montserrat)" }}
                          >
                            <LogOut className="h-4 w-4" />
                            Sign Out
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <a
                href="/auth/signin"
                className="inline-flex items-center gap-2 px-4 py-2 bg-brand-amber text-brand-navy text-sm font-semibold rounded-md hover:bg-white transition-colors"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                <LogIn className="h-4 w-4" /> Sign In
              </a>
            )}

            {/* Mobile toggle */}
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

      {/* Mobile menu — collapsible accordion style */}
      {mobileOpen && (
        <div className="lg:hidden border-t border-white/10 bg-brand-navy">
          <div className="container-page py-3 space-y-3">
            {/* Dashboard direct link */}
            {user && (
              <button
                onClick={() => handleNav("dashboard")}
                className={`flex items-center gap-2 w-full px-3 py-2.5 text-sm font-medium rounded-md ${
                  activeTab === "dashboard"
                    ? "bg-brand-amber text-brand-navy"
                    : "text-white/85 hover:bg-white/10"
                }`}
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                <LayoutDashboard className="h-4 w-4" /> Dashboard
              </button>
            )}

            {/* Mobile: render all items as flat list grouped by category */}
            {groups.map((group) => (
              <div key={group.label}>
                <div className="px-3 py-1.5 text-[10px] uppercase tracking-wider text-brand-amber font-semibold flex items-center gap-1.5">
                  <group.icon className="h-3 w-3" />
                  {group.label}
                </div>
                <div className="space-y-0.5">
                  {group.items.map((item) => (
                    <button
                      key={item.id}
                      onClick={() => handleNav(item.id)}
                      className={`flex items-center gap-2.5 w-full px-3 py-2 text-sm font-medium rounded-md ${
                        activeTab === item.id
                          ? "bg-brand-amber text-brand-navy"
                          : "text-white/85 hover:bg-white/10"
                      }`}
                      style={{ fontFamily: "var(--font-montserrat)" }}
                    >
                      <item.icon className="h-4 w-4" />
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}

            {/* Management (vet/admin only) */}
            {user && canManage && (
              <button
                onClick={() => handleNav("management")}
                className={`flex items-center gap-2 w-full px-3 py-2.5 text-sm font-medium rounded-md ${
                  activeTab === "management"
                    ? "bg-brand-amber text-brand-navy"
                    : "text-white/85 hover:bg-white/10"
                }`}
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                <Shield className="h-4 w-4" /> Management
              </button>
            )}

            {/* Profile + Sign out */}
            {user ? (
              <div className="pt-3 border-t border-white/10 space-y-1">
                <button
                  onClick={() => handleNav("profile")}
                  className={`flex items-center gap-2 w-full px-3 py-2.5 text-sm font-medium rounded-md ${
                    activeTab === "profile"
                      ? "bg-brand-amber text-brand-navy"
                      : "text-white/85 hover:bg-white/10"
                  }`}
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  <UserIcon className="h-4 w-4" /> Profile
                </button>
                <button
                  onClick={() => signOut({ callbackUrl: "/" })}
                  className="flex items-center gap-2 w-full px-3 py-2.5 text-sm font-medium rounded-md text-red-300 hover:bg-red-500/20"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  <LogOut className="h-4 w-4" /> Sign Out
                </button>
              </div>
            ) : (
              <a
                href="/auth/signin"
                className="flex items-center justify-center gap-2 w-full px-3 py-2.5 text-sm font-semibold rounded-md bg-brand-amber text-brand-navy"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                <LogIn className="h-4 w-4" /> Sign In / Sign Up
              </a>
            )}
          </div>
        </div>
      )}
    </header>
  );
}
