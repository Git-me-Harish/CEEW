"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Loader2, UserPlus, AlertCircle, Eye, EyeOff, User, Stethoscope } from "lucide-react";

export default function SignUpPage() {
  const router = useRouter();
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    phone: "",
    location: "",
    role: "FARMER" as "FARMER" | "VET",
  });
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name || !form.email || !form.password) return;
    if (form.password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Registration failed.");

      // Auto-sign in after registration
      const signRes = await signIn("credentials", {
        email: form.email,
        password: form.password,
        redirect: false,
      });
      if (signRes?.ok) {
        router.push("/");
        router.refresh();
      } else {
        router.push("/auth/signin");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-brand-navy p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-6">
          <div className="inline-flex items-center gap-2.5">
            <div className="h-10 w-10 rounded-md bg-brand-amber flex items-center justify-center">
              <span
                className="text-brand-navy font-extrabold text-xl"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                P
              </span>
            </div>
            <div className="text-left">
              <div
                className="text-lg font-bold text-white"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                PashuMitra
              </div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-brand-amber">
                Indian Bovine Platform
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-2xl p-8">
          <h1
            className="text-2xl font-bold text-brand-navy mb-1"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            Create your account
          </h1>
          <p className="text-sm text-slate-600 mb-6">
            Join PashuMitra to manage your cattle, post problems, and access government schemes.
          </p>

          {error && (
            <div className="mb-4 flex items-start gap-2 p-3 rounded-md bg-red-50 border border-red-200 text-sm text-red-700">
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          {/* Role selector */}
          <div className="mb-4">
            <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2 block">
              I am a
            </label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setForm({ ...form, role: "FARMER" })}
                className={`p-3 rounded-md border-2 text-left transition-all ${
                  form.role === "FARMER"
                    ? "border-brand-navy bg-brand-blue-50"
                    : "border-brand-line hover:border-brand-blue"
                }`}
              >
                <User className="h-5 w-5 text-brand-blue mb-1.5" />
                <div
                  className="text-sm font-bold text-brand-navy"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  Farmer
                </div>
                <div className="text-[10px] text-slate-500">I own cattle / buffalo</div>
              </button>
              <button
                type="button"
                onClick={() => setForm({ ...form, role: "VET" })}
                className={`p-3 rounded-md border-2 text-left transition-all ${
                  form.role === "VET"
                    ? "border-brand-navy bg-brand-blue-50"
                    : "border-brand-line hover:border-brand-blue"
                }`}
              >
                <Stethoscope className="h-5 w-5 text-brand-green mb-1.5" />
                <div
                  className="text-sm font-bold text-brand-navy"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  Veterinarian
                </div>
                <div className="text-[10px] text-slate-500">I provide vet services</div>
              </button>
            </div>
          </div>

          <form onSubmit={submit} className="space-y-3">
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Full Name
              </label>
              <input
                type="text"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                className="input-soft"
                placeholder="Rajesh Patel"
                required
                autoFocus
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Email
              </label>
              <input
                type="email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                className="input-soft"
                placeholder="you@example.com"
                required
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Phone
                </label>
                <input
                  type="tel"
                  value={form.phone}
                  onChange={(e) => setForm({ ...form, phone: e.target.value })}
                  className="input-soft"
                  placeholder="+91 98765 43210"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Location
                </label>
                <input
                  type="text"
                  value={form.location}
                  onChange={(e) => setForm({ ...form, location: e.target.value })}
                  className="input-soft"
                  placeholder="Anand, Gujarat"
                />
              </div>
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPw ? "text" : "password"}
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  className="input-soft pr-10"
                  placeholder="Min 6 characters"
                  required
                  minLength={6}
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-brand-navy"
                  aria-label="Toggle password visibility"
                >
                  {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <button type="submit" disabled={loading} className="btn-primary w-full mt-2">
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Creating account...
                </>
              ) : (
                <>
                  <UserPlus className="h-4 w-4" /> Create Account
                </>
              )}
            </button>
          </form>

          <div className="mt-6 pt-6 border-t border-brand-line text-center text-sm text-slate-600">
            Already have an account?{" "}
            <button
              onClick={() => router.push("/auth/signin")}
              className="text-brand-blue font-semibold hover:underline"
            >
              Sign in
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
