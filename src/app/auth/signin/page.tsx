"use client";

import { useState, Suspense } from "react";
import { signIn } from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, LogIn, AlertCircle, Eye, EyeOff } from "lucide-react";

function SignInForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const callbackUrl = searchParams.get("callbackUrl") || "/";
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) return;
    setLoading(true);
    setError(null);
    const res = await signIn("credentials", {
      email,
      password,
      redirect: false,
    });
    setLoading(false);
    if (res?.error) {
      setError("Invalid email or password. Please try again.");
    } else if (res?.ok) {
      router.push(callbackUrl);
      router.refresh();
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-brand-navy p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
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
            Welcome back
          </h1>
          <p className="text-sm text-slate-600 mb-6">
            Sign in to access your dashboard, post problems, and manage your cattle.
          </p>

          {error && (
            <div className="mb-4 flex items-start gap-2 p-3 rounded-md bg-red-50 border border-red-200 text-sm text-red-700">
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
              {error}
            </div>
          )}

          <form onSubmit={submit} className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input-soft"
                placeholder="farmer@example.com"
                required
                autoFocus
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="input-soft pr-10"
                  placeholder="••••••••"
                  required
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

            <button type="submit" disabled={loading} className="btn-primary w-full">
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Signing in...
                </>
              ) : (
                <>
                  <LogIn className="h-4 w-4" /> Sign In
                </>
              )}
            </button>
          </form>

          <div className="mt-6 pt-6 border-t border-brand-line text-center text-sm text-slate-600">
            Don&apos;t have an account?{" "}
            <button
              onClick={() => router.push("/auth/signup")}
              className="text-brand-blue font-semibold hover:underline"
            >
              Create one
            </button>
          </div>

          <div className="mt-4 p-3 rounded-md bg-brand-mist border border-brand-line text-xs text-slate-600">
            <strong className="text-brand-navy">Demo accounts:</strong>
            <br />
            Farmer — farmer@demo.in / demo123
            <br />
            Vet — vet@demo.in / demo123
            <br />
            Admin — admin@demo.in / demo123
          </div>
        </div>

        <div className="text-center mt-4 text-xs text-white/60">
          <button onClick={() => router.push("/")} className="hover:text-white">
            ← Back to home
          </button>
        </div>
      </div>
    </div>
  );
}

export default function SignInPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-brand-navy" />}>
      <SignInForm />
    </Suspense>
  );
}