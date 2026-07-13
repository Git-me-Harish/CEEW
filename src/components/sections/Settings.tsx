"use client";

import { useState, useEffect } from "react";
import {
  Loader2,
  AlertCircle,
  CheckCircle2,
  User as UserIcon,
  Lock,
  Phone,
  MapPin,
  Save,
  Shield,
  Mail,
  Eye,
  EyeOff,
  Calendar,
  Bell,
} from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { useCurrentUser } from "@/hooks/use-current-user";

export function Settings() {
  const { user, loading, refresh } = useCurrentUser();
  const [profileForm, setProfileForm] = useState({
    name: "",
    phone: "",
    location: "",
    avatarUrl: "",
  });
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: "",
    newPassword: "",
    confirmPassword: "",
  });
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [profileSuccess, setProfileSuccess] = useState(false);
  const [passwordSuccess, setPasswordSuccess] = useState(false);
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  // Populate form when user data loads
  useEffect(() => {
    if (user) {
      setProfileForm({
        name: user.name,
        phone: user.phone || "",
        location: user.location || "",
        avatarUrl: user.avatarUrl || "",
      });
    }
  }, [user]);

  if (loading) {
    return (
      <Section bg="mist">
        <div className="text-center py-16">
          <Loader2 className="h-8 w-8 mx-auto text-brand-blue animate-spin" />
        </div>
      </Section>
    );
  }

  if (!user) {
    return (
      <Section bg="mist">
        <SectionHeading eyebrow="Authentication required" title="Please sign in to access settings" />
        <div className="card-soft p-6 text-center">
          <p className="text-sm text-slate-600 mb-4">
            You need to be signed in to manage your account settings.
          </p>
          <a href="/auth/signin" className="btn-primary inline-flex">Sign In</a>
        </div>
      </Section>
    );
  }

  const saveProfile = async () => {
    setSavingProfile(true);
    setProfileError(null);
    setProfileSuccess(false);
    try {
      const res = await fetch("/api/me", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: profileForm.name,
          phone: profileForm.phone,
          location: profileForm.location,
          avatarUrl: profileForm.avatarUrl,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to update profile.");
      setProfileSuccess(true);
      await refresh();
      setTimeout(() => setProfileSuccess(false), 4000);
    } catch (err) {
      setProfileError(err instanceof Error ? err.message : "Failed to update profile.");
    } finally {
      setSavingProfile(false);
    }
  };

  const savePassword = async () => {
    setPasswordError(null);
    setPasswordSuccess(false);

    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      setPasswordError("New password and confirmation do not match.");
      return;
    }
    if (passwordForm.newPassword.length < 6) {
      setPasswordError("New password must be at least 6 characters.");
      return;
    }

    setSavingPassword(true);
    try {
      const res = await fetch("/api/me", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          currentPassword: passwordForm.currentPassword,
          newPassword: passwordForm.newPassword,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to change password.");
      setPasswordForm({ currentPassword: "", newPassword: "", confirmPassword: "" });
      setPasswordSuccess(true);
      setTimeout(() => setPasswordSuccess(false), 4000);
    } catch (err) {
      setPasswordError(err instanceof Error ? err.message : "Failed to change password.");
    } finally {
      setSavingPassword(false);
    }
  };

  const roleBadge = {
    FARMER: "pill-green",
    VET: "pill-blue",
    ADMIN: "pill-navy",
  }[user.role];

  const roleLabel = {
    FARMER: "Farmer",
    VET: "Veterinarian",
    ADMIN: "Administrator",
  }[user.role];

  return (
    <Section bg="mist">
      <SectionHeading
        eyebrow="Account Settings"
        title="Manage your account"
        subtitle="Update your profile information, change your password, and manage your account preferences."
      />

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left column — account summary */}
        <div className="lg:col-span-1 space-y-4">
          <div className="card-soft p-5">
            <div className="flex items-center gap-3 mb-4">
              <div
                className="h-14 w-14 rounded-full bg-brand-navy flex items-center justify-center text-white text-xl font-bold shrink-0"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                {user.name.charAt(0).toUpperCase()}
              </div>
              <div className="min-w-0">
                <div className="font-bold text-brand-navy truncate" style={{ fontFamily: "var(--font-montserrat)" }}>
                  {user.name}
                </div>
                <span className={`${roleBadge} text-[10px]`}>{roleLabel}</span>
              </div>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="flex items-start gap-2 text-slate-600">
                <Mail className="h-3.5 w-3.5 mt-0.5 shrink-0 text-slate-400" />
                <div className="min-w-0">
                  <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Email</div>
                  <div className="text-brand-navy truncate">{user.email}</div>
                </div>
              </div>
              {user.phone && (
                <div className="flex items-start gap-2 text-slate-600">
                  <Phone className="h-3.5 w-3.5 mt-0.5 shrink-0 text-slate-400" />
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Phone</div>
                    <div className="text-brand-navy">{user.phone}</div>
                  </div>
                </div>
              )}
              {user.location && (
                <div className="flex items-start gap-2 text-slate-600">
                  <MapPin className="h-3.5 w-3.5 mt-0.5 shrink-0 text-slate-400" />
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Location</div>
                    <div className="text-brand-navy">{user.location}</div>
                  </div>
                </div>
              )}
              <div className="flex items-start gap-2 text-slate-600">
                <Calendar className="h-3.5 w-3.5 mt-0.5 shrink-0 text-slate-400" />
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Member Since</div>
                  <div className="text-brand-navy">
                    {new Date(user.createdAt).toLocaleDateString("en-IN", {
                      day: "numeric",
                      month: "short",
                      year: "numeric",
                    })}
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-brand-line grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-lg font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                  {user._count.cattle}
                </div>
                <div className="text-[9px] uppercase tracking-wider text-slate-500 font-semibold">Cattle</div>
              </div>
              <div>
                <div className="text-lg font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                  {user._count.tickets}
                </div>
                <div className="text-[9px] uppercase tracking-wider text-slate-500 font-semibold">Problems</div>
              </div>
              <div>
                <div className="text-lg font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                  {user._count.vaccinationRequests}
                </div>
                <div className="text-[9px] uppercase tracking-wider text-slate-500 font-semibold">Vaccines</div>
              </div>
            </div>
          </div>

          {/* Quick info card */}
          <div className="card-soft p-5">
            <h3
              className="text-sm font-semibold text-brand-navy mb-2 flex items-center gap-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              <Shield className="h-4 w-4 text-brand-blue" /> Account Security
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              Keep your password safe and never share it. Use a strong, unique password
              with at least 6 characters. Change it periodically for better security.
            </p>
          </div>
        </div>

        {/* Right column — forms */}
        <div className="lg:col-span-2 space-y-6">
          {/* Edit Profile form */}
          <div className="card-soft p-6">
            <h3
              className="text-base font-bold text-brand-navy mb-1 flex items-center gap-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              <UserIcon className="h-4 w-4 text-brand-blue" /> Profile Information
            </h3>
            <p className="text-xs text-slate-500 mb-5">
              Update your name, contact details, and location. These are visible to veterinarians and admins.
            </p>

            {profileSuccess && (
              <div className="mb-4 flex items-start gap-2 p-3 rounded-md bg-green-50 border border-green-200 text-sm text-green-700">
                <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0" />
                Profile updated successfully.
              </div>
            )}
            {profileError && (
              <div className="mb-4 flex items-start gap-2 p-3 rounded-md bg-red-50 border border-red-200 text-sm text-red-700">
                <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                {profileError}
              </div>
            )}

            <div className="space-y-4">
              <div className="grid sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    Full Name
                  </label>
                  <input
                    type="text"
                    value={profileForm.name}
                    onChange={(e) => setProfileForm({ ...profileForm, name: e.target.value })}
                    className="input-soft"
                    placeholder="Your name"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    Email (read-only)
                  </label>
                  <input
                    type="email"
                    value={user.email}
                    disabled
                    className="input-soft bg-slate-50 text-slate-500 cursor-not-allowed"
                  />
                  <div className="text-[10px] text-slate-400 mt-1">Email cannot be changed.</div>
                </div>
              </div>

              <div className="grid sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    Phone Number
                  </label>
                  <input
                    type="tel"
                    value={profileForm.phone}
                    onChange={(e) => setProfileForm({ ...profileForm, phone: e.target.value })}
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
                    value={profileForm.location}
                    onChange={(e) => setProfileForm({ ...profileForm, location: e.target.value })}
                    className="input-soft"
                    placeholder="City, State"
                  />
                </div>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Avatar URL (optional)
                </label>
                <input
                  type="url"
                  value={profileForm.avatarUrl}
                  onChange={(e) => setProfileForm({ ...profileForm, avatarUrl: e.target.value })}
                  className="input-soft"
                  placeholder="https://example.com/avatar.jpg"
                />
                <div className="text-[10px] text-slate-400 mt-1">
                  Paste a direct image URL. Leave blank to use your initials avatar.
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={saveProfile}
                  disabled={savingProfile || !profileForm.name.trim()}
                  className="btn-primary"
                >
                  {savingProfile ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" /> Saving...
                    </>
                  ) : (
                    <>
                      <Save className="h-4 w-4" /> Save Changes
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Change Password form */}
          <div className="card-soft p-6">
            <h3
              className="text-base font-bold text-brand-navy mb-1 flex items-center gap-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              <Lock className="h-4 w-4 text-brand-blue" /> Change Password
            </h3>
            <p className="text-xs text-slate-500 mb-5">
              Use a strong password with at least 6 characters. You'll need to enter your current password to confirm.
            </p>

            {passwordSuccess && (
              <div className="mb-4 flex items-start gap-2 p-3 rounded-md bg-green-50 border border-green-200 text-sm text-green-700">
                <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0" />
                Password changed successfully. Use your new password next time you sign in.
              </div>
            )}
            {passwordError && (
              <div className="mb-4 flex items-start gap-2 p-3 rounded-md bg-red-50 border border-red-200 text-sm text-red-700">
                <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                {passwordError}
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Current Password
                </label>
                <div className="relative">
                  <input
                    type={showCurrent ? "text" : "password"}
                    value={passwordForm.currentPassword}
                    onChange={(e) => setPasswordForm({ ...passwordForm, currentPassword: e.target.value })}
                    className="input-soft pr-10"
                    placeholder="Enter your current password"
                    autoComplete="current-password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrent(!showCurrent)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-brand-navy"
                    aria-label="Toggle visibility"
                  >
                    {showCurrent ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              <div className="grid sm:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    New Password
                  </label>
                  <div className="relative">
                    <input
                      type={showNew ? "text" : "password"}
                      value={passwordForm.newPassword}
                      onChange={(e) => setPasswordForm({ ...passwordForm, newPassword: e.target.value })}
                      className="input-soft pr-10"
                      placeholder="Min 6 characters"
                      autoComplete="new-password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowNew(!showNew)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-brand-navy"
                      aria-label="Toggle visibility"
                    >
                      {showNew ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    Confirm New Password
                  </label>
                  <div className="relative">
                    <input
                      type={showConfirm ? "text" : "password"}
                      value={passwordForm.confirmPassword}
                      onChange={(e) => setPasswordForm({ ...passwordForm, confirmPassword: e.target.value })}
                      className="input-soft pr-10"
                      placeholder="Re-enter new password"
                      autoComplete="new-password"
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirm(!showConfirm)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-brand-navy"
                      aria-label="Toggle visibility"
                    >
                      {showConfirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              </div>

              {passwordForm.newPassword && (
                <PasswordStrength password={passwordForm.newPassword} />
              )}

              <div className="flex justify-end pt-2">
                <button
                  onClick={savePassword}
                  disabled={
                    savingPassword ||
                    !passwordForm.currentPassword ||
                    !passwordForm.newPassword ||
                    !passwordForm.confirmPassword
                  }
                  className="btn-primary"
                >
                  {savingPassword ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" /> Changing...
                    </>
                  ) : (
                    <>
                      <Shield className="h-4 w-4" /> Change Password
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Notification preferences (read-only info for now) */}
          <div className="card-soft p-6">
            <h3
              className="text-base font-bold text-brand-navy mb-1 flex items-center gap-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              <Bell className="h-4 w-4 text-brand-blue" /> Notification Preferences
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              You currently receive notifications for these events:
            </p>
            <div className="space-y-2">
              {[
                "Replies to your problems",
                "Vaccination request status updates",
                "New problems from farmers (vet/admin only)",
                "New vaccination requests (vet/admin only)",
              ].map((item, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-2.5 rounded-md bg-brand-mist border border-brand-line"
                >
                  <span className="text-sm text-slate-700">{item}</span>
                  <span className="pill-green text-[10px]">
                    <CheckCircle2 className="h-3 w-3" /> On
                  </span>
                </div>
              ))}
            </div>
            <div className="mt-3 text-[11px] text-slate-400">
              Granular notification preferences coming soon.
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function PasswordStrength({ password }: { password: string }) {
  let score = 0;
  if (password.length >= 6) score++;
  if (password.length >= 10) score++;
  if (/[A-Z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;

  const labels = ["Very Weak", "Weak", "Fair", "Good", "Strong"];
  const colors = ["bg-red-400", "bg-orange-400", "bg-amber-400", "bg-blue-400", "bg-brand-green"];
  const idx = Math.max(0, Math.min(4, score - 1));

  return (
    <div className="p-2.5 rounded-md bg-brand-mist border border-brand-line">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
          Password Strength
        </span>
        <span
          className={`text-xs font-bold ${
            score >= 4 ? "text-brand-green-dark" : score >= 3 ? "text-brand-blue" : "text-amber-700"
          }`}
          style={{ fontFamily: "var(--font-montserrat)" }}
        >
          {labels[idx]}
        </span>
      </div>
      <div className="flex gap-1">
        {[0, 1, 2, 3, 4].map((i) => (
          <div
            key={i}
            className={`h-1 flex-1 rounded-full ${
              i <= idx ? colors[idx] : "bg-slate-200"
            }`}
          />
        ))}
      </div>
      {score < 3 && (
        <div className="text-[10px] text-slate-500 mt-1.5">
          Tip: Use 10+ characters with uppercase, numbers, and symbols for a stronger password.
        </div>
      )}
    </div>
  );
}
