// Shared section heading component for consistency
"use client";

import { ReactNode } from "react";

interface SectionHeadingProps {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  align?: "left" | "center";
  action?: ReactNode;
}

export function SectionHeading({
  eyebrow,
  title,
  subtitle,
  align = "left",
  action,
}: SectionHeadingProps) {
  return (
    <div
      className={`flex flex-col sm:flex-row sm:items-end gap-4 mb-8 ${
        align === "center" ? "sm:justify-center text-center sm:text-center items-center" : "sm:justify-between"
      }`}
    >
      <div className={align === "center" ? "max-w-2xl" : "max-w-2xl"}>
        {eyebrow && <div className="section-eyebrow mb-2">{eyebrow}</div>}
        <h2 className="section-title">{title}</h2>
        {subtitle && (
          <p
            className={`section-sub mt-3 ${align === "center" ? "mx-auto" : ""}`}
          >
            {subtitle}
          </p>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

// Reusable placeholder for hero/breed images
export function ImagePlaceholder({
  label,
  hint,
  className,
}: {
  label: string;
  hint?: string;
  className?: string;
}) {
  return (
    <div className={`image-placeholder ${className ?? ""} min-h-[200px]`}>
      <div className="text-center px-4 py-8">
        <div className="text-brand-navy font-semibold text-sm mb-1">{label}</div>
        {hint && (
          <div className="text-[11px] text-slate-500 uppercase tracking-wider">{hint}</div>
        )}
      </div>
    </div>
  );
}

// Section wrapper for consistent padding and container
export function Section({
  children,
  className,
  bg = "white",
}: {
  children: ReactNode;
  className?: string;
  bg?: "white" | "mist" | "cream";
}) {
  const bgClass =
    bg === "mist" ? "bg-brand-mist" : bg === "cream" ? "bg-brand-cream" : "bg-white";
  return (
    <section className={`section-pad ${bgClass} ${className ?? ""}`}>
      <div className="container-page">{children}</div>
    </section>
  );
}
