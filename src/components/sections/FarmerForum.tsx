"use client";

import { useState, useEffect } from "react";
import { Plus, MessageSquare, ThumbsUp, ChevronRight, Loader2, X, Send, ArrowLeft, User } from "lucide-react";
import { Section, SectionHeading } from "./Section";

interface ForumPost {
  id: string;
  authorName: string;
  authorRole: string;
  topic: string;
  breedTag: string | null;
  body: string;
  upvotes: number;
  createdAt: string;
  _count?: { replies: number };
  replies?: ForumReply[];
}

interface ForumReply {
  id: string;
  authorName: string;
  authorRole: string;
  body: string;
  createdAt: string;
}

export function FarmerForum() {
  const [posts, setPosts] = useState<ForumPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [selected, setSelected] = useState<ForumPost | null>(null);
  const [form, setForm] = useState({
    authorName: "",
    authorRole: "",
    topic: "",
    breedTag: "",
    body: "",
  });
  const [submitting, setSubmitting] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/forum");
      const data = await res.json();
      setPosts(data.posts || []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const openPost = async (post: ForumPost) => {
    // For simplicity, replies aren't fetched here; in a full app, fetch by post id
    setSelected(post);
  };

  const submit = async () => {
    if (!form.authorName || !form.topic || !form.body) return;
    setSubmitting(true);
    try {
      const res = await fetch("/api/forum", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (res.ok) {
        setForm({ authorName: "", authorRole: "", topic: "", breedTag: "", body: "" });
        setShowForm(false);
        await load();
      }
    } finally {
      setSubmitting(false);
    }
  };

  if (selected) {
    return (
      <Section bg="white">
        <button
          onClick={() => setSelected(null)}
          className="text-xs text-slate-500 hover:text-brand-navy flex items-center gap-1 mb-4"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Back to forum
        </button>

        <article className="card-soft p-6">
          {selected.breedTag && (
            <div className="mb-3">
              <span className="pill-blue text-[10px]">{selected.breedTag}</span>
            </div>
          )}
          <h2
            className="text-2xl font-bold text-brand-navy mb-3"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            {selected.topic}
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-500 mb-4">
            <div className="h-7 w-7 rounded-full bg-brand-blue-50 flex items-center justify-center">
              <User className="h-3.5 w-3.5 text-brand-blue" />
            </div>
            <div>
              <div className="font-semibold text-brand-navy">{selected.authorName}</div>
              <div className="text-[11px]">{selected.authorRole}</div>
            </div>
            <span className="text-slate-300">·</span>
            <span>{new Date(selected.createdAt).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}</span>
          </div>
          <p className="text-sm text-slate-700 leading-relaxed">{selected.body}</p>
          <div className="mt-4 pt-4 border-t border-brand-line flex items-center gap-4 text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <ThumbsUp className="h-3.5 w-3.5" /> {selected.upvotes} upvotes
            </span>
            <span className="flex items-center gap-1.5">
              <MessageSquare className="h-3.5 w-3.5" /> {selected._count?.replies || 0} replies
            </span>
          </div>
        </article>

        <div className="mt-5 text-center text-xs text-slate-500">
          Reply functionality is read-only in this demo. Use the AI assistant (bottom right) for
          instant expert advice.
        </div>
      </Section>
    );
  }

  return (
    <Section bg="white">
      <SectionHeading
        eyebrow="Community"
        title="Farmer Forum"
        subtitle="Connect with fellow cattle rearers, dairy farmers, and veterinarians across India. Share experiences, ask questions, and learn from each other's practices."
        action={
          <button onClick={() => setShowForm(true)} className="btn-primary">
            <Plus className="h-4 w-4" /> New Post
          </button>
        }
      />

      {loading ? (
        <div className="text-center py-16">
          <Loader2 className="h-8 w-8 mx-auto text-brand-blue animate-spin" />
        </div>
      ) : posts.length === 0 ? (
        <div className="card-soft p-10 text-center text-sm text-slate-500">
          <MessageSquare className="h-10 w-10 mx-auto text-slate-300 mb-2" />
          No posts yet. Be the first to start a discussion!
        </div>
      ) : (
        <div className="space-y-3">
          {posts.map((p) => (
            <button
              key={p.id}
              onClick={() => openPost(p)}
              className="card-soft p-5 text-left hover:shadow-md transition-shadow w-full"
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex items-center gap-2 flex-wrap">
                  {p.breedTag && <span className="pill-blue text-[10px]">{p.breedTag}</span>}
                  <h3
                    className="text-base font-bold text-brand-navy"
                    style={{ fontFamily: "var(--font-montserrat)" }}
                  >
                    {p.topic}
                  </h3>
                </div>
                <ChevronRight className="h-4 w-4 text-slate-400 shrink-0" />
              </div>
              <p className="text-sm text-slate-600 line-clamp-2 leading-relaxed mb-3">{p.body}</p>
              <div className="flex items-center justify-between text-[11px] text-slate-500">
                <div className="flex items-center gap-2">
                  <div className="h-6 w-6 rounded-full bg-brand-blue-50 flex items-center justify-center">
                    <User className="h-3 w-3 text-brand-blue" />
                  </div>
                  <span className="font-semibold text-brand-navy">{p.authorName}</span>
                  <span>·</span>
                  <span>{p.authorRole}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1">
                    <ThumbsUp className="h-3 w-3" /> {p.upvotes}
                  </span>
                  <span className="flex items-center gap-1">
                    <MessageSquare className="h-3 w-3" /> {p._count?.replies || 0}
                  </span>
                  <span>{new Date(p.createdAt).toLocaleDateString("en-IN", { day: "numeric", month: "short" })}</span>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* New post form */}
      {showForm && (
        <div
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setShowForm(false)}
        >
          <div
            className="bg-white rounded-lg shadow-2xl max-w-lg w-full p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <h3
              className="text-lg font-bold text-brand-navy mb-4"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Start a Discussion
            </h3>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    Your Name
                  </label>
                  <input
                    type="text"
                    value={form.authorName}
                    onChange={(e) => setForm({ ...form, authorName: e.target.value })}
                    className="input-soft"
                    placeholder="Rajesh Patel"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    Your Role
                  </label>
                  <input
                    type="text"
                    value={form.authorRole}
                    onChange={(e) => setForm({ ...form, authorRole: e.target.value })}
                    className="input-soft"
                    placeholder="Dairy Farmer, Anand"
                  />
                </div>
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Topic / Title
                </label>
                <input
                  type="text"
                  value={form.topic}
                  onChange={(e) => setForm({ ...form, topic: e.target.value })}
                  className="input-soft"
                  placeholder="Best fodder mix for Gir cows in summer?"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Breed Tag (optional)
                </label>
                <input
                  type="text"
                  value={form.breedTag}
                  onChange={(e) => setForm({ ...form, breedTag: e.target.value })}
                  className="input-soft"
                  placeholder="Gir, Murrah, etc."
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Question / Details
                </label>
                <textarea
                  value={form.body}
                  onChange={(e) => setForm({ ...form, body: e.target.value })}
                  className="input-soft"
                  rows={4}
                  placeholder="Describe your question or share your experience in detail..."
                />
              </div>
            </div>
            <div className="mt-5 flex items-center justify-end gap-2">
              <button onClick={() => setShowForm(false)} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={submit}
                disabled={submitting || !form.authorName || !form.topic || !form.body}
                className="btn-primary"
              >
                {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                Post
              </button>
            </div>
          </div>
        </div>
      )}
    </Section>
  );
}
