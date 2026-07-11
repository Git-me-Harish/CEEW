"use client";

import { useState, useRef, useEffect } from "react";
import { MessageSquare, X, Send, Bot, User } from "lucide-react";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

const SUGGESTED_PROMPTS = [
  "Best indigenous breed for 50kg/day milk yield?",
  "FMD vaccination schedule for my Gir cows",
  "Summer ration for lactating Murrah buffalo",
  "Symptoms of mastitis — what to do?",
  "Government schemes for dairy farmers",
  "How to identify a purebred Sahiwal?",
];

const WELCOME_MESSAGE: ChatMessage = {
  role: "assistant",
  content:
    "Namaste! I'm PashuMitra, your AI assistant for Indian bovine management. Ask me about cattle/buffalo breeds, health, nutrition, government schemes, or any dairy farming question. How can I help today?",
};

export function AIAssistant() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading, open]);

  const send = async (text: string) => {
    if (!text.trim() || loading) return;
    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: messages.slice(-6).map((m) => ({ role: m.role, content: m.content })),
        }),
      });
      const data = await res.json();
      const reply = data.response || data.error || "Sorry, I couldn't process your request.";
      setMessages((m) => [...m, { role: "assistant", content: reply }]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content:
            "I'm having trouble connecting right now. Please check your internet connection or try again in a moment.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating button */}
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-5 right-5 z-50 h-14 w-14 rounded-full bg-brand-navy text-white shadow-xl hover:bg-brand-blue transition-colors flex items-center justify-center"
          aria-label="Open AI assistant"
        >
          <MessageSquare className="h-6 w-6" />
          <span className="absolute -top-1 -right-1 h-3.5 w-3.5 bg-brand-green rounded-full border-2 border-white" />
        </button>
      )}

      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-0 right-0 sm:bottom-5 sm:right-5 z-50 w-full sm:w-[420px] h-[100vh] sm:h-[600px] sm:max-h-[80vh] bg-white sm:rounded-lg shadow-2xl border border-brand-line flex flex-col overflow-hidden">
          {/* Header */}
          <div className="bg-brand-navy text-white px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="h-8 w-8 rounded-full bg-brand-amber flex items-center justify-center">
                <Bot className="h-4 w-4 text-brand-navy" />
              </div>
              <div>
                <div
                  className="text-sm font-semibold"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  PashuMitra AI
                </div>
                <div className="text-[11px] text-white/70 flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 bg-brand-green rounded-full" />
                  Online · bovine specialist
                </div>
              </div>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="p-1.5 rounded-md hover:bg-white/10"
              aria-label="Close chat"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Messages */}
          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto p-4 space-y-3 bg-brand-mist"
          >
            {messages.map((m, i) => (
              <div
                key={i}
                className={`flex gap-2 ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {m.role === "assistant" && (
                  <div className="h-7 w-7 rounded-full bg-brand-navy flex items-center justify-center shrink-0">
                    <Bot className="h-3.5 w-3.5 text-white" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] px-3.5 py-2.5 text-sm rounded-lg leading-relaxed ${
                    m.role === "user"
                      ? "bg-brand-navy text-white rounded-br-sm"
                      : "bg-white text-slate-800 border border-brand-line rounded-bl-sm"
                  }`}
                >
                  {m.content.split("\n").map((line, j) => (
                    <p key={j} className={j > 0 ? "mt-2" : ""}>
                      {line}
                    </p>
                  ))}
                </div>
                {m.role === "user" && (
                  <div className="h-7 w-7 rounded-full bg-brand-blue flex items-center justify-center shrink-0">
                    <User className="h-3.5 w-3.5 text-white" />
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex gap-2 justify-start">
                <div className="h-7 w-7 rounded-full bg-brand-navy flex items-center justify-center shrink-0">
                  <Bot className="h-3.5 w-3.5 text-white" />
                </div>
                <div className="bg-white border border-brand-line px-3.5 py-3 rounded-lg rounded-bl-sm">
                  <div className="flex gap-1">
                    <span className="h-1.5 w-1.5 bg-slate-400 rounded-full animate-bounce" />
                    <span
                      className="h-1.5 w-1.5 bg-slate-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.15s" }}
                    />
                    <span
                      className="h-1.5 w-1.5 bg-slate-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.3s" }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Suggested prompts */}
          {messages.length <= 1 && (
            <div className="px-3 py-2 border-t border-brand-line bg-white">
              <div className="text-[11px] text-slate-500 mb-1.5 font-medium">
                Suggested questions
              </div>
              <div className="flex flex-wrap gap-1.5">
                {SUGGESTED_PROMPTS.map((p) => (
                  <button
                    key={p}
                    onClick={() => send(p)}
                    className="text-[11px] px-2 py-1 rounded-md bg-brand-blue-50 text-brand-blue hover:bg-brand-blue hover:text-white transition-colors text-left"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <div className="border-t border-brand-line bg-white p-3 flex items-center gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") send(input);
              }}
              placeholder="Ask about breeds, health, nutrition, schemes..."
              className="flex-1 px-3 py-2 text-sm border border-brand-line rounded-md focus:outline-none focus:ring-2 focus:ring-brand-blue/30 focus:border-brand-blue"
              disabled={loading}
            />
            <button
              onClick={() => send(input)}
              disabled={loading || !input.trim()}
              className="h-9 w-9 rounded-md bg-brand-navy text-white flex items-center justify-center hover:bg-brand-blue disabled:opacity-40 transition-colors"
              aria-label="Send message"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
