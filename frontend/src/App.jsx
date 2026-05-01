import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  CheckCircle2,
  Database,
  Loader2,
  LogOut,
  Plus,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
  Zap,
} from "lucide-react";
import {
  API_BASE,
  getHealth,
  getHistory,
  getMe,
  getStoredToken,
  mediate,
  resolveConversation,
  setStoredToken,
  signIn,
  signOut as apiSignOut,
  signUp,
} from "./api.js";

const DEMO_INPUT = {
  text_a: "You took credit for my idea in the meeting without mentioning my name.",
  text_b: "I built on your idea significantly. The final version was mostly my work.",
};

const MODE_LABELS = {
  quality: "Quality",
  fast: "Fast",
  fast_production: "Production",
};

const RESOLVABILITY_LABELS = {
  resolvable: "Resolvable",
  partially_resolvable: "Partial",
  non_resolvable: "Non-resolvable",
};

const RATING_OPTIONS = [
  { value: 1, label: "Hard" },
  { value: 2, label: "Tense" },
  { value: 3, label: "Better" },
  { value: 4, label: "Good" },
  { value: 5, label: "Resolved" },
];

const LOADER_MESSAGES = [
  "Analyzing perspectives",
  "Classifying conflict",
  "Generating responses",
  "Checking for bias",
  "Finalizing",
];

function newConversationId() {
  return crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
}

function cleanType(value) {
  return String(value || "emotional").toLowerCase().replace(/[^a-z0-9_-]/g, "");
}

function titleCase(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function initials(name) {
  return String(name || "?")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "?";
}

function formatAgo(value) {
  if (!value) return "now";
  const diff = Math.floor((Date.now() - new Date(value).getTime()) / 60000);
  if (!Number.isFinite(diff) || diff < 1) return "now";
  if (diff < 60) return `${diff}m`;
  if (diff < 1440) return `${Math.floor(diff / 60)}h`;
  return new Date(value).toLocaleDateString("en", { month: "short", day: "numeric" });
}

function formatSafetyRating(value) {
  const score = Number(value);
  if (!Number.isFinite(score)) return "N/A";
  const rating = Math.max(0, Math.min(100, (1 - score) * 100));
  return rating >= 99.5 ? "99%+" : `${Math.round(rating)}%`;
}

function wordCount(value) {
  return String(value || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}

function makeRecord(result, textA, textB, mode) {
  return {
    ...result,
    id: result.request_id,
    text_a: textA,
    text_b: textB,
    mode,
    created_at: new Date().toISOString(),
  };
}

function isResolvedRecord(record) {
  return Boolean(
    record?.resolved ||
      record?.conversation_lifecycle_status === "resolved" ||
      record?.conversation_status === "resolved" ||
      record?.resolved_at
  );
}

function AuthScreen({ onAuthed }) {
  const [tab, setTab] = useState("in");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      const data =
        tab === "up"
          ? await signUp({ name, email, password })
          : await signIn({ email, password });
      onAuthed(data.user);
    } catch (err) {
      setMessage(err.message || "Authentication failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-screen">
      <form className="auth-wrap" onSubmit={submit}>
        <div className="auth-logo">
          <div className="auth-logo-mark">
            ConcordAI<span>.</span>
          </div>
          <div className="auth-logo-sub">Conflict Intelligence</div>
        </div>

        <div className="auth-tabs">
          <button
            className={tab === "in" ? "auth-tab on" : "auth-tab"}
            type="button"
            onClick={() => setTab("in")}
          >
            Sign in
          </button>
          <button
            className={tab === "up" ? "auth-tab on" : "auth-tab"}
            type="button"
            onClick={() => setTab("up")}
          >
            Register
          </button>
        </div>

        {tab === "up" && (
          <label className="field">
            <span>Full name</span>
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Your name" />
          </label>
        )}

        <label className="field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@domain.com"
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder={tab === "up" ? "Min 6 chars" : "Password"}
          />
        </label>

        <button className="auth-submit" type="submit" disabled={busy}>
          {busy ? "Please wait" : tab === "up" ? "Create account" : "Sign in"}
        </button>

        <div className="auth-msg">{message}</div>
      </form>
    </main>
  );
}

function HistoryItem({ item, active, onOpen }) {
  const summary = item.one_line_summary || item.text_a || "Untitled mediation";
  const type = cleanType(item.conflict_type);
  const resolved = isResolvedRecord(item);

  return (
    <button className={active ? "hist-item on" : "hist-item"} type="button" onClick={() => onOpen(item)}>
      <span className="hist-title">{summary}</span>
      <span className="hist-foot">
        <span className={`ctag ct-${type}`}>{type}</span>
        {resolved ? <span className="resolved-mini">Resolved</span> : null}
        <span className="hist-ago">{formatAgo(item.created_at)}</span>
      </span>
    </button>
  );
}

function ResultView({
  record,
  onNew,
  nextA,
  nextB,
  onNextA,
  onNextB,
  onContinue,
  onResolve,
  loading,
  resolving,
  resolved,
  canResolve,
  draftPresent,
}) {
  const trace = record.trace || {};
  const pct = Math.round((record.confidence || 0) * 100);
  const status = resolved ? "Resolved" : record.conversation_status || "Stable";
  const time = record.processing_time_seconds;
  const ragActive = Boolean(trace.rag_used);
  const safetyRating = formatSafetyRating(trace.safety_score ?? record.safety_score);
  const intentA = trace.intent_a || record.intent_a || "unknown";
  const intentB = trace.intent_b || record.intent_b || "unknown";
  const nextDisabled = loading || resolved;
  const intentPill = (intent) => (
    <div className="mpill intent-pill">
      <span className="mpill-l">Intent</span>
      <span className="mpill-v">{titleCase(intent)}</span>
    </div>
  );
  const sharedMetrics = (
    <>
      <div className="mpill">
        <span className="mpill-l">Type</span>
        <span className="mpill-v">{titleCase(record.conflict_type)}</span>
      </div>
      <div className="mpill">
        <span className="mpill-l">Resolvability</span>
        <span className="mpill-v">{RESOLVABILITY_LABELS[record.resolvability] || titleCase(record.resolvability)}</span>
      </div>
      <div className="mpill">
        <span className={`sdot s-${status}`}></span>
        <span className="mpill-l">Status</span>
        <span className="mpill-v">{status}</span>
      </div>
      <div className="mpill safety-pill">
        <span className="mpill-l">Safety</span>
        <span className="mpill-v">{safetyRating}</span>
      </div>
      {time ? (
        <div className="mpill">
          <span className="mpill-l">Time</span>
          <span className="mpill-v">{time.toFixed(1)}s</span>
        </div>
      ) : null}
      <div className="mpill">
        <span className="mpill-l">RAG</span>
        <span className="mpill-v">{ragActive ? `${trace.retrieved_cases || 0} cases` : "Skipped"}</span>
      </div>
      {resolved && record.resolved_turn ? (
        <div className="mpill">
          <span className="mpill-l">Turn</span>
          <span className="mpill-v">{record.resolved_turn}</span>
        </div>
      ) : null}
    </>
  );

  return (
    <section className="result-view">
      <div className="result-bar">
        <button className="back" type="button" onClick={onNew}>
          <ArrowLeft size={14} aria-hidden="true" />
          New
        </button>
        <div className="result-summ">{record.one_line_summary || "The mediation agents completed this turn."}</div>
        <div className="conf-chip">
          <span className="conf-l">Conf.</span>
          <span className="conf-v">{pct}%</span>
        </div>
        {resolved ? (
          <div className="resolved-chip">
            <CheckCircle2 size={14} aria-hidden="true" />
            Resolved
          </div>
        ) : null}
      </div>

      <div className="result-body">
        <div className="orig-row orig-row-top">
          <article className="orig-card">
            <div className="orig-l">
              <span className="orig-dot orig-dot-a"></span>
              User A input
            </div>
            <p className="orig-q">"{record.text_a}"</p>
          </article>
          <article className="orig-card">
            <div className="orig-l">
              <span className="orig-dot orig-dot-b"></span>
              User B input
            </div>
            <p className="orig-q">"{record.text_b}"</p>
          </article>
        </div>

        <article className="resp-card">
          <div className="resp-head">
            <div className="resp-badge rb-a">A</div>
            <div>
              <div className="resp-title">Response for User A</div>
              <div className="resp-sub">Addressing their view</div>
            </div>
          </div>
          <p className="resp-body">{record.response_a}</p>
        </article>

        <article className="resp-card">
          <div className="resp-head">
            <div className="resp-badge rb-b">B</div>
            <div>
              <div className="resp-title">Response for User B</div>
              <div className="resp-sub">Addressing their view</div>
            </div>
          </div>
          <p className="resp-body">{record.response_b}</p>
        </article>

        <div className="analysis-row">
          <div className="analysis-panel analysis-a">
            <div className="analysis-title">
              <span className="resp-badge rb-a">A</span>
              User A response analysis
            </div>
            <div className="analysis-pills">
              {intentPill(intentA)}
              {sharedMetrics}
            </div>
          </div>

          <div className="analysis-panel analysis-b">
            <div className="analysis-title">
              <span className="resp-badge rb-b">B</span>
              User B response analysis
            </div>
            <div className="analysis-pills">
              {intentPill(intentB)}
              {sharedMetrics}
            </div>
          </div>
        </div>

        <div className={resolved ? "next-turn locked" : "next-turn"}>
          <label className="next-card">
            <span className="next-head">
              <span className="ip-badge ba">A</span>
              User A next message
            </span>
            <textarea
              value={nextA}
              onChange={(event) => onNextA(event.target.value)}
              placeholder={resolved ? "This conversation is resolved." : "What does User A say next?"}
              disabled={nextDisabled}
            />
          </label>

          <label className="next-card">
            <span className="next-head">
              <span className="ip-badge bb">B</span>
              User B next message
            </span>
            <textarea
              value={nextB}
              onChange={(event) => onNextB(event.target.value)}
              placeholder={resolved ? "This conversation is resolved." : "What does User B say next?"}
              disabled={nextDisabled}
            />
          </label>

          <div className={resolved ? "next-actions resolved" : "next-actions"}>
            {resolved ? (
              <div className="resolved-state">
                <CheckCircle2 size={16} aria-hidden="true" />
                <span>Conversation resolved</span>
              </div>
            ) : (
              <>
                <div className="action-copy">
                  <span>Next step</span>
                  <small>{draftPresent ? "Resolving will clear the drafted next messages." : "Continue the exchange or close it as resolved."}</small>
                </div>
                <div className="action-buttons">
                  <button
                    className="resolve-btn"
                    type="button"
                    onClick={onResolve}
                    disabled={!canResolve || loading || resolving}
                  >
                    {resolving ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <Sparkles size={15} aria-hidden="true" />}
                    Conflict resolved
                  </button>
                  <button className="continue-btn" type="button" onClick={onContinue} disabled={nextDisabled}>
                    {loading ? <Loader2 className="spin" size={14} aria-hidden="true" /> : <Send size={14} aria-hidden="true" />}
                    Continue
                  </button>
                </div>
              </>
            )}
          </div>
        </div>

      </div>
    </section>
  );
}

function ResolveModal({
  stage,
  ratingA,
  ratingB,
  commentA,
  commentB,
  resolvedTurn,
  saving,
  error,
  onRatingA,
  onRatingB,
  onCommentA,
  onCommentB,
  onRate,
  onSkip,
  onSave,
}) {
  const isResolving = stage === "resolving";
  const isRating = stage === "rating";
  const title = isResolving
    ? "Marking as resolved"
    : isRating
      ? "Rate the conversation"
      : "Congratulations, conflict resolved";
  const subtitle = isResolving
    ? "Saving the final state and locking this conversation."
    : isRating
      ? "Add separate feedback for each user. Ratings are required; comments are optional."
      : `This conversation is now marked as resolved${resolvedTurn ? ` at turn ${resolvedTurn}` : ""}.`;

  return (
    <div className="modal-backdrop" role="presentation">
      <div className={`resolve-modal ${stage}`} role="dialog" aria-modal="true" aria-label="Resolve conflict">
        <div className="modal-glow"></div>
        {Array.from({ length: isResolving ? 18 : 28 }).map((_, index) => (
          <span key={index} className="particle" style={{ "--i": index }}></span>
        ))}
        <div className={isResolving ? "modal-icon working" : "modal-icon success"}>
          {isResolving ? <Loader2 className="spin" size={21} aria-hidden="true" /> : <CheckCircle2 size={22} aria-hidden="true" />}
        </div>
        <h2>{title}</h2>
        <p>{subtitle}</p>

        {isResolving ? <div className="resolve-progress">Finalizing resolution</div> : null}

        {stage === "success" ? (
          <div className="resolved-confirm">
            <div className="resolved-confirm-line">
              <CheckCircle2 size={16} aria-hidden="true" />
              Marked as resolved
            </div>
            <span>Feedback can be added now or skipped without reopening the conflict.</span>
          </div>
        ) : null}

        {isRating ? (
          <>
            <div className="rating-block">
              <div className="rating-label">User A rating</div>
              <div className="rating-row">
                {RATING_OPTIONS.map((option) => (
                  <button
                    key={`a-${option.value}`}
                    type="button"
                    className={ratingA === option.value ? "rating on" : "rating"}
                    onClick={() => onRatingA(option.value)}
                    disabled={saving}
                  >
                    <span>{option.value}</span>
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            <label className="resolve-note">
              <span>User A comment</span>
              <textarea
                value={commentA}
                onChange={(event) => onCommentA(event.target.value)}
                placeholder="What should be remembered from User A's side?"
                disabled={saving}
              />
            </label>

            <div className="rating-block">
              <div className="rating-label">User B rating</div>
              <div className="rating-row">
                {RATING_OPTIONS.map((option) => (
                  <button
                    key={`b-${option.value}`}
                    type="button"
                    className={ratingB === option.value ? "rating on" : "rating"}
                    onClick={() => onRatingB(option.value)}
                    disabled={saving}
                  >
                    <span>{option.value}</span>
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            <label className="resolve-note">
              <span>User B comment</span>
              <textarea
                value={commentB}
                onChange={(event) => onCommentB(event.target.value)}
                placeholder="What should be remembered from User B's side?"
                disabled={saving}
              />
            </label>
          </>
        ) : null}

        {error ? <div className="modal-error">{error}</div> : null}

        {!isResolving || error ? (
          <div className="modal-actions">
            <button className="modal-secondary" type="button" onClick={onSkip} disabled={saving}>
              {isRating ? "Skip" : error ? "Close" : "Skip"}
            </button>
            {isRating ? (
              <button className="modal-primary" type="button" onClick={onSave} disabled={saving || !ratingA || !ratingB}>
                {saving ? <Loader2 className="spin" size={14} aria-hidden="true" /> : <CheckCircle2 size={14} aria-hidden="true" />}
                Save feedback
              </button>
            ) : !error ? (
              <button className="modal-primary" type="button" onClick={onRate} disabled={saving}>
                <Sparkles size={14} aria-hidden="true" />
                Rate conversation
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function CelebrationBurst() {
  return (
    <div className="celebration" aria-hidden="true">
      <div className="celebration-core">
        <CheckCircle2 size={30} aria-hidden="true" />
      </div>
      {Array.from({ length: 44 }).map((_, index) => (
        <span key={index} style={{ "--i": index }}></span>
      ))}
    </div>
  );
}

function App() {
  const [booting, setBooting] = useState(true);
  const [user, setUser] = useState(null);
  const [health, setHealth] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeRecord, setActiveRecord] = useState(null);
  const [conversationId, setConversationId] = useState(newConversationId);
  const [turn, setTurn] = useState(1);
  const [mode, setMode] = useState("fast_production");
  const [inputMode, setInputMode] = useState("demo");
  const [textA, setTextA] = useState(DEMO_INPUT.text_a);
  const [textB, setTextB] = useState(DEMO_INPUT.text_b);
  const [loading, setLoading] = useState(false);
  const [loaderIndex, setLoaderIndex] = useState(0);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [nextA, setNextA] = useState("");
  const [nextB, setNextB] = useState("");
  const [resolveOpen, setResolveOpen] = useState(false);
  const [resolveStage, setResolveStage] = useState("success");
  const [ratingA, setRatingA] = useState(null);
  const [ratingB, setRatingB] = useState(null);
  const [commentA, setCommentA] = useState("");
  const [commentB, setCommentB] = useState("");
  const [resolveSaving, setResolveSaving] = useState(false);
  const [resolveError, setResolveError] = useState("");
  const [celebrating, setCelebrating] = useState(false);
  const userATextareaRef = useRef(null);

  async function refreshHealth() {
    try {
      setHealth(await getHealth());
    } catch {
      setHealth(null);
    }
  }

  async function refreshHistory() {
    const data = await getHistory();
    const rows = data.history || [];
    setHistory(rows);
    return rows;
  }

  async function completeAuth(nextUser) {
    setUser(nextUser);
    setNotice(`Welcome, ${nextUser.name.split(" ")[0] || nextUser.name}`);
    await refreshHistory();
  }

  useEffect(() => {
    async function boot() {
      await refreshHealth();
      if (!getStoredToken()) {
        setBooting(false);
        return;
      }
      try {
        const data = await getMe();
        setUser(data.user);
        await refreshHistory();
      } catch {
        setStoredToken(null);
      } finally {
        setBooting(false);
      }
    }
    boot();
  }, []);

  useEffect(() => {
    if (!loading) return undefined;
    const timer = window.setInterval(() => {
      setLoaderIndex((current) => (current + 1) % LOADER_MESSAGES.length);
    }, 1400);
    return () => window.clearInterval(timer);
  }, [loading]);

  useEffect(() => {
    if (!notice && !error) return undefined;
    const timer = window.setTimeout(() => {
      setNotice("");
      setError("");
    }, 3200);
    return () => window.clearTimeout(timer);
  }, [notice, error]);

  const kpis = useMemo(() => {
    const count = history.length;
    const improving = history.filter((item) => ["Improving", "Resolved"].includes(item.conversation_status)).length;
    const avg = count
      ? Math.round((history.reduce((total, item) => total + Number(item.confidence || 0), 0) / count) * 100)
      : null;
    return { count, improving, avg };
  }, [history]);

  function focusUserA() {
    window.requestAnimationFrame(() => userATextareaRef.current?.focus());
  }

  function resetDraft() {
    setActiveRecord(null);
    setConversationId(newConversationId());
    setTurn(1);
    setNextA("");
    setNextB("");
    setResolveOpen(false);
    setResolveStage("success");
    setRatingA(null);
    setRatingB(null);
    setCommentA("");
    setCommentB("");
    setResolveError("");
    setError("");
  }

  function loadDemoInput() {
    resetDraft();
    setInputMode("demo");
    setTextA(DEMO_INPUT.text_a);
    setTextB(DEMO_INPUT.text_b);
  }

  function startCustomInput() {
    resetDraft();
    setInputMode("custom");
    setTextA("");
    setTextB("");
    focusUserA();
  }

  function clearInputs() {
    setInputMode("custom");
    setTextA("");
    setTextB("");
    setError("");
    focusUserA();
  }

  function goHome() {
    startCustomInput();
  }

  function openHistory(item) {
    setActiveRecord(item);
    setConversationId(item.conversation_id || newConversationId());
    setTurn((item.latest_turn || item.turn || 1) + 1);
    setNextA("");
    setNextB("");
    setResolveOpen(false);
    setResolveStage("success");
    setResolveError("");
    setRatingA(item.user_a_rating || null);
    setRatingB(item.user_b_rating || null);
    setCommentA(item.user_a_comment || "");
    setCommentB(item.user_b_comment || "");
  }

  async function handleSignOut() {
    await apiSignOut();
    setUser(null);
    setHistory([]);
    setActiveRecord(null);
    setConversationId(newConversationId());
    setTurn(1);
    setInputMode("demo");
    setTextA(DEMO_INPUT.text_a);
    setTextB(DEMO_INPUT.text_b);
    setNextA("");
    setNextB("");
    setResolveOpen(false);
    setResolveStage("success");
    setRatingA(null);
    setRatingB(null);
    setCommentA("");
    setCommentB("");
  }

  async function handleStateError(err) {
    const payload = err.payload || {};
    const detail = payload.error && typeof payload.error === "object" ? payload.error : payload;
    const code = err.code || detail.code || payload.code || payload.error;
    if (code === "ALREADY_RESOLVED") {
      setActiveRecord((current) =>
        current
          ? {
              ...current,
              ...detail,
              resolved: true,
              conversation_lifecycle_status: "resolved",
            }
          : current
      );
      setNextA("");
      setNextB("");
      setResolveOpen(false);
      setNotice("Conversation already resolved.");
      await refreshHistory();
      return true;
    }
    if (code === "STALE_TURN") {
      const rows = await refreshHistory();
      const latest = rows.find((item) => item.conversation_id === conversationId);
      if (latest) {
        setActiveRecord(latest);
        setTurn((latest.latest_turn || latest.turn || detail.latest_turn || 1) + 1);
      } else if (detail.latest_turn) {
        setTurn(detail.latest_turn + 1);
      }
      setNextA("");
      setNextB("");
      setNotice("Updated to latest turn.");
      return true;
    }
    return false;
  }

  async function submit() {
    if (loading) return;
    const trimmedA = textA.trim();
    const trimmedB = textB.trim();
    if (!trimmedA || !trimmedB) {
      setError("Add both perspectives before running mediation.");
      return;
    }
    if (trimmedA.length < 10 || trimmedB.length < 10) {
      setError("Each perspective needs at least 10 characters.");
      return;
    }

    setLoading(true);
    setLoaderIndex(0);
    setError("");
    try {
      const data = await mediate({
        text_a: trimmedA,
        text_b: trimmedB,
        conversation_id: conversationId,
        turn,
        mode,
      });
      const record = makeRecord(data, trimmedA, trimmedB, mode);
      setActiveRecord(record);
      setHistory((current) => [record, ...current.filter((item) => item.request_id !== record.request_id)]);
      setTurn(data.turn + 1);
      setNextA("");
      setNextB("");
      setNotice("Mediation saved to your history.");
      await refreshHistory();
      await refreshHealth();
    } catch (err) {
      if (!(await handleStateError(err))) {
        setError(err.message || "The mediation request failed.");
      }
    } finally {
      setLoading(false);
    }
  }

  async function submitNextTurn() {
    if (loading || !activeRecord || isResolvedRecord(activeRecord)) return;
    const trimmedA = nextA.trim();
    const trimmedB = nextB.trim();
    if (!trimmedA || !trimmedB) {
      setError("Add both next messages before continuing.");
      return;
    }
    if (trimmedA.length < 10 || trimmedB.length < 10) {
      setError("Each next message needs at least 10 characters.");
      return;
    }

    setLoading(true);
    setLoaderIndex(0);
    setError("");
    try {
      const data = await mediate({
        text_a: trimmedA,
        text_b: trimmedB,
        conversation_id: activeRecord.conversation_id || conversationId,
        turn,
        mode,
      });
      const record = makeRecord(data, trimmedA, trimmedB, mode);
      setActiveRecord(record);
      setConversationId(data.conversation_id);
      setHistory((current) => [record, ...current.filter((item) => item.request_id !== record.request_id)]);
      setTurn(data.turn + 1);
      setNextA("");
      setNextB("");
      setNotice("Next turn saved.");
      await refreshHistory();
      await refreshHealth();
    } catch (err) {
      if (!(await handleStateError(err))) {
        setError(err.message || "The next turn failed.");
      }
    } finally {
      setLoading(false);
    }
  }

  async function resolveNow() {
    if (!activeRecord || isResolvedRecord(activeRecord) || loading || resolveSaving) return;
    setResolveStage("resolving");
    setResolveError("");
    setResolveOpen(true);
    setResolveSaving(true);
    try {
      const data = await resolveConversation(activeRecord.conversation_id, {});
      setActiveRecord((current) =>
        current
          ? {
              ...current,
              ...data,
              resolved: true,
              conversation_lifecycle_status: "resolved",
            }
          : current
      );
      setNextA("");
      setNextB("");
      setResolveStage("success");
      setCelebrating(true);
      setNotice("Conversation marked as resolved.");
      window.setTimeout(() => setCelebrating(false), 1800);
      await refreshHistory();
    } catch (err) {
      if (await handleStateError(err)) {
        setResolveOpen(false);
      } else {
        setResolveError(err.message || "Could not mark this conversation resolved.");
      }
    } finally {
      setResolveSaving(false);
    }
  }

  async function saveResolution() {
    if (resolveSaving || !activeRecord || !ratingA || !ratingB) return;
    setResolveSaving(true);
    setResolveError("");
    try {
      const data = await resolveConversation(activeRecord.conversation_id, {
        user_a_rating: ratingA,
        user_b_rating: ratingB,
        user_a_comment: commentA.trim() || undefined,
        user_b_comment: commentB.trim() || undefined,
      });
      setActiveRecord((current) =>
        current
          ? {
              ...current,
              ...data,
              resolved: true,
              conversation_lifecycle_status: "resolved",
            }
          : current
      );
      setResolveOpen(false);
      setResolveStage("success");
      setNotice("Resolution feedback saved.");
      await refreshHistory();
    } catch (err) {
      setResolveError(err.message || "Could not save feedback.");
    } finally {
      setResolveSaving(false);
    }
  }

  function skipResolutionFeedback() {
    if (resolveSaving) return;
    setResolveOpen(false);
    setResolveStage("success");
    setResolveError("");
  }

  if (booting) {
    return (
      <main className="boot-screen">
        <Loader2 className="spin" size={24} aria-hidden="true" />
        <span>Loading ConcordAI</span>
      </main>
    );
  }

  if (!user) {
    return <AuthScreen onAuthed={completeAuth} />;
  }

  const activeResolved = isResolvedRecord(activeRecord);
  const draftPresent = Boolean(nextA.trim() || nextB.trim());
  const canResolve = Boolean(activeRecord && activeRecord.turn >= 1 && !activeResolved);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="nav-head">
          <div className="logo">
            ConcordAI<span>.</span>
          </div>
          <div className="user-row">
            <div className="avatar">{initials(user.name)}</div>
            <div className="uname">{user.name}</div>
            <button className="sign-out" type="button" onClick={handleSignOut} title="Sign out">
              <LogOut size={14} aria-hidden="true" />
            </button>
          </div>
        </div>

        <button className="new-session" type="button" onClick={goHome}>
          <Plus size={14} aria-hidden="true" />
          New session
        </button>

        <div className="nav-label">History</div>
        <div className="hist">
          {history.length ? (
            history.map((item) => (
              <HistoryItem
                key={item.id || item.request_id}
                item={item}
                active={(activeRecord?.id || activeRecord?.request_id) === (item.id || item.request_id)}
                onOpen={openHistory}
              />
            ))
          ) : (
            <div className="no-hist">No sessions yet</div>
          )}
        </div>

        <div className="nav-foot">
          <div className="kpi">
            <div className="kpi-n">{kpis.count}</div>
            <div className="kpi-l">Sessions</div>
          </div>
          <div className="kpi">
            <div className="kpi-n">{kpis.improving}</div>
            <div className="kpi-l">Improving</div>
          </div>
          <div className="kpi">
            <div className="kpi-n">{kpis.avg === null ? "-" : `${kpis.avg}%`}</div>
            <div className="kpi-l">Avg conf.</div>
          </div>
        </div>
      </aside>

      <section className="workspace">
        {loading && (
          <div className="loader">
            <div className="loader-ring"></div>
            <div className="loader-txt">{LOADER_MESSAGES[loaderIndex]}</div>
          </div>
        )}

        {!activeRecord ? (
          <section className="home-view">
            <div className="workbench">
              <header className="home-head">
                <div>
                  <div className="home-eyebrow">Mediation Workspace</div>
                  <h1 className="home-h">
                    Turn disagreement into a clear next step
                  </h1>
                  <p className="home-sub">
                    Enter both perspectives, review the conflict analysis, and keep a history of each mediated session.
                  </p>
                </div>
                <div className="status-stack">
                  <div className={health ? "status-pill ok" : "status-pill bad"}>
                    <ShieldCheck size={14} aria-hidden="true" />
                    API {health ? "online" : "offline"}
                  </div>
                  <div className="status-pill">
                    <Database size={14} aria-hidden="true" />
                    Saved history
                  </div>
                  <div className="status-pill">
                    <Sparkles size={14} aria-hidden="true" />
                    Similar cases
                  </div>
                </div>
              </header>

              <div className="control-row">
                <div className="input-actions" aria-label="Input options">
                  <button
                    className={inputMode === "demo" ? "input-action on" : "input-action"}
                    type="button"
                    onClick={loadDemoInput}
                  >
                    Demo example
                  </button>
                  <button
                    className={inputMode === "custom" ? "input-action on" : "input-action"}
                    type="button"
                    onClick={startCustomInput}
                  >
                    Custom input
                  </button>
                  <button className="input-action ghost" type="button" onClick={clearInputs}>
                    Clear
                  </button>
                </div>

                <div className="mode-switch" aria-label="Processing mode">
                  {Object.entries(MODE_LABELS).map(([value, label]) => (
                    <button
                      key={value}
                      className={mode === value ? "on" : ""}
                      type="button"
                      onClick={() => setMode(value)}
                    >
                      {value === "fast_production" ? (
                        <Zap size={14} aria-hidden="true" />
                      ) : (
                        <RefreshCw size={14} aria-hidden="true" />
                      )}
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="input-panel">
                <div className="ip-cols">
                  <label className="ip-col">
                    <span className="ip-col-head">
                      <span className="ip-badge ba">A</span>
                      <span className="ip-col-lbl">First perspective</span>
                    </span>
                    <textarea
                      ref={userATextareaRef}
                      value={textA}
                      onChange={(event) => {
                        setTextA(event.target.value);
                        setInputMode("custom");
                        setError("");
                      }}
                      placeholder="Describe User A's concern, boundary, or statement..."
                    />
                  </label>
                  <label className="ip-col">
                    <span className="ip-col-head">
                      <span className="ip-badge bb">B</span>
                      <span className="ip-col-lbl">Second perspective</span>
                    </span>
                    <textarea
                      value={textB}
                      onChange={(event) => {
                        setTextB(event.target.value);
                        setInputMode("custom");
                        setError("");
                      }}
                      placeholder="Describe User B's response, intent, or counterpoint..."
                    />
                  </label>
                </div>
                <div className="ip-foot">
                  <div>
                    <div className="ip-chars">
                      A: {textA.length} | B: {textB.length}
                    </div>
                    {error ? <div className="inline-error">{error}</div> : null}
                  </div>
                  <button className="go-btn" type="button" onClick={submit} disabled={loading}>
                    {loading ? (
                      <Loader2 className="spin" size={14} aria-hidden="true" />
                    ) : (
                      <Send size={14} aria-hidden="true" />
                    )}
                    Mediate
                  </button>
                </div>
              </div>
            </div>
          </section>
        ) : (
          <ResultView
            record={activeRecord}
            onNew={goHome}
            nextA={nextA}
            nextB={nextB}
            onNextA={(value) => {
              setNextA(value);
              setError("");
            }}
            onNextB={(value) => {
              setNextB(value);
              setError("");
            }}
            onContinue={submitNextTurn}
            onResolve={resolveNow}
            loading={loading}
            resolving={resolveSaving}
            resolved={activeResolved}
            canResolve={canResolve}
            draftPresent={draftPresent}
          />
        )}
      </section>

      {resolveOpen && (
        <ResolveModal
          stage={resolveStage}
          ratingA={ratingA}
          ratingB={ratingB}
          commentA={commentA}
          commentB={commentB}
          resolvedTurn={activeRecord?.resolved_turn}
          saving={resolveSaving}
          error={resolveError}
          onRatingA={setRatingA}
          onRatingB={setRatingB}
          onCommentA={setCommentA}
          onCommentB={setCommentB}
          onRate={() => {
            setResolveError("");
            setResolveStage("rating");
          }}
          onSkip={skipResolutionFeedback}
          onSave={saveResolution}
        />
      )}
      {celebrating ? <CelebrationBurst /> : null}
      {(notice || error) && <div className={error ? "toast show err" : "toast show"}>{error || notice}</div>}
      <div className="api-base">{API_BASE}</div>
    </main>
  );
}

export default App;
