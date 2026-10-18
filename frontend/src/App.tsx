import { useEffect, useState } from "react";
import { useAgent } from "./agent/useAgent";
import { Chat } from "./components/Chat";
import { Detections } from "./components/Detections";
import { displayName, listNames } from "./results";

const TOKEN_KEY = "cat-detector.admin-token";

type Tab = "assistant" | "detections";
const tabFromHash = (): Tab => (window.location.hash === "#detections" ? "detections" : "assistant");

// The token is a per-device convenience; storage may be unavailable (private mode).
function loadToken(): string {
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

function saveToken(token: string) {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // kept for this visit only
  }
}

export function App() {
  const [token, setTokenState] = useState(loadToken);
  const setToken = (value: string) => {
    setTokenState(value);
    saveToken(value);
  };
  const agent = useAgent({ token, setToken });
  const known = agent.cats?.cats.map((c) => displayName(c.name)) ?? [];
  const [tab, setTab] = useState<Tab>(tabFromHash);
  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const detections = agent.messages.reduce((n, m) => n + (m.role === "assistant" && m.check ? m.check.result.findings.length : 0), 0);

  return (
    <div className="app">
      <header className="header">
        <div className="header__who">
          <span className="header__avatar" aria-hidden="true">
            🐱
          </span>
          <div>
            <h1 className="header__title">Cat assistant</h1>
            <p className="header__status">
              <span className="status-dot" aria-hidden="true" />
              {agent.cats === undefined
                ? "Connecting…"
                : known.length
                  ? `Knows ${listNames(known)}`
                  : agent.cats === null
                    ? "Finds cats in photos"
                    : "Finds cats · knows none of yours yet"}
            </p>
          </div>
        </div>
        <nav className="header__links small">
          {token && (
            <button type="button" className="link" onClick={() => setToken("")} title="Forget the admin token saved on this device">
              Forget token
            </button>
          )}
          <a href="/docs">API</a>
        </nav>
      </header>
      <nav className="tabs" aria-label="Views">
        <a href="#assistant" className={`tabs__tab${tab === "assistant" ? " tabs__tab--active" : ""}`} aria-current={tab === "assistant" ? "page" : undefined}>
          Assistant
        </a>
        <a href="#detections" className={`tabs__tab${tab === "detections" ? " tabs__tab--active" : ""}`} aria-current={tab === "detections" ? "page" : undefined}>
          Detections{detections > 0 && <span className="badge">{detections}</span>}
        </a>
      </nav>
      {/* The chat stays mounted so its draft and scroll survive switching tabs. */}
      <div className="view" hidden={tab !== "assistant"}>
        <Chat agent={agent} />
      </div>
      {tab === "detections" && (
        <div className="view view--scroll">
          <Detections messages={agent.messages} onCheck={(files) => agent.send("", files)} />
        </div>
      )}
    </div>
  );
}
