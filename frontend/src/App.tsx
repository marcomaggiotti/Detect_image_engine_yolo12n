import { useState } from "react";
import { useAgent } from "./agent/useAgent";
import { Chat } from "./components/Chat";
import { displayName, listNames } from "./results";

const TOKEN_KEY = "cat-detector.admin-token";

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
      <Chat agent={agent} />
    </div>
  );
}
