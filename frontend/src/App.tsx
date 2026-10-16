import { useCallback, useEffect, useState } from "react";
import { listCats, type CatsResponse } from "./api";
import { CheckPhoto } from "./components/CheckPhoto";
import { MyCats } from "./components/MyCats";

type Tab = "check" | "cats";
const TOKEN_KEY = "cat-detector.admin-token";

function tabFromHash(): Tab {
  return window.location.hash === "#cats" ? "cats" : "check";
}

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
    // keep it for this visit only
  }
}

export function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  // undefined: loading; null: the server can't tell cats apart.
  const [cats, setCats] = useState<CatsResponse | null | undefined>(undefined);
  const [loadError, setLoadError] = useState("");
  const [token, setTokenState] = useState(loadToken);

  const refresh = useCallback(async () => {
    try {
      setCats(await listCats());
      setLoadError("");
    } catch (error) {
      setLoadError((error as Error).message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  function setToken(value: string) {
    setTokenState(value);
    saveToken(value);
  }

  const count = cats?.cats.length ?? 0;

  return (
    <div className="app">
      <header className="header">
        <h1 className="header__title">
          <span aria-hidden="true">🐱</span> Cat Detector
        </h1>
        <nav className="tabs" aria-label="Sections">
          <a href="#check" className={`tabs__tab${tab === "check" ? " tabs__tab--active" : ""}`} aria-current={tab === "check" ? "page" : undefined}>
            Check a photo
          </a>
          <a href="#cats" className={`tabs__tab${tab === "cats" ? " tabs__tab--active" : ""}`} aria-current={tab === "cats" ? "page" : undefined}>
            My cats{count > 0 && <span className="badge">{count}</span>}
          </a>
        </nav>
      </header>

      <main className="main">
        {loadError && (
          <p className="notice notice--error" role="alert">
            {loadError}{" "}
            <button type="button" className="link" onClick={refresh}>
              Try again
            </button>
          </p>
        )}
        {tab === "check" ? (
          <CheckPhoto cats={cats} />
        ) : (
          <MyCats cats={cats} refresh={refresh} token={token} setToken={setToken} />
        )}
      </main>

      <footer className="footer muted small">
        YOLO12n finds the cats; your photos tell them apart. <a href="/docs">API</a>
      </footer>
    </div>
  );
}
