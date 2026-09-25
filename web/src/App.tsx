import { useEffect, useState } from "react";
import DatasetTab from "./components/DatasetTab";
import { DatabaseIcon, MonitorIcon, MoonIcon, SearchIcon, SunIcon } from "./components/icons";
import SearchTab, { type SearchRequest } from "./components/SearchTab";
import styles from "./App.module.scss";

type Tab = "search" | "dataset";
const TABS = [
  { id: "search", label: "Tìm kiếm", icon: SearchIcon },
  { id: "dataset", label: "Dataset", icon: DatabaseIcon },
] as const;

type Theme = "system" | "light" | "dark";
const THEMES: Theme[] = ["system", "light", "dark"];
const THEME_ICON = { system: MonitorIcon, light: SunIcon, dark: MoonIcon };
const THEME_LABEL = { system: "Theo hệ thống", light: "Sáng", dark: "Tối" };
const THEME_KEY = "vsearch.theme";

const tabFromHash = (): Tab => (location.hash === "#dataset" ? "dataset" : "search");

function savedTheme(): Theme {
  try {
    const t = localStorage.getItem(THEME_KEY);
    return THEMES.includes(t as Theme) ? (t as Theme) : "system";
  } catch {
    return "system";
  }
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [theme, setTheme] = useState<Theme>(savedTheme);
  // Set by the Dataset tab's "search with this question"; the nonce re-triggers an identical query.
  const [request, setRequest] = useState<SearchRequest | null>(null);

  useEffect(() => {
    history.replaceState(null, "", tab === "search" ? location.pathname + location.search : `#${tab}`);
  }, [tab]);

  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // Storage unavailable (private window): the choice just won't persist.
    }
  }, [theme]);

  const searchFor = (text: string) => {
    setRequest({ text, nonce: Date.now() });
    setTab("search");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const nextTheme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  const ThemeIcon = THEME_ICON[theme];

  return (
    <>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.brand}>
            <span className={styles.logo}>
              <SearchIcon size={18} strokeWidth={2.5} />
            </span>
            <div>
              <div className={styles.name}>VieQuAD Search</div>
              <div className={styles.tagline}>Tìm kiếm ngữ nghĩa hai tầng · Wikipedia tiếng Việt</div>
            </div>
          </div>
          <nav className={styles.tabs} aria-label="Chuyển tab">
            {TABS.map(({ id, label, icon: TabIcon }) => (
              <button
                key={id}
                type="button"
                className={id === tab ? styles.active : undefined}
                aria-current={id === tab ? "page" : undefined}
                onClick={() => setTab(id)}
              >
                <TabIcon size={16} />
                <span>{label}</span>
              </button>
            ))}
          </nav>
          <button
            type="button"
            className={styles.themeButton}
            onClick={() => setTheme(nextTheme)}
            title={`Giao diện: ${THEME_LABEL[theme]} (bấm để chuyển sang ${THEME_LABEL[nextTheme]})`}
            aria-label={`Giao diện: ${THEME_LABEL[theme]}`}
          >
            <ThemeIcon size={18} />
          </button>
        </div>
      </header>

      <main className={styles.page}>
        {/* Both tabs stay mounted so switching back keeps their state. */}
        <div hidden={tab !== "search"}>
          <SearchTab request={request} />
        </div>
        <div hidden={tab !== "dataset"}>
          <DatasetTab active={tab === "dataset"} onSearch={searchFor} />
        </div>
      </main>

      <footer className={styles.footer}>
        BM25 (pg_search) + Dense (multilingual-e5) → RRF → Cross-encoder rerank (bge-reranker-v2-m3)
      </footer>
    </>
  );
}
