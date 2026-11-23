import React from "react";

export default function Navbar({ activeTab, onSelectTab, serverOnline }) {
  const tabs = [
    { id: "home", label: "Home" },
    { id: "council", label: "LLM Council" },
    { id: "architecture", label: "Architecture" },
    { id: "about", label: "About" },
  ];

  return (
    <header className="navbar-container">
      <nav className="navbar glass" aria-label="Main Navigation">
        <div className="navbar__brand" onClick={() => onSelectTab("home")} role="button" tabIndex={0}>
          <div className="navbar__logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              <circle cx="12" cy="11" r="3"/>
            </svg>
          </div>
          <div className="navbar__title-wrap">
            <span className="navbar__title">CODE AUDIT</span>
            <span className="navbar__badge">AICD MultiView</span>
          </div>
        </div>

        <div className="navbar__links">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`navbar__link ${activeTab === tab.id ? "navbar__link--active" : ""}`}
              onClick={() => onSelectTab(tab.id)}
            >
              {tab.label}
              {activeTab === tab.id && <span className="navbar__indicator" />}
            </button>
          ))}
        </div>

        <div className="navbar__status">
          <span className={`status-dot ${serverOnline ? "status-dot--online" : "status-dot--offline"}`} />
          <span className="status-text">{serverOnline ? "System Ready" : "Connecting..."}</span>
        </div>
      </nav>
    </header>
  );
}
