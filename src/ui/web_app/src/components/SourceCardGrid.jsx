// Compact numbered source-card grid rendered beneath an assistant message.
// Additive to (not a replacement for) the drawer's Kaynaklar tab — this uses
// the exact same `sources` array, just surfaced inline for quicker scanning.
// The numbers here are array-order labels only; they are not cross-referenced
// with numbered markers inside the prose (the backend doesn't emit those —
// see CLAUDE.md/plan notes on parseMarkdown's citation format).
export default function SourceCardGrid({ sources }) {
  if (!sources || sources.length === 0) return null;

  return (
    <div>
      <span className="source-card-grid-label">KAYNAKLAR</span>
      <div className="source-card-grid">
        {sources.map((s, idx) => (
          <div className="surface-card source-card" key={idx}>
            <span className="source-card-badge">{idx + 1}</span>
            <div className="source-card-info">
              <span className="source-card-title">{s.source_name || s.title || `Kaynak ${idx + 1}`}</span>
              <span className="source-card-meta">
                {[s.date, s.document_type].filter(Boolean).join(' · ') || 'Tarih belirtilmemiş'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
