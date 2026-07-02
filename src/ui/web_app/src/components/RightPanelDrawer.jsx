import { Brain, Terminal, FileText, Copy, Check, ChevronDown, ChevronUp, Layers, X, BookOpen, ThumbsUp, ThumbsDown } from 'lucide-react';
import { phaseLabel } from '../utils';

const TERM_STATUS_LABELS = { pending: 'Bekleyen', approved: 'Onaylı', rejected: 'Reddedilen' };
const TERM_EMPTY_LABELS = {
  pending: 'Bekleyen terim yok.',
  approved: 'Onaylanmış terim yok.',
  rejected: 'Reddedilmiş terim yok.',
};

// Side panel that pushes the chat (shrinks .messages-area) instead of
// overlaying it — width animates via .details-panel.open, no backdrop.
export default function RightPanelDrawer({
  isOpen,
  onClose,
  currentTab,
  onTabChange,
  currentMemory,
  memoryViewMode,
  currentTrace,
  expandedTraceIdx,
  onToggleTraceExpand,
  currentSources,
  expandedSourceIdx,
  onToggleSourceExpand,
  copyStatus,
  onCopyMarkdown,
  termCandidates,
  termCandidatesFilter,
  onTermCandidatesFilterChange,
  pendingTermCount,
  onApproveTermCandidate,
  onRejectTermCandidate,
}) {
  return (
    <div className={`details-panel ${isOpen ? 'open' : ''}`}>
      <div className="drawer-header">
        <span>DEBUG / AUDIT</span>
        <button className="drawer-close-btn" onClick={onClose} title="Kapat">
          <X size={16} />
        </button>
      </div>

      <div className="panel-tabs">
        <button
          className={`tab ${currentTab === 'memory' ? 'active' : ''}`}
          onClick={() => onTabChange('memory')}
        >
          <Brain size={16} />
          Memory ({currentMemory.history.length})
        </button>
        <button
          className={`tab ${currentTab === 'trace' ? 'active' : ''}`}
          onClick={() => onTabChange('trace')}
        >
          <Terminal size={16} />
          Debug / Trace
        </button>
        <button
          className={`tab ${currentTab === 'sources' ? 'active' : ''}`}
          onClick={() => onTabChange('sources')}
        >
          <FileText size={16} />
          Kaynaklar ({currentSources.length})
        </button>
        <button
          className={`tab ${currentTab === 'terms' ? 'active' : ''}`}
          onClick={() => onTabChange('terms')}
          title="Sistemin arama sırasında keşfettiği terim karşılıkları"
        >
          <BookOpen size={16} />
          Öğrenilen Terimler ({pendingTermCount})
        </button>
        <button
          className="copy-markdown-btn"
          title="Tüm akışı Markdown olarak kopyala"
          onClick={onCopyMarkdown}
        >
          {copyStatus === 'copied' ? <Check size={15} /> : <Copy size={15} />}
          {copyStatus === 'copied' ? 'Kopyalandı' : 'Kopyala'}
        </button>
      </div>

      <div className="panel-content">
        {currentTab === 'memory' && (
          <div>
            <div className="memory-info-banner">
              <Brain size={14} />
              <span>
                {memoryViewMode === 'historical' ? (
                  <>Bu yanıt üretilirken LLM'e gönderilen son <strong>{currentMemory.max_turns}</strong> tur (user + assistant).</>
                ) : (
                  <>Son <strong>{currentMemory.max_turns}</strong> tur (user + assistant) bir sonraki mesajda LLM'e gönderilecek.</>
                )}
              </span>
            </div>
            {currentMemory.history.length === 0 ? (
              <div style={{ color: 'var(--ink-3)', textAlign: 'center', marginTop: '60px', fontSize: '13px' }}>
                Henüz hafıza verisi yok — ilk mesajda geçmiş olmadığı için boş.
              </div>
            ) : (
              currentMemory.history.map((m, idx) => (
                <div key={idx} className={`memory-item memory-${m.role}`}>
                  <div className="memory-item-header">
                    <span className={`memory-role-badge ${m.role}`}>
                      {m.role === 'user' ? '👤 User' : '🤖 Assistant'}
                    </span>
                    <span className="memory-turn-idx">#{idx + 1}</span>
                  </div>
                  <div className="memory-item-content">
                    {m.content.length > 400 ? m.content.slice(0, 400) + '…' : m.content}
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {currentTab === 'trace' && (
          <div>
            {currentTrace.length === 0 ? (
              <div style={{ color: 'var(--ink-3)', textAlign: 'center', marginTop: '60px', fontSize: '13px' }}>
                Henüz debug/trace adımı bulunmuyor.
              </div>
            ) : (
              currentTrace.map((t, idx) => (
                <div key={idx} className="trace-item">
                  <div
                    className="trace-phase"
                    style={{ cursor: 'pointer' }}
                    onClick={() => onToggleTraceExpand(idx)}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      {t.status === 'success' ? '✅' : '⏳'}
                      {phaseLabel(t.phase)}
                    </span>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="trace-elapsed">{t.elapsed ? t.elapsed.toFixed(2) + 's' : ''}</span>
                      {expandedTraceIdx[idx] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </span>
                  </div>
                  {t.model && (
                    <div className="trace-model">
                      <Layers size={11} />
                      Model: {t.model}
                    </div>
                  )}
                  {t.details && (t.details.reasoning || t.details.thinking) && (
                    <div className="trace-thinking">
                      <Terminal size={11} />
                      <span className="trace-thinking-text">{t.details.thinking || t.details.reasoning}</span>
                    </div>
                  )}
                  {t.details && t.details.answer_preview && (
                    <div className="trace-answer">{t.details.answer_preview}</div>
                  )}
                  {t.details && expandedTraceIdx[idx] && (
                    <div className="trace-details" style={{ marginTop: '10px' }}>
                      <pre>{JSON.stringify(t.details, null, 2)}</pre>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {currentTab === 'sources' && (
          <div>
            {currentSources.length === 0 ? (
              <div style={{ color: 'var(--ink-3)', textAlign: 'center', marginTop: '60px', fontSize: '13px' }}>
                Bu sorgu için başvurulmuş kaynak bulunmuyor.
              </div>
            ) : (
              currentSources.map((s, idx) => (
                <div
                  key={idx}
                  className="trace-item source-item clickable-source"
                  onClick={() => onToggleSourceExpand(idx)}
                >
                  <div className="trace-phase">
                    <span>Kaynak #{idx + 1}</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="trace-elapsed">
                        {s.document_type || 'Belge'}
                      </span>
                      {expandedSourceIdx[idx] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </div>
                  </div>
                  <div style={{ margin: '8px 0', fontSize: '13px', lineHeight: '1.4' }}>
                    <strong>{s.source_name || 'Gazete/Tutanak'}</strong> | {s.date || 'Tarih Belirtilmemiş'}
                  </div>
                  {s.title && <div style={{ fontStyle: 'italic', marginBottom: '8px', color: 'var(--ink-3)' }}>"{s.title}"</div>}
                  <div style={{ fontSize: '12px', color: 'var(--ink-3)' }}>
                    Yazar/Konuşmacı: {s.author || 'Belirtilmemiş'}
                  </div>
                  {expandedSourceIdx[idx] && (
                    <div className="source-text-block animate-fade-in" onClick={(e) => e.stopPropagation()}>
                      <div className="source-text-header">📚 Belge Parçası / Metin:</div>
                      <div className="source-text-content">
                        {s.text || "Belge metni bulunamadı."}
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}

        {currentTab === 'terms' && (
          <div>
            <div className="memory-info-banner">
              <BookOpen size={14} />
              <span>
                Sistemin arama sırasında keşfettiği terim karşılıkları (örn. "kadük" →
                "hükümsüz sayılan kanun teklifleri"). Onayladığınız terimler ~1 dakika
                içinde canlı aramaya yansır; kod değişikliği gerekmez.
              </span>
            </div>
            <div className="term-status-filter">
              {Object.keys(TERM_STATUS_LABELS).map((s) => (
                <button
                  key={s}
                  className={`term-filter-btn ${termCandidatesFilter === s ? 'active' : ''}`}
                  onClick={() => onTermCandidatesFilterChange(s)}
                >
                  {TERM_STATUS_LABELS[s]}
                </button>
              ))}
            </div>
            {termCandidates.length === 0 ? (
              <div style={{ color: 'var(--ink-3)', textAlign: 'center', marginTop: '60px', fontSize: '13px' }}>
                {TERM_EMPTY_LABELS[termCandidatesFilter]}
              </div>
            ) : (
              termCandidates.map((tc) => (
                <div key={tc.id} className="trace-item">
                  <div className="trace-phase">
                    <span>"{tc.term}" → "{tc.hypothesis}"</span>
                    <span className="trace-elapsed">{tc.times_seen}× görüldü</span>
                  </div>
                  {tc.last_source_query && (
                    <div style={{ fontSize: '12px', color: 'var(--ink-3)', margin: '6px 0' }}>
                      Örnek sorgu: "{tc.last_source_query}"
                    </div>
                  )}
                  {tc.status === 'pending' && (
                    <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                      <button className="msg-action-btn term-approve-btn" onClick={() => onApproveTermCandidate(tc.id)}>
                        <ThumbsUp size={13} /> Onayla
                      </button>
                      <button className="msg-action-btn term-reject-btn" onClick={() => onRejectTermCandidate(tc.id)}>
                        <ThumbsDown size={13} /> Reddet
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
