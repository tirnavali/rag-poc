import { PanelRight } from 'lucide-react';
import { formatTime } from '../utils';

export default function ChatHeader({
  title,
  sessionCreatedAt,
  mufettisMode,
  onToggleMufettis,
  rightPanelOpen,
  onToggleDrawer,
  hasTraceData,
}) {
  const createdLabel = formatTime(sessionCreatedAt);

  return (
    <div className="chat-header">
      <div className="header-left">
        <h2>{title}</h2>
        {createdLabel && <span className="chip mono">{createdLabel}</span>}
      </div>
      <div className="mode-toggle">
        <label className="mufettis-switch">
          <input
            type="checkbox"
            checked={mufettisMode}
            onChange={(e) => onToggleMufettis(e.target.checked)}
          />
          Müfettiş Modu
        </label>
        <button
          className={`drawer-toggle-btn ${rightPanelOpen ? 'open' : ''}`}
          title="Memory / Trace / Kaynaklar panelini aç"
          onClick={onToggleDrawer}
        >
          <PanelRight size={16} />
          {hasTraceData && !rightPanelOpen && <span className="drawer-toggle-badge" />}
        </button>
      </div>
    </div>
  );
}
