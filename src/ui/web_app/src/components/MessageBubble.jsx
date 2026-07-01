import { Clock, Terminal, Compass } from 'lucide-react';
import { parseMarkdown, formatTime, formatRelativeTime } from '../utils';
import SourceCardGrid from './SourceCardGrid';
import MessageActions from './MessageActions';

const RabbitHoles = ({ suggestions, onSend, disabled }) => {
  if (!suggestions || suggestions.length === 0) return null;
  return (
    <div className="rabbit-holes">
      <div className="rabbit-holes-label">
        <Compass size={13} />
        <span>İlgili olabilir — derinleşmek için:</span>
      </div>
      <div className="rabbit-holes-chips">
        {suggestions.map((s, i) => (
          <button
            key={i}
            className="suggestion-chip"
            disabled={disabled}
            onClick={(e) => { e.stopPropagation(); onSend(s); }}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
};

// Single component for both a persisted historical message and the live,
// still-streaming answer (isLive) — kept as one component so the two render
// paths can't silently drift apart from each other over time.
export default function MessageBubble({
  message,
  isSelected,
  isLive,
  isGenerating,
  isPrintTarget,
  onClick,
  onCopyMessage,
  onRegenerate,
  onPrintMessage,
  onSendSuggestion,
}) {
  if (message.role === 'user') {
    return (
      <div
        className={`message user ${isSelected ? 'selected-msg' : ''}`}
        onClick={onClick}
        style={{ cursor: onClick ? 'pointer' : 'default' }}
      >
        <div className="message-bubble" dangerouslySetInnerHTML={{ __html: parseMarkdown(message.content) }} />
        {message.created_at && <span className="message-time">{formatTime(message.created_at)}</span>}
      </div>
    );
  }

  const modelLabel = (message.trace || []).find(
    (t) => t.phase === 'answering' || t.phase === 'generation'
  )?.model || null;

  return (
    <div
      className={`message assistant ${isSelected ? 'selected-msg' : ''} ${isPrintTarget ? 'print-target' : ''}`}
      onClick={onClick}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      <div className="message-header-row">
        <div className="crest">T</div>
        <span className="message-role-label">
          ASİSTAN{message.created_at ? ` · ${formatRelativeTime(message.created_at)}` : (isLive ? ' · şimdi' : '')}
        </span>
      </div>

      {isLive && message.status && (
        <div className="status-indicator">
          <Clock size={14} />
          {message.status}
          <span className="status-dots">● ● ●</span>
        </div>
      )}

      {isLive && message.thinking && (
        <div className="thinking-box">
          <div className="thinking-header">
            <Terminal size={14} />
            <span>Agent Düşünce Süreci</span>
          </div>
          <div>{message.thinking}</div>
        </div>
      )}

      {message.content && (
        <div className="message-bubble" dangerouslySetInnerHTML={{ __html: parseMarkdown(message.content) }} />
      )}

      <SourceCardGrid sources={message.sources} />

      {message.created_at && <span className="message-time">{formatTime(message.created_at)}</span>}

      {!isLive && message.content && (
        <MessageActions
          sourceCount={(message.sources || []).length}
          modelLabel={modelLabel}
          onCopy={() => onCopyMessage(message)}
          onRegenerate={onRegenerate}
          onPrint={onPrintMessage}
        />
      )}

      <RabbitHoles suggestions={message.suggestions} onSend={onSendSuggestion} disabled={isGenerating} />
    </div>
  );
}
