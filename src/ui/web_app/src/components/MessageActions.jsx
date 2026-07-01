import { useState } from 'react';
import { Copy, Check, RotateCcw, Printer } from 'lucide-react';

// Per-message action bar. Deliberately separate from the drawer's "Kopyala"
// (full Markdown/trace export, App.jsx handleCopyMarkdown) — this copies just
// this message's own text + source list, with its own local "copied" flash so
// the two same-looking buttons never share (or appear to share) state.
export default function MessageActions({ sourceCount, modelLabel, onCopy, onRegenerate, onPrint }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e) => {
    e.stopPropagation();
    await onCopy();
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const stop = (fn) => (e) => { e.stopPropagation(); fn(); };

  return (
    <div className="message-actions-bar">
      <button className={`msg-action-btn ${copied ? 'copied' : ''}`} onClick={handleCopy}>
        {copied ? <Check size={13} /> : <Copy size={13} />}
        {copied ? 'Kopyalandı' : 'Kopyala'}
      </button>
      <button
        className="msg-action-btn"
        onClick={stop(onRegenerate)}
        title="Yeniden sor (yeni mesaj olarak eklenir, eskisinin yerine geçmez)"
      >
        <RotateCcw size={13} />
        Yeniden
      </button>
      <button className="msg-action-btn" onClick={stop(onPrint)} title="Bu yanıtı yazdır / PDF olarak kaydet">
        <Printer size={13} />
        PDF
      </button>
      <span className="message-meta-footer">
        {sourceCount > 0 ? `${sourceCount} kaynak` : 'kaynak yok'}{modelLabel ? ` · ${modelLabel}` : ''}
      </span>
    </div>
  );
}
