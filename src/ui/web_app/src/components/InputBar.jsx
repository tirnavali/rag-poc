import { Send } from 'lucide-react';

export default function InputBar({ value, disabled, onChange, onKeyDown, onSend }) {
  return (
    <div className="input-area">
      <div className="input-container">
        <input
          type="text"
          placeholder="Parlamenter belgeler hakkında bir soru yazın…"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={disabled}
        />
        <button className="send-btn" onClick={onSend} disabled={disabled || !value.trim()}>
          <Send size={16} />
        </button>
      </div>
      <div className="input-hints">
        <span>Enter ile gönder</span>
        <span>Yanıtlar belge kaynaklı olup doğrulanmalıdır.</span>
      </div>
    </div>
  );
}
