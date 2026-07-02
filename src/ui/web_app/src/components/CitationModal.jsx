import { useState } from 'react';
import { X, Copy, Check, Calendar, User, FileText, Bookmark } from 'lucide-react';

export default function CitationModal({ isOpen, onClose, source, citationText }) {
  const [copied, setCopied] = useState(false);

  if (!isOpen || !source) return null;

  const handleCopyText = async () => {
    try {
      await navigator.clipboard.writeText(source.text || '');
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Copy failed', e);
    }
  };

  // Prevent click on modal content from closing the modal (bubbling to overlay)
  const handleContentClick = (e) => {
    e.stopPropagation();
  };

  return (
    <div className="citation-modal-overlay" onClick={onClose}>
      <div className="citation-modal-content" onClick={handleContentClick}>
        <div className="citation-modal-header">
          <span className="citation-modal-title">Kaynak Atıf Detayı</span>
          <button className="citation-modal-close-btn" onClick={onClose} title="Kapat">
            <X size={18} />
          </button>
        </div>

        <div className="citation-modal-body">
          <div className="citation-modal-meta">
            <div className="citation-meta-item">
              <span className="citation-meta-label">
                <Bookmark size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                Kaynak Adı
              </span>
              <span className="citation-meta-value">
                {source.source_name || source.title || 'Belirtilmemiş'}
              </span>
            </div>

            <div className="citation-meta-item">
              <span className="citation-meta-label">
                <Calendar size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                Tarih
              </span>
              <span className="citation-meta-value">
                {source.date || 'Tarih Belirtilmemiş'}
              </span>
            </div>

            <div className="citation-meta-item">
              <span className="citation-meta-label">
                <User size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                Yazar / Konuşmacı
              </span>
              <span className="citation-meta-value">
                {source.author || 'Belirtilmemiş'}
              </span>
            </div>

            <div className="citation-meta-item">
              <span className="citation-meta-label">
                <FileText size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                Belge Türü
              </span>
              <span className="citation-meta-value" style={{ textTransform: 'capitalize' }}>
                {source.document_type || 'Belge Parçası'}
              </span>
            </div>
          </div>

          {source.title && (
            <div style={{ marginBottom: '14px' }}>
              <span className="citation-modal-text-title">Başlık / Konu</span>
              <div style={{ fontSize: '13.5px', fontWeight: '600', color: 'var(--ink)', fontStyle: 'italic' }}>
                "{source.title}"
              </div>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <span className="citation-modal-text-title">Alıntılanan Metin (Chunk)</span>
            <div className="citation-modal-text">
              {source.text || 'Bu kaynak için detay metni bulunmuyor.'}
            </div>
          </div>
        </div>

        <div className="citation-modal-footer">
          <button 
            className={`msg-action-btn ${copied ? 'copied' : ''}`}
            onClick={handleCopyText}
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? 'Kopyalandı!' : 'Metni Kopyala'}
          </button>
          <button 
            className="send-btn" 
            onClick={onClose}
            style={{ width: 'auto', padding: '0 20px', fontSize: '13px', fontWeight: '600' }}
          >
            Kapat
          </button>
        </div>
      </div>
    </div>
  );
}
