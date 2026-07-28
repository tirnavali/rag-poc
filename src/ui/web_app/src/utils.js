// Shared pure helpers used across App.jsx and the components/ tree.

export const PHASE_LABELS = {
  bad_words_filter: 'Uygunsuz İçerik Filtresi',
  classification: 'Niyet Analizi',
  planning: 'Planlama',
  filter_extraction: 'Filtre Çıkarımı',
  policy: 'Politika Kontrolü',
  budget: 'Arama Bütçesi',
  retrieval: 'Arşiv Taraması',
  assembly: 'Bağlam Oluşturma',
  judge: 'Kanıt Değerlendirme',
  expansion: 'Sorgu Genişletme',
  judge_post_expand: 'Kanıt Değerlendirme (Genişletme Sonrası)',
  reflect: 'Yansıtma / Yeniden Planlama',
  answering: 'Yanıt Üretimi',
  validation: 'Yanıt Doğrulama',
  citation: 'Kaynak Atıflandırma',
  rabbit_holes: 'İlgili Öneriler',
  suggestion: 'Öneri Üretimi',
  // Legacy phase names — no longer emitted by the current orchestrator, but old
  // sessions have them baked into their persisted trace JSON (messages.trace).
  // Kept so historical Debug/Trace tabs render a real label instead of falling
  // through to phaseLabel()'s raw-uppercase fallback. See src/agent/flow_diagram.html
  // (tracer.py module note) for why each one stopped firing.
  allocation: 'Arama Bütçesi', // renamed to `budget`, same step
  probe: 'Ön Tarama (kaldırıldı)', // old clarification probe pass; folded into retrieval reuse
  clarification: 'Netleştirme (eski akış)', // old blocking Q&A flow; replaced by non-blocking rabbit_holes
};
export const phaseLabel = (phase) => PHASE_LABELS[phase] || (phase ? phase.toUpperCase() : '');

export const parseMarkdown = (text) => {
  if (!text) return '';

  // 1. Escape HTML
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // 2. Bold: **text**
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // 3. Italic: *text*
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // 4. Inline Citation Badges: (Kaynak: ...)
  html = html.replace(/\((Kaynak:\s*[^)]+)\)/gi, (match, p1) => {
    return `<span class="inline-citation" title="${p1}">${p1}</span>`;
  });

  // 5. Bullet Lists & Paragraphs
  const lines = html.split('\n');
  let inList = false;
  const processedLines = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith('* ') || trimmed.startsWith('- ')) {
      const content = trimmed.substring(2);
      if (!inList) {
        inList = true;
        processedLines.push('<ul class="markdown-list">');
      }
      processedLines.push(`<li>${content}</li>`);
    } else {
      if (inList) {
        inList = false;
        processedLines.push('</ul>');
      }
      if (trimmed) {
        processedLines.push(`<p>${line}</p>`);
      } else {
        processedLines.push('<br />');
      }
    }
  }

  if (inList) {
    processedLines.push('</ul>');
  }

  return processedLines.join('\n');
};

export const findMatchingSource = (citationText, sources) => {
  if (!sources || sources.length === 0) return null;
  const cleanCitation = citationText.replace(/^Kaynak:\s*/i, '').toLowerCase();
  
  let bestSource = null;
  let bestScore = 0;
  
  for (let i = 0; i < sources.length; i++) {
    const s = sources[i];
    let score = 0;
    
    // Match source name
    if (s.source_name && cleanCitation.includes(s.source_name.toLowerCase())) {
      score += 10;
    }
    // Match date
    if (s.date && cleanCitation.includes(s.date.toLowerCase())) {
      score += 10;
    }
    // Match author
    if (s.author && cleanCitation.includes(s.author.toLowerCase())) {
      score += 10;
    }
    // Match document_type
    if (s.document_type && cleanCitation.includes(s.document_type.toLowerCase())) {
      score += 2;
    }
    // Match title
    if (s.title && cleanCitation.includes(s.title.toLowerCase())) {
      score += 5;
    }

    // Also match against index if the model somehow uses it (like "Kaynak 1" or similar)
    const indexStr = `${i + 1}`;
    if (cleanCitation.includes(indexStr)) {
      score += 1;
    }
    
    if (score > bestScore) {
      bestScore = score;
      bestSource = s;
    }
  }
  
  // If we found a source with some match score, return it; otherwise return first source as fallback
  return bestSource || sources[0];
};

// Parses both SQLite ("YYYY-MM-DD HH:MM:SS", stored as UTC) and ISO timestamps.
// Returns null (not a throw) on anything unparseable, so callers doing date
// bucketing/comparison don't need their own try/catch.
export const parseFlexibleDate = (isoOrSqlString) => {
  if (!isoOrSqlString) return null;
  try {
    let date;
    if (isoOrSqlString.includes(' ') && !isoOrSqlString.includes('T')) {
      const parts = isoOrSqlString.split(' ');
      const dateParts = parts[0].split('-');
      const timeParts = parts[1].split(':');
      date = new Date(Date.UTC(
        parseInt(dateParts[0]),
        parseInt(dateParts[1]) - 1,
        parseInt(dateParts[2]),
        parseInt(timeParts[0]),
        parseInt(timeParts[1]),
        parseInt(timeParts[2] || 0)
      ));
    } else {
      date = new Date(isoOrSqlString);
    }
    return isNaN(date.getTime()) ? null : date;
  } catch (e) {
    return null;
  }
};

export const formatTime = (isoOrSqlString) => {
  const date = parseFlexibleDate(isoOrSqlString);
  if (!date) return '';
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// "2 sn önce" / "5 dk önce" / "3 sa önce" — falls back to the absolute
// formatTime() once older than a day, since "27 sa önce" stops being useful.
export const formatRelativeTime = (isoOrSqlString) => {
  const date = parseFlexibleDate(isoOrSqlString);
  if (!date) return '';
  const diffSec = Math.round((Date.now() - date.getTime()) / 1000);
  if (diffSec < 5) return 'az önce';
  if (diffSec < 60) return `${diffSec} sn önce`;
  const diffMin = Math.round(diffSec / 60);
  if (diffMin < 60) return `${diffMin} dk önce`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr} sa önce`;
  return formatTime(isoOrSqlString);
};

export const truncate = (text, max = 500) =>
  !text ? '' : (text.length > max ? text.slice(0, max) + '…' : text);

// navigator.clipboard needs a secure context (https, or the loopback origins
// localhost/127.0.0.1) — plain http://<lan-ip> doesn't qualify, so the API is
// undefined there. Fall back to the legacy execCommand path, which has no
// such restriction.
export const copyToClipboard = async (text) => {
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(text);
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.cssText = 'position:fixed;opacity:0;left:-9999px';
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  try {
    if (!document.execCommand('copy')) throw new Error('execCommand("copy") başarısız');
  } finally {
    document.body.removeChild(ta);
  }
};

// Exports the whole observed flow (question, plan/strategy, every pipeline
// stage, sources, memory) as Markdown — for eyeballing the run or pasting
// into an LLM for further analysis. Works both for a live-just-generated
// answer and for a historical message the user clicked to inspect.
export const buildMarkdownExport = ({ question, answer, trace, sources, memory }) => {
  const lines = [];
  lines.push('# RAG Debug Export', '', `_Oluşturulma: ${new Date().toLocaleString('tr-TR')}_`, '');
  lines.push('## Soru', question || '_(soru bulunamadı)_', '');
  lines.push('## Yanıt', answer || '_(yanıt bulunamadı)_', '');

  const planningEvent = (trace || []).find(t => t.phase === 'planning');
  if (planningEvent && planningEvent.details) {
    const d = planningEvent.details;
    lines.push('## Planlama Özeti');
    lines.push(`- **Sorgu Tipi:** ${d.query_type || '-'}`);
    lines.push(`- **Strateji:** ${d.strategy || '-'}`);
    lines.push(`- **Kapsamlı mı:** ${d.comprehensive ? 'Evet' : 'Hayır'}`);
    lines.push(`- **Koleksiyonlar:** ${(d.collections || []).join(', ') || '-'}`);
    if (d.reasoning) lines.push(`- **Gerekçe:** ${d.reasoning}`);
    lines.push('');
  }

  lines.push('## Pipeline Adımları');
  (trace || []).forEach((t, i) => {
    lines.push(`### ${i + 1}. ${phaseLabel(t.phase)}`);
    const meta = [];
    if (t.elapsed) meta.push(`⏱ ${t.elapsed.toFixed(2)}s`);
    if (t.model) meta.push(`🧠 ${t.model}`);
    if (meta.length) lines.push(meta.join(' · '));
    const d = t.details || {};
    if (d.reasoning || d.thinking) lines.push('', `> ${(d.reasoning || d.thinking).replace(/\n/g, '\n> ')}`);
    if (d.answer_preview) lines.push('', `**Önizleme:** ${d.answer_preview}`);
    lines.push('', '```json', JSON.stringify(d, null, 2), '```', '');
  });
  if (!trace || trace.length === 0) lines.push('_Trace verisi yok._');

  lines.push('## Kaynaklar');
  const srcList = sources || [];
  const formatSource = (s, i) => {
    lines.push(`${i + 1}. **${s.source_name || 'Bilinmeyen Kaynak'}** — ${s.date || 'tarih yok'} — ${s.author || 'Belirtilmemiş'}${s.title ? ` — _"${s.title}"_` : ''}`);
    if (s.text) lines.push(`   > ${truncate(s.text, 300).replace(/\n/g, ' ')}`);
  };
  if (srcList.length === 0) {
    lines.push('_Kaynak yok._');
  } else if (srcList.some(s => typeof s.cited === 'boolean')) {
    // Backend flags which retrieved chunks the prose actually cites (best-effort
    // match on inline "(Kaynak: ..., Tarih, Yazar)" markers) — split so 40
    // retrieved-but-unused chunks don't read as 40 sources the answer drew on.
    const used = srcList.filter(s => s.cited);
    const rest = srcList.filter(s => !s.cited);
    lines.push('', `### Yanıtta Kullanılan Kaynaklar (${used.length})`, '');
    if (used.length) used.forEach(formatSource); else lines.push('_Yok._');
    lines.push('', `### Ek Taranan Kaynaklar (${rest.length})`, '', '_Retrieval kapsamına girdi ama yanıt metninde doğrudan atıf tespit edilmedi (heuristik eşleşme kaçırmış olabilir)._', '');
    if (rest.length) rest.forEach(formatSource); else lines.push('_Yok._');
  } else {
    srcList.forEach(formatSource);
  }
  lines.push('');

  lines.push('## Hafıza (Bağlam)');
  const memHistory = (memory && memory.history) || [];
  memHistory.forEach((m, i) => lines.push(`${i + 1}. **${m.role === 'user' ? 'Kullanıcı' : 'Asistan'}:** ${truncate(m.content, 400)}`));
  if (memHistory.length === 0) lines.push('_Hafıza verisi yok._');

  return lines.join('\n');
};
