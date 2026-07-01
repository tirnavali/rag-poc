// Shared pure helpers used across App.jsx and the components/ tree.

export const PHASE_LABELS = {
  bad_words_filter: 'Uygunsuz İçerik Filtresi',
  classification: 'Niyet Analizi',
  planning: 'Planlama',
  filter_extraction: 'Filtre Çıkarımı',
  policy: 'Politika Kontrolü',
  allocation: 'Kaynak Tahsisi',
  retrieval: 'Arşiv Taraması',
  assembly: 'Bağlam Oluşturma',
  judge: 'Kanıt Değerlendirme',
  expansion: 'Sorgu Genişletme',
  judge_post_expand: 'Kanıt Değerlendirme (Genişletme Sonrası)',
  answering: 'Yanıt Üretimi',
  validation: 'Yanıt Doğrulama',
  citation: 'Kaynak Atıflandırma',
  rabbit_holes: 'İlgili Öneriler',
  suggestion: 'Öneri Üretimi',
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
  (sources || []).forEach((s, i) => {
    lines.push(`${i + 1}. **${s.source_name || 'Bilinmeyen Kaynak'}** — ${s.date || 'tarih yok'} — ${s.author || 'Belirtilmemiş'}${s.title ? ` — _"${s.title}"_` : ''}`);
    if (s.text) lines.push(`   > ${truncate(s.text, 300).replace(/\n/g, ' ')}`);
  });
  if (!sources || sources.length === 0) lines.push('_Kaynak yok._');
  lines.push('');

  lines.push('## Hafıza (Bağlam)');
  const memHistory = (memory && memory.history) || [];
  memHistory.forEach((m, i) => lines.push(`${i + 1}. **${m.role === 'user' ? 'Kullanıcı' : 'Asistan'}:** ${truncate(m.content, 400)}`));
  if (memHistory.length === 0) lines.push('_Hafıza verisi yok._');

  return lines.join('\n');
};
