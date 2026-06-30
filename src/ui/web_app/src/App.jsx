import { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Plus, 
  MessageSquare, 
  Terminal, 
  Layers, 
  FileText, 
  Compass, 
  AlertTriangle,
  Clock,
  ChevronDown,
  ChevronUp,
  Brain
} from 'lucide-react';

const API_BASE = '/api';
const WS_BASE = `ws://${window.location.host}/api/chat/stream`;

const parseMarkdown = (text) => {
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
  
  // 4. Inline Citation Pills: (Kaynak: ...) or (Kaynak: ...)
  html = html.replace(/\((Kaynak:\s*[^)]+)\)/gi, (match, p1) => {
    return `<span class="inline-citation" title="${p1}">📌 ${p1}</span>`;
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

const formatTime = (isoOrSqlString) => {
  if (!isoOrSqlString) return '';
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
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch (e) {
    return '';
  }
};

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [mufettisMode, setMufettisMode] = useState(false);
  const [currentTab, setCurrentTab] = useState('trace'); // trace | sources | memory
  const [selectedMessageIdx, setSelectedMessageIdx] = useState(null);
  
  // Streaming state variables
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('');
  const [currentThinking, setCurrentThinking] = useState('');
  const [currentAnswer, setCurrentAnswer] = useState('');
  
  // Traces, Sources, and Memory states
  const [currentTrace, setCurrentTrace] = useState([]);
  const [currentSources, setCurrentSources] = useState([]);
  const [currentMemory, setCurrentMemory] = useState({ history: [], max_turns: 5 });
  const [expandedTraceIdx, setExpandedTraceIdx] = useState({});
  const [expandedSourceIdx, setExpandedSourceIdx] = useState({});
  // Facet-grounded "rabbit hole" drill-down suggestions (clickable chips).
  const [currentSuggestions, setCurrentSuggestions] = useState([]);

  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    init();
  }, []);

  useEffect(() => {
    if (activeSessionId) {
      loadSessionMessages(activeSessionId);
    }
  }, [activeSessionId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, currentThinking, currentAnswer, currentStatus]);

  const init = async () => {
    await fetchSessions();
  };

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`);
      const data = await res.json();
      setSessions(data);
      if (data.length > 0 && !activeSessionId) {
        setActiveSessionId(data[0].id);
      }
    } catch (e) {
      console.error("Failed to load sessions", e);
    }
  };

  const loadSessionMessages = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/sessions/${id}/messages`);
      const data = await res.json();
      setMessages(data);
      
      // Load last assistant message's trace, sources, and memory
      let lastAssistantIdx = -1;
      for (let i = data.length - 1; i >= 0; i--) {
        if (data[i].role === 'assistant') {
          lastAssistantIdx = i;
          break;
        }
      }

      if (lastAssistantIdx !== -1) {
        setSelectedMessageIdx(lastAssistantIdx);
        const lastMsg = data[lastAssistantIdx];
        setCurrentTrace(lastMsg.trace || []);
        setCurrentSources(lastMsg.sources || []);
        setCurrentMemory({
          history: lastMsg.memory || [],
          max_turns: 5
        });
      } else {
        setSelectedMessageIdx(null);
        setCurrentTrace([]);
        setCurrentSources([]);
        setCurrentMemory({ history: [], max_turns: 5 });
      }
      setExpandedTraceIdx({});
      setExpandedSourceIdx({});
    } catch (e) {
      console.error("Failed to load messages", e);
    }
  };

  const handleMessageClick = (idx) => {
    if (isGenerating) return;
    const msg = messages[idx];
    if (!msg) return;

    let assistantMsg = null;
    let targetIdx = idx;
    if (msg.role === 'user') {
      // Find the subsequent assistant message
      for (let i = idx + 1; i < messages.length; i++) {
        if (messages[i].role === 'assistant') {
          assistantMsg = messages[i];
          targetIdx = i;
          break;
        }
      }
    } else if (msg.role === 'assistant') {
      assistantMsg = msg;
    }

    if (assistantMsg) {
      setSelectedMessageIdx(targetIdx);
      setCurrentTrace(assistantMsg.trace || []);
      setCurrentSources(assistantMsg.sources || []);
      setCurrentMemory({
        history: assistantMsg.memory || [],
        max_turns: 5
      });
      setExpandedTraceIdx({});
      setExpandedSourceIdx({});
    }
  };

  const toggleSourceExpand = (idx) => {
    setExpandedSourceIdx(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  const createNewSession = async () => {
    try {
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'Yeni Sohbet' })
      });
      const data = await res.json();
      setActiveSessionId(data.id);
      setMessages([]);
      setSelectedMessageIdx(null);
      setCurrentTrace([]);
      setCurrentSources([]);
      setCurrentMemory({ history: [], max_turns: 5 });
      await fetchSessions();
    } catch (e) {
      console.error("Failed to create session", e);
    }
  };

  const sendMessage = (queryText) => {
    // queryText is a string when called from a suggestion chip; when used as a
    // button onClick handler the arg is a SyntheticEvent, so fall back to input.
    const raw = typeof queryText === 'string' ? queryText : input;
    if (!raw.trim() || isGenerating || !activeSessionId) return;

    const query = raw.trim();
    setInput('');
    setIsGenerating(true);
    setSelectedMessageIdx(null);
    setCurrentStatus('Arşiv asistanı başlatılıyor...');
    setCurrentThinking('');
    setCurrentAnswer('');
    setCurrentTrace([]);
    setCurrentSources([]);
    setCurrentMemory({ history: [], max_turns: 5 });
    setExpandedTraceIdx({});
    setCurrentSuggestions([]);

    // Optimistically update message list
    setMessages(prev => [...prev, { role: 'user', content: query, created_at: new Date().toISOString() }]);

    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}${API_BASE}/chat/stream`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({
        session_id: activeSessionId,
        query: query,
        mufettis_mode: mufettisMode
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'status') {
        setCurrentStatus(data.content);
      } else if (data.type === 'thinking') {
        setCurrentStatus('');
        setCurrentThinking(prev => prev + data.content);
      } else if (data.type === 'content') {
        setCurrentStatus('');
        setCurrentAnswer(prev => prev + data.content);
      } else if (data.type === 'trace_phase') {
        // Live per-stage trace: append each phase as it completes.
        setCurrentStatus('');
        setCurrentTrace(prev => [...prev, data.event]);
      } else if (data.type === 'trace') {
        // Final authoritative trace (reconcile only if richer than what streamed).
        setCurrentTrace(prev => (data.events && data.events.length >= prev.length ? data.events : prev));
      } else if (data.type === 'sources') {
        setCurrentSources(data.sources);
      } else if (data.type === 'suggestions') {
        setCurrentSuggestions(data.suggestions || []);
      } else if (data.type === 'memory') {
        setCurrentMemory({ history: data.history || [], max_turns: data.max_turns || 5 });
      } else if (data.type === 'error') {
        setCurrentStatus(`Hata oluştu: ${data.content}`);
      }
    };

    ws.onclose = async () => {
      setIsGenerating(false);
      setCurrentStatus('');
      // Persisted message (from DB) carries its own suggestions; clear live ones.
      setCurrentSuggestions([]);
      await fetchSessions();
      await loadSessionMessages(activeSessionId);
    };
  };

  const toggleTraceExpand = (idx) => {
    setExpandedTraceIdx(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  // Facet-grounded "rabbit hole" drill-down chips; clicking one fires it as a query.
  const renderSuggestions = (list) => {
    if (!list || list.length === 0) return null;
    return (
      <div className="rabbit-holes">
        <div className="rabbit-holes-label">
          <Compass size={13} />
          <span>İlgili olabilir — derinleşmek için:</span>
        </div>
        <div className="rabbit-holes-chips">
          {list.map((s, i) => (
            <button
              key={i}
              className="suggestion-chip"
              disabled={isGenerating}
              onClick={(e) => { e.stopPropagation(); sendMessage(s); }}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const activeSessionTitle = sessions.find(s => s.id === activeSessionId)?.title || 'Sohbet';

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <Compass size={24} />
          <h1>TBMM Arşivi</h1>
        </div>
        
        <button className="new-chat-btn" onClick={createNewSession}>
          <Plus size={18} />
          Yeni Sohbet
        </button>

        <div className="session-list-container">
          {sessions.map(s => (
            <div 
              key={s.id} 
              className={`session-item ${s.id === activeSessionId ? 'active' : ''}`}
              onClick={() => setActiveSessionId(s.id)}
            >
              <MessageSquare size={16} />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {s.title || 'Yeni Sohbet'}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Main Workspace */}
      <div className="main-content">
        {/* Header */}
        <div className="chat-header">
          <h2>{activeSessionTitle}</h2>
          <div className="mode-toggle">
            <label className="mufettis-switch">
              <input 
                type="checkbox" 
                checked={mufettisMode} 
                onChange={(e) => setMufettisMode(e.target.checked)}
              />
              Müfettiş (Derin Araştırma) Modu
            </label>
          </div>
        </div>

        {/* Chat Body & Panels */}
        <div className="chat-body">
          {/* Messages Area */}
          <div className="messages-area">
            {messages.length === 0 && !isGenerating ? (
              <div className="empty-state">
                <Compass size={64} />
                <h3>RAG Arşiv Portalı</h3>
                <p>Sorunuzu yazın. Yapay zeka arşiv asistanı belgeleri tarayıp doğrulanmış cevabı hazırlayacaktır.</p>
              </div>
            ) : (
              messages.map((m, idx) => {
                const isSelected = selectedMessageIdx === idx || 
                                   (m.role === 'user' && selectedMessageIdx === idx + 1);
                return (
                  <div 
                    key={idx} 
                    className={`message ${m.role} ${isSelected ? 'selected-msg' : ''}`}
                    onClick={() => handleMessageClick(idx)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div
                      className="message-bubble"
                      dangerouslySetInnerHTML={{ __html: parseMarkdown(m.content) }}
                    />
                    {m.role === 'assistant' && renderSuggestions(m.suggestions)}
                    {m.created_at && (
                      <span className="message-time">
                        {formatTime(m.created_at)}
                      </span>
                    )}
                  </div>
                );
              })
            )}

            {/* Live Generation Block */}
            {isGenerating && (
              <div className="message assistant">
                {currentStatus && (
                  <div className="status-indicator pulse">
                    <Clock size={16} />
                    {currentStatus}
                  </div>
                )}
                {currentThinking && (
                  <div className="thinking-box">
                    <div className="thinking-header">
                      <Terminal size={14} />
                      <span>Agent Düşünce Süreci</span>
                    </div>
                    <div>{currentThinking}</div>
                  </div>
                )}
                {currentAnswer && (
                  <>
                    <div 
                      className="message-bubble"
                      dangerouslySetInnerHTML={{ __html: parseMarkdown(currentAnswer) }}
                    />
                    <span className="message-time">
                      {formatTime(new Date().toISOString())}
                    </span>
                  </>
                )}
                {renderSuggestions(currentSuggestions)}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Details / Debug Trace Panel */}
          <div className="details-panel">
            <div className="panel-tabs">
              <button 
                className={`tab ${currentTab === 'memory' ? 'active' : ''}`}
                onClick={() => setCurrentTab('memory')}
              >
                <Brain size={16} />
                Memory ({currentMemory.history.length})
              </button>
              <button 
                className={`tab ${currentTab === 'trace' ? 'active' : ''}`}
                onClick={() => setCurrentTab('trace')}
              >
                <Terminal size={16} />
                Debug / Trace
              </button>
              <button 
                className={`tab ${currentTab === 'sources' ? 'active' : ''}`}
                onClick={() => setCurrentTab('sources')}
              >
                <FileText size={16} />
                Kaynaklar ({currentSources.length})
              </button>
            </div>

            <div className="panel-content">
              {currentTab === 'memory' && (
                <div>
                  <div className="memory-info-banner">
                    <Brain size={14} />
                    <span>
                      Son <strong>{currentMemory.max_turns}</strong> tur (user + assistant) LLM'e gönderiliyor.
                    </span>
                  </div>
                  {currentMemory.history.length === 0 ? (
                    <div style={{ color: '#94a3b8', textAlign: 'center', marginTop: '60px', fontSize: '13px' }}>
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
                    <div style={{ color: '#94a3b8', textAlign: 'center', marginTop: '60px', fontSize: '13px' }}>
                      Henüz debug/trace adımı bulunmuyor.
                    </div>
                  ) : (
                    currentTrace.map((t, idx) => (
                      <div key={idx} className="trace-item">
                        <div 
                          className="trace-phase" 
                          style={{ cursor: 'pointer' }}
                          onClick={() => toggleTraceExpand(idx)}
                        >
                          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                            {t.status === 'success' ? '✅' : '⏳'} 
                            {t.phase.toUpperCase()}
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
                        {/* Per-stage LLM reasoning/thinking — always visible (the headline of this stage) */}
                        {t.details && (t.details.reasoning || t.details.thinking) && (
                          <div className="trace-thinking">
                            <Terminal size={11} />
                            <span>{t.details.thinking || t.details.reasoning}</span>
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
                    <div style={{ color: '#94a3b8', textAlign: 'center', marginTop: '60px', fontSize: '13px' }}>
                      Bu sorgu için başvurulmuş kaynak bulunmuyor.
                    </div>
                  ) : (
                    currentSources.map((s, idx) => (
                      <div 
                        key={idx} 
                        className="trace-item source-item clickable-source"
                        onClick={() => toggleSourceExpand(idx)}
                      >
                        <div className="trace-phase">
                          <span>Kaynak #{idx + 1}</span>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className="trace-elapsed" style={{ background: '#e0f2fe', color: '#0369a1' }}>
                              {s.document_type || 'Belge'}
                            </span>
                            {expandedSourceIdx[idx] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          </div>
                        </div>
                        <div style={{ margin: '8px 0', fontSize: '13px', lineHeight: '1.4' }}>
                          <strong>{s.source_name || 'Gazete/Tutanak'}</strong> | {s.date || 'Tarih Belirtilmemiş'}
                        </div>
                        {s.title && <div style={{ fontStyle: 'italic', marginBottom: '8px', color: '#475569' }}>"{s.title}"</div>}
                        <div style={{ fontSize: '12px', color: '#64748b' }}>
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
            </div>
          </div>
        </div>

        {/* Input area */}
        <div className="input-area">
          <div className="input-container">
            <input 
              type="text" 
              placeholder="Arşivde aramak istediğiniz konuyu yazın..." 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isGenerating}
            />
            <button 
              className="send-btn" 
              onClick={sendMessage}
              disabled={isGenerating || !input.trim()}
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
