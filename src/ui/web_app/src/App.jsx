import { useState, useEffect, useRef } from 'react';
import { buildMarkdownExport } from './utils';
import Sidebar from './components/Sidebar';
import ChatHeader from './components/ChatHeader';
import MessageList from './components/MessageList';
import RightPanelDrawer from './components/RightPanelDrawer';
import InputBar from './components/InputBar';

const API_BASE = '/api';

const MAX_MEMORY_TURNS = 5;

// Mirrors the backend's own windowing logic (src/api/server.py, chat_history
// build): last MAX_MEMORY_TURNS*2 user/assistant messages. Derives the CURRENT
// forward-looking memory preview directly from the messages list, instead of
// trusting a message's stored (retrospective, one-turn-stale) `memory` field.
const buildRecentMemory = (messagesList, maxTurns = MAX_MEMORY_TURNS) => {
  const history = (messagesList || [])
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .slice(-(maxTurns * 2))
    .map(m => ({ role: m.role, content: m.content }));
  return { history, max_turns: maxTurns };
};

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [mufettisMode, setMufettisMode] = useState(false);
  const [sessionSearchQuery, setSessionSearchQuery] = useState('');
  const [currentTab, setCurrentTab] = useState('trace'); // trace | sources | memory
  const [selectedMessageIdx, setSelectedMessageIdx] = useState(null);
  const [rightPanelOpen, setRightPanelOpen] = useState(false); // drawer defaults to closed
  const [printTargetIdx, setPrintTargetIdx] = useState(null);

  // Streaming state variables
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('');
  const [currentThinking, setCurrentThinking] = useState('');
  const [currentAnswer, setCurrentAnswer] = useState('');

  // Traces, Sources, and Memory states
  const [currentTrace, setCurrentTrace] = useState([]);
  const [currentSources, setCurrentSources] = useState([]);
  const [currentMemory, setCurrentMemory] = useState({ history: [], max_turns: 5 });
  // 'live': currentMemory previews what WILL be sent on the next message.
  // 'historical': currentMemory shows what WAS actually sent for a past,
  // user-clicked answer (audit view) — see handleMessageClick.
  const [memoryViewMode, setMemoryViewMode] = useState('live');
  const [expandedTraceIdx, setExpandedTraceIdx] = useState({});
  const [expandedSourceIdx, setExpandedSourceIdx] = useState({});
  // Facet-grounded "rabbit hole" drill-down suggestions (clickable chips).
  const [currentSuggestions, setCurrentSuggestions] = useState([]);
  // 'idle' | 'copied' — brief visual confirmation for the "Copy as Markdown" button.
  const [copyStatus, setCopyStatus] = useState('idle');

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

      // Load last assistant message's trace and sources
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
      } else {
        setSelectedMessageIdx(null);
        setCurrentTrace([]);
        setCurrentSources([]);
      }
      // Forward-looking memory preview: derived from the conversation itself
      // (what will be sent next), not the last message's stored (one-turn-stale)
      // `memory` field — fixes the "memory appears empty right after the first
      // answer" bug.
      setCurrentMemory(buildRecentMemory(data));
      setMemoryViewMode('live');
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
      setMemoryViewMode('historical');
      setExpandedTraceIdx({});
      setExpandedSourceIdx({});
      // Surface the drawer if it's closed — otherwise this click silently
      // updates panel content the user can't see.
      setRightPanelOpen(true);
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
      setMemoryViewMode('live');
      await fetchSessions();
    } catch (e) {
      console.error("Failed to create session", e);
    }
  };

  const sendMessage = (queryText) => {
    // queryText is a string when called from a suggestion chip or the
    // regenerate action; when used as a button onClick handler the arg is a
    // SyntheticEvent, so fall back to input.
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
    // Preview of what this request will actually send as chat_history (backend
    // re-derives the same window fresh from the DB — see server.py) — computed
    // from `messages` BEFORE the optimistic push below, so it matches exactly.
    setCurrentMemory(buildRecentMemory(messages));
    setMemoryViewMode('live');
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
        setMemoryViewMode('live');
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

  // Exports the whole observed flow (question, plan/strategy, every pipeline
  // stage, sources, memory) as Markdown — for eyeballing the run or pasting
  // into an LLM for further analysis. Works both for a live-just-generated
  // answer and for a historical message the user clicked to inspect.
  const handleCopyMarkdown = async () => {
    let question = '';
    let answer = '';

    if (selectedMessageIdx !== null && messages[selectedMessageIdx]) {
      answer = messages[selectedMessageIdx].content || '';
      const prev = messages[selectedMessageIdx - 1];
      question = (prev && prev.role === 'user') ? prev.content : '';
    } else {
      answer = currentAnswer || '';
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'user') { question = messages[i].content; break; }
      }
    }

    const markdown = buildMarkdownExport({
      question, answer,
      trace: currentTrace,
      sources: currentSources,
      memory: currentMemory,
    });

    try {
      await navigator.clipboard.writeText(markdown);
      setCopyStatus('copied');
      setTimeout(() => setCopyStatus('idle'), 1500);
    } catch (e) {
      console.error('Copy failed', e);
    }
  };

  // Copies just this one message's answer text + its own source list — kept
  // deliberately separate from handleCopyMarkdown's full trace/debug export
  // above, so the two differently-scoped "Copy" buttons never get merged.
  const handleCopyMessage = async (message) => {
    const sourceLines = (message.sources || []).map((s, i) =>
      `${i + 1}. ${s.source_name || 'Kaynak'}${s.date ? ' — ' + s.date : ''}`
    );
    const text = [message.content, ...(sourceLines.length ? ['', 'Kaynaklar:', ...sourceLines] : [])].join('\n');
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      console.error('Copy failed', e);
    }
  };

  // Resends the preceding question as a NEW turn through the normal sendMessage
  // flow. This appends rather than replacing the old answer in place — true
  // in-place regeneration would need a backend delete/update endpoint, which
  // doesn't exist today. The action bar's tooltip makes this explicit.
  const handleRegenerate = (messageIdx) => {
    const prevUserMsg = messages[messageIdx - 1];
    if (prevUserMsg && prevUserMsg.role === 'user') {
      sendMessage(prevUserMsg.content);
    }
  };

  // Client-side PDF export via the browser's native print, scoped to a single
  // message by the .print-target rule in index.css. No backend/new dependency.
  const handlePrintMessage = (messageIdx) => {
    setPrintTargetIdx(messageIdx);
    requestAnimationFrame(() => {
      window.print();
      setTimeout(() => setPrintTargetIdx(null), 300);
    });
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  const activeSession = sessions.find(s => s.id === activeSessionId);
  const activeSessionTitle = activeSession?.title || 'Sohbet';

  return (
    <div className="app-container">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        searchQuery={sessionSearchQuery}
        onSearchChange={setSessionSearchQuery}
        onSelectSession={setActiveSessionId}
        onNewSession={createNewSession}
      />

      <div className="main-content">
        <ChatHeader
          title={activeSessionTitle}
          sessionCreatedAt={activeSession?.created_at}
          mufettisMode={mufettisMode}
          onToggleMufettis={setMufettisMode}
          rightPanelOpen={rightPanelOpen}
          onToggleDrawer={() => setRightPanelOpen(o => !o)}
          hasTraceData={currentTrace.length > 0 || currentSources.length > 0}
        />

        <div className="chat-body">
          <MessageList
            messages={messages}
            selectedMessageIdx={selectedMessageIdx}
            isGenerating={isGenerating}
            currentStatus={currentStatus}
            currentThinking={currentThinking}
            currentAnswer={currentAnswer}
            currentSources={currentSources}
            currentSuggestions={currentSuggestions}
            printTargetIdx={printTargetIdx}
            messagesEndRef={messagesEndRef}
            onMessageClick={handleMessageClick}
            onCopyMessage={handleCopyMessage}
            onRegenerate={handleRegenerate}
            onPrintMessage={handlePrintMessage}
            onSendSuggestion={sendMessage}
          />

          <RightPanelDrawer
            isOpen={rightPanelOpen}
            onClose={() => setRightPanelOpen(false)}
            currentTab={currentTab}
            onTabChange={setCurrentTab}
            currentMemory={currentMemory}
            memoryViewMode={memoryViewMode}
            currentTrace={currentTrace}
            expandedTraceIdx={expandedTraceIdx}
            onToggleTraceExpand={toggleTraceExpand}
            currentSources={currentSources}
            expandedSourceIdx={expandedSourceIdx}
            onToggleSourceExpand={toggleSourceExpand}
            copyStatus={copyStatus}
            onCopyMarkdown={handleCopyMarkdown}
          />
        </div>

        <InputBar
          value={input}
          disabled={isGenerating}
          onChange={setInput}
          onKeyDown={handleKeyDown}
          onSend={sendMessage}
        />
      </div>
    </div>
  );
}
