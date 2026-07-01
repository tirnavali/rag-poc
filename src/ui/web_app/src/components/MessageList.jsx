import { Compass } from 'lucide-react';
import MessageBubble from './MessageBubble';

export default function MessageList({
  messages,
  selectedMessageIdx,
  isGenerating,
  currentStatus,
  currentThinking,
  currentAnswer,
  currentSources,
  currentSuggestions,
  printTargetIdx,
  messagesEndRef,
  onMessageClick,
  onCopyMessage,
  onRegenerate,
  onPrintMessage,
  onSendSuggestion,
}) {
  const isEmpty = messages.length === 0 && !isGenerating;

  return (
    <div className="messages-area">
      {isEmpty ? (
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
            <MessageBubble
              key={idx}
              message={m}
              isSelected={isSelected}
              isLive={false}
              isGenerating={isGenerating}
              isPrintTarget={printTargetIdx === idx}
              onClick={() => onMessageClick(idx)}
              onCopyMessage={onCopyMessage}
              onRegenerate={() => onRegenerate(idx)}
              onPrintMessage={() => onPrintMessage(idx)}
              onSendSuggestion={onSendSuggestion}
            />
          );
        })
      )}

      {isGenerating && (
        <MessageBubble
          message={{
            role: 'assistant',
            content: currentAnswer,
            status: currentStatus,
            thinking: currentThinking,
            sources: currentSources,
            suggestions: currentSuggestions,
            created_at: null,
          }}
          isSelected={false}
          isLive
          isGenerating={isGenerating}
          isPrintTarget={false}
          onClick={undefined}
          onSendSuggestion={onSendSuggestion}
        />
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}
