import { Plus, Search, MessageSquare, MoreHorizontal } from 'lucide-react';
import { parseFlexibleDate } from '../utils';

// Placeholder profile shown at the bottom of the sidebar. There is no auth/user
// system in this app yet — this is a static, visually-complete stand-in.
// Colocated here since nothing else needs it; replace with a real user object
// in one line once auth exists.
const CURRENT_USER = {
  displayName: 'Ahmet Y.',
  roleLabel: 'BİM',
  idLabel: '24178',
  initials: 'AY',
};

const DATE_GROUP_ORDER = ['BUGÜN', 'DÜN', 'BU HAFTA', 'DAHA ESKİ'];

const groupSessionsByDate = (sessions) => {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfYesterday = new Date(startOfToday);
  startOfYesterday.setDate(startOfYesterday.getDate() - 1);
  const startOfWeek = new Date(startOfToday);
  startOfWeek.setDate(startOfWeek.getDate() - 7);

  const buckets = { 'BUGÜN': [], 'DÜN': [], 'BU HAFTA': [], 'DAHA ESKİ': [] };
  for (const s of sessions) {
    const d = parseFlexibleDate(s.created_at);
    if (!d) {
      buckets['DAHA ESKİ'].push(s);
    } else if (d >= startOfToday) {
      buckets['BUGÜN'].push(s);
    } else if (d >= startOfYesterday) {
      buckets['DÜN'].push(s);
    } else if (d >= startOfWeek) {
      buckets['BU HAFTA'].push(s);
    } else {
      buckets['DAHA ESKİ'].push(s);
    }
  }
  return DATE_GROUP_ORDER
    .map(label => [label, buckets[label]])
    .filter(([, items]) => items.length > 0);
};

export default function Sidebar({ sessions, activeSessionId, searchQuery, onSearchChange, onSelectSession, onNewSession }) {
  const filtered = sessions.filter(s =>
    (s.title || 'Yeni Sohbet').toLowerCase().includes(searchQuery.toLowerCase())
  );
  const grouped = groupSessionsByDate(filtered);

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="crest" style={{ width: 24, height: 24, fontSize: 12 }}>T</div>
        <h1>TBMM Asistan</h1>
      </div>

      <button className="new-chat-btn" onClick={onNewSession}>
        <Plus size={15} />
        Yeni Sohbet
      </button>

      <div className="sidebar-search">
        <Search size={13} />
        <input
          type="text"
          placeholder="Sohbet ara…"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
        />
      </div>

      <div className="session-list-container">
        {grouped.length === 0 ? (
          <div className="session-empty-hint">
            {searchQuery ? 'Eşleşen sohbet bulunamadı.' : 'Henüz sohbet yok.'}
          </div>
        ) : (
          grouped.map(([label, items]) => (
            <div className="session-date-group" key={label}>
              <span className="session-date-label">{label}</span>
              {items.map(s => (
                <div
                  key={s.id}
                  className={`session-item ${s.id === activeSessionId ? 'active' : ''}`}
                  onClick={() => onSelectSession(s.id)}
                >
                  <MessageSquare size={14} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {s.title || 'Yeni Sohbet'}
                  </span>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      <div className="sidebar-footer">
        <div className="user-avatar">{CURRENT_USER.initials}</div>
        <div className="sidebar-footer-info">
          <span className="sidebar-footer-name">{CURRENT_USER.displayName}</span>
          <span className="sidebar-footer-role">{CURRENT_USER.roleLabel} · {CURRENT_USER.idLabel}</span>
        </div>
        <span className="sidebar-footer-menu"><MoreHorizontal size={15} /></span>
      </div>
    </div>
  );
}
