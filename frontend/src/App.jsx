import { useState, useRef, useEffect } from 'react';
import { LogOut, Globe, Menu, X } from 'lucide-react';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import { dict } from './i18n';

// Custom hook for persistent state
function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      if (item) {
        return JSON.parse(item);
      } else {
        window.localStorage.setItem(key, JSON.stringify(initialValue));
        return initialValue;
      }
    } catch (error) {
      console.error(error);
      return initialValue;
    }
  });

  const setValue = (value) => {
    try {
      setStoredValue((prevStoredValue) => {
        const valueToStore = value instanceof Function ? value(prevStoredValue) : value;
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
        return valueToStore;
      });
    } catch (error) {
      console.error(error);
    }
  };

  return [storedValue, setValue];
}

export default function App() {
  const [lang, setLang] = useLocalStorage('yt-insight-lang', 'vi');
  const t = dict[lang];

  const DEFAULT_GREETING = { 
    role: 'ai', 
    content: t.placeholderPaste 
  };
  const [isAuthenticated, setIsAuthenticated] = useLocalStorage('yt-insight-auth', false);
  const [userId, setUserId] = useLocalStorage('yt-insight-user-id', 'demouser');
  
  // Use environment variable for API URL or fallback to localhost for local dev
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // WAKE UP PING: Render's free tier sleeps after 15 mins.
  // This background ping wakes up the backend the moment the user opens the frontend.
  useEffect(() => {
    fetch(`${API_URL}/api/health`).catch(() => {});
  }, [API_URL]);

  // Migrate legacy anonymous user IDs to unified 'demouser' account
  useEffect(() => {
    if (isAuthenticated && userId && userId.startsWith('user-')) {
      setUserId('demouser');
    }
  }, [isAuthenticated, userId, setUserId]);

  
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  
  // Session Management via Backend
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useLocalStorage('yt-insight-active-session', null);
  const [currentChatHistory, setCurrentChatHistory] = useState([]);
  const [sessionVideos, setSessionVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState('');
  
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  
  const chatEndRef = useRef(null);

  // Fetch sessions on mount
  const fetchSessions = async () => {
    try {
      const res = await fetch(`${API_URL}/api/sessions?user_id=${userId}`);
      if (res.ok) {
        const data = await res.json();
        setSessions(data);
        if (data.length > 0) {
          // Auto select first session if none selected, OR if the selected one doesn't exist anymore
          if (!activeSessionId || !data.find(s => s.id === activeSessionId)) {
            setActiveSessionId(data[0].id);
          }
        } else {
          setActiveSessionId(null);
        }
      }
    } catch (e) {
      console.error("Failed to fetch sessions", e);
    }
  };

  useEffect(() => {
    if (isAuthenticated && userId) {
      fetchSessions();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAuthenticated, userId]);

  // Fetch messages and videos when active session changes
  useEffect(() => {
    const fetchMessagesAndVideos = async () => {
      if (!activeSessionId) return;
      try {
        // Fetch Messages
        const resMsg = await fetch(`${API_URL}/api/sessions/${activeSessionId}/messages`);
        if (resMsg.ok) {
          const data = await resMsg.json();
          setCurrentChatHistory(prev => {
            // Prevent race condition: if we are currently showing a loading message from ingest, don't overwrite it
            if (prev.some(m => m.content.includes(t.analyzing) || m.content.includes('http'))) {
              return prev;
            }
            if (data.length === 0) {
              return [DEFAULT_GREETING];
            } else {
              return data;
            }
          });
        }
        
        // Fetch Videos
        const resVideos = await fetch(`${API_URL}/api/sessions/${activeSessionId}/videos`);
        if (resVideos.ok) {
          const data = await resVideos.json();
          setSessionVideos(data);
          
          if (data.length > 0) {
            // Auto-select first video if none selected or if currently selected is not in the list
            if (!selectedVideo || !data.find(v => v.video_id === selectedVideo)) {
              setSelectedVideo(data[0].video_id);
            }
          } else {
            setSelectedVideo('');
          }
        }
      } catch (e) {
        console.error("Failed to fetch session messages/videos", e);
      }
    };
    fetchMessagesAndVideos();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  // Fetch Stats when activeSessionId or selectedVideo changes
  useEffect(() => {
    const fetchStats = async () => {
      if (!activeSessionId) return;
      try {
        const url = new URL(`${API_URL}/api/sessions/${activeSessionId}/stats`);
        if (selectedVideo) {
          url.searchParams.append('video_id', selectedVideo);
        }
        
        const resStats = await fetch(url.toString());
        if (resStats.ok) {
          const statsData = await resStats.json();
          setSessions(prev => prev.map(s => 
            s.id === activeSessionId ? { ...s, sentimentCounts: statsData.sentimentCounts } : s
          ));
        }
      } catch (e) {
        console.error("Failed to fetch session stats", e);
      }
    };
    fetchStats();
  }, [activeSessionId, selectedVideo]);

  const activeSession = sessions.find(s => s.id === activeSessionId) || null;

  useEffect(() => {
    // Add a slight delay to ensure the DOM has updated and height is recalculated before scrolling
    const timeoutId = setTimeout(() => {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
    return () => clearTimeout(timeoutId);
  }, [currentChatHistory, isTyping]);

  const handleNewChat = async () => {
    try {
      const res = await fetch(`${API_URL}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, title: 'New Workspace' })
      });
      if (res.ok) {
        const data = await res.json();
        await fetchSessions();
        setActiveSessionId(data.id);
        setIsSidebarOpen(false);
      }
    } catch (e) {
      console.error("Failed to create session", e);
    }
  };

  const deleteSession = async (id, e) => {
    e.stopPropagation(); // prevent selecting the chat
    try {
      const res = await fetch(`${API_URL}/api/sessions/${id}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        setSessions(prev => prev.filter(s => s.id !== id));
        if (activeSessionId === id) {
          setActiveSessionId(null); // will auto-select first in next effect if exists
          fetchSessions();
        }
      }
    } catch (e) {
      console.error("Failed to delete session", e);
    }
  };

  const renameSession = async (id, currentTitle, e) => {
    e.stopPropagation();
    const newTitle = window.prompt("Enter new workspace name:", currentTitle);
    if (!newTitle || newTitle.trim() === '' || newTitle === currentTitle) return;
    try {
      const res = await fetch(`${API_URL}/api/sessions/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle.trim() })
      });
      if (res.ok) {
        setSessions(prev => prev.map(s => s.id === id ? { ...s, title: newTitle.trim() } : s));
      }
    } catch (e) {
      console.error("Failed to rename session", e);
    }
  };

  const processIngest = async (targetUrl) => {
    if (!targetUrl.trim()) return;
    
    setIsIngesting(true);
    setIngestStatus(null);
    
    let targetSessionId = activeSessionId;
    
    // Auto-create session if none is active
    if (!targetSessionId) {
      try {
        const res = await fetch(`${API_URL}/api/sessions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, title: 'New Workspace' })
        });
        if (res.ok) {
          const data = await res.json();
          targetSessionId = data.id;
          setActiveSessionId(targetSessionId);
          setIsSidebarOpen(false);
          // Also fetch sessions so the sidebar updates
          await fetchSessions();
        } else {
          setIngestStatus({ type: 'error', message: 'Failed to create workspace' });
          setIsIngesting(false);
          return;
        }
      } catch (err) {
        setIngestStatus({ type: 'error', message: 'Could not connect to the server' });
        setIsIngesting(false);
        return;
      }
    }
    
    // Add optimistic message
    setCurrentChatHistory(prev => [...prev, { role: 'ai', content: `${t.analyzing} ${targetUrl}... ${t.pleaseWait}` }]);
    
    try {
      const response = await fetch(`${API_URL}/api/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoUrl: targetUrl, session_id: targetSessionId })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setCurrentChatHistory(prev => [...prev, { role: 'ai', content: `${t.successIngest} ${data.commentsProcessed} ${t.commentsFrom} ${data.videoTitle || data.videoId}. ${t.dashboardReady}` }]);
        
        // Update local session title and fetch new stats
        setSessions(prev => prev.map(s => {
          if (s.id === targetSessionId) {
            const prevCounts = s.sentimentCounts || { POSITIVE: 0, NEGATIVE: 0, NEUTRAL: 0 };
            const newCounts = data.sentimentCounts || { POSITIVE: 0, NEGATIVE: 0, NEUTRAL: 0 };
            return { 
              ...s, 
              title: s.title === 'New Workspace' ? data.videoTitle || data.videoId : s.title,
              sentimentCounts: {
                POSITIVE: prevCounts.POSITIVE + newCounts.POSITIVE,
                NEGATIVE: prevCounts.NEGATIVE + newCounts.NEGATIVE,
                NEUTRAL: prevCounts.NEUTRAL + newCounts.NEUTRAL
              }
            };
          }
          return s;
        }));
        
        // Refetch videos for the dropdown
        const resVideos = await fetch(`${API_URL}/api/sessions/${targetSessionId}/videos`);
        if (resVideos.ok) {
          const vData = await resVideos.json();
          setSessionVideos(vData);
        }
        
      } else {
        setIngestStatus({ type: 'error', message: data.detail || 'Failed to ingest video' });
        setCurrentChatHistory(prev => [...prev, { role: 'ai', content: `${t.errorIngest} ${data.detail || 'Failed to ingest'}` }]);
      }
    } catch (err) {
      setIngestStatus({ type: 'error', message: 'Could not connect to the server' });
      setCurrentChatHistory(prev => [...prev, { role: 'ai', content: t.connectionError }]);
    } finally {
      setIsIngesting(false);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!message.trim() || isTyping) return;
    
    const userMsg = message.trim();
    
    // Check if the user pasted a YouTube URL directly into the chat
    const urlPattern = /(https?:\/\/(?:www\.)?(?:youtube\.com|youtu\.be)[^\s]+)/;
    const match = userMsg.match(urlPattern);
    
    if (match) {
      setMessage('');
      await processIngest(match[1]);
      return;
    }
    
    if (!activeSessionId) {
      setMessage('');
      setCurrentChatHistory([{ role: 'ai', content: t.placeholderPaste }]);
      return;
    }
    
    setMessage('');
    
    setCurrentChatHistory(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsTyping(true);
    
    try {
      const payload = { 
        message: userMsg,
        session_id: activeSessionId
      };
      
      if (selectedVideo) {
        payload.video_id = selectedVideo;
      }
      
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setCurrentChatHistory(prev => [...prev, { role: 'ai', content: data.reply }]);
      } else {
        setCurrentChatHistory(prev => [...prev, { role: 'ai', content: `${t.errorIngest} ${data.detail || 'Failed to ingest'}` }]);
      }
    } catch (err) {
      setCurrentChatHistory(prev => [...prev, { role: 'ai', content: t.connectionError }]);
    } finally {
      setIsTyping(false);
    }
  };

  if (!isAuthenticated) {
    return <Login onLogin={() => setIsAuthenticated(true)} />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-slate-900 text-slate-100 font-sans relative">
      
      {/* Backdrop for mobile */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-20 md:hidden transition-opacity duration-300"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`fixed inset-y-0 left-0 z-30 w-80 bg-slate-800 border-r border-slate-700 flex flex-col shadow-xl transition-transform duration-300 ease-in-out md:relative md:translate-x-0 ${
        isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
      } shrink-0`}>
        <div className="p-5 border-b border-slate-700 bg-slate-800 flex justify-between items-start">
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
              {t.title}
            </h1>
            <p className="text-sm text-slate-400 mt-1">{t.subtitle}</p>
          </div>
          <div className="flex items-center gap-1.5">
            <button 
              onClick={() => setLang(lang === 'vi' ? 'en' : 'vi')}
              className="p-1.5 bg-slate-700/50 hover:bg-slate-600 rounded-md text-slate-300 transition-colors flex items-center gap-1 text-xs font-medium"
              title="Toggle Language"
            >
              <Globe size={14} />
              {lang.toUpperCase()}
            </button>
            <button
              onClick={() => setIsSidebarOpen(false)}
              className="md:hidden p-1.5 bg-slate-700/50 hover:bg-slate-600 rounded-md text-slate-300 transition-colors"
              aria-label="Close sidebar"
            >
              <X size={14} />
            </button>
          </div>
        </div>
        
        {/* Chat History Section */}
        <div className="flex-1 overflow-y-auto flex flex-col p-3">
          <div className="flex items-center justify-between px-2 mb-2 mt-2">
            <h2 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              {t.chatHistory}
            </h2>
            <button 
              onClick={handleNewChat}
              className="text-xs bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded text-slate-200 transition-colors"
              aria-label="Start New Chat"
            >
              {t.newChat}
            </button>
          </div>
          
          <ul className="space-y-1 flex-1 overflow-y-auto pr-1">
            {sessions.map((session) => (
              <li key={session.id}>
                <button
                  onClick={() => {
                    setActiveSessionId(session.id);
                    setIsSidebarOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-all group flex justify-between items-center ${
                    activeSessionId === session.id 
                      ? 'bg-blue-600/20 text-blue-300 border border-blue-500/30' 
                      : 'hover:bg-slate-700/50 text-slate-400 hover:text-slate-200 border border-transparent'
                  }`}
                  aria-current={activeSessionId === session.id ? 'page' : undefined}
                >
                  <span className="truncate pr-2 flex items-center gap-2">
                    {session.status === 'processing' && (
                      <span className="w-2.5 h-2.5 rounded-full border-2 border-slate-400 border-t-blue-400 animate-spin inline-block"></span>
                    )}
                    {session.status === 'error' && (
                      <span className="w-2 h-2 rounded-full bg-rose-500 inline-block"></span>
                    )}
                    {session.title}
                  </span>
                  <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-all">
                    <div 
                      role="button"
                      tabIndex={0}
                      aria-label="Rename Chat"
                      onClick={(e) => renameSession(session.id, session.title, e)}
                      onKeyDown={(e) => { if (e.key === 'Enter') renameSession(session.id, session.title, e); }}
                      className="hover:text-blue-400 p-1 rounded-md text-slate-500 focus:opacity-100"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>
                    </div>
                    <div 
                      role="button"
                      tabIndex={0}
                      aria-label="Delete Chat"
                      onClick={(e) => deleteSession(session.id, e)}
                      onKeyDown={(e) => { if (e.key === 'Enter') deleteSession(session.id, e); }}
                      className="hover:text-rose-400 p-1 -mr-1 rounded-md text-slate-500 focus:opacity-100"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>
                    </div>
                  </div>
                </button>
              </li>
            ))}
            {sessions.length === 0 && (
              <div className="text-center py-6 text-slate-500 text-sm">
                {t.noHistory}
              </div>
            )}
          </ul>

          <div className="mt-4 pt-4 border-t border-slate-700">
            <button
              onClick={() => {
                setIsAuthenticated(false);
                setIsSidebarOpen(false);
              }}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-sm font-medium text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            >
              <LogOut size={16} />
              {t.signOut}
            </button>
          </div>
        </div>
      </aside>
      
      {/* Main Content: Chat Interface */}
      <main className="flex-1 flex flex-col bg-slate-900 relative">
        <header className="h-16 border-b border-slate-800 flex items-center px-4 md:px-6 bg-slate-900/80 backdrop-blur-md absolute top-0 w-full z-10 shadow-sm">
          <div className="flex items-center gap-3 w-full">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="md:hidden p-2 -ml-2 bg-slate-800/40 hover:bg-slate-800 border border-slate-700/50 rounded-lg text-slate-300 transition-colors mr-1"
              aria-label="Open sidebar"
            >
              <Menu size={18} />
            </button>
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)] animate-pulse shrink-0" aria-hidden="true"></div>
            <h2 className="font-medium text-slate-200 truncate">
              {activeSession ? activeSession.title : t.agentName}
            </h2>
          </div>
        </header>
        
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 pt-20 sm:pt-24 pb-32 space-y-6 scroll-smooth" role="log" aria-live="polite">
          {activeSession?.status === 'processing' && (
            <div className="flex flex-col items-center justify-center p-12 text-slate-400">
               <div className="w-10 h-10 border-4 border-slate-700 border-t-blue-500 rounded-full animate-spin mb-4"></div>
               <p className="animate-pulse">Extracting and analyzing comments... This takes about 10-15 seconds.</p>
            </div>
          )}
          
          {activeSession?.sentimentCounts && activeSession?.status !== 'processing' && (
            <Dashboard sentimentCounts={activeSession.sentimentCounts} lang={lang} />
          )}

          {!activeSession && sessions.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-slate-500 animate-in fade-in duration-700">
               <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 text-center max-w-md">
                 <h3 className="text-xl font-bold text-slate-300 mb-2">{t.title}</h3>
                 <p className="text-sm">{t.placeholderPaste}</p>
               </div>
            </div>
          )}
          
          {currentChatHistory.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-2xl px-5 py-3.5 leading-relaxed shadow-sm ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-tr-sm' 
                  : 'bg-slate-800 text-slate-200 border border-slate-700/50 rounded-tl-sm'
              }`}>
                <div className="whitespace-pre-wrap text-[15px]">{msg.content}</div>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-slate-800 border border-slate-700/50 rounded-2xl rounded-tl-sm px-5 py-4 shadow-sm flex items-center gap-2" aria-label="Agent is typing">
                <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-1.5 h-1.5 bg-slate-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
        
        <div className="absolute bottom-0 w-full p-4 sm:p-6 bg-gradient-to-t from-slate-900 via-slate-900 to-transparent pt-12 pointer-events-none">
          <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto relative group pointer-events-auto">

            <label htmlFor="chat-input" className="sr-only">Type your message</label>
            <input
              id="chat-input"
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder={activeSessionId && sessionVideos.length > 0 ? t.placeholderAsk : t.placeholderPaste}
              disabled={isTyping}
              className="w-full bg-slate-800 border border-slate-700 rounded-xl pl-4 sm:pl-5 pr-20 sm:pr-24 py-4 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all text-sm sm:text-[15px] shadow-lg disabled:opacity-50 placeholder:text-slate-500"
            />
            <button
              type="submit"
              disabled={!message.trim() || isTyping}
              aria-label="Send message"
              className="absolute right-2 top-2 bottom-2 px-4 sm:px-5 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 text-white font-medium rounded-lg transition-all active:scale-95 flex items-center justify-center text-sm"
            >
              {t.send}
            </button>
          </form>
          <div className="text-center mt-3 text-xs text-slate-500 pointer-events-auto">
            {t.footer}
          </div>
        </div>
      </main>
      
    </div>
  );
}
