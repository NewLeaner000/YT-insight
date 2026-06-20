import React from 'react';
import { dict } from '../i18n';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend
} from 'recharts';
import { ThumbsUp, ThumbsDown, Minus, Activity, MessageCircle } from 'lucide-react';

const COLORS = {
  positive: '#10b981', // emerald-500
  neutral: '#64748b',  // slate-500
  negative: '#f43f5e', // rose-500
};

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  componentDidCatch(error, errorInfo) {
    console.error("Dashboard caught an error:", error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return <div className="p-4 bg-rose-500/10 text-rose-400 rounded-xl border border-rose-500/20">Error loading dashboard chart. Please refresh.</div>;
    }
    return this.props.children;
  }
}

export default function DashboardWrapper(props) {
  return (
    <ErrorBoundary>
      <DashboardInner {...props} />
    </ErrorBoundary>
  );
}

function DashboardInner({ sentimentCounts, lang = 'vi' }) {
  if (!sentimentCounts) return null;

  const positive = sentimentCounts.POSITIVE || sentimentCounts.positive || 0;
  const neutral = sentimentCounts.NEUTRAL || sentimentCounts.neutral || 0;
  const negative = sentimentCounts.NEGATIVE || sentimentCounts.negative || 0;
  const total = positive + neutral + negative;

  const t = dict[lang];

  const data = [
    { name: t.positive, value: positive, color: COLORS.positive },
    { name: t.neutral, value: neutral, color: COLORS.neutral },
    { name: t.negative, value: negative, color: COLORS.negative },
  ];

  return (
    <div className="bg-slate-800/50 backdrop-blur-md border border-slate-700 rounded-2xl p-6 shadow-xl mb-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-500/20 text-blue-400 rounded-lg">
          <Activity size={24} />
        </div>
        <div>
          <h3 className="text-lg font-bold text-slate-100">{t.dashboardOverview}</h3>
          <p className="text-sm text-slate-400">{t.analysisOf} {total} {t.comments}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-slate-800 border border-slate-700 p-4 rounded-xl flex items-center gap-4 hover:border-white/50 transition-colors group cursor-default">
          <div className="bg-emerald-500/10 text-emerald-400 p-3 rounded-lg group-hover:scale-110 group-hover:text-white transition-all">
            <ThumbsUp size={24} />
          </div>
          <div>
            <p className="text-sm text-slate-400 font-medium group-hover:text-white/80 transition-colors">{t.positive}</p>
            <p className="text-2xl font-bold text-white transition-colors">{positive}</p>
          </div>
        </div>

        <div className="bg-slate-800 border border-slate-700 p-4 rounded-xl flex items-center gap-4 hover:border-white/50 transition-colors group cursor-default">
          <div className="bg-slate-500/10 text-slate-400 p-3 rounded-lg group-hover:scale-110 group-hover:text-white transition-all">
            <Minus size={24} />
          </div>
          <div>
            <p className="text-sm text-slate-400 font-medium group-hover:text-white/80 transition-colors">{t.neutral}</p>
            <p className="text-2xl font-bold text-white transition-colors">{neutral}</p>
          </div>
        </div>

        <div className="bg-slate-800 border border-slate-700 p-4 rounded-xl flex items-center gap-4 hover:border-white/50 transition-colors group cursor-default">
          <div className="bg-rose-500/10 text-rose-400 p-3 rounded-lg group-hover:scale-110 group-hover:text-white transition-all">
            <ThumbsDown size={24} />
          </div>
          <div>
            <p className="text-sm text-slate-400 font-medium group-hover:text-white/80 transition-colors">{t.negative}</p>
            <p className="text-2xl font-bold text-white transition-colors">{negative}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-72">
        {/* Bar Chart */}
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50 flex flex-col h-full hover:border-white/30 transition-colors">
          <h4 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <MessageCircle size={16} /> {t.volumeDist}
          </h4>
          <div className="flex-1" style={{ minWidth: 0, minHeight: 200 }}>
            <ResponsiveContainer width="99%" height="100%">
              <BarChart data={data} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pie Chart */}
        <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-700/50 flex flex-col h-full hover:border-white/30 transition-colors">
          <h4 className="text-sm font-semibold text-slate-300 mb-2">{t.sentimentRatio}</h4>
          <div className="flex-1" style={{ minWidth: 0, minHeight: 200 }}>
            <ResponsiveContainer width="99%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                />
                <Legend verticalAlign="bottom" height={36} iconType="circle" />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
