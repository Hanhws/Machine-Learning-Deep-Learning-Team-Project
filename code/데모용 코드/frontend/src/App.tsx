import { Activity, BarChart3, Camera, Globe2, Landmark, Map as MapIcon, PlusCircle, Settings2 } from 'lucide-react'
import { Component, type ReactNode } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import { api } from './api/client'
import { ToastProvider } from './components/ui'
import { usePolling } from './lib/usePolling'
import AppsPage from './pages/AppsPage'
import CameraDetailPage from './pages/CameraDetailPage'
import CamerasPage from './pages/CamerasPage'
import MapPage from './pages/MapPage'
import OverviewPage from './pages/OverviewPage'
import RegisterPage from './pages/RegisterPage'
import StatsPage from './pages/StatsPage'
import SystemPage from './pages/SystemPage'

const NAV = [
  { to: '/', label: '지도 관제', icon: MapIcon },
  { to: '/overview', label: '전국 현황', icon: Globe2 },
  { to: '/cameras', label: '전체 CCTV', icon: Camera },
  { to: '/register', label: 'CCTV 등록', icon: PlusCircle },
  { to: '/stats', label: '점유율 통계', icon: BarChart3 },
  { to: '/apps', label: '응용 분석', icon: Landmark },
  { to: '/system', label: '시스템', icon: Settings2 },
]

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    if (this.state.error)
      return (
        <div className="page">
          <h1>화면 오류</h1>
          <pre className="error" style={{ whiteSpace: 'pre-wrap' }}>{String(this.state.error.stack || this.state.error)}</pre>
          <button onClick={() => this.setState({ error: null })}>다시 시도</button>
        </div>
      )
    return this.props.children
  }
}

function SystemStatus() {
  const { data, error } = usePolling(() => api.system(), 10000)
  const workers = data ? Object.keys(data.workers).length : 0
  return (
    <div className="sys">
      <div className="line"><span className={`status-dot ${error ? 'err' : data ? 'ok' : ''}`} />{error ? <NavLink to="/system">백엔드 연결 안 됨 → 주소 설정</NavLink> : data ? `백엔드 정상 · 워커 ${workers}` : '연결 중'}</div>
      {data && <div className="line"><span className={`status-dot ${data.its_configured ? 'ok' : 'warn'}`} />ITS 키 {data.its_configured ? '설정됨' : '미설정'}</div>}
      {data && <div className="line"><span className={`status-dot ${data.models.road.backend === 'sam3' ? 'ok' : 'warn'}`} />도로 SAM: {data.models.road.backend}</div>}
    </div>
  )
}

export default function App() {
  const links = NAV.map((n) => (
    <NavLink key={n.to} to={n.to} end={n.to === '/'} className={({ isActive }) => (isActive ? 'active' : '')} title={n.label}>
      <n.icon />
      <span>{n.label}</span>
    </NavLink>
  ))
  return (
    <ToastProvider>
      <div className="shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="logo"><Activity size={16} /></div>
            <span>
              도로 CCTV 혼잡도
              <small>Segmentation 기반 점유율</small>
            </span>
          </div>
          <nav>{links}</nav>
          <div className="grow" />
          <SystemStatus />
        </aside>
        <div className="main">
          <div className="mobile-bar"><Activity size={18} color="var(--accent)" /><span className="brand">도로 CCTV 혼잡도</span></div>
          <nav className="mobile-nav" style={{ display: undefined }}>{links}</nav>
          <main className="content">
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<MapPage />} />
                <Route path="/overview" element={<OverviewPage />} />
                <Route path="/cameras" element={<CamerasPage />} />
                <Route path="/cameras/:id" element={<CameraDetailPage />} />
                <Route path="/register" element={<RegisterPage />} />
                <Route path="/stats" element={<StatsPage />} />
                <Route path="/apps" element={<AppsPage />} />
                <Route path="/system" element={<SystemPage />} />
              </Routes>
            </ErrorBoundary>
          </main>
        </div>
      </div>
    </ToastProvider>
  )
}
