/* 공용 UI 프리미티브: 로딩/빈 상태/오류/토스트/모달/KPI/세그먼트 */
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react'
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'

export function Spinner({ lg = false }: { lg?: boolean }) {
  return <span className={`spinner${lg ? ' lg' : ''}`} aria-label="loading" />
}

export function Loading({ label = '불러오는 중…', lg = false }: { label?: string; lg?: boolean }) {
  return (
    <div className="loading">
      <Spinner lg={lg} /> {label}
    </div>
  )
}

export function Skeleton({ h = 16, w = '100%', style }: { h?: number; w?: number | string; style?: React.CSSProperties }) {
  return <div className="skeleton" style={{ height: h, width: w, ...style }} />
}

export function EmptyState({ icon, title, description, action }: { icon?: ReactNode; title: string; description?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      {icon}
      <h3>{title}</h3>
      {description && <p>{description}</p>}
      {action && <div className="row" style={{ marginTop: 6, justifyContent: 'center' }}>{action}</div>}
    </div>
  )
}

export function Banner({ kind = 'info', children, onRetry, onClose }: { kind?: 'info' | 'warn' | 'error'; children: ReactNode; onRetry?: () => void; onClose?: () => void }) {
  const Icon = kind === 'error' ? AlertTriangle : kind === 'warn' ? AlertTriangle : Info
  return (
    <div className={`banner ${kind}`} role={kind === 'error' ? 'alert' : undefined}>
      <Icon />
      <div className="grow">{children}</div>
      {onRetry && <button className="sm" onClick={onRetry}>다시 시도</button>}
      {onClose && <button className="sm ghost icon" onClick={onClose} aria-label="닫기"><X /></button>}
    </div>
  )
}

export function StatCard({ label, value, unit, sub, icon, tone }: { label: string; value: ReactNode; unit?: string; sub?: ReactNode; icon?: ReactNode; tone?: string }) {
  return (
    <div className="card stat-card">
      {icon && <div className={`ic ${tone ?? ''}`}>{icon}</div>}
      <div style={{ minWidth: 0 }}>
        <div className="label">{label}</div>
        <div className="value">{value}{unit && <small>{unit}</small>}</div>
        {sub && <div className="sub">{sub}</div>}
      </div>
    </div>
  )
}

export function Segmented<T extends string | number>({ options, value, onChange }: { options: { v: T; l: ReactNode; title?: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="segmented" role="tablist">
      {options.map((o) => (
        <button key={String(o.v)} role="tab" aria-selected={o.v === value} className={o.v === value ? 'active' : ''} onClick={() => onChange(o.v)} title={o.title}>
          {o.l}
        </button>
      ))}
    </div>
  )
}

export function Modal({ open, title, onClose, children, width = 640 }: { open: boolean; title: ReactNode; onClose: () => void; children: ReactNode; width?: number }) {
  useEffect(() => {
    if (!open) return
    const h = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="modal-bg" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ ['--modal-w' as string]: `${width}px` }} role="dialog" aria-modal>
        <div className="head">
          <h2>{title}</h2>
          <button className="ghost icon" onClick={onClose} aria-label="닫기"><X /></button>
        </div>
        <div className="body">{children}</div>
      </div>
    </div>
  )
}

export function Card({ title, icon, actions, children, className = '', style }: { title?: ReactNode; icon?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string; style?: React.CSSProperties }) {
  return (
    <div className={`card ${className}`} style={style}>
      {(title || actions) && (
        <div className="card-head">
          {title && <h3>{icon}{title}</h3>}
          {actions && <div className="row" style={{ gap: 6 }}>{actions}</div>}
        </div>
      )}
      {children}
    </div>
  )
}

export function PageHeader({ title, description, actions }: { title: ReactNode; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="page-header">
      <div>
        <h1>{title}</h1>
        {description && <div className="desc">{description}</div>}
      </div>
      {actions && <div className="actions">{actions}</div>}
    </div>
  )
}

/* ---- 토스트 ---- */
interface Toast { id: number; kind: 'ok' | 'error' | 'info'; text: string }
const ToastCtx = createContext<(kind: Toast['kind'], text: string) => void>(() => {})

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([])
  const push = useCallback((kind: Toast['kind'], text: string) => {
    const id = Date.now() + Math.random()
    setItems((t) => [...t, { id, kind, text }])
    setTimeout(() => setItems((t) => t.filter((x) => x.id !== id)), kind === 'error' ? 6000 : 3500)
  }, [])
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts">
        {items.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>
            {t.kind === 'ok' ? <CheckCircle2 size={16} color="var(--free)" /> : t.kind === 'error' ? <AlertTriangle size={16} color="var(--jam)" /> : <Info size={16} color="var(--accent)" />}
            {t.text}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  )
}

export function useToast() {
  return useContext(ToastCtx)
}

/** 비동기 동작 실행 + 토스트. 실패 메시지는 자동 표시 */
export function useAction() {
  const toast = useToast()
  const [busy, setBusy] = useState<string | null>(null)
  const run = useCallback(
    async <T,>(label: string, fn: () => Promise<T>, okMsg?: string): Promise<T | undefined> => {
      setBusy(label)
      try {
        const r = await fn()
        if (okMsg) toast('ok', okMsg)
        return r
      } catch (e: any) {
        toast('error', `${label} 실패: ${e.message ?? e}`)
        return undefined
      } finally {
        setBusy(null)
      }
    },
    [toast],
  )
  return { run, busy }
}

/** 간단한 스파크라인 (0~1 값) */
export function Sparkline({ values, width = 120, height = 28, color = 'var(--accent)', max }: { values: (number | null)[]; width?: number; height?: number; color?: string; max?: number }) {
  const vals = values.map((v) => (v == null ? null : v))
  const nums = vals.filter((v): v is number => v != null)
  if (nums.length < 2) return <svg className="spark" width={width} height={height} />
  const mx = max ?? Math.max(0.1, ...nums)
  const pts = vals.map((v, i) => (v == null ? null : [(i / (vals.length - 1)) * width, height - 2 - (v / mx) * (height - 4)]))
  let d = ''
  pts.forEach((p) => {
    if (!p) { d += ' M'; return }
    d += (d.endsWith('M') || d === '' ? ` M${p[0]},${p[1]}` : ` L${p[0]},${p[1]}`)
  })
  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <path d={d.replace(/ M(?= M| L)/g, '').trim()} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
    </svg>
  )
}

/** 요일×시간 히트맵 */
export function HeatGrid({ grid, weekdays, max = 0.35 }: { grid: (number | null)[][]; weekdays: string[]; max?: number }) {
  return (
    <div className="heat">
      <div />
      {Array.from({ length: 24 }, (_, h) => <div key={h} style={{ textAlign: 'center' }}>{h % 3 === 0 ? h : ''}</div>)}
      {grid.map((row, d) => (
        <span key={d} style={{ display: 'contents' }}>
          <div className="lab">{weekdays[d]}</div>
          {row.map((v, h) => {
            const t = v == null ? 0 : Math.min(1, v / max)
            const bg = v == null ? 'var(--panel-2)' : `rgba(${t < 0.5 ? 47 + (242 - 47) * t * 2 : 242 + (229 - 242) * (t - 0.5) * 2}, ${t < 0.5 ? 191 + (193 - 191) * t * 2 : 193 + (72 - 193) * (t - 0.5) * 2}, ${t < 0.5 ? 113 + (78 - 113) * t * 2 : 78 + (77 - 78) * (t - 0.5) * 2}, ${0.25 + 0.75 * t})`
            return <div key={h} className="cell" style={{ background: bg }} title={v == null ? '데이터 없음' : `${weekdays[d]} ${h}시 ${(v * 100).toFixed(1)}%`} />
          })}
        </span>
      ))}
    </div>
  )
}
