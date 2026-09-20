import { levelColor, pct } from '../lib/format'

export default function OccupancyBar({ value, level, color, label, max = 0.5, compact = false }: { value: number | null | undefined; level?: string | null; color?: string; label?: string; max?: number; compact?: boolean }) {
  const w = value == null ? 0 : Math.min(100, (value / max) * 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: compact ? 12 : 13 }}>
      {label && (
        <span style={{ minWidth: compact ? 70 : 84, display: 'inline-flex', alignItems: 'center', gap: 6, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {color && <span className="dot" style={{ background: color, width: 8, height: 8 }} />}
          {label}
        </span>
      )}
      <div className="bar" style={{ flex: 1, height: compact ? 6 : 8 }}>
        <div style={{ width: `${w}%`, background: levelColor(level) }} />
      </div>
      <span className="num" style={{ minWidth: 46, textAlign: 'right' }}>{pct(value)}</span>
    </div>
  )
}
