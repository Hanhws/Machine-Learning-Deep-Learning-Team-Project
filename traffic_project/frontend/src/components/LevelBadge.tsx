import { LEVEL_CLASS } from '../lib/format'

export default function LevelBadge({ level, size }: { level: string | null | undefined; size?: 'sm' }) {
  const style = size === 'sm' ? { fontSize: 11, padding: '1px 6px' } : undefined
  if (!level) return <span className="badge none" style={style}>대기</span>
  return <span className={`badge ${LEVEL_CLASS[level] ?? 'none'}`} style={style}>{level}</span>
}
