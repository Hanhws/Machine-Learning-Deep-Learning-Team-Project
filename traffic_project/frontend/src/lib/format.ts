export const LEVELS = ['원활', '서행', '지체', '정체'] as const
export type Level = (typeof LEVELS)[number]

export const LEVEL_CLASS: Record<string, string> = { 원활: 'free', 서행: 'slow', 지체: 'delay', 정체: 'jam' }
export const LEVEL_COLOR: Record<string, string> = {
  원활: '#2fbf71',
  서행: '#f2c14e',
  지체: '#f28c28',
  정체: '#e5484d',
}

export function levelOf(occ: number | null | undefined, thresholds: number[]): Level | null {
  if (occ == null || Number.isNaN(occ)) return null
  for (let i = 0; i < thresholds.length; i++) if (occ < thresholds[i]) return LEVELS[i]
  return LEVELS[LEVELS.length - 1]
}

export function levelColor(level: string | null | undefined): string {
  return level ? (LEVEL_COLOR[level] ?? '#666') : '#666'
}

export function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return '–'
  return (v * 100).toFixed(digits) + '%'
}

export function fmtTime(iso: string | number | null | undefined): string {
  if (iso == null) return '–'
  if (typeof iso === 'number') {
    const m = Math.floor(iso / 60)
    const s = Math.floor(iso % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }
  const d = new Date(iso)
  return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export const CLASS_COLORS: Record<string, string> = { car: '#50c850', bus: '#ffa03c', truck: '#2878dc' }
export const CLASS_LABELS: Record<string, string> = { car: '승용차', bus: '버스', truck: '트럭' }
export const DIRECTION_PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#c850c8', '#dcc828', '#fa7878', '#3cc8dc', '#a0a0ff']

/** 추론 주기 선택지 (초). 0 = 실시간(서버 INFER_FPS) */
export const INTERVAL_OPTIONS: { v: number; l: string }[] = [
  { v: 0, l: '실시간 (연속)' },
  { v: 1, l: '1초마다' },
  { v: 5, l: '5초마다' },
  { v: 10, l: '10초마다 (기본)' },
  { v: 15, l: '15초마다' },
  { v: 30, l: '30초마다' },
  { v: 60, l: '1분마다' },
  { v: 300, l: '5분마다' },
  { v: 600, l: '10분마다' },
]
export const DEFAULT_INTERVAL = 10
export function intervalLabel(v: number | null | undefined): string {
  if (v == null) v = DEFAULT_INTERVAL
  if (v === 0) return '실시간'
  return INTERVAL_OPTIONS.find((o) => o.v === v)?.l ?? (v >= 60 ? `${Math.round(v / 60)}분마다` : `${v}초마다`)
}
