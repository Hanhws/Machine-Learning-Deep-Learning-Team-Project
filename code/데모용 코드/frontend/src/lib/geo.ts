/* 지도 방향 표시 유틸: 진행 각도(북=0, 시계방향) ↔ 좌표, 방향 화살표 아이콘 */
import L from 'leaflet'
import type { Direction } from '../api/client'

const R = 6371000

/** a → b 방위각 (도) */
export function bearing(a: [number, number], b: [number, number]): number {
  const [la1, lo1] = a.map((v) => (v * Math.PI) / 180)
  const [la2, lo2] = b.map((v) => (v * Math.PI) / 180)
  const y = Math.sin(lo2 - lo1) * Math.cos(la2)
  const x = Math.cos(la1) * Math.sin(la2) - Math.sin(la1) * Math.cos(la2) * Math.cos(lo2 - lo1)
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360
}

/** 시작점에서 heading 방향으로 dist(m) 떨어진 점 */
export function destination(p: [number, number], headingDeg: number, dist: number): [number, number] {
  const d = dist / R
  const h = (headingDeg * Math.PI) / 180
  const la1 = (p[0] * Math.PI) / 180
  const lo1 = (p[1] * Math.PI) / 180
  const la2 = Math.asin(Math.sin(la1) * Math.cos(d) + Math.cos(la1) * Math.sin(d) * Math.cos(h))
  const lo2 = lo1 + Math.atan2(Math.sin(h) * Math.sin(d) * Math.cos(la1), Math.cos(d) - Math.sin(la1) * Math.sin(la2))
  return [(la2 * 180) / Math.PI, (lo2 * 180) / Math.PI]
}

export const COMPASS = ['북', '북동', '동', '남동', '남', '남서', '서', '북서']
/** 진행 각도 → '북동행' 같은 기본 이름 */
export function headingName(h: number): string {
  return `${COMPASS[Math.round(h / 45) % 8]}행`
}
/** 여러 진행 각도의 도로 축 (0~180, 양방향 구분 없이) */
export function axisOf(headings: number[]): number | null {
  if (!headings.length) return null
  let c = 0, s = 0
  headings.forEach((h) => { const r = (2 * h * Math.PI) / 180; c += Math.cos(r); s += Math.sin(r) })
  return ((((Math.atan2(s, c) * 180) / Math.PI) / 2) + 180) % 180
}
/** 두 각도 차이 (0~180) */
export function angleDiff(a: number, b: number): number {
  const d = Math.abs(((a - b) % 360) + 360) % 360
  return d > 180 ? 360 - d : d
}
/** 작은 회전 화살표 SVG 문자열 (배지 안에 넣는 용도) */
export function miniArrowSvg(heading: number, color: string, size = 12): string {
  return `<svg width="${size}" height="${size}" viewBox="0 0 32 32" style="transform:rotate(${heading}deg);flex:none"><path d="M16 2 L28 20 L20 18 L20 30 L12 30 L12 18 L4 20 Z" fill="${color}" stroke="#0b0b0b" stroke-width="2" stroke-linejoin="round"/></svg>`
}
export function compass(h: number | null | undefined): string {
  if (h == null) return '미지정'
  return `${COMPASS[Math.round(h / 45) % 8]} ${Math.round(h)}°`
}

/** 방향 화살표의 지도 위치 (없으면 카메라 좌표) */
export function anchorOf(d: Direction, cam: { lat: number | null; lon: number | null }): [number, number] | null {
  if (d.lat != null && d.lon != null) return [d.lat, d.lon]
  if (cam.lat != null && cam.lon != null) return [cam.lat, cam.lon]
  return null
}

/**
 * 진행 방향으로 회전한 화살표 + 숫자 라벨.
 * 한국은 우측통행이므로 화살표를 진행 방향 오른쪽으로 offsetPx 만큼 밀어 상·하행이 도로 양옆에 나란히 보이게 한다.
 */
export function arrowIcon(opts: { heading: number; color: string; label?: string; offsetPx?: number; size?: number; dim?: boolean; selected?: boolean; textColor?: string }) {
  const { heading, color, label, offsetPx = 14, size = 30, dim = false, selected = false, textColor = '#0b0b0b' } = opts
  const t = (heading * Math.PI) / 180
  // 화면 좌표(x 오른쪽, y 아래)에서 진행 방향의 오른쪽 수직 벡터 = (cos t, sin t)
  const ox = Math.cos(t) * offsetPx
  const oy = Math.sin(t) * offsetPx
  // 라벨은 화살표 꼬리 뒤쪽(진행 반대 방향)에 두어 화살표 몸통·반대편 화살표와 겹치지 않게
  const lx = -Math.sin(t) * (size * 0.85)
  const ly = Math.cos(t) * (size * 0.85)
  const stroke = selected ? '#4f8cff' : '#ffffff'
  const html = `
<div style="position:absolute;left:0;top:0;transform:translate(${ox}px,${oy}px);opacity:${dim ? 0.45 : 1}">
  <svg width="${size}" height="${size}" viewBox="0 0 32 32" style="position:absolute;left:${-size / 2}px;top:${-size / 2}px;transform:rotate(${heading}deg);filter:drop-shadow(0 1px 3px rgba(0,0,0,.6))">
    <path d="M16 2 L28 20 L20 18 L20 30 L12 30 L12 18 L4 20 Z" fill="${color}" stroke="${stroke}" stroke-width="${selected ? 3 : 2}" stroke-linejoin="round"/>
  </svg>
  ${label ? `<div style="position:absolute;left:${lx}px;top:${ly}px;transform:translate(-50%,-50%);background:${color};color:${textColor};border:1.5px solid #fff;border-radius:999px;padding:0 6px;font:700 11px/16px 'Noto Sans KR',sans-serif;white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.5);font-variant-numeric:tabular-nums">${label}</div>` : ''}
</div>`
  return L.divIcon({ className: '', html, iconSize: [0, 0], iconAnchor: [0, 0] })
}

/** 도로 이름으로 방향 묶기 (도로 미지정은 '(도로 미지정)') */
export function groupByRoad<T extends { road?: string | null }>(dirs: T[]): [string, T[]][] {
  const m = new Map<string, T[]>()
  dirs.forEach((d) => {
    const k = d.road?.trim() || '(도로 미지정)'
    m.set(k, [...(m.get(k) ?? []), d])
  })
  return [...m.entries()]
}

/** 줌·위도에서 1픽셀이 몇 m 인지 (Web Mercator) */
export function metersPerPixel(zoom: number, lat: number): number {
  return (156543.03392 * Math.cos((lat * Math.PI) / 180)) / Math.pow(2, zoom)
}

/** 진행 방향 순서의 선을 오른쪽으로 meters 만큼 평행 이동 (우측통행: 그 방면 차선 쪽) */
export function offsetLine(coords: number[][], meters: number): [number, number][] {
  if (coords.length < 2) return coords.map((c) => [c[0], c[1]] as [number, number])
  const lat0 = coords[0][0]
  const kx = Math.cos((lat0 * Math.PI) / 180) * 111320
  const ky = 110540
  const xy = coords.map(([la, lo]) => [lo * kx, la * ky])
  const out: [number, number][] = []
  for (let i = 0; i < xy.length; i++) {
    const a = xy[Math.max(0, i - 1)], b = xy[Math.min(xy.length - 1, i + 1)]
    let dx = b[0] - a[0], dy = b[1] - a[1]
    const L = Math.hypot(dx, dy) || 1
    dx /= L; dy /= L
    // 진행 방향 (dx, dy) 의 오른쪽 법선 = (dy, -dx)
    const x = xy[i][0] + dy * meters, y = xy[i][1] - dx * meters
    out.push([y / ky, x / kx])
  }
  return out
}
