import type { Direction, LiveState } from '../api/client'
import { CLASS_COLORS, CLASS_LABELS, DIRECTION_PALETTE, dirLabel, pct } from '../lib/format'
import { compass } from '../lib/geo'
import LevelBadge from './LevelBadge'
import OccupancyBar from './OccupancyBar'

/** 방향별 점유율/대수 패널. showClasses=true 면 차종별 구분 표시. */
export default function DirectionPanel({ live, directions, showClasses, compact = false }: { live: LiveState | null | undefined; directions: Direction[]; showClasses: boolean; compact?: boolean }) {
  if (!live || !live.directions?.length)
    return (
      <div className="muted">
        {live?.status === 'starting' ? '워커 시작 중…' : live?.status === 'reconnecting' ? '스트림 재연결 중…' : '아직 측정값이 없습니다 (정지 상태)'}
        {live?.error ? <div className="error" style={{ marginTop: 4 }}>{live.error}</div> : null}
      </div>
    )
  const liveIdx = new Set(live.directions.map((d) => d.direction_index))
  const unmeasured = directions.filter((d) => !liveIdx.has(d.index))
  const colorOf = (idx: number) => directions.find((d) => d.index === idx)?.color ?? DIRECTION_PALETTE[(idx - 1) % DIRECTION_PALETTE.length]
  const roadOf = (idx: number) => directions.find((d) => d.index === idx)?.road?.trim() || ''
  const headingOf = (idx: number) => directions.find((d) => d.index === idx)?.heading_deg
  const multiRoad = new Set(live.directions.filter((d) => d.direction_index !== 0).map((d) => roadOf(d.direction_index))).size > 1
  // 도로가 여러 개면 전체 → 도로별로 묶어서
  const ordered = [...live.directions].sort((a, b) => (a.direction_index === 0 ? -1 : b.direction_index === 0 ? 1 : roadOf(a.direction_index).localeCompare(roadOf(b.direction_index)) || a.direction_index - b.direction_index))
  return (
    <div className="stack" style={{ gap: compact ? 6 : 10 }}>
      {ordered.map((d, i) => (
        <div key={d.direction_index}>
          {multiRoad && d.direction_index !== 0 && roadOf(d.direction_index) !== roadOf(ordered[i - 1]?.direction_index ?? -1) && (
            <div className="muted" style={{ fontWeight: 600, marginTop: 4, marginBottom: 2 }}>{roadOf(d.direction_index) || '(도로 미지정)'}</div>
          )}
          <div className="row" style={{ gap: 8, marginBottom: 2 }}>
            {d.direction_index !== 0 && <span className="dot" style={{ background: colorOf(d.direction_index) }} />}
            <strong style={{ fontSize: 13 }}>{d.direction_index === 0 ? (unmeasured.length ? '측정 방면 전체' : '전체') : dirLabel(directions.find((x) => x.index === d.direction_index) ?? { name: d.name, destination: d.destination })}</strong>
            <LevelBadge level={d.level} size="sm" />
            <span className="muted">{d.n_vehicles}대</span>
            {d.direction_index !== 0 && headingOf(d.direction_index) != null && <span className="muted">→ {compass(headingOf(d.direction_index))}</span>}
            {!compact && <span className="muted">도로 {Math.round(d.road_px / 1000)}k px</span>}
          </div>
          <OccupancyBar value={d.occupancy} level={d.level} compact={compact} />
          {showClasses && (
            <div className="row" style={{ gap: 6, marginTop: 4 }}>
              {Object.keys(CLASS_LABELS).map((c) => (
                <span key={c} className="pill" style={{ padding: '1px 7px' }}>
                  <span className="dot" style={{ background: CLASS_COLORS[c], width: 8, height: 8 }} />
                  {CLASS_LABELS[c]} {d.counts?.[c] ?? 0}
                  {!compact && d.road_px > 0 && <span className="muted">({pct((d.class_px?.[c] ?? 0) / d.road_px)})</span>}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
      {unmeasured.map((d) => (
        <div key={`u${d.index}`} className="row" style={{ gap: 8, opacity: 0.75 }}>
          <span className="dot" style={{ background: '#555' }} />
          <strong style={{ fontSize: 13 }}>{dirLabel(d)}</strong>
          <span className="badge none" style={{ fontSize: 11 }}>미측정</span>
          {!compact && <span className="muted">마스크에 이 방면 차로를 칠하지 않음</span>}
        </div>
      ))}
    </div>
  )
}
