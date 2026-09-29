import { Marker, Tooltip, useMapEvents } from 'react-leaflet'
import type { Direction } from '../api/client'
import { levelColor, pct } from '../lib/format'
import { anchorOf, arrowIcon, compass } from '../lib/geo'

export interface DirValue {
  value: number | null
  level: string | null
  vehicles?: number | null
}

/** 지도 줌을 상태로 받는다 */
export function ZoomWatcher({ onZoom }: { onZoom: (z: number) => void }) {
  const map = useMapEvents({ zoomend: () => onZoom(map.getZoom()) })
  return null
}

/**
 * 카메라 한 대의 방향별 화살표. 진행 각도가 지정된 방향만 그린다.
 * 색 = 그 방향의 혼잡 단계, 숫자 = 점유율(%) 또는 차량 수.
 */
export default function DirectionArrows({
  cam,
  values,
  metric = 'occupancy',
  selected = false,
  dim = false,
  onClick,
}: {
  cam: { id: number; name: string; lat: number | null; lon: number | null; directions: Direction[] }
  values: Record<number, DirValue>
  metric?: 'occupancy' | 'vehicles'
  selected?: boolean
  dim?: boolean
  onClick?: () => void
}) {
  return (
    <>
      {cam.directions.map((d) => {
        const a = anchorOf(d, cam)
        if (!a || d.heading_deg == null) return null
        const v = values[d.index] ?? { value: null, level: null }
        if (d.measured === false) return null // 미측정 방면은 화살표를 그리지 않는다
        const num = metric === 'occupancy' ? (v.value == null ? '–' : `${Math.round(v.value * 100)}%`) : v.vehicles == null ? '–' : `${Math.round(v.vehicles)}대`
        const label = num
        return (
          <Marker
            key={`${cam.id}-${d.index}`}
            position={a}
            icon={arrowIcon({ heading: d.heading_deg, color: levelColor(v.level), label, dim, selected, textColor: v.level === '정체' ? '#fff' : '#0b0b0b' })}
            zIndexOffset={Math.round((v.value ?? 0) * 1000)}
            eventHandlers={onClick ? { click: onClick } : undefined}
          >
            <Tooltip direction="top" offset={[0, -16]}>
              <b>{cam.name}</b>
              <br />
              {d.road ? `${d.road} · ` : ''}
              {d.name} ({compass(d.heading_deg)})
              <br />
              점유율 {pct(v.value)} · {v.level ?? '–'}
              {v.vehicles != null ? ` · 차량 ${Math.round(v.vehicles)}대` : ''}
            </Tooltip>
          </Marker>
        )
      })}
    </>
  )
}

export const hasHeading = (c: { directions: Direction[] }) => c.directions.some((d) => d.heading_deg != null)
export const allHeading = (c: { directions: Direction[] }) => c.directions.length > 0 && c.directions.every((d) => d.heading_deg != null)
