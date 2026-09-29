import { Crosshair, FlipVertical2, MapPin, RefreshCw, Wand2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap, useMapEvents } from 'react-leaflet'
import { api, imageSrc, type Direction } from '../api/client'
import { dirLabel } from '../lib/format'
import { anchorOf, arrowIcon, bearing, compass, groupByRoad } from '../lib/geo'

/**
 * 방면 설정기.
 * 방향(= 마스크의 색 영역) 마다 ① 노선 ② 방면(화면 표지판의 목적지, 예: 서울) 을 적으면
 * 노선 선 위에서 그 목적지 쪽을 찾아 지도 진행 방향이 자동으로 정해진다.
 * 자동 결과가 틀리면 '뒤집기' 나 '지도에서 직접' 으로 바꾸고(수동), '자동으로' 로 되돌린다.
 */
interface Props {
  directions: Direction[]
  onChange: (d: Direction[]) => void
  cameraLat: number | null
  cameraLon: number | null
  previewUrl?: string
  height?: number
  route?: string | null
  roadType?: string | null
  roadPx?: Record<string, number> | null // 방향별 칠한 픽셀 수 (0 = 미측정)
}

function Recenter({ center }: { center: [number, number] | null }) {
  const map = useMap()
  const done = useRef(false)
  useEffect(() => {
    if (done.current || !center) return
    map.setView(center, 15)
    done.current = true
  }, [center, map])
  return null
}
function ClickCatcher({ onClick }: { onClick: (p: [number, number]) => void }) {
  useMapEvents({ click: (e) => onClick([e.latlng.lat, e.latlng.lng]) })
  return null
}

export default function DirectionMapEditor({ directions, onChange, cameraLat, cameraLon, previewUrl, height = 420, route, roadType, roadPx }: Props) {
  const [active, setActive] = useState<number | null>(null)
  const [first, setFirst] = useState<[number, number] | null>(null)
  const [preview, setPreview] = useState('')
  const [line, setLine] = useState<number[][] | null>(null)
  const [lineErr, setLineErr] = useState<string | null>(null)
  const [reasons, setReasons] = useState<Record<number, string>>({})
  const [busy, setBusy] = useState(false)
  const dirsRef = useRef(directions)
  dirsRef.current = directions
  const cam = { lat: cameraLat, lon: cameraLon }
  const center: [number, number] | null = cameraLat != null && cameraLon != null ? [cameraLat, cameraLon] : null

  useEffect(() => {
    if (!previewUrl) return
    let u = ''
    imageSrc(previewUrl).then((s) => { u = s; setPreview(s) }).catch(() => setPreview(''))
    return () => { if (u.startsWith('blob:')) URL.revokeObjectURL(u) }
  }, [previewUrl])
  // 카메라 주변 노선 선
  const mainRoute = directions.find((d) => d.road?.trim())?.road?.trim() || route || ''
  useEffect(() => {
    if (!mainRoute || cameraLat == null || cameraLon == null) { setLine(null); return }
    setLineErr(null)
    api.routes.near({ route: mainRoute, lat: cameraLat, lon: cameraLon, road_type: roadType }).then((r) => setLine(r.coords)).catch((e) => { setLine(null); setLineErr(e.message) })
  }, [mainRoute, cameraLat, cameraLon, roadType])

  const patch = (idx: number, p: Partial<Direction>) => onChange(dirsRef.current.map((d) => (d.index === idx ? { ...d, ...p } : d)))
  const measured = (d: Direction) => (roadPx ? (roadPx[String(d.index)] ?? 0) > 0 : d.measured !== false)

  /** 방면 → 진행 방향 자동 계산 (수동 지정은 건드리지 않음) */
  const resolveAll = async (only?: number) => {
    if (cameraLat == null || cameraLon == null) return
    const cur = dirsRef.current
    const targets = cur.filter((d) => d.destination?.trim() && d.heading_source !== 'manual' && (only == null || d.index === only || (d.road ?? '') === (cur.find((x) => x.index === only)?.road ?? '')))
    if (!targets.length) return
    setBusy(true)
    try {
      const groups = new Map<string, Direction[]>()
      targets.forEach((d) => { const k = d.road?.trim() || route || ''; groups.set(k, [...(groups.get(k) ?? []), d]) })
      const updates: Record<number, Partial<Direction>> = {}
      const why: Record<number, string> = {}
      for (const [r, ds] of groups) {
        const res = await api.routes.resolve({ route: r, lat: cameraLat, lon: cameraLon, road_type: roadType, destinations: ds.map((d) => d.destination!.trim()) })
        ds.forEach((d) => {
          const x = res[d.destination!.trim()]
          if (x?.ok) { updates[d.index] = { heading_deg: x.heading, heading_source: 'auto', lat: null, lon: null }; why[d.index] = x.reason }
          else { updates[d.index] = { heading_deg: null, heading_source: null }; why[d.index] = x?.reason ?? '계산 실패' }
        })
      }
      onChange(dirsRef.current.map((d) => (updates[d.index] ? { ...d, ...updates[d.index] } : d)))
      setReasons((r) => ({ ...r, ...why }))
    } catch (e: any) {
      setLineErr(e.message)
    } finally {
      setBusy(false)
    }
  }
  // 방면을 적고 잠시 멈추면 자동 계산
  const timer = useRef<number | undefined>(undefined)
  const onDest = (idx: number, v: string) => {
    const name = v.trim() ? `${v.trim()} 방면` : `방향 ${idx}`
    patch(idx, { destination: v, name, heading_source: dirsRef.current.find((d) => d.index === idx)?.heading_source === 'manual' ? 'manual' : null })
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => resolveAll(idx), 700)
  }

  const onMapClick = (p: [number, number]) => {
    if (active == null) return
    if (!first) { setFirst(p); return }
    patch(active, { lat: first[0], lon: first[1], heading_deg: Math.round(bearing(first, p) * 10) / 10, heading_source: 'manual' })
    setReasons((r) => ({ ...r, [active]: '지도에서 직접 지정' }))
    setFirst(null)
    setActive(null)
  }
  const routes = useMemo(() => [...new Set([route, ...directions.map((d) => d.road?.trim())].filter(Boolean))] as string[], [route, directions])
  const groups = groupByRoad(directions.map((d) => ({ ...d, road: d.road?.trim() || route || '' })))
  const activeDir = directions.find((d) => d.index === active)

  return (
    <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.2fr) minmax(320px, 1fr)', alignItems: 'start' }}>
      <div className="stack" style={{ gap: 8 }}>
        <div className="card pad-0" style={{ position: 'relative', height }}>
          <MapContainer center={center ?? [36.4, 127.8]} zoom={center ? 15 : 7} style={{ height: '100%', cursor: active != null ? 'crosshair' : undefined }}>
            <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={19} />
            <Recenter center={center} />
            <ClickCatcher onClick={onMapClick} />
            {line && <Polyline positions={line as [number, number][]} pathOptions={{ color: '#0b0e14', weight: 12, opacity: 0.5, lineJoin: 'round' }} interactive={false} />}
            {line && <Polyline positions={line as [number, number][]} pathOptions={{ color: '#c3cad8', weight: 8, opacity: 0.9, lineJoin: 'round' }}><Tooltip sticky>{mainRoute} (ITS CCTV 좌표로 만든 노선 선)</Tooltip></Polyline>}
            {center && <CircleMarker center={center} radius={6} pathOptions={{ color: '#fff', fillColor: '#0b0e14', fillOpacity: 1, weight: 2 }}><Tooltip>CCTV 위치</Tooltip></CircleMarker>}
            {directions.map((d) => {
              const a = anchorOf(d, cam)
              if (!a || d.heading_deg == null) return null
              return <Marker key={d.index} position={a} icon={arrowIcon({ heading: d.heading_deg, color: measured(d) ? d.color : '#666', label: dirLabel(d), offsetPx: 16, selected: d.index === active, textColor: '#fff', dim: !measured(d) })} />
            })}
            {first && <CircleMarker center={first} radius={7} pathOptions={{ color: '#fff', fillColor: activeDir?.color ?? '#4f8cff', fillOpacity: 1, weight: 2 }} />}
          </MapContainer>
          <div className="map-overlay small" style={{ bottom: 22, left: 10, right: 10 }}>
            {active != null
              ? !first ? <><b style={{ color: activeDir?.color }}>{activeDir && dirLabel(activeDir)}</b>: ① 이 방면 차로가 지나는 지점 클릭</> : <><b style={{ color: activeDir?.color }}>{activeDir && dirLabel(activeDir)}</b>: ② 차가 가는 쪽 클릭</>
              : lineErr ? <span className="error">{lineErr}</span>
              : '밝은 회색 띠 = 노선, 화살표 = 방면별 진행 방향 (우측통행이라 진행 방향 오른쪽에 표시)'}
          </div>
        </div>
        {preview && (
          <div className="card pad-0">
            <img src={preview} alt="" style={{ width: '100%', display: 'block' }} />
            <div className="muted" style={{ padding: '6px 10px' }}>색 영역마다 화면 속 방면 표지(예: "⬇ 부산 · 서울 ⬆")를 보고 방면을 적으세요.</div>
          </div>
        )}
      </div>

      <div className="stack" style={{ gap: 10 }}>
        <div className="banner info" style={{ alignItems: 'center' }}>
          <span className="grow">색 영역마다 <b>방면</b>(표지판의 목적지)만 적으면 지도 방향은 노선 선에서 자동으로 정해집니다.</span>
          <button className="sm" onClick={() => resolveAll()} disabled={busy || cameraLat == null}><RefreshCw />{busy ? '계산 중…' : '다시 계산'}</button>
        </div>
        <datalist id="road-names">{routes.map((r) => <option key={r} value={r} />)}</datalist>
        {groups.map(([road, dirs]) => (
          <div key={road} className="card" style={{ padding: 10 }}>
            <div className="row" style={{ gap: 6, marginBottom: 6 }}><MapPin size={14} color="var(--muted)" /><b>{road || '(노선 미지정)'}</b><span className="muted">{dirs.length}개 방면</span></div>
            <div className="stack" style={{ gap: 10 }}>
              {dirs.map((g) => {
                const d = directions.find((x) => x.index === g.index)!
                const ok = measured(d)
                return (
                  <div key={d.index} style={{ borderLeft: `3px solid ${ok ? d.color : '#555'}`, paddingLeft: 8, opacity: ok ? 1 : 0.75 }}>
                    <div className="row" style={{ gap: 6 }}>
                      <span className="kbd">{d.index}</span>
                      <input value={d.destination ?? ''} onChange={(e) => onDest(d.index, e.target.value)} placeholder="방면 (예: 서울)" style={{ width: 110, padding: '4px 8px' }} />
                      <input value={d.road ?? ''} onChange={(e) => patch(d.index, { road: e.target.value })} onBlur={() => resolveAll(d.index)} list="road-names" placeholder={route ? `노선 (기본: ${route})` : '노선 (예: 경부선)'} style={{ flex: 1, minWidth: 100, padding: '4px 8px' }} />
                    </div>
                    <div className="row" style={{ gap: 6, marginTop: 5 }}>
                      {!ok ? <span className="badge none" title="마스크에서 이 색으로 칠한 픽셀이 없습니다">미측정 · 칠하지 않음</span>
                        : d.heading_deg == null ? <span className="pill">방향 미정</span>
                        : <span className="pill">→ {compass(d.heading_deg)} <span className="muted">{d.heading_source === 'auto' ? '자동 (방면)' : '수동'}</span></span>}
                      <button className="sm icon" title="반대로 뒤집기 (수동)" disabled={d.heading_deg == null} onClick={() => { patch(d.index, { heading_deg: ((d.heading_deg ?? 0) + 180) % 360, heading_source: 'manual' }); setReasons((r) => ({ ...r, [d.index]: '뒤집음 (수동)' })) }}><FlipVertical2 /></button>
                      <button className={`sm icon ${active === d.index ? 'active' : ''}`} title="지도에서 직접 지정 (수동)" onClick={() => { setActive(active === d.index ? null : d.index); setFirst(null) }}><Crosshair /></button>
                      {d.heading_source === 'manual' && d.destination && <button className="sm" title="방면으로 다시 자동 계산" onClick={() => { patch(d.index, { heading_source: null }); window.setTimeout(() => resolveAll(d.index), 0) }}><Wand2 />자동으로</button>}
                    </div>
                    {reasons[d.index] && <div className="muted" style={{ marginTop: 3 }}>{reasons[d.index]}</div>}
                  </div>
                )
              })}
            </div>
          </div>
        ))}
        <div className="muted">
          한 화면에 노선이 여러 개면(본선 + 램프, 교차 도로) 영역마다 노선 이름을 다르게 적으세요. 칠하지 않은 방면은 "미측정" 으로 남고 지도·통계에 0% 로 나오지 않습니다.
          빠진 방면을 재려면 "도로 마스크 편집" 에서 그 차로를 해당 색으로 칠하세요.
        </div>
      </div>
    </div>
  )
}
