import L from 'leaflet'
import { ArrowLeftRight, Crosshair, FlipVertical2, MapPin, RotateCcw, Sparkles } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap, useMapEvents } from 'react-leaflet'
import { api, imageSrc, type Direction } from '../api/client'
import { anchorOf, arrowIcon, bearing, compass, destination, groupByRoad, headingName } from '../lib/geo'

/**
 * 방향 ↔ 지도 연결 편집기.
 * 방향마다 "지도에서 지정" 을 누르고 ① 그 차로가 지나는 지점 ② 차가 가는 쪽 을 차례로 클릭하면
 * 화살표 위치(lat/lon)와 진행 각도(heading_deg)가 정해진다. 한 CCTV 에 도로가 여러 개면 도로 이름으로 묶는다.
 */
interface Props {
  directions: Direction[]
  onChange: (d: Direction[]) => void
  cameraLat: number | null
  cameraLon: number | null
  previewUrl?: string // 방향 색이 칠해진 스냅샷 (어느 라벨이 어느 차로인지 보면서 지정)
  height?: number
  route?: string | null // 노선 (자동 제안 시 같은 노선 이웃 CCTV 만 사용)
  roadType?: string | null // ex | its
}

function Recenter({ center, zoom }: { center: [number, number] | null; zoom: number }) {
  const map = useMap()
  const [done, setDone] = useState(false)
  useEffect(() => {
    if (done || !center) return
    map.setView(center, zoom)
    setDone(true)
  }, [center, zoom, map, done])
  return null
}

function ClickCatcher({ onClick }: { onClick: (p: [number, number]) => void }) {
  useMapEvents({ click: (e) => onClick([e.latlng.lat, e.latlng.lng]) })
  return null
}

export default function DirectionMapEditor({ directions, onChange, cameraLat, cameraLon, previewUrl, height = 420, route, roadType }: Props) {
  const [suggest, setSuggest] = useState<{ axis: number; linearity: number; neighbors: { name: string; lat: number; lon: number }[]; note: string } | null>(null)
  const [suggestBusy, setSuggestBusy] = useState(false)
  const [suggestErr, setSuggestErr] = useState<string | null>(null)
  const [active, setActive] = useState<number | null>(null) // 지정 중인 방향 index
  const [first, setFirst] = useState<[number, number] | null>(null) // 첫 클릭(위치)
  const [preview, setPreview] = useState('')
  const cam = { lat: cameraLat, lon: cameraLon }
  const center: [number, number] | null = useMemo(() => {
    const pts = directions.map((d) => anchorOf(d, cam)).filter(Boolean) as [number, number][]
    if (pts.length) return [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length]
    return cameraLat != null && cameraLon != null ? [cameraLat, cameraLon] : null
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!previewUrl) return
    let u = ''
    imageSrc(previewUrl).then((s) => { u = s; setPreview(s) }).catch(() => setPreview(''))
    return () => { if (u.startsWith('blob:')) URL.revokeObjectURL(u) }
  }, [previewUrl])

  const patch = (idx: number, p: Partial<Direction>) => onChange(directions.map((d) => (d.index === idx ? { ...d, ...p } : d)))
  const onMapClick = (p: [number, number]) => {
    if (active == null) return
    if (!first) { setFirst(p); return }
    const h = bearing(first, p)
    patch(active, { lat: first[0], lon: first[1], heading_deg: Math.round(h * 10) / 10 })
    setFirst(null)
    // 다음 미지정 방향으로 자동 이동
    const next = directions.find((d) => d.index !== active && d.heading_deg == null)
    setActive(next ? next.index : null)
  }
  const isDefaultName = (n: string) => /^방향\s*\d+/.test(n.trim()) || /^[북남동서]+행$/.test(n.trim())
  const runSuggest = async () => {
    if (cameraLat == null || cameraLon == null) return
    setSuggestBusy(true)
    setSuggestErr(null)
    try {
      const r = await api.its.suggestHeading({ lat: cameraLat, lon: cameraLon, route, road_type: roadType ?? 'ex' })
      setSuggest({ axis: r.axis_deg, linearity: r.linearity, neighbors: r.neighbors, note: r.used_route ? `같은 노선(${r.route}) 이웃 CCTV ${r.neighbors.length}개 기준` : `주변 CCTV ${r.neighbors.length}개 기준 (노선 일치 없음)` })
      // 도로가 비었거나 노선과 같은 방향 중 앞의 두 개에 축/반대 적용
      const main = directions.filter((d) => !d.road?.trim() || d.road.trim() === (route ?? '').trim()).slice(0, 2)
      const hs = [r.axis_deg, r.opposite_deg]
      onChange(directions.map((d) => {
        const i = main.findIndex((m) => m.index === d.index)
        if (i < 0) return d
        return { ...d, heading_deg: hs[i], lat: null, lon: null, road: d.road?.trim() || route || d.road, name: isDefaultName(d.name) ? headingName(hs[i]) : d.name }
      }))
    } catch (e: any) {
      setSuggestErr(e.message)
    } finally {
      setSuggestBusy(false)
    }
  }
  const swapPair = (road: string) => {
    const pair = directions.filter((d) => (d.road?.trim() || '(도로 미지정)') === road && d.heading_deg != null).slice(0, 2)
    if (pair.length < 2) return
    const [a, b] = pair
    onChange(directions.map((d) => (d.index === a.index ? { ...d, heading_deg: b.heading_deg, lat: b.lat, lon: b.lon, name: isDefaultName(d.name) && b.heading_deg != null ? headingName(b.heading_deg) : d.name } : d.index === b.index ? { ...d, heading_deg: a.heading_deg, lat: a.lat, lon: a.lon, name: isDefaultName(d.name) && a.heading_deg != null ? headingName(a.heading_deg) : d.name } : d)))
  }
  const roads = [...new Set(directions.map((d) => d.road?.trim()).filter(Boolean))] as string[]
  const groups = groupByRoad(directions)
  const activeDir = directions.find((d) => d.index === active)

  return (
    <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.3fr) minmax(300px, 1fr)', alignItems: 'start' }}>
      <div className="stack" style={{ gap: 8 }}>
        <div className="card pad-0" style={{ position: 'relative', height }}>
          <MapContainer center={center ?? [36.4, 127.8]} zoom={center ? 17 : 7} style={{ height: '100%', cursor: active != null ? 'crosshair' : undefined }}>
            <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" maxZoom={19} />
            <Recenter center={center} zoom={17} />
            <ClickCatcher onClick={onMapClick} />
            {cameraLat != null && cameraLon != null && (
              <CircleMarker center={[cameraLat, cameraLon]} radius={6} pathOptions={{ color: '#fff', fillColor: '#8b96ad', fillOpacity: 1, weight: 2 }}>
                <Tooltip>CCTV 위치</Tooltip>
              </CircleMarker>
            )}
            {directions.map((d) => {
              const a = anchorOf(d, cam)
              if (!a || d.heading_deg == null) return null
              const tip = destination(a, d.heading_deg, 40)
              return (
                <span key={d.index}>
                  <Polyline positions={[a, tip]} pathOptions={{ color: d.color, weight: 3, dashArray: '4 4', opacity: 0.8 }} />
                  <Marker position={a} icon={arrowIcon({ heading: d.heading_deg, color: d.color, label: d.name, offsetPx: 14, selected: d.index === active, textColor: '#fff' })} eventHandlers={{ click: () => setActive(d.index) }} />
                </span>
              )
            })}
            {suggest?.neighbors.map((n, i) => (
              <CircleMarker key={i} center={[n.lat, n.lon]} radius={5} pathOptions={{ color: '#fff', fillColor: '#5a6680', fillOpacity: 1, weight: 1 }}>
                <Tooltip>{n.name}</Tooltip>
              </CircleMarker>
            ))}
            {suggest && cameraLat != null && cameraLon != null && (
              <Polyline positions={[destination([cameraLat, cameraLon], suggest.axis, 600), destination([cameraLat, cameraLon], suggest.axis + 180, 600)]} pathOptions={{ color: '#ffe600', weight: 2, dashArray: '6 6', opacity: 0.8 }} />
            )}
            {first && <CircleMarker center={first} radius={7} pathOptions={{ color: '#fff', fillColor: activeDir?.color ?? '#4f8cff', fillOpacity: 1, weight: 2 }} />}
          </MapContainer>
          <div className="map-overlay small" style={{ bottom: 22, left: 10, right: 10 }}>
            {active == null
              ? '오른쪽 목록에서 방향의 "지도에서 지정" 을 누르세요.'
              : !first
                ? <><b style={{ color: activeDir?.color }}>{activeDir?.name}</b>: ① 이 방향 차로가 지나는 지점을 클릭</>
                : <><b style={{ color: activeDir?.color }}>{activeDir?.name}</b>: ② 차가 <b>가는 쪽</b>(예: 서울 쪽 도로 위)을 클릭 <button className="sm ghost" onClick={(e) => { e.stopPropagation(); setFirst(null) }}>다시</button></>}
          </div>
        </div>
        {preview && (
          <div className="card pad-0">
            <img src={preview} alt="" style={{ width: '100%', display: 'block' }} />
            <div className="muted" style={{ padding: '6px 10px' }}>CCTV 화면의 방향 색. 화면 속 표지판(예: "⬆ 서울")을 보고 지도에서 같은 쪽을 고르세요.</div>
          </div>
        )}
      </div>

      <div className="stack" style={{ gap: 10 }}>
        <div className="card" style={{ padding: 10 }}>
          <div className="row between">
            <b className="row" style={{ gap: 6 }}><Sparkles size={14} />도로 축 자동 제안</b>
            <button className="sm primary" onClick={runSuggest} disabled={suggestBusy || cameraLat == null || cameraLon == null}>{suggestBusy ? '계산 중…' : '제안 받기'}</button>
          </div>
          <div className="muted" style={{ marginTop: 4 }}>
            {cameraLat == null ? 'CCTV 좌표가 있어야 합니다.' : suggest ? `${compass(suggest.axis)} ↔ ${compass((suggest.axis + 180) % 360)} · ${suggest.note} · 직선도 ${Math.round(suggest.linearity * 100)}%` : 'ITS 의 같은 노선 이웃 CCTV 위치로 도로가 뻗은 방향을 계산해 방향 1·2 에 넣습니다. 화면 속 표지판과 반대면 "두 방향 바꾸기" 를 누르세요.'}
          </div>
          {suggestErr && <div className="error" style={{ marginTop: 4 }}>{suggestErr}</div>}
        </div>
        <datalist id="road-names">{roads.map((r) => <option key={r} value={r} />)}</datalist>
        {groups.map(([road, dirs]) => (
          <div key={road} className="card" style={{ padding: 10 }}>
            <div className="row" style={{ gap: 6, marginBottom: 6 }}>
              <MapPin size={14} color="var(--muted)" /><b>{road}</b><span className="muted">{dirs.length}개 방향</span>
              {dirs.filter((d) => d.heading_deg != null).length >= 2 && <button className="sm" style={{ marginLeft: 'auto' }} onClick={() => swapPair(road)} title="화면 속 차로와 지도 방향이 반대로 잡혔을 때"><ArrowLeftRight />두 방향 바꾸기</button>}
            </div>
            <div className="stack" style={{ gap: 8 }}>
              {dirs.map((d) => (
                <div key={d.index} style={{ borderLeft: `3px solid ${d.color}`, paddingLeft: 8, outline: active === d.index ? '1px solid var(--accent)' : 'none', borderRadius: 4 }}>
                  <div className="row" style={{ gap: 6 }}>
                    <span className="kbd">{d.index}</span>
                    <input value={d.name} onChange={(e) => patch(d.index, { name: e.target.value })} placeholder="예: 서울 방면" style={{ width: 120, padding: '4px 8px' }} />
                    <input value={d.road ?? ''} onChange={(e) => patch(d.index, { road: e.target.value })} list="road-names" placeholder="도로 (예: 경부선 본선)" style={{ flex: 1, minWidth: 110, padding: '4px 8px' }} />
                  </div>
                  <div className="row" style={{ gap: 6, marginTop: 6 }}>
                    <span className="pill" title="진행 방향">{d.heading_deg == null ? '방향 미지정' : `→ ${compass(d.heading_deg)}`}</span>
                    <button className={`sm ${active === d.index ? 'active' : ''}`} onClick={() => { setActive(active === d.index ? null : d.index); setFirst(null) }}><Crosshair />지도에서 지정</button>
                    <button className="sm icon" title="반대 방향으로 (180° 회전)" disabled={d.heading_deg == null} onClick={() => patch(d.index, { heading_deg: ((d.heading_deg ?? 0) + 180) % 360 })}><FlipVertical2 /></button>
                    <button className="sm icon" title="지도 지정 지우기" disabled={d.heading_deg == null && d.lat == null} onClick={() => patch(d.index, { heading_deg: null, lat: null, lon: null })}><RotateCcw /></button>
                  </div>
                  {d.heading_deg == null && directions.some((o) => o.index !== d.index && o.heading_deg != null && (o.road ?? '') === (d.road ?? '')) && (
                    <button className="sm ghost" style={{ marginTop: 4 }} onClick={() => {
                      const o = directions.find((x) => x.index !== d.index && x.heading_deg != null && (x.road ?? '') === (d.road ?? ''))!
                      patch(d.index, { heading_deg: ((o.heading_deg ?? 0) + 180) % 360, lat: o.lat, lon: o.lon })
                    }}>같은 도로 "{directions.find((x) => x.index !== d.index && x.heading_deg != null && (x.road ?? '') === (d.road ?? ''))?.name}" 의 반대 방향으로</button>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
        <div className="muted">
          한 화면에 도로가 여러 개면(본선 + 램프, 교차 도로) 방향마다 <b>도로</b> 이름을 다르게 적으세요. 같은 도로의 상·하행은 같은 이름으로 묶입니다.
          지도에는 방향마다 진행 방향 화살표가 그려지고, 우측통행이라 상·하행이 도로 양옆에 나란히 표시됩니다.
        </div>
      </div>
    </div>
  )
}

// L 은 react-leaflet 이 내부에서 쓰는 전역과 같은 인스턴스
void L
