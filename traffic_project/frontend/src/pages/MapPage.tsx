import L from 'leaflet'
import { Activity, AlertTriangle, Camera, Car, Gauge, Pause, Play, PlusCircle, Search, Sparkles, Video } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { MapContainer, Marker, TileLayer, Tooltip, useMap } from 'react-leaflet'
import { Link, useNavigate } from 'react-router-dom'
import { api, type LiveCamera } from '../api/client'
import DirectionArrows, { allHeading, hasHeading, ZoomWatcher, type DirValue } from '../components/DirectionArrows'
import DirectionPanel from '../components/DirectionPanel'
import LevelBadge from '../components/LevelBadge'
import LiveImage from '../components/LiveImage'
import OccupancyBar from '../components/OccupancyBar'
import { Banner, EmptyState, Loading, Segmented, Sparkline, StatCard, useAction } from '../components/ui'
import { fmtTime, LEVEL_CLASS, LEVELS, levelColor, levelOf, pct } from '../lib/format'
import { usePolling } from '../lib/usePolling'

const DEFAULT_CENTER: [number, number] = [36.5, 127.8]

function FitOnce({ points }: { points: [number, number][] }) {
  const map = useMap()
  const done = useRef(false)
  useEffect(() => {
    if (done.current || !points.length) return
    done.current = true
    map.fitBounds(L.latLngBounds(points).pad(0.25), { maxZoom: 13 })
  }, [points, map])
  return null
}

function markerIcon(level: string | null, running: boolean, selected: boolean) {
  const color = levelColor(level)
  const size = selected ? 22 : 16
  return L.divIcon({
    className: '',
    html: `<div class="cam-marker ${running ? 'running' : ''} ${selected ? 'selected' : ''}" style="width:${size}px;height:${size}px;color:${color};position:relative">${running ? '<div class="ring"></div>' : ''}<div class="pin" style="background:${color};opacity:${running ? 1 : 0.45}"></div></div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    tooltipAnchor: [0, -size / 2],
  })
}

export default function MapPage() {
  const nav = useNavigate()
  const { data, error } = usePolling(() => api.metrics.live(), 2000)
  const [selected, setSelected] = useState<number | null>(null)
  const [zoom, setZoom] = useState(7)
  const ARROW_ZOOM = 12
  const [showClasses, setShowClasses] = useState(false)
  const [q, setQ] = useState('')
  const [route, setRoute] = useState('')
  const [levelFilter, setLevelFilter] = useState<string | null>(null)
  const [onlyRunning, setOnlyRunning] = useState(false)
  // 타임라인 재생
  const [tlMinutes, setTlMinutes] = useState(60)
  const [tl, setTl] = useState<Awaited<ReturnType<typeof api.metrics.timeline>> | null>(null)
  const [tlIdx, setTlIdx] = useState<number | null>(null) // null = 실시간
  const [playing, setPlaying] = useState(false)
  const { run, busy } = useAction()

  useEffect(() => {
    let alive = true
    const load = () => api.metrics.timeline(tlMinutes, tlMinutes >= 720 ? 900 : tlMinutes >= 180 ? 300 : 60).then((t) => alive && setTl(t)).catch(() => {})
    load()
    const id = setInterval(load, 30000)
    return () => { alive = false; clearInterval(id) }
  }, [tlMinutes])
  useEffect(() => {
    if (!playing || !tl) return
    const id = setInterval(() => setTlIdx((i) => (i == null || i >= tl.buckets.length - 1 ? 0 : i + 1)), 500)
    return () => clearInterval(id)
  }, [playing, tl])

  const thresholds = data?.thresholds ?? [0.08, 0.15, 0.25]
  const cams = data?.cameras ?? []
  const routes = useMemo(() => [...new Set(cams.map((c) => c.route).filter(Boolean))] as string[], [cams])

  // 표시 값: 실시간 또는 타임라인 시점
  const valueOf = (c: LiveCamera): { occ: number | null; level: string | null } => {
    if (tlIdx != null && tl) {
      const v = tl.cameras[String(c.id)]?.[tlIdx] ?? null
      return { occ: v, level: levelOf(v, thresholds) }
    }
    return { occ: c.occupancy, level: c.level }
  }
  const filtered = cams.filter((c) => {
    if (q && !`${c.name} ${c.route ?? ''} ${c.region ?? ''} ${c.section ?? ''}`.toLowerCase().includes(q.toLowerCase())) return false
    if (route && c.route !== route) return false
    if (onlyRunning && !c.running) return false
    if (levelFilter && valueOf(c).level !== levelFilter) return false
    return true
  })
  const withPos = filtered.filter((c) => c.lat != null && c.lon != null)
  const sorted = useMemo(() => [...filtered].sort((a, b) => (valueOf(b).occ ?? -1) - (valueOf(a).occ ?? -1)), [filtered, tlIdx, tl]) // eslint-disable-line react-hooks/exhaustive-deps
  const sel = cams.find((c) => c.id === selected) ?? null
  const points = useMemo(() => withPos.map((c) => [c.lat!, c.lon!] as [number, number]), [withPos.length]) // eslint-disable-line react-hooks/exhaustive-deps

  // KPI
  const running = cams.filter((c) => c.running)
  const occs = running.map((c) => c.occupancy).filter((v): v is number => v != null)
  const meanOcc = occs.length ? occs.reduce((a, b) => a + b, 0) / occs.length : null
  const nJam = running.filter((c) => c.level === '정체' || c.level === '지체').length
  const nVeh = running.reduce((a, c) => a + (c.n_vehicles ?? 0), 0)
  const worst = [...running].sort((a, b) => (b.occupancy ?? -1) - (a.occupancy ?? -1))[0]

  if (error && !data) return <div className="page"><Banner kind="error">백엔드에 연결할 수 없습니다: {error}</Banner></div>
  if (!data) return <div className="page"><Loading lg /></div>

  if (!cams.length)
    return (
      <div className="page">
        <EmptyState
          icon={<Video />}
          title="아직 등록된 CCTV 가 없습니다"
          description={
            <span>
              세 단계로 시작합니다. ① 영상 소스(ITS 실시간 CCTV / 업로드) 선택 → ② SAM 으로 도로 영역과 방향 지정 → ③ 저장하면 실시간 점유율이 이 지도에 표시됩니다.
              <br />먼저 감을 잡으려면 샘플 영상으로 데모 카메라를 만들어 보세요.
            </span>
          }
          action={
            <>
              <button className="primary" disabled={!!busy} onClick={() => run('샘플 카메라 생성', () => api.demo(), '샘플 카메라를 만들고 모니터링을 시작했습니다')}><Sparkles />샘플로 시작</button>
              <Link to="/register"><button><PlusCircle />CCTV 등록하기</button></Link>
            </>
          }
        />
      </div>
    )

  return (
    <div className="page full" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="grid kpi-strip">
        <StatCard label="등록 CCTV" value={cams.length} unit="대" sub={`모니터링 ${running.length}대`} icon={<Camera />} />
        <StatCard label="평균 점유율" value={pct(meanOcc)} sub="모니터링 중인 카메라 전체" icon={<Gauge />} tone={meanOcc != null ? LEVEL_CLASS[levelOf(meanOcc, thresholds) ?? ''] : ''} />
        <StatCard label="지체·정체" value={nJam} unit="곳" sub={worst ? `최고 ${worst.name} ${pct(worst.occupancy)}` : '—'} icon={<AlertTriangle />} tone={nJam ? 'jam' : 'free'} />
        <StatCard label="검출 차량" value={Math.round(nVeh)} unit="대" sub="현재 프레임 합계" icon={<Car />} />
        <StatCard label="시점" value={tlIdx != null && tl ? fmtTime(tl.buckets[tlIdx]) : '실시간'} sub={tlIdx != null ? '타임라인 재생 중' : '2초마다 갱신'} icon={<Activity />} tone={tlIdx != null ? 'slow' : 'free'} />
      </div>
      <div className="map-layout">
        <div className="card pad-0 map-area">
          <MapContainer center={DEFAULT_CENTER} zoom={7} style={{ height: '100%', minHeight: 420 }}>
            <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <FitOnce points={points} />
            <ZoomWatcher onZoom={setZoom} />
            {withPos.map((c) => {
              const v = valueOf(c)
              const arrows = zoom >= ARROW_ZOOM && hasHeading(c)
              // 방향별 값: 실시간이면 방향별, 타임라인 재생 중이면 전체 값으로
              const dv: Record<number, DirValue> = {}
              c.directions.forEach((d) => {
                const ld = tlIdx == null ? c.live?.directions?.find((x) => x.direction_index === d.index) : undefined
                dv[d.index] = ld ? { value: ld.occupancy, level: ld.level, vehicles: ld.n_vehicles } : { value: v.occ, level: v.level }
              })
              return (
                <span key={c.id}>
                  {arrows && <DirectionArrows cam={c} values={dv} selected={c.id === selected} dim={!c.running} onClick={() => setSelected(c.id)} />}
                  {(!arrows || !allHeading(c)) && (
                    <Marker position={[c.lat!, c.lon!]} icon={markerIcon(v.level, c.running, c.id === selected)} eventHandlers={{ click: () => setSelected(c.id) }}>
                      <Tooltip direction="top"><b>{c.name}</b> · {pct(v.occ)} {v.level ?? ''}{zoom < ARROW_ZOOM && hasHeading(c) ? ' · 확대하면 방향별 화살표' : ''}</Tooltip>
                    </Marker>
                  )}
                </span>
              )
            })}
          </MapContainer>
          <div className="map-overlay" style={{ top: 10, left: 10, display: 'grid', gap: 8, maxWidth: 'calc(100% - 20px)' }}>
            <div className="row" style={{ gap: 6 }}>
              <div style={{ position: 'relative' }}>
                <Search size={14} style={{ position: 'absolute', left: 8, top: 9, color: 'var(--muted)' }} />
                <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="이름·노선·구간 검색" style={{ paddingLeft: 26, width: 200 }} />
              </div>
              <select value={route} onChange={(e) => setRoute(e.target.value)}>
                <option value="">모든 노선</option>
                {routes.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
              <label className="check small"><input type="checkbox" checked={onlyRunning} onChange={(e) => setOnlyRunning(e.target.checked)} />모니터링 중만</label>
            </div>
            <div className="muted small">{zoom >= ARROW_ZOOM ? '화살표 = 방향별 진행 방향 (우측통행, 색 = 혼잡 단계)' : `줌 ${ARROW_ZOOM} 이상 확대하면 방향별 화살표로 표시`}</div>
            <div className="legend">
              {LEVELS.map((l, i) => (
                <span key={l} className={`pill clickable ${levelFilter === l ? 'on' : ''}`} onClick={() => setLevelFilter(levelFilter === l ? null : l)}>
                  <span className="dot" style={{ background: levelColor(l) }} />
                  {l} {i < thresholds.length ? `< ${Math.round(thresholds[i] * 100)}%` : `≥ ${Math.round(thresholds[i - 1] * 100)}%`}
                </span>
              ))}
            </div>
          </div>
          {tl && tl.buckets.length > 1 && (
            <div className="map-overlay" style={{ bottom: 10, left: 10, right: 10, display: 'flex', gap: 10, alignItems: 'center' }}>
              <button className="icon" onClick={() => { if (playing) { setPlaying(false) } else { setPlaying(true); if (tlIdx == null) setTlIdx(0) } }} title="타임라인 재생">{playing ? <Pause /> : <Play />}</button>
              <input type="range" min={0} max={tl.buckets.length - 1} value={tlIdx ?? tl.buckets.length - 1} onChange={(e) => { setPlaying(false); setTlIdx(+e.target.value) }} style={{ flex: 1 }} />
              <span className="mono small" style={{ minWidth: 64 }}>{tlIdx != null ? fmtTime(tl.buckets[tlIdx]) : '실시간'}</span>
              <Segmented options={[{ v: 60, l: '1시간' }, { v: 180, l: '3시간' }, { v: 720, l: '12시간' }, { v: 1440, l: '24시간' }]} value={tlMinutes} onChange={(v) => { setTlMinutes(v); setTlIdx(null); setPlaying(false) }} />
              {tlIdx != null && <button className="sm" onClick={() => { setTlIdx(null); setPlaying(false) }}>실시간으로</button>}
            </div>
          )}
        </div>
        <aside className="map-aside">
          <div className="row between">
            <h2>{tlIdx != null ? '해당 시점 점유율' : '실시간 점유율'} <span className="muted">({filtered.length})</span></h2>
            <label className="check small"><input type="checkbox" checked={showClasses} onChange={(e) => setShowClasses(e.target.checked)} />차종 구분</label>
          </div>
          {sel && (
            <div className="card" style={{ outline: '1px solid var(--accent)' }}>
              <div className="row between" style={{ marginBottom: 6 }}>
                <div style={{ minWidth: 0 }}>
                  <h3 style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{sel.name}</h3>
                  <div className="muted">{[sel.route, sel.region, sel.section].filter(Boolean).join(' · ') || sel.source_type}</div>
                </div>
                <button className="sm primary" onClick={() => nav(`/cameras/${sel.id}`)}>상세</button>
              </div>
              <LiveImage cameraId={sel.id} mode={showClasses ? 'class' : 'vehicle'} maxFps={5} running={sel.running} />
              <div style={{ marginTop: 8 }}>
                <DirectionPanel live={sel.live} directions={sel.directions} showClasses={showClasses} compact />
              </div>
              {tl?.cameras[String(sel.id)] && (
                <div className="row between" style={{ marginTop: 8 }}>
                  <span className="muted">최근 {tlMinutes >= 60 ? `${tlMinutes / 60}시간` : `${tlMinutes}분`} 추이</span>
                  <Sparkline values={tl.cameras[String(sel.id)]} width={180} height={32} color={levelColor(sel.level)} />
                </div>
              )}
            </div>
          )}
          {sorted.map((c) => {
            const v = valueOf(c)
            return (
              <div key={c.id} className="card" style={{ padding: '10px 12px', cursor: 'pointer', outline: c.id === selected ? '1px solid var(--accent)' : 'none' }} onClick={() => setSelected(c.id)}>
                <div className="row between" style={{ gap: 6, marginBottom: 4 }}>
                  <span style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{c.name}</span>
                  {tl?.cameras[String(c.id)] && <Sparkline values={tl.cameras[String(c.id)]} width={70} height={20} color={levelColor(v.level)} />}
                  <LevelBadge level={v.level} size="sm" />
                  {!c.running && <span className="badge outline" style={{ fontSize: 11 }}>정지</span>}
                </div>
                <OccupancyBar value={v.occ} level={v.level} compact />
                {tlIdx == null && c.live?.directions?.filter((d) => d.direction_index !== 0).map((d) => (
                  <OccupancyBar key={d.direction_index} value={d.occupancy} level={d.level} label={d.name} color={c.directions.find((x) => x.index === d.direction_index)?.color} compact />
                ))}
              </div>
            )
          })}
          {!sorted.length && <div className="muted" style={{ padding: 12 }}>조건에 맞는 카메라가 없습니다.</div>}
        </aside>
      </div>
    </div>
  )
}
