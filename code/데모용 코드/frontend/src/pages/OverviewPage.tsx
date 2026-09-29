import L from 'leaflet'
import { Compass, Globe2, Layers } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { MapContainer, Marker, TileLayer, Tooltip, useMap } from 'react-leaflet'
import { Link, useNavigate } from 'react-router-dom'
import { API_BASE, api, type Camera, type Direction, type LiveCamera } from '../api/client'
import DirectionArrows, { hasHeading, ZoomWatcher, type DirValue } from '../components/DirectionArrows'
import DirectionMapEditor from '../components/DirectionMapEditor'
import LevelBadge from '../components/LevelBadge'
import { Banner, EmptyState, Loading, Modal, Segmented, useAction, useToast } from '../components/ui'
import { LEVELS, levelColor, levelOf, pct } from '../lib/format'
import { angleDiff, axisOf, compass, miniArrowSvg } from '../lib/geo'
import { usePolling } from '../lib/usePolling'

/**
 * 전국 현황: 지점마다 점유율 숫자를 혼잡 단계 색으로 찍어 전국 교통 그림을 한눈에 본다.
 * 방향: 배지 아래에 방향별 "회전 화살표 + 숫자" (화살표가 가리키는 쪽으로 진행), 확대하면 도로 위 화살표로 펼친다.
 * 칠하지 않은(측정하지 않는) 방향은 그리지 않는다. 노선/지역 묶음은 도로 축 기준 두 방향으로 평균낸다.
 */
type Period = 0 | 15 | 60 | 1440
type Group = 'camera' | 'route' | 'region'
type Metric = 'occupancy' | 'vehicles'
type Rank = 'direction' | 'point'

interface DirVal { index: number; value: number | null; level: string | null; vehicles?: number | null; heading: number | null }
interface Point { key: string; name: string; lat: number; lon: number; value: number | null; vehicles: number | null; level: string | null; n: number; ids: number[]; dirs: DirVal[] }

const ARROW_ZOOM = 11

function badgeIcon(p: Point, metric: Metric, big: boolean, split: boolean) {
  const color = levelColor(p.level)
  const fmt = (v: number | null | undefined, veh: number | null | undefined) => (metric === 'occupancy' ? (v == null ? '–' : `${Math.round(v * 100)}`) : veh == null ? '–' : `${Math.round(veh)}`)
  const unit = metric === 'occupancy' ? '%' : '대'
  const chips = split && p.dirs.length
    ? `<div style="display:flex;gap:3px;margin-top:3px;justify-content:center">${p.dirs.slice(0, 4).map((d) => {
        const c = levelColor(d.level)
        const arrow = d.heading != null ? miniArrowSvg(d.heading, c, 13) : `<span style="font-size:9px;opacity:.8">${d.index}</span>`
        return `<span style="display:inline-flex;align-items:center;gap:2px;background:rgba(11,14,20,.88);border:1px solid ${c};border-radius:6px;padding:0 4px 0 2px;color:#fff;font:700 10px/15px 'Noto Sans KR',sans-serif;font-variant-numeric:tabular-nums">${arrow}${fmt(d.value, d.vehicles)}</span>`
      }).join('')}</div>`
    : ''
  const html = `<div style="display:flex;flex-direction:column;align-items:center;transform:translate(-50%,-50%)">
    <div style="background:${color};color:${p.level === '정체' ? '#fff' : '#0b0b0b'};border:2px solid #fff;border-radius:999px;padding:${big ? '4px 10px' : '2px 7px'};font-weight:800;font-size:${big ? 14 : 12}px;line-height:1.2;box-shadow:0 2px 8px rgba(0,0,0,.6);white-space:nowrap;font-variant-numeric:tabular-nums">${fmt(p.value, p.vehicles)}<span style="font-size:9px;font-weight:600">${unit}</span>${p.n > 1 ? `<span style="font-size:9px;opacity:.8"> ×${p.n}</span>` : ''}</div>${chips}</div>`
  return L.divIcon({ className: '', html, iconSize: [0, 0], iconAnchor: [0, 0] })
}

function FitOnce({ points }: { points: [number, number][] }) {
  const map = useMap()
  const [done, setDone] = useState(false)
  useEffect(() => {
    if (done || !points.length) return
    setDone(true)
    map.fitBounds(L.latLngBounds(points).pad(0.3), { maxZoom: 10 })
  }, [points, map, done])
  return null
}

const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)

export default function OverviewPage() {
  const nav = useNavigate()
  const [period, setPeriod] = useState<Period>(0)
  const [group, setGroup] = useState<Group>('camera')
  const [metric, setMetric] = useState<Metric>('occupancy')
  const [split, setSplit] = useState(true)
  const [rank, setRank] = useState<Rank>('direction')
  const [levelFilter, setLevelFilter] = useState<string | null>(null)
  const [zoom, setZoom] = useState(7)
  const [setupOpen, setSetupOpen] = useState(false)
  const { data: live, error, setData: setLive } = usePolling(() => api.metrics.live(), period === 0 ? 5000 : 30000)
  const { data: summ } = usePolling(() => (period ? api.metrics.summary('camera', period) : Promise.resolve(null)), 30000, [period])
  const thresholds = live?.thresholds ?? [0.08, 0.15, 0.25]

  /** (카메라, 방향) → 값 */
  const dirValue = useMemo(() => {
    const m = new Map<string, { value: number | null; level: string | null; vehicles: number | null }>()
    ;(live?.cameras ?? []).forEach((c) => {
      if (period === 0) (c.live?.directions ?? []).forEach((d) => m.set(`${c.id}:${d.direction_index}`, { value: d.occupancy, level: d.level, vehicles: d.n_vehicles }))
      else {
        const pc = summ?.groups?.flatMap((g: any) => g.cameras).find((x: any) => x.id === c.id)
        if (pc?.overall) m.set(`${c.id}:0`, { value: pc.overall.occupancy, level: pc.overall.level, vehicles: pc.overall.n_vehicles })
        pc?.directions?.forEach((d: any) => m.set(`${c.id}:${d.direction_index}`, { value: d.occupancy, level: d.level, vehicles: d.n_vehicles }))
      }
    })
    return m
  }, [live, summ, period])

  /** 측정하는(칠한) 방향만 */
  const camDirs = (c: LiveCamera): DirVal[] =>
    c.directions.filter((d) => d.measured !== false).map((d) => {
      const v = dirValue.get(`${c.id}:${d.index}`)
      return { index: d.index, value: v?.value ?? null, level: v?.level ?? null, vehicles: v?.vehicles ?? null, heading: d.heading_deg ?? null }
    })

  const points: Point[] = useMemo(() => {
    const cams = (live?.cameras ?? []).filter((c) => c.lat != null && c.lon != null)
    if (group === 'camera')
      return cams.map((c) => {
        const o = dirValue.get(`${c.id}:0`)
        const dirs = camDirs(c)
        return { key: String(c.id), name: c.name, lat: c.lat!, lon: c.lon!, value: o?.value ?? null, vehicles: o?.vehicles ?? null, level: levelOf(o?.value, thresholds), n: 1, ids: [c.id], dirs: dirs.length > 1 || dirs.some((d) => d.heading != null) ? dirs : [] }
      })
    // 노선/지역 묶음: 방향을 도로 축 기준 정·역 두 방향으로 모은다
    const m = new Map<string, LiveCamera[]>()
    cams.forEach((c) => { const k = (group === 'route' ? c.route : c.region) || '(미지정)'; m.set(k, [...(m.get(k) ?? []), c]) })
    return [...m.entries()].map(([k, cs]) => {
      const v = avg(cs.map((c) => dirValue.get(`${c.id}:0`)?.value).filter((x): x is number => x != null))
      const vehs = cs.map((c) => dirValue.get(`${c.id}:0`)?.vehicles).filter((x): x is number => x != null)
      const all = cs.flatMap((c) => camDirs(c)).filter((d) => d.heading != null && d.value != null)
      const axis = axisOf(all.map((d) => d.heading!))
      const dirs: DirVal[] = []
      if (axis != null) {
        const fwd = all.filter((d) => angleDiff(d.heading!, axis) <= 90)
        const back = all.filter((d) => angleDiff(d.heading!, axis) > 90)
        const mk = (i: number, h: number, xs: DirVal[]): DirVal => { const a = avg(xs.map((x) => x.value!)); return { index: i, value: a, level: levelOf(a, thresholds), heading: h, vehicles: avg(xs.map((x) => x.vehicles ?? 0)) } }
        if (fwd.length) dirs.push(mk(1, axis, fwd))
        if (back.length) dirs.push(mk(2, (axis + 180) % 360, back))
      }
      return { key: k, name: k, lat: avg(cs.map((c) => c.lat!))!, lon: avg(cs.map((c) => c.lon!))!, value: v, vehicles: vehs.length ? vehs.reduce((a, b) => a + b, 0) : null, level: levelOf(v, thresholds), n: cs.length, ids: cs.map((c) => c.id), dirs }
    })
  }, [live, dirValue, group, thresholds]) // eslint-disable-line react-hooks/exhaustive-deps

  const shown = points.filter((p) => !levelFilter || p.level === levelFilter || p.dirs.some((d) => d.level === levelFilter))
  const fitPts = useMemo(() => points.map((p) => [p.lat, p.lon] as [number, number]), [points.length]) // eslint-disable-line react-hooks/exhaustive-deps
  const counts = LEVELS.map((l) => points.filter((p) => p.level === l).length)
  // 측정하는 방향인데 진행 방향이 없는 카메라만 안내 (칠하지 않은 방향은 문제 삼지 않는다)
  const needSetup = (live?.cameras ?? []).filter((c) => c.directions.some((d) => d.measured !== false && d.heading_deg == null))

  const ranked = useMemo(() => {
    if (rank === 'point' || group !== 'camera')
      return [...points].filter((p) => p.value != null).sort((a, b) => b.value! - a.value!).map((p) => ({ key: p.key, title: p.name, sub: '', value: p.value, level: p.level, heading: null as number | null, camId: p.ids.length === 1 ? p.ids[0] : null }))
    return (live?.cameras ?? []).flatMap((c) => camDirs(c).filter((d) => d.value != null).map((d) => ({ key: `${c.id}-${d.index}`, title: c.name, sub: d.heading != null ? `${compass(d.heading)} 방향` : `방향 ${d.index}`, value: d.value, level: d.level, heading: d.heading, camId: c.id }))).sort((a, b) => b.value! - a.value!)
  }, [points, rank, group, live, dirValue]) // eslint-disable-line react-hooks/exhaustive-deps

  if (error && !live) return <div className="page"><Banner kind="error">{error}</Banner></div>
  if (!live) return <div className="page"><Loading lg /></div>
  if (!live.cameras.length) return <div className="page"><EmptyState icon={<Globe2 />} title="등록된 CCTV 가 없습니다" description="CCTV 를 등록하고 좌표를 넣으면 전국 지도에 지점별 점유율이 숫자로 표시됩니다." action={<Link to="/register"><button className="primary">CCTV 등록</button></Link>} /></div>

  return (
    <div className="page full" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="row between" style={{ padding: '12px 16px 0' }}>
        <div>
          <h1 className="row" style={{ gap: 8 }}><Globe2 size={20} />전국 현황</h1>
          <div className="muted">지점별 점유율을 숫자와 색으로. 배지 아래 작은 화살표 = 방향별 진행 방향과 점유율. 확대하면 도로 위 화살표로 펼쳐집니다.</div>
        </div>
        <div className="row">
          <Segmented options={[{ v: 0, l: '실시간' }, { v: 15, l: '15분 평균' }, { v: 60, l: '1시간 평균' }, { v: 1440, l: '24시간 평균' }]} value={period} onChange={(v) => setPeriod(v as Period)} />
          <Segmented options={[{ v: 'camera', l: '지점별' }, { v: 'route', l: '노선 묶음' }, { v: 'region', l: '지역 묶음' }]} value={group} onChange={(v) => setGroup(v as Group)} />
          <Segmented options={[{ v: 'occupancy', l: '점유율 %' }, { v: 'vehicles', l: '차량 수' }]} value={metric} onChange={(v) => setMetric(v as Metric)} />
          <label className="check small"><input type="checkbox" checked={split} onChange={(e) => setSplit(e.target.checked)} />방향별</label>
        </div>
      </div>
      {needSetup.length > 0 && (
        <div style={{ padding: '10px 16px 0' }}>
          <Banner kind="warn">
            <span>카메라 {needSetup.length}대의 진행 방향이 정해지지 않았습니다: {needSetup.map((c) => c.name).join(', ')}</span>
            <button className="sm primary" style={{ marginLeft: 8 }} onClick={() => setSetupOpen(true)}><Compass />방향 지정하기</button>
          </Banner>
        </div>
      )}
      <div style={{ display: 'flex', flex: 1, minHeight: 0, padding: 14, gap: 12 }}>
        <div className="card pad-0" style={{ flex: 1, minWidth: 0, position: 'relative', minHeight: 480 }}>
          <MapContainer center={[36.3, 127.8]} zoom={7} style={{ height: '100%' }}>
            <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <FitOnce points={fitPts} />
            <ZoomWatcher onZoom={setZoom} />
            {shown.map((p) => {
              const cam = group === 'camera' ? live.cameras.find((c) => c.id === p.ids[0]) : undefined
              if (cam && zoom >= ARROW_ZOOM && hasHeading(cam)) {
                const dv: Record<number, DirValue> = {}
                p.dirs.forEach((d) => (dv[d.index] = { value: d.value, level: d.level, vehicles: d.vehicles }))
                return (
                  <span key={p.key}>
                    <DirectionArrows cam={cam} values={dv} metric={metric} onClick={() => nav(`/cameras/${cam.id}`)} />
                    <Marker position={[p.lat, p.lon]} icon={badgeIcon({ ...p, dirs: [] }, metric, false, false)} eventHandlers={{ click: () => nav(`/cameras/${cam.id}`) }} />
                  </span>
                )
              }
              return (
                <Marker key={p.key} position={[p.lat, p.lon]} icon={badgeIcon(p, metric, group !== 'camera', split)} zIndexOffset={Math.round((p.value ?? 0) * 1000)} eventHandlers={{ click: () => (p.ids.length === 1 ? nav(`/cameras/${p.ids[0]}`) : undefined) }}>
                  <Tooltip direction="top" offset={[0, -14]}>
                    <b>{p.name}</b>{p.n > 1 ? ` (${p.n}대 평균)` : ''}<br />점유율 {pct(p.value)} · {p.level ?? '–'} · 차량 {p.vehicles == null ? '–' : Math.round(p.vehicles)}대
                    {p.dirs.map((d) => <div key={d.index}>{d.heading != null ? `${compass(d.heading)} 방향` : `방향 ${d.index}`}: {pct(d.value)} {d.level ?? ''}</div>)}
                  </Tooltip>
                </Marker>
              )
            })}
          </MapContainer>
          <div className="map-overlay" style={{ top: 10, left: 10 }}>
            <div className="legend" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 4 }}>
              {LEVELS.map((l, i) => (
                <span key={l} className={`pill clickable ${levelFilter === l ? 'on' : ''}`} onClick={() => setLevelFilter(levelFilter === l ? null : l)}>
                  <span className="dot" style={{ background: levelColor(l) }} />
                  {l} {i < thresholds.length ? `< ${Math.round(thresholds[i] * 100)}%` : `≥ ${Math.round(thresholds[i - 1] * 100)}%`} <span className="muted">{counts[i]}</span>
                </span>
              ))}
              <span className="muted">큰 숫자 = {group === 'camera' ? '지점' : '묶음 평균'} {metric === 'occupancy' ? '점유율(%)' : '차량 수'}</span>
              {split && <span className="muted">작은 화살표 = 방향별 (가리키는 쪽으로 진행)</span>}
              <span className="muted">{zoom >= ARROW_ZOOM ? '확대: 도로 위 방향 화살표 (우측통행)' : `줌 ${ARROW_ZOOM} 이상이면 도로 위 화살표`}</span>
            </div>
          </div>
        </div>
        <aside style={{ width: 320, flex: 'none', overflow: 'auto', display: 'grid', gap: 6, alignContent: 'start' }}>
          <div className="row between">
            <h2 className="row" style={{ gap: 6 }}><Layers size={16} />순위 <span className="muted">{ranked.length}</span></h2>
            {group === 'camera' && <Segmented options={[{ v: 'direction', l: '방향별' }, { v: 'point', l: '지점별' }]} value={rank} onChange={(v) => setRank(v as Rank)} />}
          </div>
          {ranked.map((r, i) => (
            <div key={r.key} className="card" style={{ padding: '8px 10px', cursor: r.camId ? 'pointer' : 'default' }} onClick={() => r.camId && nav(`/cameras/${r.camId}`)}>
              <div className="row between" style={{ gap: 6, flexWrap: 'nowrap' }}>
                <span className="row" style={{ gap: 6, minWidth: 0, flexWrap: 'nowrap' }}>
                  <span className="muted num" style={{ width: 20 }}>{i + 1}</span>
                  {r.heading != null && <span dangerouslySetInnerHTML={{ __html: miniArrowSvg(r.heading, levelColor(r.level), 14) }} />}
                  <span style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.title}</span>
                </span>
                <span className="row" style={{ gap: 6, flex: 'none' }}><b className="num" style={{ color: levelColor(r.level) }}>{pct(r.value, 0)}</b><LevelBadge level={r.level} size="sm" /></span>
              </div>
              {r.sub && <div className="muted" style={{ marginTop: 2, marginLeft: 26 }}>{r.sub}</div>}
            </div>
          ))}
          {!ranked.length && <div className="muted">{period ? '해당 기간에 기록이 없습니다.' : '측정값이 아직 없습니다.'}</div>}
        </aside>
      </div>
      <SetupModal open={setupOpen} onClose={() => setSetupOpen(false)} cams={needSetup} allCams={live.cameras} onSaved={async () => setLive(await api.metrics.live())} />
    </div>
  )
}

function SetupModal({ open, onClose, cams, allCams, onSaved }: { open: boolean; onClose: () => void; cams: LiveCamera[]; allCams: LiveCamera[]; onSaved: () => Promise<void> }) {
  const [sel, setSel] = useState<number | null>(null)
  const [cam, setCam] = useState<Camera | null>(null)
  const [dirs, setDirs] = useState<Direction[]>([])
  const { run, busy } = useAction()
  const toast = useToast()
  useEffect(() => { if (open && sel == null && cams.length) setSel(cams[0].id) }, [open, cams, sel])
  useEffect(() => {
    if (sel == null) return
    setCam(null)
    api.cameras.get(sel).then((c) => { setCam(c); setDirs(c.directions.map((d) => ({ ...d }))) }).catch(() => {})
  }, [sel])
  const done = (c: LiveCamera) => !c.directions.some((d) => d.measured !== false && d.heading_deg == null)
  const list = [...cams, ...allCams.filter((c) => !cams.some((x) => x.id === c.id))]
  return (
    <Modal open={open} title="방향 지정" onClose={onClose} width={1300}>
      <div style={{ display: 'grid', gridTemplateColumns: '220px minmax(0, 1fr)', gap: 12 }}>
        <div className="stack" style={{ gap: 4, alignContent: 'start', maxHeight: '72vh', overflow: 'auto' }}>
          {list.map((c) => (
            <div key={c.id} className={`pill clickable ${sel === c.id ? 'on' : ''}`} style={{ justifyContent: 'space-between', borderRadius: 8, padding: '6px 8px', whiteSpace: 'normal' }} onClick={() => setSel(c.id)}>
              <span>{c.name}</span>
              {done(c) ? <span className="ok">✓</span> : <span className="muted">방향 미정</span>}
            </div>
          ))}
        </div>
        <div>
          {!cam ? <Loading /> : (
            <div className="stack">
              <div className="row between">
                <div><b>{cam.name}</b> <span className="muted">{cam.route ?? ''}</span></div>
                <div className="row">
                  <Link to={`/cameras/${cam.id}`} className="small">상세 페이지</Link>
                  <button className="primary" disabled={!!busy} onClick={() => run('방향 저장', async () => {
                    const saved = await api.cameras.putDirections(cam.id, dirs)
                    await onSaved()
                    void saved
                    toast('ok', `"${cam.name}" 저장`)
                    // 목록 순서대로, 아직 방향이 없거나 방면 이름이 없는 다음 카메라로
                    const i = list.findIndex((c) => c.id === cam.id)
                    const next = [...list.slice(i + 1), ...list.slice(0, i)].find((c) => !done(c))
                    if (next) setSel(next.id)
                    else { toast('ok', '모든 카메라의 방향을 지정했습니다'); onClose() }
                  })}>저장 후 다음</button>
                </div>
              </div>
              <DirectionMapEditor key={cam.id} directions={dirs} onChange={setDirs} cameraLat={cam.lat} cameraLon={cam.lon} previewUrl={`${API_BASE}/api/cameras/${cam.id}/preview.jpg`} route={cam.route} roadType={cam.its_road_type ?? 'ex'} roadPx={cam.meta?.road_px as Record<string, number> | undefined} height={380} />
            </div>
          )}
        </div>
      </div>
    </Modal>
  )
}

void compass
