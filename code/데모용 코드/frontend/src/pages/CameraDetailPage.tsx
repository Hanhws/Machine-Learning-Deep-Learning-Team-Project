import { AlertTriangle, ArrowDownRight, Compass, ArrowUpRight, Camera as CameraIcon, Download, FileVideo, Grid3X3, Images, Minus, Pencil, Play, RefreshCw, Square, Trash2, TrendingUp } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { API_BASE, api, imageSrc, type AlertEpisode, type Camera, type Capture, type Direction, type HistoryPoint, type Job } from '../api/client'
import DirectionPanel from '../components/DirectionPanel'
import HlsPlayer from '../components/HlsPlayer'
import LevelBadge from '../components/LevelBadge'
import LiveImage from '../components/LiveImage'
import DirectionMapEditor from '../components/DirectionMapEditor'
import MaskEditor, { type MaskEditorHandle } from '../components/MaskEditor'
import TimeSeriesChart from '../components/TimeSeriesChart'
import { Banner, Card, HeatGrid, Loading, Modal, PageHeader, Segmented, StatCard, useAction } from '../components/ui'
import { DEFAULT_INTERVAL, DIRECTION_PALETTE, fmtTime, INTERVAL_OPTIONS, intervalLabel, LEVEL_CLASS, pct } from '../lib/format'
import { usePolling } from '../lib/usePolling'

const MODES = [{ v: 'class', l: '차종별' }, { v: 'vehicle', l: '차량(단일)' }, { v: 'road', l: '도로만' }, { v: 'none', l: '원본' }]

export default function CameraDetailPage() {
  const { id } = useParams()
  const cid = Number(id)
  const nav = useNavigate()
  const { data: cam, error, setData } = usePolling(() => api.cameras.get(cid), 2000, [cid])
  const [mode, setMode] = useState('class')
  const [hud, setHud] = useState(true)
  const [showOriginal, setShowOriginal] = useState(false)
  const [editing, setEditing] = useState(false)
  const [editingDirs, setEditingDirs] = useState(false)
  const [thresholds, setThresholds] = useState<number[]>([0.08, 0.15, 0.25])
  const [recent, setRecent] = useState<Record<number, number> | null>(null) // 최근 15분 방향별 평균
  const { run, busy } = useAction()

  useEffect(() => { api.system().then((s) => setThresholds(s.thresholds)).catch(() => {}) }, [])
  useEffect(() => {
    let alive = true
    const load = () => api.metrics.summary('camera', 15).then((s) => {
      if (!alive) return
      const pc = s.groups.flatMap((g: any) => g.cameras).find((c: any) => c.id === cid)
      if (!pc) return setRecent(null)
      const m: Record<number, number> = {}
      if (pc.overall) m[0] = pc.overall.occupancy
      pc.directions.forEach((d: any) => (m[d.direction_index] = d.occupancy))
      setRecent(m)
    }).catch(() => {})
    load()
    const t = setInterval(load, 15000)
    return () => { alive = false; clearInterval(t) }
  }, [cid])

  const refresh = async () => setData(await api.cameras.get(cid))
  if (error && !cam) return <div className="page"><Banner kind="error">{error}</Banner></div>
  if (!cam) return <div className="page"><Loading lg /></div>
  const overall = cam.live?.directions?.find((d) => d.direction_index === 0)
  const srcLabel = cam.source_type === 'its' ? `ITS 실시간 · ${cam.its_cctv_name}` : cam.source_type === 'upload' ? `업로드 · ${cam.video_path?.split('/').pop()}` : `URL · ${cam.stream_url}`
  const trend = (idx: number, cur: number | undefined) => {
    const base = recent?.[idx]
    if (cur == null || base == null) return null
    const diff = cur - base
    if (Math.abs(diff) < 0.01) return <span className="muted row" style={{ gap: 2 }}><Minus size={12} />15분 평균 대비 보합</span>
    return <span className="row" style={{ gap: 2, color: diff > 0 ? 'var(--jam)' : 'var(--free)' }}>{diff > 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}{diff > 0 ? '+' : ''}{(diff * 100).toFixed(1)}%p vs 15분 평균</span>
  }

  return (
    <div className="page">
      <PageHeader
        title={<span className="row" style={{ gap: 8 }}>{cam.name} <LevelBadge level={overall?.level} />{cam.running ? <span className="live-badge"><span className="rec" />LIVE</span> : <span className="badge outline">정지</span>}</span>}
        description={<span>{[cam.route, cam.region, cam.section].filter(Boolean).join(' · ')}{cam.route || cam.region ? ' · ' : ''}{srcLabel}</span>}
        actions={
          <>
            <button onClick={() => run(cam.running ? '정지' : '시작', async () => { await (cam.running ? api.cameras.stop(cid) : api.cameras.start(cid)); await refresh() })} disabled={!cam.mask_path || !!busy}>{cam.running ? <><Square />모니터링 정지</> : <><Play />모니터링 시작</>}</button>
            <button onClick={() => run('스냅샷 갱신', async () => { await api.cameras.refreshSnapshot(cid); await refresh() }, '스냅샷을 갱신했습니다')} disabled={!!busy}><RefreshCw />스냅샷 갱신</button>
            <button className={editing ? 'active' : ''} onClick={() => setEditing(!editing)}><Pencil />도로 마스크 편집</button>
            <button className={editingDirs ? 'active' : ''} onClick={() => setEditingDirs(!editingDirs)} disabled={!cam.directions.length}><Compass />방향 · 지도 표시</button>
            <button className="danger" onClick={() => { if (confirm('이 CCTV 와 모든 기록을 삭제할까요?')) run('삭제', () => api.cameras.remove(cid)).then(() => nav('/cameras')) }}><Trash2 />삭제</button>
          </>
        }
      />
      {!cam.mask_path && <div style={{ marginBottom: 10 }}><Banner kind="warn">도로 마스크가 없어 모니터링할 수 없습니다. "도로 마스크 편집" 으로 지정하세요.</Banner></div>}
      {cam.live?.error && <div style={{ marginBottom: 10 }}><Banner kind="error">스트림 오류: {cam.live.error} — ITS URL 은 24시간 후 만료되며 자동 갱신을 시도합니다.</Banner></div>}

      <div className="grid cols-4" style={{ marginBottom: 12 }}>
        <StatCard label="전체 점유율" value={pct(overall?.occupancy)} sub={trend(0, overall?.occupancy)} icon={<TrendingUp />} tone={LEVEL_CLASS[overall?.level ?? ''] ?? ''} />
        {cam.live?.directions?.filter((d) => d.direction_index !== 0).slice(0, 2).map((d) => (
          <StatCard key={d.direction_index} label={d.name} value={pct(d.occupancy)} sub={trend(d.direction_index, d.occupancy)} icon={<span className="dot" style={{ width: 14, height: 14, background: cam.directions.find((x) => x.index === d.direction_index)?.color }} />} tone={LEVEL_CLASS[d.level ?? ''] ?? ''} />
        ))}
        <StatCard label="검출 차량 (도로 영역 내)" value={overall?.n_vehicles ?? '—'} unit="대" sub={cam.live ? `${intervalLabel(cam.infer_interval_s)} · ${cam.live.infer_ms} ms · ${fmtTime(cam.live.ts)}` : '정지'} icon={<CameraIcon />} />
      </div>

      <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(300px, 1fr)' }}>
        <Card
          title="실시간 세그멘테이션"
          actions={
            <>
              <Segmented options={MODES} value={mode} onChange={setMode} />
              <label className="check small"><input type="checkbox" checked={hud} onChange={(e) => setHud(e.target.checked)} />HUD</label>
              {cam.source_type === 'its' && cam.stream_url && <label className="check small"><input type="checkbox" checked={showOriginal} onChange={(e) => setShowOriginal(e.target.checked)} />원본 HLS</label>}
              <button className="sm" title="현재 프레임과 지표를 저장" disabled={!cam.running || !!busy} onClick={() => run('캡처', () => api.captures.create(cid, mode), '캡처를 저장했습니다')}><Images />캡처</button>
            </>
          }
        >
          <LiveImage cameraId={cid} mode={mode} hud={hud} maxFps={10} running={cam.running} label={cam.live?.status === 'reconnecting' ? '재연결 중' : undefined} />
          {showOriginal && cam.stream_url && <div style={{ marginTop: 8 }}><HlsPlayer url={cam.stream_url} /></div>}
        </Card>
        <Card title="방향별 점유율 (실시간)">
          <DirectionPanel live={cam.live} directions={cam.directions} showClasses={mode === 'class'} />
          <div className="divider" />
          <dl className="kv">
            <dt>도로 픽셀</dt><dd>{cam.meta?.road_px ? Object.entries(cam.meta.road_px as Record<string, number>).map(([k, v]) => `${cam.directions.find((d) => d.index === +k)?.name ?? k} ${Math.round(v / 1000)}k`).join(' · ') : '–'}</dd>
            <dt>프레임</dt><dd>{cam.frame_width}×{cam.frame_height} · 도로 비율 {pct(cam.meta?.road_coverage as number)}</dd>
            {cam.source_type === 'its' && <><dt>ITS URL</dt><dd>{cam.stream_url_fetched_at ? new Date(cam.stream_url_fetched_at).toLocaleString('ko-KR') : '–'} 발급 (24h 유효, 자동 갱신)</dd></>}
            <dt>등록</dt><dd>{new Date(cam.created_at).toLocaleString('ko-KR')}</dd>
            <dt>추론 주기</dt>
            <dd>
              <select value={cam.infer_interval_s ?? DEFAULT_INTERVAL} onChange={(e) => run('추론 주기 변경', async () => { await api.cameras.update(cid, { infer_interval_s: +e.target.value }); await refresh() }, '추론 주기를 바꿨습니다 (워커 재시작)')} style={{ padding: '3px 24px 3px 8px' }}>
                {INTERVAL_OPTIONS.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}
              </select>
            </dd>
          </dl>
          {cam.source_type === 'upload' && <AnalysisPanel cam={cam} />}
        </Card>
      </div>

      {editing && (
        <Card title="도로 마스크 편집" style={{ marginTop: 12 }}>
          <MaskEditSection cam={cam} onSaved={() => { setEditing(false); refresh() }} />
        </Card>
      )}

      {(editingDirs || (cam.directions.length > 0 && cam.directions.every((d) => d.heading_deg == null) && !editing)) && (
        <Card title="방향 · 도로 · 지도 표시" icon={<Compass size={16} />} style={{ marginTop: 12 }} actions={!editingDirs ? <span className="muted">진행 방향이 아직 지정되지 않아 지도에 점으로만 표시됩니다</span> : undefined}>
          <DirectionsSection cam={cam} onSaved={() => { setEditingDirs(false); refresh() }} />
        </Card>
      )}

      <div style={{ marginTop: 12 }}><HistorySection cam={cam} thresholds={thresholds} /></div>

      <div className="grid cols-2" style={{ marginTop: 12 }}>
        <AlertsSection cid={cid} />
        <HeatmapSection cid={cid} />
      </div>
      <div style={{ marginTop: 12 }}><CapturesSection cid={cid} /></div>
      <div style={{ marginTop: 12 }}><Link to="/cameras">← 전체 CCTV</Link></div>
    </div>
  )
}

function DirectionsSection({ cam, onSaved }: { cam: Camera; onSaved: () => void }) {
  const [directions, setDirections] = useState<Direction[]>(cam.directions.map((d) => ({ ...d })))
  const { run, busy } = useAction()
  return (
    <div className="stack">
      <DirectionMapEditor directions={directions} onChange={setDirections} cameraLat={cam.lat} cameraLon={cam.lon} previewUrl={`${API_BASE}/api/cameras/${cam.id}/preview.jpg`} route={cam.route} roadType={cam.its_road_type ?? 'ex'} roadPx={cam.meta?.road_px as Record<string, number> | undefined} />
      <div className="row">
        <button className="primary" disabled={!!busy} onClick={() => run('방향 저장', async () => { await api.cameras.putDirections(cam.id, directions); onSaved() }, '방향 정보를 저장했습니다')}>방향 저장</button>
        <span className="muted">마스크 라벨(칠한 픽셀)은 그대로이고 이름·도로·지도 방향만 바뀝니다.</span>
      </div>
    </div>
  )
}

function MaskEditSection({ cam, onSaved }: { cam: Camera; onSaved: () => void }) {
  const [directions, setDirections] = useState<Direction[]>(cam.directions.map((d) => ({ ...d })))
  const ref = useRef<MaskEditorHandle | null>(null)
  const onReady = useCallback((h: MaskEditorHandle) => { ref.current = h }, [])
  const { run, busy } = useAction()
  if (!cam.frame_width || !cam.frame_height) return <div className="muted">스냅샷이 없습니다</div>
  return (
    <div>
      <MaskEditor imageUrl={`/api/cameras/${cam.id}/snapshot.jpg?_=${encodeURIComponent(cam.snapshot_path ?? '')}`} width={cam.frame_width} height={cam.frame_height} cameraId={cam.id} directions={directions} onDirectionsChange={setDirections} initialMaskUrl={cam.mask_path ? `/api/cameras/${cam.id}/mask.png?_=${Date.now()}` : undefined} onReady={onReady} />
      <div className="row" style={{ marginTop: 8 }}>
        <button className="primary" disabled={!!busy} onClick={() => run('마스크 저장', async () => { await api.cameras.putMask(cam.id, ref.current!.exportPng(), directions); onSaved() }, '마스크를 저장하고 워커를 재시작했습니다')}>마스크 저장 (워커 재시작)</button>
      </div>
    </div>
  )
}

function AnalysisPanel({ cam }: { cam: Camera }) {
  const [stride, setStride] = useState(5)
  const { data: jobs, setData } = usePolling(() => api.cameras.jobs(cam.id), 3000, [cam.id])
  const { run, busy } = useAction()
  const latest: Job | undefined = jobs?.[0]
  return (
    <div style={{ marginTop: 12, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
      <h3 className="row" style={{ gap: 6 }}><FileVideo size={15} />영상 전체 분석 (오프라인)</h3>
      <div className="row" style={{ marginTop: 6 }}>
        <select value={stride} onChange={(e) => setStride(+e.target.value)}>{[1, 2, 5, 10, 15, 30].map((s) => <option key={s} value={s}>{s}프레임마다</option>)}</select>
        <button className="primary sm" disabled={latest?.status === 'running' || !!busy} onClick={() => run('분석 시작', async () => { await api.cameras.analyze(cam.id, stride); setData(await api.cameras.jobs(cam.id)) })}>분석 실행</button>
        <a className="small" href={api.metrics.historyCsvUrl(cam.id, 0, 'file')}>결과 CSV</a>
      </div>
      {latest && (
        <div style={{ marginTop: 8, fontSize: 13 }}>
          <div className="row" style={{ gap: 6 }}>상태 <span className={`badge ${latest.status === 'done' ? 'free' : latest.status === 'error' ? 'jam' : 'outline'}`}>{latest.status}</span> {latest.status === 'running' && `${Math.round(latest.progress * 100)}%`} {latest.error && <span className="error">{latest.error}</span>}</div>
          {latest.status === 'running' && <div className="bar" style={{ marginTop: 4 }}><div style={{ width: `${latest.progress * 100}%`, background: 'var(--accent)' }} /></div>}
          {latest.summary && (
            <dl className="kv" style={{ marginTop: 6 }}>
              <dt>길이</dt><dd>{latest.summary.duration_s}s · {latest.summary.processed}프레임 처리</dd>
              {Object.entries(latest.summary.directions as Record<string, any>).map(([d, s]) => (
                <span key={d} style={{ display: 'contents' }}><dt>{d === '0' ? '전체' : cam.directions.find((x) => x.index === +d)?.name ?? d}</dt><dd>평균 {pct(s.mean)} · 최대 {pct(s.max)} · p90 {pct(s.p90)}</dd></span>
              ))}
            </dl>
          )}
        </div>
      )}
    </div>
  )
}

function HistorySection({ cam, thresholds }: { cam: Camera; thresholds: number[] }) {
  const [source, setSource] = useState<'live' | 'file'>('live')
  const [minutes, setMinutes] = useState(30)
  const [bucket, setBucket] = useState(30)
  const [rows, setRows] = useState<Record<string, any>[]>([])
  const [loading, setLoading] = useState(true)
  const [compare, setCompare] = useState<number | ''>('')
  const [others, setOthers] = useState<Camera[]>([])
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const dirs = useMemo(() => [{ index: 0, name: '전체', color: '#e9edf5' }, ...cam.directions], [cam.directions])
  useEffect(() => { api.cameras.list().then((l) => setOthers(l.filter((c) => c.id !== cam.id))).catch(() => {}) }, [cam.id])

  useEffect(() => {
    let alive = true
    const load = async () => {
      const reqs = dirs.map((d) => api.metrics.history({ camera_id: cam.id, direction: d.index, minutes, bucket, source }))
      if (compare !== '' && source === 'live') reqs.push(api.metrics.history({ camera_id: compare as number, direction: 0, minutes, bucket, source }))
      const res = await Promise.all(reqs)
      if (!alive) return
      const byT = new Map<string | number, Record<string, any>>()
      res.forEach((r, i) => {
        const key = i < dirs.length ? `d${dirs[i].index}` : 'cmp'
        r.points.forEach((p: HistoryPoint) => {
          const row = byT.get(p.t) ?? { t: p.t }
          row[key] = p.occupancy
          if (key === 'd0') row.n = p.n_vehicles
          byT.set(p.t, row)
        })
      })
      setRows([...byT.values()].sort((a, b) => (a.t > b.t ? 1 : -1)))
      setLoading(false)
    }
    setLoading(true)
    load().catch(() => setLoading(false))
    const t = setInterval(load, 10000)
    return () => { alive = false; clearInterval(t) }
  }, [cam.id, dirs, minutes, bucket, source, compare])

  const series = [
    ...dirs.filter((d) => !hidden.has(`d${d.index}`)).map((d) => ({ key: `d${d.index}`, name: d.name, color: d.color, area: d.index === 0 })),
    ...(compare !== '' ? [{ key: 'cmp', name: `비교: ${others.find((o) => o.id === compare)?.name ?? compare}`, color: '#c850c8' }] : []),
  ]
  return (
    <Card
      title="점유율 추이"
      icon={<TrendingUp size={16} />}
      actions={
        <>
          {cam.source_type === 'upload' && <Segmented options={[{ v: 'live', l: '실시간(반복 재생)' }, { v: 'file', l: '영상 전체 분석' }]} value={source} onChange={setSource} />}
          {source === 'live' && (
            <select value={minutes} onChange={(e) => setMinutes(+e.target.value)}>
              {[10, 30, 60, 180, 720, 1440, 10080].map((m) => <option key={m} value={m}>최근 {m >= 1440 ? `${m / 1440}일` : m >= 60 ? `${m / 60}시간` : `${m}분`}</option>)}
            </select>
          )}
          <select value={bucket} onChange={(e) => setBucket(+e.target.value)}>{[5, 10, 30, 60, 300, 900, 3600].map((b) => <option key={b} value={b}>{b >= 60 ? `${b / 60}분` : `${b}초`} 평균</option>)}</select>
          {source === 'live' && (
            <select value={compare} onChange={(e) => setCompare(e.target.value === '' ? '' : +e.target.value)} title="다른 카메라의 전체 점유율을 겹쳐 봅니다">
              <option value="">비교 없음</option>
              {others.map((o) => <option key={o.id} value={o.id}>비교: {o.name}</option>)}
            </select>
          )}
          <a href={api.metrics.historyCsvUrl(cam.id, minutes, source)} className="small row" style={{ gap: 4 }} title="원본 샘플(5초 단위) CSV"><Download size={14} />CSV</a>
        </>
      }
    >
      <div className="legend" style={{ marginBottom: 6 }}>
        {dirs.map((d) => <span key={d.index} className={`pill clickable ${hidden.has(`d${d.index}`) ? '' : 'on'}`} onClick={() => { const s = new Set(hidden); s.has(`d${d.index}`) ? s.delete(`d${d.index}`) : s.add(`d${d.index}`); setHidden(s) }}><span className="dot" style={{ background: d.color }} />{d.name}</span>)}
        <span className="muted">막대 = 검출 차량 수 · 배경 밴드 = 혼잡 단계</span>
      </div>
      {loading && !rows.length ? <Loading /> : <TimeSeriesChart data={rows} series={series} thresholds={thresholds} xIsVideoTime={source === 'file'} countKey="n" showLegend={false} height={280} />}
    </Card>
  )
}

function AlertsSection({ cid }: { cid: number }) {
  const [minLevel, setMinLevel] = useState(2)
  const { data } = usePolling(() => api.metrics.alerts({ minutes: 1440, camera_id: cid, min_level: minLevel, min_duration: 30 }), 15000, [cid, minLevel])
  const eps: AlertEpisode[] = data?.episodes ?? []
  return (
    <Card title="혼잡 경보 (최근 24시간)" icon={<AlertTriangle size={16} />} actions={<Segmented options={[{ v: 1, l: '서행↑' }, { v: 2, l: '지체↑' }, { v: 3, l: '정체' }]} value={minLevel} onChange={setMinLevel} />}>
      {!eps.length && <div className="muted">30초 이상 지속된 {['', '서행', '지체', '정체'][minLevel]} 이상 구간이 없습니다.</div>}
      {eps.length > 0 && (
        <div className="table-wrap" style={{ maxHeight: 260 }}>
          <table>
            <thead><tr><th>시작</th><th>지속</th><th>방향</th><th className="num">최고</th><th className="num">평균</th><th></th></tr></thead>
            <tbody>
              {eps.slice(0, 30).map((e, i) => (
                <tr key={i}>
                  <td className="num">{new Date(e.start).toLocaleString('ko-KR', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</td>
                  <td className="num">{e.duration_s >= 3600 ? `${(e.duration_s / 3600).toFixed(1)}h` : `${Math.round(e.duration_s / 60)}분`}</td>
                  <td>{e.direction_name}</td>
                  <td className="num">{pct(e.peak)}</td>
                  <td className="num">{pct(e.mean)}</td>
                  <td>{e.ongoing ? <span className="badge jam">진행 중</span> : <LevelBadge level={e.level} size="sm" />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

function HeatmapSection({ cid }: { cid: number }) {
  const [days, setDays] = useState(7)
  const { data } = usePolling(() => api.metrics.heatmap(cid, days), 60000, [cid, days])
  const has = data?.grid.some((r) => r.some((v) => v != null))
  return (
    <Card title="요일 × 시간대 프로필" icon={<Grid3X3 size={16} />} actions={<Segmented options={[{ v: 7, l: '7일' }, { v: 30, l: '30일' }]} value={days} onChange={setDays} />}>
      {!data ? <Loading /> : !has ? <div className="muted">아직 누적된 기록이 없습니다. 모니터링을 켜 두면 요일·시간대별 패턴이 쌓입니다.</div> : <HeatGrid grid={data.grid} weekdays={data.weekdays} />}
      <div className="muted" style={{ marginTop: 6 }}>KST 기준 전체 점유율 평균. 초록 → 빨강.</div>
    </Card>
  )
}

function CapturesSection({ cid }: { cid: number }) {
  const { data, setData } = usePolling(() => api.captures.list(cid), 10000, [cid])
  const [open, setOpen] = useState<Capture | null>(null)
  const { run } = useAction()
  const list = data ?? []
  return (
    <Card title={<>저장한 캡처 <span className="muted">{list.length}</span></>} icon={<Images size={16} />}>
      {!list.length && <div className="muted">실시간 화면의 "캡처" 버튼으로 현재 프레임과 지표를 저장하면 여기에 모입니다 (보고서·비교용).</div>}
      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))' }}>
        {list.map((c) => (
          <div key={c.name} className="card pad-0" style={{ cursor: 'pointer' }} onClick={() => setOpen(c)}>
            <BlobImg url={API_BASE + c.url} style={{ width: '100%', aspectRatio: '16/9', objectFit: 'cover', display: 'block' }} />
            <div style={{ padding: '6px 8px' }} className="row between">
              <span className="small num">{fmtTime(c.ts)}</span>
              <span className="row" style={{ gap: 4 }}><LevelBadge level={c.level} size="sm" /><span className="muted">{pct(c.occupancy)}</span></span>
            </div>
          </div>
        ))}
      </div>
      <Modal open={!!open} title={open ? `${new Date(open.ts).toLocaleString('ko-KR')} · ${pct(open.occupancy)} ${open.level ?? ''}` : ''} onClose={() => setOpen(null)} width={960}>
        {open && (
          <div className="stack">
            <BlobImg url={API_BASE + open.url} style={{ width: '100%', borderRadius: 8 }} />
            <div className="row between">
              <a href={API_BASE + open.url} download className="row small" style={{ gap: 4 }}><Download size={14} />이미지 저장</a>
              <button className="sm danger" onClick={() => run('캡처 삭제', async () => { await api.captures.remove(cid, open.name); setOpen(null); setData(await api.captures.list(cid)) })}><Trash2 />삭제</button>
            </div>
          </div>
        )}
      </Modal>
    </Card>
  )
}

function BlobImg({ url, style }: { url: string; style?: React.CSSProperties }) {
  const [src, setSrc] = useState('')
  useEffect(() => { let u = ''; imageSrc(url).then((s) => { u = s; setSrc(s) }).catch(() => {}); return () => { if (u.startsWith('blob:')) URL.revokeObjectURL(u) } }, [url])
  return <img src={src} alt="" style={style} />
}

// DIRECTION_PALETTE 는 방향 색 fallback 용으로 유지
void DIRECTION_PALETTE
