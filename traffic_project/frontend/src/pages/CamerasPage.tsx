import { Camera as CameraIcon, LayoutGrid, List, Pencil, Play, PlusCircle, RefreshCw, Square, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, type Camera, type LiveCamera } from '../api/client'
import DirectionPanel from '../components/DirectionPanel'
import LevelBadge from '../components/LevelBadge'
import LiveImage from '../components/LiveImage'
import OccupancyBar from '../components/OccupancyBar'
import { Banner, EmptyState, Loading, Modal, PageHeader, Segmented, useAction } from '../components/ui'
import { DEFAULT_INTERVAL, fmtTime, INTERVAL_OPTIONS, intervalLabel, pct } from '../lib/format'
import { usePolling } from '../lib/usePolling'

type View = 'grid' | 'table'

export default function CamerasPage() {
  const nav = useNavigate()
  const { data, error, setData } = usePolling(() => api.metrics.live(), 2000)
  const [view, setView] = useState<View>('grid')
  const [showClasses, setShowClasses] = useState(false)
  const [hud, setHud] = useState(true)
  const [cols, setCols] = useState(3)
  const [q, setQ] = useState('')
  const [editing, setEditing] = useState<LiveCamera | null>(null)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const { run, busy } = useAction()
  const cams = data?.cameras ?? []
  const filtered = useMemo(() => cams.filter((c) => !q || `${c.name} ${c.route ?? ''} ${c.region ?? ''} ${c.section ?? ''}`.toLowerCase().includes(q.toLowerCase())), [cams, q])
  const refresh = async () => setData(await api.metrics.live())

  const toggle = (id: number, running: boolean) => run(running ? '정지' : '시작', async () => { await (running ? api.cameras.stop(id) : api.cameras.start(id)); await refresh() })
  const bulk = (action: 'start' | 'stop' | 'delete') =>
    run(
      { start: '일괄 시작', stop: '일괄 정지', delete: '일괄 삭제' }[action],
      async () => {
        if (action === 'delete' && !confirm(`${selected.size}대를 삭제할까요? 기록도 함께 삭제됩니다.`)) return
        for (const id of selected) await (action === 'start' ? api.cameras.start(id) : action === 'stop' ? api.cameras.stop(id) : api.cameras.remove(id))
        setSelected(new Set())
        await refresh()
      },
      '완료했습니다',
    )

  if (error && !data) return <div className="page"><Banner kind="error">{error}</Banner></div>
  if (!data) return <div className="page"><Loading lg /></div>

  return (
    <div className="page">
      <PageHeader
        title={<>전체 CCTV <span className="muted">{cams.length}대 · 모니터링 {cams.filter((c) => c.running).length}대</span></>}
        description="등록된 모든 CCTV 의 실시간 세그멘테이션을 한 화면에서 봅니다. 표 보기에서 이름·지역 편집, 일괄 시작/정지/삭제가 가능합니다."
        actions={
          <>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="검색" style={{ width: 160 }} />
            <Segmented options={[{ v: 'grid', l: <LayoutGrid size={14} />, title: '격자' }, { v: 'table', l: <List size={14} />, title: '표' }]} value={view} onChange={setView} />
            {view === 'grid' && (
              <>
                <label className="check small"><input type="checkbox" checked={showClasses} onChange={(e) => setShowClasses(e.target.checked)} />차종 구분</label>
                <label className="check small"><input type="checkbox" checked={hud} onChange={(e) => setHud(e.target.checked)} />HUD</label>
                <Segmented options={[1, 2, 3, 4].map((c) => ({ v: c, l: `${c}열` }))} value={cols} onChange={setCols} />
              </>
            )}
            <Link to="/register"><button className="primary"><PlusCircle />등록</button></Link>
          </>
        }
      />
      {!cams.length && <EmptyState icon={<CameraIcon />} title="등록된 CCTV 가 없습니다" description="ITS 실시간 CCTV 또는 영상 파일을 등록하면 여기에 실시간 세그멘테이션이 표시됩니다." action={<Link to="/register"><button className="primary"><PlusCircle />CCTV 등록</button></Link>} />}

      {view === 'grid' && (
        <div className="grid" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
          {filtered.map((c) => (
            <div key={c.id} className="card" style={{ padding: 12 }}>
              <div className="row between" style={{ marginBottom: 8, gap: 6 }}>
                <Link to={`/cameras/${c.id}`} style={{ color: 'var(--text)', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.name}</Link>
                <span className="row" style={{ gap: 6, flex: 'none' }}>
                  <LevelBadge level={c.level} size="sm" />
                  <button className="sm icon" title={c.running ? '정지' : '시작'} disabled={!c.has_mask || !!busy} onClick={() => toggle(c.id, c.running)}>{c.running ? <Square /> : <Play />}</button>
                </span>
              </div>
              <LiveImage cameraId={c.id} mode={showClasses ? 'class' : 'vehicle'} hud={hud} maxFps={cols >= 3 ? 4 : 8} running={c.running} onClick={() => nav(`/cameras/${c.id}`)} label={c.live?.status === 'reconnecting' ? '재연결 중' : undefined} />
              <div style={{ marginTop: 8 }}>
                <DirectionPanel live={c.live} directions={c.directions} showClasses={showClasses} compact />
              </div>
              <div className="row between" style={{ marginTop: 8 }}>
                <span className="muted" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{[c.route, c.region, c.section].filter(Boolean).join(' · ') || { its: 'ITS 실시간', upload: '업로드 영상', url: '스트림 URL' }[c.source_type]}</span>
                <span className="muted num">{intervalLabel(c.infer_interval_s)}{c.live?.ts ? ` · ${fmtTime(c.live.ts)}` : ''}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {view === 'table' && cams.length > 0 && (
        <>
          <div className="row" style={{ marginBottom: 8 }}>
            <span className="muted">{selected.size}대 선택</span>
            <button className="sm" disabled={!selected.size || !!busy} onClick={() => bulk('start')}><Play />시작</button>
            <button className="sm" disabled={!selected.size || !!busy} onClick={() => bulk('stop')}><Square />정지</button>
            <button className="sm danger" disabled={!selected.size || !!busy} onClick={() => bulk('delete')}><Trash2 />삭제</button>
            <button className="sm ghost" onClick={refresh}><RefreshCw />새로고침</button>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th><input type="checkbox" checked={selected.size === filtered.length && filtered.length > 0} onChange={(e) => setSelected(e.target.checked ? new Set(filtered.map((c) => c.id)) : new Set())} /></th>
                  <th>이름</th><th>소스</th><th>노선 / 지역 / 구간</th><th>방향</th><th>상태</th><th>주기</th><th className="num">점유율</th><th className="num">차량</th><th></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((c) => (
                  <tr key={c.id}>
                    <td><input type="checkbox" checked={selected.has(c.id)} onChange={(e) => { const s = new Set(selected); e.target.checked ? s.add(c.id) : s.delete(c.id); setSelected(s) }} /></td>
                    <td><Link to={`/cameras/${c.id}`}>{c.name}</Link>{!c.has_mask && <span className="badge outline" style={{ marginLeft: 6 }}>마스크 없음</span>}</td>
                    <td className="muted">{{ its: 'ITS', upload: '업로드', url: 'URL' }[c.source_type]}</td>
                    <td className="muted">{[c.route, c.region, c.section].filter(Boolean).join(' / ') || '—'}</td>
                    <td>{c.directions.map((d) => <span key={d.index} className="pill" style={{ marginRight: 4 }}><span className="dot" style={{ background: d.color, width: 8, height: 8 }} />{d.name}</span>)}</td>
                    <td>{c.running ? <span className="row" style={{ gap: 6 }}><span className="status-dot ok" />{c.live?.status ?? 'running'}</span> : <span className="row" style={{ gap: 6 }}><span className="status-dot" />정지</span>}</td>
                    <td className="muted">{intervalLabel(c.infer_interval_s)}</td>
                    <td className="num" style={{ minWidth: 160 }}><OccupancyBar value={c.occupancy} level={c.level} compact /></td>
                    <td className="num">{c.n_vehicles ?? '—'}</td>
                    <td>
                      <span className="row" style={{ gap: 4, flexWrap: 'nowrap' }}>
                        <button className="sm icon" title="편집" onClick={() => setEditing(c)}><Pencil /></button>
                        <button className="sm icon" title={c.running ? '정지' : '시작'} disabled={!c.has_mask} onClick={() => toggle(c.id, c.running)}>{c.running ? <Square /> : <Play />}</button>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <EditModal cam={editing} onClose={() => setEditing(null)} onSaved={async () => { setEditing(null); await refresh() }} />
    </div>
  )
}

function EditModal({ cam, onClose, onSaved }: { cam: LiveCamera | null; onClose: () => void; onSaved: () => Promise<void> }) {
  const [f, setF] = useState<Partial<Camera>>({})
  const { run, busy } = useAction()
  const key = cam?.id
  useMemo(() => { if (cam) setF({ name: cam.name, route: cam.route ?? '', region: cam.region ?? '', section: cam.section ?? '', lon: cam.lon, lat: cam.lat, infer_interval_s: cam.infer_interval_s ?? DEFAULT_INTERVAL }) }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!cam) return null
  return (
    <Modal open title={`편집 — ${cam.name}`} onClose={onClose} width={560}>
      <div className="form">
        <label>이름</label><input value={f.name ?? ''} onChange={(e) => setF({ ...f, name: e.target.value })} />
        <label>노선</label><input value={f.route ?? ''} onChange={(e) => setF({ ...f, route: e.target.value })} placeholder="예: 경부선" />
        <label>지역</label><input value={f.region ?? ''} onChange={(e) => setF({ ...f, region: e.target.value })} placeholder="예: 경기 수원" />
        <label>구간</label><input value={f.section ?? ''} onChange={(e) => setF({ ...f, section: e.target.value })} placeholder="예: 신갈JC~수원IC" />
        <label>경도 / 위도</label>
        <div className="row"><input value={f.lon ?? ''} onChange={(e) => setF({ ...f, lon: e.target.value === '' ? null : +e.target.value })} style={{ width: 130 }} /><input value={f.lat ?? ''} onChange={(e) => setF({ ...f, lat: e.target.value === '' ? null : +e.target.value })} style={{ width: 130 }} /></div>
        <label>추론 주기</label>
        <select value={f.infer_interval_s ?? DEFAULT_INTERVAL} onChange={(e) => setF({ ...f, infer_interval_s: +e.target.value })}>{INTERVAL_OPTIONS.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}</select>
      </div>
      <div className="row" style={{ marginTop: 14, justifyContent: 'flex-end' }}>
        <button onClick={onClose}>취소</button>
        <button className="primary" disabled={!!busy} onClick={() => run('저장', async () => { await api.cameras.update(cam.id, { ...f, route: f.route || null, region: f.region || null, section: f.section || null, infer_interval_s: f.infer_interval_s ?? DEFAULT_INTERVAL }); await onSaved() }, '저장했습니다')}>저장</button>
      </div>
      <div className="muted" style={{ marginTop: 10 }}>현재 점유율 {pct(cam.occupancy)} · 도로 마스크 편집은 상세 페이지에서</div>
    </Modal>
  )
}
