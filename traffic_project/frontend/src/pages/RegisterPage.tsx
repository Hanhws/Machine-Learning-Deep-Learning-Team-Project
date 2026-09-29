import L from 'leaflet'
import { Check, Compass, FileVideo, Link2, MapPin, RadioTower, Save, Search, Upload } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CircleMarker, MapContainer, TileLayer, Tooltip, useMap, useMapEvents } from 'react-leaflet'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { API_BASE, api, imageSrc, type Camera, type Direction, type ItsCctv } from '../api/client'
import DirectionMapEditor from '../components/DirectionMapEditor'
import MaskEditor, { type MaskEditorHandle } from '../components/MaskEditor'
import { Banner, Card, PageHeader, Segmented, Spinner, useToast } from '../components/ui'
import { DEFAULT_INTERVAL, DIRECTION_PALETTE, INTERVAL_OPTIONS } from '../lib/format'

type SourceTab = 'its' | 'upload' | 'url'
interface Snapshot { snapshot_id: string; width: number; height: number; url: string }

function MapBridge({ onMap, onMove }: { onMap: (m: L.Map) => void; onMove: (m: L.Map) => void }) {
  const map = useMap()
  const once = useRef(false)
  useEffect(() => {
    if (once.current) return
    once.current = true
    onMap(map)
  }, [map, onMap])
  useMapEvents({ moveend: () => onMove(map), zoomend: () => onMove(map) })
  return null
}

const STEPS = ['영상 소스', '도로 영역 · 방향', '정보 입력 · 저장']

export default function RegisterPage() {
  const nav = useNavigate()
  const toast = useToast()
  const [params] = useSearchParams()
  const [step, setStep] = useState(1)
  const [tab, setTab] = useState<SourceTab>('its')
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  // ---- ITS ----
  const [itsConfigured, setItsConfigured] = useState<boolean | null>(null)
  const [roadType, setRoadType] = useState('all')
  const [cctvType, setCctvType] = useState('1')
  const [results, setResults] = useState<ItsCctv[]>([])
  const [listQ, setListQ] = useState('')
  const [selectedIts, setSelectedIts] = useState<ItsCctv | null>(null)
  const [zoomHint, setZoomHint] = useState<string | null>(null)
  const [totalFound, setTotalFound] = useState(0)
  const [existing, setExisting] = useState<Camera[]>([])
  const mapRef = useRef<L.Map | null>(null)
  const searchTimer = useRef<number | undefined>(undefined)
  const [center, setCenterState] = useState<[number, number] | null>(null)
  // ---- URL / 업로드 ----
  const [urlInput, setUrlInput] = useState('')
  const [upload, setUpload] = useState<{ path: string; kind: string; filename: string; info: any } | null>(null)
  const [over, setOver] = useState(false)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  // ---- 마스크 ----
  const [directions, setDirections] = useState<Direction[]>([
    { index: 1, name: '방향 1', color: DIRECTION_PALETTE[0], destination: '' },
    { index: 2, name: '방향 2', color: DIRECTION_PALETTE[1], destination: '' },
  ])
  const editorRef = useRef<MaskEditorHandle | null>(null)
  const onEditorReady = useCallback((h: MaskEditorHandle) => { editorRef.current = h }, [])
  // ---- 메타 ----
  const [meta, setMeta] = useState({ name: '', route: '', region: '', section: '', lon: '', lat: '' })
  const [interval, setInterval_] = useState(DEFAULT_INTERVAL)

  const preLat = params.get('lat'), preLon = params.get('lon'), preLabel = params.get('label')
  const initialCenter: [number, number] = preLat && preLon ? [+preLat, +preLon] : [37.5, 127.0]

  useEffect(() => {
    api.its.status().then((s) => setItsConfigured(s.configured)).catch(() => setItsConfigured(false))
    api.cameras.list().then(setExisting).catch(() => {})
  }, [])
  useEffect(() => { if (preLabel) setMeta((m) => ({ ...m, section: preLabel, name: m.name || preLabel })) }, [preLabel])

  const registeredNames = useMemo(() => new Set(existing.map((c) => c.its_cctv_name).filter(Boolean)), [existing])

  const searchIts = useCallback(async (m: L.Map) => {
    if (itsConfigured === false) return
    const b = m.getBounds()
    const area = (b.getEast() - b.getWest()) * (b.getNorth() - b.getSouth())
    if (m.getZoom() < 10 || area > 1) { setZoomHint('지도를 더 확대하면 이 영역의 CCTV 를 자동으로 검색합니다 (줌 10 이상)'); setResults([]); return }
    setZoomHint(null)
    setBusy('ITS 검색 중…')
    setErr(null)
    try {
      const r = await api.its.search({ road_type: roadType, cctv_type: cctvType, min_x: b.getWest(), max_x: b.getEast(), min_y: b.getSouth(), max_y: b.getNorth() })
      const c = m.getCenter()
      const sorted = r.items.sort((x, y) => (x.lat - c.lat) ** 2 + (x.lon - c.lng) ** 2 - ((y.lat - c.lat) ** 2 + (y.lon - c.lng) ** 2))
      setTotalFound(sorted.length)
      setResults(sorted.slice(0, 400)) // 지도 성능을 위해 중심에서 가까운 400개만
    } catch (e: any) { setErr(e.message) } finally { setBusy(null) }
  }, [roadType, cctvType, itsConfigured])
  const onMove = useCallback((m: L.Map) => {
    setCenterState([m.getCenter().lat, m.getCenter().lng])
    window.clearTimeout(searchTimer.current)
    searchTimer.current = window.setTimeout(() => searchIts(m), 500)
  }, [searchIts])
  useEffect(() => { if (mapRef.current && itsConfigured && tab === 'its') searchIts(mapRef.current) }, [roadType, cctvType, itsConfigured, searchIts, tab])

  const takeSnapshot = async (source: string) => {
    setErr(null)
    setBusy('스냅샷 가져오는 중… (실시간 HLS 는 5~15초 걸릴 수 있습니다)')
    try {
      const s = await api.snapshots.fromSource(source)
      setSnapshot(s)
      setStep(2)
    } catch (e: any) { setErr(e.message) } finally { setBusy(null) }
  }
  const onFile = async (f: File | null) => {
    if (!f) return
    setErr(null)
    setBusy(`업로드 중… ${f.name} (${(f.size / 1e6).toFixed(1)} MB)`)
    try {
      const r = await api.snapshots.upload(f)
      setUpload({ path: r.path, kind: r.kind, filename: r.filename, info: r.info })
      setSnapshot({ snapshot_id: r.snapshot_id, width: r.width, height: r.height, url: r.url })
      setMeta((m) => ({ ...m, name: m.name || f.name.replace(/\.[^.]+$/, '') }))
      setStep(2)
    } catch (e: any) { setErr(e.message) } finally { setBusy(null) }
  }
  const selectIts = (it: ItsCctv) => {
    setSelectedIts(it)
    setMeta((m) => ({ ...m, name: it.name, route: it.route ?? '', lon: String(it.lon), lat: String(it.lat) }))
    mapRef.current?.panTo([it.lat, it.lon])
  }

  const save = async () => {
    if (!snapshot || !editorRef.current) return
    if (!meta.name.trim()) return setErr('이름을 입력하세요')
    const st = editorRef.current.stats()
    if (!Object.values(st).some((v) => v > 0)) return setErr('도로 영역이 비어 있습니다. 2단계로 돌아가 도로를 지정하세요.')
    setErr(null)
    setBusy('카메라 생성 중…')
    // 카메라 좌표가 비어 있으면 방향 화살표 위치의 평균으로
    const anchors = directions.filter((d) => d.lat != null && d.lon != null)
    if ((!meta.lat || !meta.lon) && anchors.length) {
      meta.lat = String(anchors.reduce((a, d) => a + d.lat!, 0) / anchors.length)
      meta.lon = String(anchors.reduce((a, d) => a + d.lon!, 0) / anchors.length)
    }
    try {
      const body: Record<string, unknown> = {
        name: meta.name, source_type: tab, snapshot_id: snapshot.snapshot_id,
        route: meta.route || null, region: meta.region || null, section: meta.section || null,
        lon: meta.lon ? +meta.lon : null, lat: meta.lat ? +meta.lat : null,
        infer_interval_s: interval,
      }
      if (tab === 'its' && selectedIts) Object.assign(body, { stream_url: selectedIts.url, its_cctv_name: selectedIts.name, its_road_type: selectedIts.road_type, its_cctv_type: cctvType })
      if (tab === 'url') Object.assign(body, { stream_url: urlInput })
      if (tab === 'upload' && upload) Object.assign(body, { video_path: upload.path, meta: { upload_kind: upload.kind, filename: upload.filename } })
      const cam = await api.cameras.create(body)
      setBusy('도로 마스크 저장 중…')
      await api.cameras.putMask(cam.id, editorRef.current.exportPng(), directions)
      if (!(tab === 'upload' && upload?.kind === 'image')) { setBusy('모니터링 시작 중…'); await api.cameras.start(cam.id) }
      toast('ok', `"${cam.name}" 등록 완료`)
      nav(`/cameras/${cam.id}`)
    } catch (e: any) { setErr(e.message) } finally { setBusy(null) }
  }

  const filteredResults = results.filter((r) => !listQ || r.name.toLowerCase().includes(listQ.toLowerCase()))

  return (
    <div className="page">
      <PageHeader title="CCTV 등록" description="영상 소스를 고르고, 도로 영역을 방향별로 지정한 뒤 저장하면 바로 모니터링이 시작됩니다." />
      <div className="stepper">
        {STEPS.map((s, i) => {
          const n = i + 1
          return (
            <button key={n} className={`step ${step === n ? 'active' : ''} ${step > n ? 'done' : ''}`} disabled={n > 1 && !snapshot} onClick={() => setStep(n)}>
              <span className="n">{step > n ? <Check size={14} /> : n}</span>
              <span className="t">{s}</span>
            </button>
          )
        })}
      </div>
      {err && <div style={{ marginBottom: 10 }}><Banner kind="error" onClose={() => setErr(null)}>{err}</Banner></div>}
      {busy && <div className="row muted" style={{ marginBottom: 10 }}><Spinner />{busy}</div>}

      {step === 1 && (
        <div className="stack">
          <Segmented
            options={[{ v: 'its', l: <><RadioTower size={14} /> ITS 실시간 CCTV</> }, { v: 'upload', l: <><Upload size={14} /> 영상/이미지 업로드</> }, { v: 'url', l: <><Link2 size={14} /> 스트림 URL</> }]}
            value={tab}
            onChange={setTab}
          />
          {tab === 'its' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 380px', gap: 12, minHeight: 560 }}>
              <div className="card pad-0" style={{ position: 'relative', minHeight: 420 }}>
                <MapContainer center={initialCenter} zoom={preLat ? 14 : 11} style={{ height: '100%', minHeight: 420 }}>
                  <MapBridge onMap={(m) => { mapRef.current = m }} onMove={onMove} />
                  <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  {results.map((it, i) => {
                    const reg = registeredNames.has(it.name)
                    return (
                      <CircleMarker key={i} center={[it.lat, it.lon]} radius={selectedIts === it ? 9 : 6} pathOptions={{ color: selectedIts === it ? '#fff' : reg ? '#666' : it.road_type === 'ex' ? '#4f8cff' : '#f2c14e', fillColor: reg ? '#666' : it.road_type === 'ex' ? '#4f8cff' : '#f2c14e', fillOpacity: 0.9, weight: selectedIts === it ? 2 : 1 }} eventHandlers={{ click: () => selectIts(it) }}>
                        <Tooltip>{it.name}{reg ? ' (등록됨)' : ''}</Tooltip>
                      </CircleMarker>
                    )
                  })}
                </MapContainer>
                <div className="map-overlay row" style={{ top: 10, left: 10 }}>
                  <select value={roadType} onChange={(e) => setRoadType(e.target.value)}>
                    <option value="all">고속도로+국도</option><option value="ex">고속도로</option><option value="its">국도</option>
                  </select>
                  <select value={cctvType} onChange={(e) => setCctvType(e.target.value)} title="https 페이지(Vercel 등)에서 원본 재생이 필요하면 https">
                    <option value="1">HLS (http)</option><option value="4">HLS (https)</option>
                  </select>
                  <button className="sm" onClick={() => mapRef.current && searchIts(mapRef.current)} disabled={!!busy || itsConfigured === false}><Search />다시 검색</button>
                </div>
                {zoomHint && <div className="map-overlay small" style={{ bottom: 10, left: 10 }}>{zoomHint}</div>}
                {!zoomHint && center && <div className="map-overlay small muted" style={{ bottom: 10, left: 10 }}>지도를 움직이면 자동 검색 · 파랑=고속도로, 노랑=국도, 회색=이미 등록</div>}
              </div>
              <Card title={<>검색 결과 <span className="muted">{filteredResults.length}건{totalFound > results.length ? ` (전체 ${totalFound}건 중 가까운 순)` : ''}</span></>} icon={<MapPin size={16} />} style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                {itsConfigured === false && (
                  <Banner kind="error">ITS_API_KEY 가 설정되지 않았습니다. <code>.env</code> 에 키를 넣고 서버를 재시작하세요. (<a href="https://www.its.go.kr/opendata/opendataList?service=cctv" target="_blank" rel="noreferrer">키 발급</a>)</Banner>
                )}
                <input value={listQ} onChange={(e) => setListQ(e.target.value)} placeholder="이름으로 필터 (예: 한남, 경부)" style={{ marginBottom: 8 }} />
                <div style={{ overflow: 'auto', maxHeight: 300, display: 'grid', gap: 4, alignContent: 'start' }}>
                  {filteredResults.slice(0, 300).map((it, i) => {
                    const reg = registeredNames.has(it.name)
                    return (
                      <div key={i} className={`pill clickable ${selectedIts === it ? 'on' : ''}`} style={{ justifyContent: 'space-between', opacity: reg ? 0.55 : 1 }} onClick={() => selectIts(it)}>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{it.name}</span>
                        <span className="muted">{reg ? '등록됨' : it.road_type === 'ex' ? '고속' : '국도'}</span>
                      </div>
                    )
                  })}
                  {!filteredResults.length && !busy && <div className="muted" style={{ padding: 8 }}>{zoomHint ?? '이 영역에 CCTV 가 없습니다. 지도를 이동해 보세요.'}</div>}
                </div>
                {selectedIts && (
                  <div style={{ marginTop: 10, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
                    <div style={{ fontWeight: 600 }}>{selectedIts.name}</div>
                    <div className="muted">{selectedIts.lat.toFixed(5)}, {selectedIts.lon.toFixed(5)} · {selectedIts.format} · {selectedIts.road_type === 'ex' ? '고속도로' : '국도'}</div>
                    {registeredNames.has(selectedIts.name) && <Banner kind="warn">이미 등록된 CCTV 입니다. 중복 등록하면 워커가 2개 돌아갑니다.</Banner>}
                    <button className="primary" style={{ marginTop: 8, width: '100%' }} onClick={() => takeSnapshot(selectedIts.url)} disabled={!!busy}>스냅샷 가져와서 다음 →</button>
                  </div>
                )}
              </Card>
            </div>
          )}

          {tab === 'upload' && (
            <Card title="영상 또는 이미지 업로드" icon={<FileVideo size={16} />}>
              <p className="muted" style={{ marginBottom: 10 }}>mp4 / mov / avi / mkv 영상 또는 jpg / png 이미지. 영상은 등록 후 실시간처럼 반복 재생되고, 상세 페이지의 "전체 분석" 으로 영상 전체의 점유율 시계열을 얻을 수 있습니다.</p>
              <label className={`dropzone ${over ? 'over' : ''}`} onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); onFile(e.dataTransfer.files?.[0] ?? null) }} style={{ display: 'block' }}>
                <Upload />
                <div>여기에 파일을 끌어다 놓거나 클릭해서 선택</div>
                <input type="file" accept="video/*,image/*" style={{ display: 'none' }} onChange={(e) => onFile(e.target.files?.[0] ?? null)} disabled={!!busy} />
              </label>
              <div className="muted" style={{ marginTop: 8 }}>테스트용 샘플: <code>data/uploads/sample_highway.mp4</code> — 스트림 URL 탭에 그 경로를 넣어도 됩니다.</div>
            </Card>
          )}

          {tab === 'url' && (
            <Card title="스트림 URL / 파일 경로" icon={<Link2 size={16} />}>
              <p className="muted" style={{ marginBottom: 10 }}>HLS(.m3u8), RTSP, 서버의 로컬 파일 경로 등 OpenCV(FFmpeg)가 열 수 있는 소스.</p>
              <div className="row">
                <input value={urlInput} onChange={(e) => setUrlInput(e.target.value)} placeholder="http://.../stream.m3u8 또는 /path/to/video.mp4" style={{ flex: 1 }} />
                <button className="primary" onClick={() => { if (!urlInput.startsWith('http') && !urlInput.startsWith('rtsp')) setTab('upload'); takeSnapshot(urlInput) }} disabled={!urlInput || !!busy}>스냅샷 가져와서 다음 →</button>
              </div>
            </Card>
          )}
        </div>
      )}

      {step === 2 && snapshot && (
        <div className="stack">
          <Banner kind="info">화면 속 방면 표지(예: "⬇ 부산 · 서울 ⬆")를 보고 <b>방면마다 다른 색</b>으로 차로를 칠하고, 아래 색 칩에 그 방면(예: 서울)을 적으세요. 보이는 방면은 <b>모두</b> 칠해야 합니다. 칠하지 않은 방면은 "미측정" 으로 남습니다. 분모는 차량이 덮을 수 있는 노면 전체입니다(갓길·중앙분리대 제외).</Banner>
          <MaskEditor imageUrl={snapshot.url} width={snapshot.width} height={snapshot.height} snapshotId={snapshot.snapshot_id} directions={directions} onDirectionsChange={setDirections} onReady={onEditorReady} />
          <div className="row between">
            <button onClick={() => setStep(1)}>← 소스 다시 선택</button>
            <button className="primary" onClick={() => setStep(3)}>다음: 정보 입력 →</button>
          </div>
        </div>
      )}

      {step === 3 && snapshot && (
        <div className="stack">
        <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) 360px' }}>
          <Card title="정보 입력" icon={<Save size={16} />}>
            <div className="form">
              <label>이름 *</label><input value={meta.name} onChange={(e) => setMeta({ ...meta, name: e.target.value })} autoFocus />
              <label>노선</label><input value={meta.route} onChange={(e) => setMeta({ ...meta, route: e.target.value })} placeholder="예: 경부선 (통계 그룹핑)" />
              <label>지역</label><input value={meta.region} onChange={(e) => setMeta({ ...meta, region: e.target.value })} placeholder="예: 경기 수원" />
              <label>구간</label><input value={meta.section} onChange={(e) => setMeta({ ...meta, section: e.target.value })} placeholder="예: 신갈JC~수원IC / 한남대교" />
              <label>경도 / 위도</label>
              <div className="row">
                <input value={meta.lon} onChange={(e) => setMeta({ ...meta, lon: e.target.value })} placeholder="127.0" style={{ width: 130 }} />
                <input value={meta.lat} onChange={(e) => setMeta({ ...meta, lat: e.target.value })} placeholder="37.5" style={{ width: 130 }} />
                <span className="muted">지도 표시용</span>
              </div>
              <label>방향</label>
              <div className="row">{directions.map((d) => <span key={d.index} className="pill"><span className="dot" style={{ background: d.color }} />{d.name}</span>)}</div>
              <label>추론 주기</label>
              <div className="row">
                <select value={interval} onChange={(e) => setInterval_(+e.target.value)}>{INTERVAL_OPTIONS.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}</select>
                <span className="muted">기본 5초마다 1장. 실시간은 GPU 를 계속 쓰므로 카메라가 많으면 부담이 큽니다 (나중에 변경 가능).</span>
              </div>
            </div>
            <div className="row" style={{ marginTop: 16 }}>
              <button onClick={() => setStep(2)}>← 도로 영역 수정</button>
              <button className="primary" onClick={save} disabled={!!busy}><Save />저장하고 모니터링 시작</button>
              <Link to="/" className="muted">취소</Link>
            </div>
          </Card>
          <Card title="미리보기">
            <MaskPreview imageUrl={snapshot.url} width={snapshot.width} height={snapshot.height} handle={editorRef.current} directions={directions} />
            <div className="muted" style={{ marginTop: 8 }}>
              {tab === 'its' ? `ITS 실시간 · ${selectedIts?.name ?? ''}` : tab === 'upload' ? `업로드 · ${upload?.filename ?? ''}` : `URL · ${urlInput}`}
            </div>
          </Card>
        </div>
          <Card title="방향 · 도로 · 지도 표시" icon={<Compass size={16} />} actions={<span className="muted">선택 사항 · 나중에 상세 페이지에서도 지정 가능</span>}>
            <DirectionMapEditor directions={directions} onChange={setDirections} cameraLat={meta.lat ? +meta.lat : null} cameraLon={meta.lon ? +meta.lon : null} route={meta.route || null} roadType={selectedIts?.road_type ?? 'ex'} roadPx={editorRef.current ? Object.fromEntries(Object.entries(editorRef.current.stats()).map(([k, v]) => [String(k), v])) : null} />
          </Card>
        </div>
      )}
    </div>
  )
}

function MaskPreview({ imageUrl, width, height, handle, directions }: { imageUrl: string; width: number; height: number; handle: MaskEditorHandle | null; directions: Direction[] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const cv = ref.current
    if (!cv || !handle) return
    const ctx = cv.getContext('2d')!
    const img = new Image()
    img.crossOrigin = 'anonymous'
    imageSrc(imageUrl.startsWith('/') ? API_BASE + imageUrl : imageUrl).then((u) => (img.src = u)).catch(() => {})
    img.onload = () => {
      ctx.drawImage(img, 0, 0, width, height)
      const id = ctx.getImageData(0, 0, width, height)
      const lab = handle.label
      const cols: Record<number, number[]> = {}
      directions.forEach((d) => { const h = d.color.replace('#', ''); cols[d.index] = [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)] })
      for (let i = 0; i < lab.length; i++) { const v = lab[i]; if (!v) continue; const c = cols[v] ?? [255, 255, 255]; id.data[i * 4] = (id.data[i * 4] + c[0] * 1.2) / 2.2; id.data[i * 4 + 1] = (id.data[i * 4 + 1] + c[1] * 1.2) / 2.2; id.data[i * 4 + 2] = (id.data[i * 4 + 2] + c[2] * 1.2) / 2.2 }
      ctx.putImageData(id, 0, 0)
    }
  }, [imageUrl, width, height, handle, directions])
  return <canvas ref={ref} width={width} height={height} style={{ width: '100%', borderRadius: 8, background: '#000' }} />
}
