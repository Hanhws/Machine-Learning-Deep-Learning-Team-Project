import { Cpu, Database, Globe, KeyRound, Play, RadioTower, Settings2, Sparkles } from 'lucide-react'
import { Link } from 'react-router-dom'
import { useState } from 'react'
import { API_BASE, API_BASE_SOURCE, api, setApiBase } from '../api/client'
import { Banner, Card, Loading, PageHeader, useAction } from '../components/ui'
import { usePolling } from '../lib/usePolling'

export default function SystemPage() {
  const { data, error, setData } = usePolling(() => api.system(), 5000)
  const { data: cams } = usePolling(() => api.cameras.list(), 5000)
  const { run, busy } = useAction()
  if (error && !data)
    return (
      <div className="page">
        <Banner kind="error">백엔드에 연결할 수 없습니다: {error}</Banner>
        <div style={{ marginTop: 12, maxWidth: 640 }}><ApiBaseCard /></div>
      </div>
    )
  if (!data) return <div className="page"><Loading lg /></div>
  const road = data.models.road
  const workers = Object.entries(data.workers) as [string, string][]
  return (
    <div className="page">
      <PageHeader title="시스템" description="모델·연동·워커 상태. 설정값은 프로젝트 루트의 .env 로 바꾸고 서버를 재시작합니다." actions={<button className="primary" disabled={!!busy} onClick={() => run('데모 카메라 생성', () => api.demo(), '샘플 카메라를 시작했습니다')}><Sparkles />샘플 카메라 만들기</button>} />
      <div className="grid auto">
        <ApiBaseCard />
        <Card title="차량 세그멘테이션 (YOLO-seg, 파인튜닝)" icon={<Cpu size={16} />}>
          <dl className="kv">
            <dt>가중치</dt><dd className="mono">{data.models.yolo_weights}</dd>
            <dt>상태</dt><dd>{data.models.yolo_loaded ? <span className="ok">로드됨 · {data.models.yolo_device}</span> : <span className="muted">첫 추론 시 로드</span>}</dd>
            <dt>클래스</dt><dd>{(data.models.yolo_classes ?? data.vehicle_classes).join(', ')}</dd>
            <dt>기본 추론 주기</dt><dd>{data.default_infer_interval_s ? `${data.default_infer_interval_s}초마다 1장` : '실시간'} (DEFAULT_INFER_INTERVAL_S)</dd>
            <dt>실시간 모드</dt><dd>{data.infer_fps} fps / 카메라 (INFER_FPS)</dd>
          </dl>
        </Card>
        <Card title="도로 세그멘테이션 (등록 시 1회)" icon={<Cpu size={16} />}>
          <dl className="kv">
            <dt>백엔드</dt><dd>{road.backend === 'sam3' ? <span className="ok">SAM3</span> : <span style={{ color: 'var(--slow)' }}>SAM2.1 (대체 동작)</span>}</dd>
            <dt>텍스트 프롬프트</dt><dd>{road.text_prompt ? '사용 가능' : '불가 — sam3.pt 필요'}</dd>
            <dt>점/박스 프롬프트</dt><dd>사용 가능</dd>
            <dt>디바이스</dt><dd>{road.device}</dd>
            <dt>SAM3 경로</dt><dd className="mono">{road.sam3_weights}</dd>
          </dl>
          {!road.text_prompt && <div style={{ marginTop: 8 }}><Banner kind="info">SAM3 승인 후 <code>python scripts/download_sam3.py --token hf_xxx</code> 로 받아 두면 서버 재시작 시 자동 전환됩니다.</Banner></div>}
        </Card>
        <Card title="ITS Open API" icon={<RadioTower size={16} />}>
          <dl className="kv">
            <dt>인증키</dt><dd>{data.its_configured ? <span className="ok">설정됨</span> : <span className="error">미설정 (.env ITS_API_KEY)</span>}</dd>
            <dt>영상 URL</dt><dd>24시간 유효 · 워커가 만료 전 자동 재조회</dd>
          </dl>
          <div className="row" style={{ marginTop: 8 }}><Link to="/register"><button className="sm"><Play />ITS CCTV 등록</button></Link></div>
        </Card>
        <Card title="혼잡 단계 임계값" icon={<Settings2 size={16} />}>
          <dl className="kv">
            {data.levels.map((l: string, i: number) => (
              <span key={l} style={{ display: 'contents' }}><dt><span className={`badge ${['free', 'slow', 'delay', 'jam'][i]}`}>{l}</span></dt><dd>{i < data.thresholds.length ? `점유율 < ${(data.thresholds[i] * 100).toFixed(0)}%` : `점유율 ≥ ${(data.thresholds[i - 1] * 100).toFixed(0)}%`}</dd></span>
            ))}
          </dl>
          <p className="muted" style={{ marginTop: 8 }}>.env 의 CONGESTION_THRESHOLDS 로 변경. 기준값은 AI-Hub 샘플에서 비혼잡 8% / 혼잡 18% 였던 도로 대비 면적비를 바탕으로 잡았습니다.</p>
        </Card>
        <Card title="저장소" icon={<Database size={16} />}>
          <dl className="kv">
            <dt>DB</dt><dd>PostgreSQL · 5초 단위 점유율 샘플, 카메라/방향/그룹/분석 작업</dd>
            <dt>파일</dt><dd className="mono">data/cameras/{'{id}'}/ (snapshot.jpg, road_mask.png, captures/)</dd>
          </dl>
        </Card>
        <Card title={<>워커 <span className="muted">{workers.length}</span></>} icon={<KeyRound size={16} />}>
          {!workers.length && <div className="muted">실행 중인 워커가 없습니다.</div>}
          {workers.length > 0 && (
            <table>
              <thead><tr><th>카메라</th><th>상태</th><th className="num">추론</th></tr></thead>
              <tbody>
                {workers.map(([id, st]) => {
                  const c = cams?.find((x) => x.id === +id)
                  return (
                    <tr key={id}>
                      <td><Link to={`/cameras/${id}`}>{c?.name ?? `camera ${id}`}</Link></td>
                      <td><span className="row" style={{ gap: 6 }}><span className={`status-dot ${st === 'running' ? 'ok' : st === 'error' ? 'err' : 'warn'}`} />{st}</span></td>
                      <td className="num muted">{c?.live ? `${c.live.infer_ms} ms · ${c.live.fps} fps` : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
          <div style={{ marginTop: 8 }}><button className="sm ghost" onClick={async () => setData(await api.system())}>새로고침</button></div>
        </Card>
      </div>
    </div>
  )
}


function ApiBaseCard() {
  const [v, setV] = useState(API_BASE)
  const srcLabel = { browser: '브라우저에 저장된 값', env: '빌드 환경변수 VITE_API_BASE', 'same-origin': '같은 서버 (/api)' }[API_BASE_SOURCE]
  return (
    <Card title="백엔드 주소" icon={<Globe size={16} />}>
      <dl className="kv" style={{ marginBottom: 8 }}>
        <dt>현재</dt><dd className="mono">{API_BASE || '(같은 서버) /api'}</dd>
        <dt>출처</dt><dd>{srcLabel}</dd>
      </dl>
      <div className="row">
        <input value={v} onChange={(e) => setV(e.target.value)} placeholder="https://xxxx.ngrok-free.app 또는 http://localhost:8000" style={{ flex: 1, minWidth: 220 }} />
        <button className="primary" onClick={() => setApiBase(v)}>적용</button>
        {API_BASE_SOURCE === 'browser' && <button onClick={() => setApiBase('')}>초기화</button>}
      </div>
      <p className="muted" style={{ marginTop: 8 }}>
        이 값은 이 브라우저에만 저장됩니다. 다른 사람에게 링크로 알려주려면 <code>?api=백엔드주소</code> 를 붙여 주세요 (한 번 열면 저장됨). 비워 두면 빌드 환경변수 → 같은 서버 순으로 씁니다.
      </p>
    </Card>
  )
}
