import { CalendarDays, Download, FlaskConical, Landmark, Lightbulb, Pencil, Plus, Printer, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type Camera, type Group } from '../api/client'
import LevelBadge from '../components/LevelBadge'
import OccupancyBar from '../components/OccupancyBar'
import TimeSeriesChart from '../components/TimeSeriesChart'
import { Banner, Card, EmptyState, Loading, Modal, PageHeader, Segmented, StatCard, useAction } from '../components/ui'
import { DIRECTION_PALETTE, LEVEL_CLASS, levelColor, levelOf, pct } from '../lib/format'
import { usePolling } from '../lib/usePolling'

interface Bridge { name: string; lat: number; lon: number }
const plabel = (m: number) => (m >= 1440 ? `${m / 1440}일` : m >= 60 ? `${m / 60}시간` : `${m}분`)

/** 응용 분석: CCTV 를 목적별로 묶어(그룹) 비교·정책용 리포트. 프리셋 "한강 대교". */
export default function AppsPage() {
  const [groups, setGroups] = useState<Group[]>([])
  const [cams, setCams] = useState<Camera[]>([])
  const [bridges, setBridges] = useState<Bridge[]>([])
  const [sel, setSel] = useState<number | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const { run, busy } = useAction()

  const load = async () => {
    const [g, c, b] = await Promise.all([api.apps.groups(), api.cameras.list(), api.apps.hanRiver()])
    setGroups(g); setCams(c); setBridges(b.bridges); setLoaded(true)
    setSel((s) => (s == null && g.length ? g[0].id : s))
  }
  useEffect(() => { load().catch((e) => setErr(e.message)) }, [])
  const createHanRiver = () => run('프리셋 생성', async () => { const g = await api.apps.createGroup({ name: '한강 대교 교통량', kind: 'han_river', description: '한강 교량별 CCTV 점유율 비교 — 교량 간 통행 분산, 시간대별 혼잡 패턴, 가변차로·우회 유도 등 정책 검토용', members: [] }); await load(); setSel(g.id) })
  const createCustom = () => { const name = prompt('그룹 이름 (예: 수도권 순환선, 부산항 진입로)'); if (name) run('그룹 생성', async () => { const g = await api.apps.createGroup({ name, kind: 'custom', description: '', members: [] }); await load(); setSel(g.id) }) }
  const group = groups.find((g) => g.id === sel) ?? null

  return (
    <div className="page">
      <PageHeader
        title="응용 분석"
        description="측정 결과를 정책·운영에 쓰기 위한 그룹 리포트. 예: 한강 대교별 교통량 비교로 교량 간 통행 분산 정책의 근거를 만듭니다."
        actions={<>{!groups.some((g) => g.kind === 'han_river') && <button className="primary" disabled={!!busy} onClick={createHanRiver}><Landmark />한강 대교 프리셋</button>}<button onClick={createCustom} disabled={!!busy}><Plus />새 그룹</button></>}
      />
      {err && <Banner kind="error">{err}</Banner>}
      {!loaded && <Loading lg />}
      {loaded && !groups.length && <EmptyState icon={<Landmark />} title="아직 그룹이 없습니다" description="한강 대교 프리셋을 만들면 26개 교량 목록과 함께 리포트 틀이 생깁니다. 각 교량 근처의 ITS CCTV 를 등록해 연결하세요." action={<button className="primary" onClick={createHanRiver}><Landmark />한강 대교 프리셋 만들기</button>} />}
      {loaded && groups.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: '240px minmax(0, 1fr)', gap: 12 }}>
          <aside className="stack" style={{ alignContent: 'start' }}>
            {groups.map((g) => (
              <div key={g.id} className="card" style={{ padding: '10px 12px', cursor: 'pointer', outline: sel === g.id ? '1px solid var(--accent)' : 'none' }} onClick={() => setSel(g.id)}>
                <div style={{ fontWeight: 600 }}>{g.name}</div>
                <div className="muted">{g.kind === 'han_river' ? '한강 대교 프리셋' : '사용자 정의'} · {g.members.length}개</div>
              </div>
            ))}
          </aside>
          <section>{group ? <GroupView key={group.id} group={group} cams={cams} bridges={bridges} onChange={load} /> : null}</section>
        </div>
      )}
    </div>
  )
}

function GroupView({ group, cams, bridges, onChange }: { group: Group; cams: Camera[]; bridges: Bridge[]; onChange: () => Promise<void> }) {
  const [minutes, setMinutes] = useState(60)
  const [editing, setEditing] = useState(group.members.length === 0)
  const [members, setMembers] = useState(group.members.map((m) => ({ camera_id: m.camera_id, label: m.label ?? '', order: m.order })))
  const [tab, setTab] = useState<'now' | 'daily' | 'scenario'>('now')
  const { data: report, error } = usePolling(() => api.apps.report(group.id, minutes, minutes >= 720 ? 1800 : minutes >= 180 ? 600 : 120), 5000, [group.id, minutes])
  const { run, busy } = useAction()
  const isHan = group.kind === 'han_river'

  const saveMembers = () => run('구성 저장', async () => { await api.apps.updateGroup(group.id, { members: members.filter((m) => m.camera_id).map((m, i) => ({ ...m, order: i, label: m.label || null })) }); await onChange(); setEditing(false) }, '저장했습니다')
  const remove = () => { if (confirm('그룹을 삭제할까요? (카메라는 유지됩니다)')) run('그룹 삭제', async () => { await api.apps.deleteGroup(group.id); await onChange() }) }

  const chartData = useMemo(() => (report?.members ?? []).map((m: any) => { const row: Record<string, any> = { label: m.label, 전체: m.mean ?? 0, level: m.level }; m.directions.forEach((d: any) => (row[d.name] = d.mean ?? 0)); return row }), [report])
  const dirNames = useMemo(() => { const s = new Set<string>(); (report?.members ?? []).forEach((m: any) => m.directions.forEach((d: any) => s.add(d.name))); return [...s] }, [report])
  const hourly = useMemo(() => Array.from({ length: 24 }, (_, h) => { const row: Record<string, any> = { hour: `${h}시` }; (report?.members ?? []).forEach((m: any) => (row[m.label] = m.hourly[h])); return row }), [report])
  const series = useMemo(() => { const byT = new Map<string, Record<string, any>>(); (report?.members ?? []).forEach((m: any) => m.series.forEach((p: any) => { const r = byT.get(p.t) ?? { t: p.t }; r[m.label] = p.occupancy; byT.set(p.t, r) })); return [...byT.values()].sort((a, b) => (a.t > b.t ? 1 : -1)) }, [report])
  const insights = useMemo(() => buildInsights(report, minutes, isHan), [report, minutes, isHan])
  const exportCsv = () => {
    if (!report) return
    const rows = [['label', 'camera', 'live', 'mean', 'max', 'level', 'n_vehicles', 'samples', ...dirNames.map((d) => `dir:${d}`)]]
    report.members.forEach((m: any) => rows.push([m.label, m.name, m.live ?? '', m.mean ?? '', m.max ?? '', m.level ?? '', m.n_vehicles ?? '', m.samples, ...dirNames.map((d) => m.directions.find((x: any) => x.name === d)?.mean ?? '')]))
    const blob = new Blob(['﻿' + rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n')], { type: 'text/csv;charset=utf-8' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `${group.name}_${minutes}m.csv`; a.click()
  }

  return (
    <div className="stack">
      <div className="row between">
        <div><h2>{group.name}</h2><div className="muted">{group.description}</div></div>
        <div className="row no-print">
          <select value={minutes} onChange={(e) => setMinutes(+e.target.value)}>{[15, 60, 180, 720, 1440, 10080].map((m) => <option key={m} value={m}>최근 {plabel(m)}</option>)}</select>
          <button className={editing ? 'active' : ''} onClick={() => setEditing(!editing)}><Pencil />구성 편집</button>
          <button onClick={exportCsv} disabled={!report}><Download />CSV</button>
          <button onClick={() => window.print()}><Printer />인쇄/PDF</button>
          <button className="danger" onClick={remove} disabled={!!busy}><Trash2 /></button>
        </div>
      </div>

      <Modal open={editing} title={`구성원 편집 — ${group.name}`} onClose={() => setEditing(false)} width={760}>
        <p className="muted" style={{ marginBottom: 10 }}>{isHan ? '교량별로 이미 등록된 CCTV 를 고르세요. 아직 없다면 아래 교량 링크로 ITS 검색 지도가 해당 위치에서 열립니다.' : '라벨(표시 이름)과 카메라를 짝지으세요.'}</p>
        <table>
          <thead><tr><th>{isHan ? '교량' : '라벨'}</th><th>카메라</th><th></th></tr></thead>
          <tbody>
            {members.map((m, i) => (
              <tr key={i}>
                <td>{isHan ? <select value={m.label} onChange={(e) => setMembers(members.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)))}><option value="">(선택)</option>{bridges.map((b) => <option key={b.name} value={b.name}>{b.name}</option>)}</select> : <input value={m.label} onChange={(e) => setMembers(members.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)))} placeholder="표시 이름" />}</td>
                <td><select value={m.camera_id} onChange={(e) => setMembers(members.map((x, j) => (j === i ? { ...x, camera_id: +e.target.value } : x)))}><option value={0}>(카메라 선택)</option>{cams.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</select></td>
                <td><button className="sm icon" onClick={() => setMembers(members.filter((_, j) => j !== i))}><Trash2 /></button></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="row" style={{ marginTop: 10 }}>
          <button onClick={() => setMembers([...members, { camera_id: 0, label: '', order: members.length }])}><Plus />행 추가</button>
          <button className="primary" onClick={saveMembers} disabled={!!busy}>저장</button>
        </div>
        {isHan && (
          <div style={{ marginTop: 14 }}>
            <div className="muted" style={{ marginBottom: 6 }}>교량 근처 CCTV 등록 바로가기 (ITS 지도가 해당 위치에서 열립니다)</div>
            <div className="row" style={{ gap: 6 }}>{bridges.map((b) => <Link key={b.name} to={`/register?lat=${b.lat}&lon=${b.lon}&label=${encodeURIComponent(b.name)}`} className="pill clickable">{b.name}</Link>)}</div>
          </div>
        )}
      </Modal>

      {error && <Banner kind="error">{error}</Banner>}
      {!report && <Loading />}
      {report && report.members.length === 0 && <EmptyState icon={<Landmark />} title="구성원이 없습니다" description="구성 편집에서 카메라를 연결하세요." action={<button className="primary" onClick={() => setEditing(true)}><Pencil />구성 편집</button>} />}
      {report && report.members.length > 0 && (
        <>
          <div className="grid cols-4">
            <StatCard label="그룹 평균 점유율" value={pct(report.overall.mean)} sub={`최근 ${plabel(minutes)}`} icon={<Landmark />} tone={LEVEL_CLASS[report.overall.level ?? ''] ?? ''} />
            <StatCard label="가장 혼잡" value={<span style={{ fontSize: 17 }}>{report.overall.most_congested ?? '–'}</span>} icon={<Landmark />} tone="jam" />
            <StatCard label="가장 원활" value={<span style={{ fontSize: 17 }}>{report.overall.least_congested ?? '–'}</span>} icon={<Landmark />} tone="free" />
            <StatCard label="모니터링 중" value={report.overall.n_running} unit={`/ ${report.overall.n_members}`} icon={<Landmark />} />
          </div>
          <Segmented options={[{ v: 'now', l: '현황 비교' }, { v: 'daily', l: <><CalendarDays size={14} /> 일별 리포트</> }, { v: 'scenario', l: <><FlaskConical size={14} /> 정책 시나리오</> }]} value={tab} onChange={setTab} />

          {tab === 'now' && (
            <>
              <Card title={`${isHan ? '교량별' : '구성원별'} 평균 점유율 — 방향 비교`}>
                <ResponsiveContainer width="100%" height={Math.max(200, 44 * chartData.length)}>
                  <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
                    <CartesianGrid stroke="#253046" strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="#8b96ad" fontSize={11} />
                    <YAxis type="category" dataKey="label" width={110} stroke="#8b96ad" fontSize={12} />
                    <Tooltip contentStyle={{ background: '#141924', border: '1px solid #31405a', borderRadius: 8, fontSize: 12 }} formatter={(v: any) => `${(Number(v) * 100).toFixed(1)}%`} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="전체" fill="#e9edf5" isAnimationActive={false} barSize={10} />
                    {dirNames.map((n, i) => <Bar key={n} dataKey={n} fill={DIRECTION_PALETTE[i % DIRECTION_PALETTE.length]} isAnimationActive={false} barSize={10} />)}
                  </BarChart>
                </ResponsiveContainer>
              </Card>
              <div className="grid cols-2">
                <Card title="시간대별 프로필 (누적 전체 기간, KST)">
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={hourly}>
                      <CartesianGrid stroke="#253046" strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="hour" stroke="#8b96ad" fontSize={11} interval={2} tickLine={false} />
                      <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="#8b96ad" fontSize={11} width={44} tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: '#141924', border: '1px solid #31405a', borderRadius: 8, fontSize: 12 }} formatter={(v: any) => `${(Number(v) * 100).toFixed(1)}%`} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      {report.members.map((m: any, i: number) => <Line key={m.camera_id} dataKey={m.label} stroke={DIRECTION_PALETTE[i % DIRECTION_PALETTE.length]} dot={false} connectNulls isAnimationActive={false} strokeWidth={2} />)}
                    </LineChart>
                  </ResponsiveContainer>
                </Card>
                <Card title={`최근 ${plabel(minutes)} 추이`}>
                  <TimeSeriesChart data={series} series={report.members.map((m: any, i: number) => ({ key: m.label, name: m.label, color: DIRECTION_PALETTE[i % DIRECTION_PALETTE.length] }))} thresholds={report.thresholds} height={240} />
                </Card>
              </div>
              <Card title="정책 인사이트 (자동 생성)" icon={<Lightbulb size={16} />}>
                <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.8 }}>{insights.map((s, i) => <li key={i}>{s}</li>)}</ul>
              </Card>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>{isHan ? '교량' : '구성원'}</th><th>카메라</th><th>실시간</th><th>평균</th><th className="num">최대</th><th>방향별</th><th className="num">샘플</th></tr></thead>
                  <tbody>
                    {report.members.map((m: any) => (
                      <tr key={m.camera_id}>
                        <td><b>{m.label}</b></td>
                        <td><Link to={`/cameras/${m.camera_id}`}>{m.name}</Link> {!m.running && <span className="badge outline">정지</span>}</td>
                        <td>{pct(m.live)} <LevelBadge level={m.live_level} size="sm" /></td>
                        <td style={{ minWidth: 160 }}><OccupancyBar value={m.mean} level={m.level} compact /></td>
                        <td className="num">{pct(m.max)}</td>
                        <td>{m.directions.map((d: any) => <div key={d.index} className="small"><span className="dot" style={{ background: d.color, width: 8, height: 8, marginRight: 4 }} />{d.name}: {pct(d.mean)} <span style={{ color: levelColor(d.level) }}>{d.level ?? ''}</span></div>)}</td>
                        <td className="num muted">{m.samples}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {tab === 'daily' && <DailyReport group={group} />}
          {tab === 'scenario' && <Scenario report={report} isHan={isHan} />}
        </>
      )}
    </div>
  )
}

function DailyReport({ group }: { group: Group }) {
  const [days, setDays] = useState(7)
  const { data } = usePolling(() => api.apps.daily(group.id, days), 60000, [group.id, days])
  if (!data) return <Loading />
  const thr: number[] = data.thresholds
  return (
    <Card title="일별 리포트 (KST)" icon={<CalendarDays size={16} />} actions={<Segmented options={[{ v: 7, l: '7일' }, { v: 14, l: '14일' }, { v: 30, l: '30일' }]} value={days} onChange={setDays} />}>
      {!data.days.length && <div className="muted">해당 기간에 기록이 없습니다.</div>}
      {data.days.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>구성원</th>{data.days.map((d: string) => <th key={d} className="num">{d.slice(5)}</th>)}</tr></thead>
            <tbody>
              {data.members.map((m: any) => (
                <tr key={m.camera_id}>
                  <td><b>{m.label}</b></td>
                  {m.days.map((d: any) => (
                    <td key={d.day} className="num" style={{ background: d.mean == null ? undefined : `${levelColor(levelOf(d.mean, thr))}22` }} title={d.mean == null ? '' : `평균 ${pct(d.mean)} · 최대 ${pct(d.max)} · 피크 ${d.peak_hour}시 · 정체 비율 ${pct(d.jam_ratio, 0)} · 차량 ${d.n_vehicles?.toFixed(1)}대\n${d.directions.map((x: any) => `${x.name} ${pct(x.mean)}`).join(' / ')}`}>
                      {d.mean == null ? <span className="muted">—</span> : <><div>{pct(d.mean)}</div><div className="muted">피크 {d.peak_hour}시</div></>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="muted" style={{ marginTop: 6 }}>셀 색 = 그날 평균의 혼잡 단계. 마우스를 올리면 최대·피크 시간·정체 비율·방향별 평균.</div>
    </Card>
  )
}

function Scenario({ report, isHan }: { report: any; isHan: boolean }) {
  const ms = report.members.filter((m: any) => m.mean != null)
  const [from, setFrom] = useState<number>(ms[0]?.camera_id ?? 0)
  const [to, setTo] = useState<number>(ms[ms.length - 1]?.camera_id ?? 0)
  const [ratio, setRatio] = useState(20)
  const [cap, setCap] = useState(1)
  const f = ms.find((m: any) => m.camera_id === from), t = ms.find((m: any) => m.camera_id === to)
  const thr: number[] = report.thresholds
  if (ms.length < 2) return <Banner kind="info">시나리오 비교에는 기록이 있는 구성원이 2개 이상 필요합니다.</Banner>
  const moved = f ? f.mean * (ratio / 100) : 0
  const newF = f ? f.mean - moved : 0
  const newT = t ? t.mean + moved / cap : 0
  return (
    <Card title="정책 시나리오: 통행 분산" icon={<FlaskConical size={16} />}>
      <p className="muted" style={{ marginBottom: 10 }}>
        가장 단순한 선형 가정입니다: {isHan ? '한 교량' : '한 지점'}의 교통량 일부가 다른 곳으로 옮겨가면 점유율도 같은 비율로 이동한다고 봅니다. (용량 배수 = 목적지 도로가 상대적으로 넓을수록 큼)
      </p>
      <div className="row" style={{ gap: 12 }}>
        <label className="field"><span>혼잡 지점</span><select value={from} onChange={(e) => setFrom(+e.target.value)}>{ms.map((m: any) => <option key={m.camera_id} value={m.camera_id}>{m.label} ({pct(m.mean)})</option>)}</select></label>
        <label className="field"><span>우회 목적지</span><select value={to} onChange={(e) => setTo(+e.target.value)}>{ms.map((m: any) => <option key={m.camera_id} value={m.camera_id}>{m.label} ({pct(m.mean)})</option>)}</select></label>
        <label className="field"><span>전환 비율 {ratio}%</span><input type="range" min={0} max={60} value={ratio} onChange={(e) => setRatio(+e.target.value)} /></label>
        <label className="field"><span>목적지 용량 배수 {cap.toFixed(1)}×</span><input type="range" min={0.5} max={3} step={0.1} value={cap} onChange={(e) => setCap(+e.target.value)} /></label>
      </div>
      {f && t && f !== t && (
        <div className="grid cols-2" style={{ marginTop: 14 }}>
          <div className="card">
            <div className="muted">{f.label}</div>
            <div className="row" style={{ gap: 10 }}><span className="stat-card"><span className="value">{pct(f.mean)}</span></span><span>→</span><span className="value" style={{ fontSize: 22, fontWeight: 700, color: levelColor(levelOf(newF, thr)) }}>{pct(newF)}</span><LevelBadge level={levelOf(newF, thr)} /></div>
          </div>
          <div className="card">
            <div className="muted">{t.label}</div>
            <div className="row" style={{ gap: 10 }}><span className="stat-card"><span className="value">{pct(t.mean)}</span></span><span>→</span><span className="value" style={{ fontSize: 22, fontWeight: 700, color: levelColor(levelOf(newT, thr)) }}>{pct(newT)}</span><LevelBadge level={levelOf(newT, thr)} /></div>
          </div>
        </div>
      )}
      {f && t && f !== t && (
        <Banner kind={levelOf(newT, thr) === '정체' ? 'warn' : 'info'}>
          {levelOf(newT, thr) === '정체' ? `${t.label} 이(가) 정체 단계에 들어갑니다. 전환 비율을 낮추거나 다른 목적지를 고려하세요.` : `${f.label} 의 혼잡을 ${(f.mean - newF > 0 ? (f.mean - newF) * 100 : 0).toFixed(1)}%p 낮추는 동안 ${t.label} 은 ${levelOf(newT, thr)} 수준을 유지합니다.`}
        </Banner>
      )}
    </Card>
  )
}

function buildInsights(report: any, minutes: number, isHan: boolean): string[] {
  if (!report?.members?.length) return []
  const ms = report.members.filter((m: any) => m.mean != null)
  if (!ms.length) return ['아직 집계된 샘플이 없습니다. 카메라를 모니터링 상태로 두면 5초마다 기록됩니다.']
  const out: string[] = []
  const sorted = [...ms].sort((a, b) => b.mean - a.mean)
  const top = sorted[0], bottom = sorted[sorted.length - 1]
  const unit = isHan ? '교량' : '지점'
  out.push(`최근 ${plabel(minutes)} 기준 가장 혼잡한 ${unit}은 ${top.label}(평균 ${pct(top.mean)}, ${top.level})이고, 가장 원활한 ${unit}은 ${bottom.label}(${pct(bottom.mean)})입니다.`)
  if (ms.length >= 2 && top.mean / Math.max(bottom.mean, 0.005) >= 2) out.push(`${top.label} 의 점유율이 ${bottom.label} 의 ${(top.mean / Math.max(bottom.mean, 0.005)).toFixed(1)}배입니다. ${isHan ? '인접 교량으로의 우회 유도(가변 안내표지·내비 연계)를 검토할 여지가 있습니다 — "정책 시나리오" 탭에서 효과를 가늠해 보세요.' : '통행 분산 정책의 우선 검토 대상입니다.'}`)
  const jam = ms.filter((m: any) => m.level === '정체' || m.level === '지체')
  if (jam.length) out.push(`지체 이상 ${unit}: ${jam.map((m: any) => m.label).join(', ')} (${jam.length}/${ms.length}).`)
  ms.forEach((m: any) => {
    const ds = m.directions.filter((d: any) => d.mean != null)
    if (ds.length >= 2) { const s = [...ds].sort((a, b) => b.mean - a.mean); if (s[0].mean >= 0.08 && s[0].mean / Math.max(s[1].mean, 0.005) >= 1.8) out.push(`${m.label}: ${s[0].name}(${pct(s[0].mean)})이 ${s[1].name}(${pct(s[1].mean)})보다 뚜렷이 혼잡합니다 — 방향별 가변차로/신호 배분 검토 대상.`) }
  })
  ms.forEach((m: any) => {
    const hs = (m.hourly as (number | null)[]).map((v, h) => ({ v, h })).filter((x) => x.v != null) as { v: number; h: number }[]
    if (hs.length >= 6) { const peak = hs.reduce((a, b) => (b.v > a.v ? b : a)); out.push(`${m.label} 의 일중 피크는 ${peak.h}시(평균 ${pct(peak.v)})입니다.`) }
  })
  out.push('점유율은 카메라 화각·원근의 영향을 받으므로 지점 간 절대 비교보다 같은 지점의 시간대·방향 간 상대 비교가 더 신뢰할 수 있습니다.')
  return out
}
