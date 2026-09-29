import { AlertTriangle, BarChart3, Download, Layers, TrendingUp } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, type AlertEpisode } from '../api/client'
import LevelBadge from '../components/LevelBadge'
import OccupancyBar from '../components/OccupancyBar'
import TimeSeriesChart from '../components/TimeSeriesChart'
import { Banner, Card, EmptyState, Loading, PageHeader, Segmented, Sparkline, StatCard } from '../components/ui'
import { DIRECTION_PALETTE, LEVEL_CLASS, levelColor, levelOf, pct } from '../lib/format'
import { usePolling } from '../lib/usePolling'

const BY = [{ v: 'route', l: '노선별' }, { v: 'region', l: '지역별' }, { v: 'section', l: '구간별' }, { v: 'camera', l: '카메라별' }]
const PERIODS = [5, 15, 30, 60, 180, 720, 1440, 10080]
const plabel = (m: number) => (m >= 1440 ? `${m / 1440}일` : m >= 60 ? `${m / 60}시간` : `${m}분`)
type SortKey = 'occupancy' | 'max' | 'n_vehicles' | 'key' | 'n_cameras'

export default function StatsPage() {
  const [by, setBy] = useState('route')
  const [minutes, setMinutes] = useState(15)
  const [open, setOpen] = useState<string | null>(null)
  const [sort, setSort] = useState<{ k: SortKey; d: 1 | -1 }>({ k: 'occupancy', d: -1 })
  const { data, error } = usePolling(() => api.metrics.summary(by, minutes), 5000, [by, minutes])
  const { data: alerts } = usePolling(() => api.metrics.alerts({ minutes, min_level: 2, min_duration: 30 }), 15000, [minutes])
  const [tl, setTl] = useState<Awaited<ReturnType<typeof api.metrics.timeline>> | null>(null)
  useEffect(() => {
    const bucket = minutes >= 1440 ? 1800 : minutes >= 180 ? 300 : 60
    api.metrics.timeline(Math.max(minutes, 10), bucket).then(setTl).catch(() => {})
    const t = setInterval(() => api.metrics.timeline(Math.max(minutes, 10), bucket).then(setTl).catch(() => {}), 30000)
    return () => clearInterval(t)
  }, [minutes])

  const groups: any[] = useMemo(() => {
    const g = [...(data?.groups ?? [])]
    g.sort((a, b) => { const va = a[sort.k] ?? -1, vb = b[sort.k] ?? -1; return (va > vb ? 1 : va < vb ? -1 : 0) * sort.d })
    return g
  }, [data, sort])
  const thresholds: number[] = data?.thresholds ?? [0.08, 0.15, 0.25]
  const camSpark = (id: number) => tl?.cameras[String(id)]
  // 그룹별 추이 (타임라인을 그룹 평균으로)
  const groupSeries = useMemo(() => {
    if (!tl || !data) return { rows: [] as Record<string, any>[], keys: [] as string[] }
    const keys = groups.slice(0, 8).map((g) => g.key as string)
    const rows = tl.buckets.map((t, i) => {
      const row: Record<string, any> = { t }
      groups.slice(0, 8).forEach((g) => {
        const vals = g.cameras.map((c: any) => tl.cameras[String(c.id)]?.[i]).filter((v: any) => v != null) as number[]
        row[g.key] = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
      })
      return row
    })
    return { rows, keys }
  }, [tl, data, groups])
  const eps: AlertEpisode[] = alerts?.episodes ?? []
  const allCams = groups.flatMap((g) => g.cameras)
  const occs = allCams.map((c) => c.overall?.occupancy).filter((v) => v != null) as number[]
  const mean = occs.length ? occs.reduce((a, b) => a + b, 0) / occs.length : null
  const th = (k: SortKey, label: string, cls = '') => (
    <th className={`sortable ${cls}`} onClick={() => setSort({ k, d: sort.k === k ? (sort.d === 1 ? -1 : 1) : -1 })}>{label}{sort.k === k ? (sort.d === -1 ? ' ▾' : ' ▴') : ''}</th>
  )

  return (
    <div className="page">
      <PageHeader
        title="점유율 통계"
        description={`최근 ${plabel(minutes)} 동안 기록된 5초 단위 샘플의 평균. 방향별 값은 각 방향의 도로 영역 기준입니다.`}
        actions={
          <>
            <Segmented options={BY} value={by} onChange={setBy} />
            <select value={minutes} onChange={(e) => setMinutes(+e.target.value)}>{PERIODS.map((m) => <option key={m} value={m}>최근 {plabel(m)}</option>)}</select>
            <a href={api.metrics.summaryCsvUrl(by, minutes)}><button><Download />CSV</button></a>
          </>
        }
      />
      {error && <Banner kind="error">{error}</Banner>}
      {!data && <Loading lg />}
      {data && !groups.length && <EmptyState icon={<BarChart3 />} title="이 기간에 기록이 없습니다" description="카메라를 모니터링 상태로 두면 5초마다 기록되어 통계가 쌓입니다. 기간을 늘리거나 전체 CCTV 에서 모니터링을 시작하세요." action={<Link to="/cameras"><button className="primary">전체 CCTV</button></Link>} />}
      {data && groups.length > 0 && (
        <>
          <div className="grid cols-4" style={{ marginBottom: 12 }}>
            <StatCard label={`${BY.find((b) => b.v === by)?.l.replace('별', '')} 수`} value={groups.length} sub={`카메라 ${allCams.length}대`} icon={<Layers />} />
            <StatCard label="전체 평균 점유율" value={pct(mean)} sub={`최근 ${plabel(minutes)}`} icon={<TrendingUp />} tone={LEVEL_CLASS[levelOf(mean, thresholds) ?? ''] ?? ''} />
            <StatCard label="가장 혼잡" value={<span style={{ fontSize: 16 }}>{groups[0]?.key}</span>} sub={`${pct(groups.slice().sort((a, b) => (b.occupancy ?? 0) - (a.occupancy ?? 0))[0]?.occupancy)} 평균`} icon={<AlertTriangle />} tone="jam" />
            <StatCard label="지체 이상 경보" value={eps.length} unit="건" sub={`${eps.filter((e) => e.ongoing).length}건 진행 중 · 30초 이상 지속`} icon={<AlertTriangle />} tone={eps.length ? 'delay' : 'free'} />
          </div>
          <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', marginBottom: 12 }}>
            <Card title={`${BY.find((b) => b.v === by)?.l} 평균 점유율`} icon={<BarChart3 size={16} />}>
              <ResponsiveContainer width="100%" height={Math.max(180, 30 * Math.min(groups.length, 12) + 40)}>
                <BarChart data={groups.slice(0, 12)} layout="vertical" margin={{ left: 10, right: 30 }}>
                  <CartesianGrid stroke="#253046" strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="#8b96ad" fontSize={11} />
                  <YAxis type="category" dataKey="key" width={130} stroke="#8b96ad" fontSize={12} />
                  <Tooltip contentStyle={{ background: '#141924', border: '1px solid #31405a', borderRadius: 8, fontSize: 12 }} formatter={(v: any) => [`${(Number(v) * 100).toFixed(1)}%`, '평균 점유율']} />
                  <Bar dataKey="occupancy" isAnimationActive={false} radius={[0, 4, 4, 0]} barSize={16}>{groups.slice(0, 12).map((g, i) => <Cell key={i} fill={levelColor(g.level)} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="추이 (상위 8개)" icon={<TrendingUp size={16} />}>
              <TimeSeriesChart data={groupSeries.rows} series={groupSeries.keys.map((k, i) => ({ key: k, name: k, color: DIRECTION_PALETTE[i % DIRECTION_PALETTE.length] }))} thresholds={thresholds} height={Math.max(180, 30 * Math.min(groups.length, 12) + 40)} />
            </Card>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>{th('key', BY.find((b) => b.v === by)?.l.replace('별', '') ?? '')}{th('n_cameras', '카메라', 'num')}{th('occupancy', '평균 점유율')}{th('max', '최대', 'num')}<th>혼잡도</th>{th('n_vehicles', '평균 차량', 'num')}<th>추이</th></tr>
              </thead>
              <tbody>
                {groups.map((g) => (
                  <GroupRows key={g.key} g={g} open={open === g.key} onToggle={() => setOpen(open === g.key ? null : g.key)} spark={camSpark} />
                ))}
              </tbody>
            </table>
          </div>
          {eps.length > 0 && (
            <Card title="지체 이상 경보 목록" icon={<AlertTriangle size={16} />} style={{ marginTop: 12 }}>
              <div className="table-wrap" style={{ maxHeight: 300 }}>
                <table>
                  <thead><tr><th>카메라</th><th>방향</th><th>시작</th><th className="num">지속</th><th className="num">최고</th><th></th></tr></thead>
                  <tbody>
                    {eps.slice(0, 50).map((e, i) => (
                      <tr key={i}>
                        <td><Link to={`/cameras/${e.camera_id}`}>{e.camera_name}</Link></td><td>{e.direction_name}</td>
                        <td className="num">{new Date(e.start).toLocaleString('ko-KR', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</td>
                        <td className="num">{e.duration_s >= 3600 ? `${(e.duration_s / 3600).toFixed(1)}h` : `${Math.round(e.duration_s / 60)}분`}</td>
                        <td className="num">{pct(e.peak)}</td><td>{e.ongoing ? <span className="badge jam">진행 중</span> : <LevelBadge level={e.level} size="sm" />}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

function GroupRows({ g, open, onToggle, spark }: { g: any; open: boolean; onToggle: () => void; spark: (id: number) => (number | null)[] | undefined }) {
  const merged = useMemo(() => {
    const arrs = g.cameras.map((c: any) => spark(c.id)).filter(Boolean) as (number | null)[][]
    if (!arrs.length) return null
    const n = Math.max(...arrs.map((a) => a.length))
    return Array.from({ length: n }, (_, i) => { const v = arrs.map((a) => a[i]).filter((x) => x != null) as number[]; return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null })
  }, [g, spark])
  return (
    <>
      <tr className="clickable" onClick={onToggle}>
        <td><b>{open ? '▾' : '▸'} {g.key}</b></td>
        <td className="num">{g.n_cameras}</td>
        <td style={{ minWidth: 200 }}><OccupancyBar value={g.occupancy} level={g.level} compact /></td>
        <td className="num">{pct(g.max)}</td>
        <td><LevelBadge level={g.level} size="sm" /></td>
        <td className="num">{g.n_vehicles?.toFixed(1)}</td>
        <td>{merged && <Sparkline values={merged} width={110} height={24} color={levelColor(g.level)} />}</td>
      </tr>
      {open && g.cameras.map((c: any) => (
        <tr key={c.id} style={{ background: 'var(--bg-2)' }}>
          <td style={{ paddingLeft: 28 }}><Link to={`/cameras/${c.id}`}>{c.name}</Link><div className="muted">{[c.route, c.region, c.section].filter(Boolean).join(' · ')}</div></td>
          <td className="muted">{c.running ? `실시간 ${pct(c.live_occupancy)}` : '정지'}</td>
          <td>
            <OccupancyBar value={c.overall?.occupancy} level={c.overall?.level} label="전체" compact />
            {c.directions.map((d: any) => <OccupancyBar key={d.direction_index} value={d.occupancy} level={d.level} label={d.name} color={c.directions_meta?.[d.direction_index] ?? DIRECTION_PALETTE[(d.direction_index - 1) % DIRECTION_PALETTE.length]} compact />)}
          </td>
          <td className="num">{pct(c.overall?.max)}</td>
          <td><LevelBadge level={c.overall?.level} size="sm" /></td>
          <td className="num">{c.overall?.n_vehicles?.toFixed(1)}</td>
          <td>{spark(c.id) && <Sparkline values={spark(c.id)!} width={110} height={24} color={levelColor(c.overall?.level)} />}</td>
        </tr>
      ))}
    </>
  )
}
