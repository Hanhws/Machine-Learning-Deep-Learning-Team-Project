import { Area, Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { fmtTime, LEVEL_COLOR, LEVELS } from '../lib/format'

export interface SeriesDef {
  key: string
  name: string
  color: string
  area?: boolean
}

/** 점유율 시계열. thresholds 가 있으면 배경에 혼잡 단계 밴드를 깐다. countKey 로 차량 수 막대를 보조축에 그린다. */
export default function TimeSeriesChart({ data, series, thresholds, height = 260, xIsVideoTime = false, countKey, countName = '차량 수', showLegend = true }: { data: Record<string, any>[]; series: SeriesDef[]; thresholds?: number[]; height?: number; xIsVideoTime?: boolean; countKey?: string; countName?: string; showLegend?: boolean }) {
  if (!data.length) return <div className="muted" style={{ padding: 28, textAlign: 'center' }}>표시할 데이터가 없습니다</div>
  const maxVal = Math.max(0.3, ...data.flatMap((r) => series.map((s) => Number(r[s.key]) || 0)))
  const yMax = Math.ceil(maxVal * 10) / 10
  const bands = thresholds ? [0, ...thresholds, yMax] : []
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: countKey ? 8 : 16, left: 0, bottom: 0 }}>
        <defs>
          {series.map((s) => (
            <linearGradient key={s.key} id={`g-${s.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={s.color} stopOpacity={0.35} />
              <stop offset="100%" stopColor={s.color} stopOpacity={0.02} />
            </linearGradient>
          ))}
        </defs>
        {thresholds && bands.slice(0, -1).map((b, i) => <ReferenceArea key={i} y1={b} y2={bands[i + 1]} yAxisId="occ" fill={LEVEL_COLOR[LEVELS[i]]} fillOpacity={0.05} strokeOpacity={0} />)}
        <CartesianGrid stroke="#253046" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="t" tickFormatter={(v) => fmtTime(v)} stroke="#8b96ad" fontSize={11} minTickGap={36} tickLine={false} />
        <YAxis yAxisId="occ" tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="#8b96ad" fontSize={11} width={44} domain={[0, yMax]} tickLine={false} axisLine={false} />
        {countKey && <YAxis yAxisId="cnt" orientation="right" stroke="#8b96ad" fontSize={11} width={36} tickLine={false} axisLine={false} allowDecimals={false} />}
        <Tooltip
          contentStyle={{ background: '#141924', border: '1px solid #31405a', borderRadius: 8, fontSize: 12 }}
          labelFormatter={(v) => (xIsVideoTime ? `영상 ${fmtTime(v as number)}` : fmtTime(v as string))}
          formatter={(v: any, name: any) => [name === countName ? `${Number(v).toFixed(1)}대` : `${(Number(v) * 100).toFixed(1)}%`, name]}
        />
        {showLegend && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {countKey && <Bar yAxisId="cnt" dataKey={countKey} name={countName} fill="#31405a" barSize={6} isAnimationActive={false} />}
        {series.map((s) =>
          s.area ? (
            <Area key={s.key} yAxisId="occ" type="monotone" dataKey={s.key} name={s.name} stroke={s.color} fill={`url(#g-${s.key})`} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
          ) : (
            <Line key={s.key} yAxisId="occ" type="monotone" dataKey={s.key} name={s.name} stroke={s.color} dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
          ),
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
