/* 백엔드 REST 클라이언트.
 * 백엔드 주소(API_BASE) 결정 순서:
 *   1. URL 쿼리 ?api=https://... (한 번 열면 브라우저에 저장됨)
 *   2. 브라우저에 저장된 값 (시스템 페이지 > 백엔드 주소)
 *   3. 빌드 시 환경변수 VITE_API_BASE
 *   4. 비어 있으면 같은 서버의 /api (개발 프록시 / 단일 서버 배포)
 */
const STORAGE_KEY = 'traffic.api_base'
function resolveApiBase(): string {
  let v: string | null = null
  try {
    const q = new URLSearchParams(window.location.search).get('api')
    if (q != null) {
      v = q
      localStorage.setItem(STORAGE_KEY, q)
      const u = new URL(window.location.href)
      u.searchParams.delete('api')
      window.history.replaceState({}, '', u.toString())
    } else v = localStorage.getItem(STORAGE_KEY)
  } catch {
    /* localStorage 사용 불가 */
  }
  if (v == null || v === '') v = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''
  return v.trim().replace(/\/+$/, '')
}
export const API_BASE: string = resolveApiBase()
/** ngrok 무료 도메인은 브라우저 요청에 경고 페이지를 끼워 넣는다. 이 헤더를 붙이면 건너뛴다. */
export const NEEDS_BYPASS = /ngrok/.test(API_BASE)
export const EXTRA_HEADERS: Record<string, string> = NEEDS_BYPASS ? { 'ngrok-skip-browser-warning': '1' } : {}

/** <img src> 로 직접 못 쓰는 경우(ngrok 경고 페이지)를 위해 fetch 로 받아 blob URL 로 바꾼다. */
export async function imageSrc(url: string): Promise<string> {
  if (!NEEDS_BYPASS) return url
  const r = await fetch(url, { headers: EXTRA_HEADERS })
  if (!r.ok) throw new Error(`이미지 요청 실패 ${r.status}`)
  return URL.createObjectURL(await r.blob())
}
export const API_BASE_SOURCE: 'browser' | 'env' | 'same-origin' = (() => {
  try {
    const st = localStorage.getItem(STORAGE_KEY)
    if (st) return 'browser'
  } catch { /* ignore */ }
  return (import.meta.env.VITE_API_BASE as string | undefined) ? 'env' : 'same-origin'
})()
/** 백엔드 주소를 브라우저에 저장하고 새로고침 (빈 값 = 환경변수/같은 서버로 복귀) */
export function setApiBase(v: string) {
  try {
    if (v.trim()) localStorage.setItem(STORAGE_KEY, v.trim())
    else localStorage.removeItem(STORAGE_KEY)
  } catch { /* ignore */ }
  window.location.reload()
}
export const apiUrl = (path: string) => `${API_BASE}${path}`


export interface Direction {
  id?: number
  index: number
  name: string
  color: string
  road?: string | null // 소속 도로 (한 CCTV 에 도로가 여러 개일 때)
  heading_deg?: number | null // 진행 방향 (북=0, 시계방향)
  lat?: number | null // 지도 화살표 위치
  lon?: number | null
}

export interface DirectionLive {
  direction_index: number
  name: string
  occupancy: number
  vehicle_px: number
  road_px: number
  n_vehicles: number
  counts: Record<string, number>
  class_px: Record<string, number>
  level: string | null
  road?: string | null
}

export interface LiveState {
  camera_id: number
  status: string
  error: string | null
  ts: string | null
  seq: number
  infer_ms: number
  fps: number
  video_time: number | null
  directions: DirectionLive[]
  name?: string
  interval_s?: number | null
}

export interface Camera {
  id: number
  name: string
  source_type: 'its' | 'upload' | 'url'
  its_cctv_name: string | null
  its_road_type: string | null
  stream_url: string | null
  stream_url_fetched_at: string | null
  video_path: string | null
  lon: number | null
  lat: number | null
  route: string | null
  region: string | null
  section: string | null
  snapshot_path: string | null
  mask_path: string | null
  frame_width: number | null
  frame_height: number | null
  enabled: boolean
  infer_interval_s: number | null
  meta: Record<string, unknown> | null
  created_at: string
  directions: Direction[]
  running: boolean
  live: LiveState | null
}

export interface ItsCctv {
  name: string
  lon: number
  lat: number
  url: string
  format: string | null
  cctvtype: string | null
  resolution: string | null
  road_type: string
  route: string | null
}

export interface LiveCamera {
  id: number
  name: string
  source_type: string
  lon: number | null
  lat: number | null
  route: string | null
  region: string | null
  section: string | null
  enabled: boolean
  has_mask: boolean
  infer_interval_s: number | null
  directions: Direction[]
  running: boolean
  live: LiveState | null
  occupancy: number | null
  level: string | null
  n_vehicles: number | null
}

export interface LiveAll {
  thresholds: number[]
  levels: string[]
  cameras: LiveCamera[]
}

export interface HistoryPoint {
  t: string | number
  occupancy: number
  max: number
  n_vehicles: number
  n_car: number
  n_bus: number
  n_truck: number
  frames: number
  level: string | null
}

export interface MaskOut {
  mask_png_base64: string
  coverage: number
  width: number
  height: number
  backend: string
}

export interface Job {
  id: number
  camera_id: number
  status: string
  progress: number
  frame_stride: number
  started_at: string | null
  finished_at: string | null
  summary: Record<string, any> | null
  error: string | null
}

export interface Group {
  id: number
  name: string
  kind: string
  description: string | null
  created_at: string
  members: { id: number; camera_id: number; label: string | null; order: number; camera_name: string | null }[]
}

export interface AlertEpisode {
  camera_id: number
  camera_name: string
  direction_index: number
  direction_name: string
  start: string
  end: string
  duration_s: number
  peak: number
  mean: number
  level: string | null
  n_vehicles: number
  ongoing: boolean
}

export interface Capture {
  name: string
  ts: string
  mode: string
  note: string
  occupancy: number | null
  level: string | null
  n_vehicles: number | null
  url: string
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { ...init, headers: { 'Content-Type': 'application/json', ...EXTRA_HEADERS, ...(init?.headers || {}) } })
  if (!r.ok) {
    let msg = r.statusText
    try {
      const j = await r.json()
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch {
      /* ignore */
    }
    throw new ApiError(r.status, msg)
  }
  if (r.status === 204) return undefined as T
  return (await r.json()) as T
}

const json = (body: unknown) => ({ body: JSON.stringify(body) })

export const api = {
  system: () => req<any>(apiUrl('/api/system/status')),
  its: {
    status: () => req<{ configured: boolean }>(apiUrl('/api/its/status')),
    suggestHeading: (p: { lat: number; lon: number; route?: string | null; road_type?: string }) =>
      req<{ axis_deg: number; opposite_deg: number; linearity: number; used_route: boolean; route: string | null; neighbors: { name: string; lat: number; lon: number }[] }>(
        apiUrl(`/api/its/suggest-heading?` + new URLSearchParams(Object.fromEntries(Object.entries(p).filter(([, v]) => v != null && v !== '').map(([k, v]) => [k, String(v)])))),
      ),
    search: (p: { road_type: string; cctv_type: string; min_x: number; max_x: number; min_y: number; max_y: number }) =>
      req<{ count: number; items: ItsCctv[] }>(apiUrl(`/api/its/search?` + new URLSearchParams(Object.fromEntries(Object.entries(p).map(([k, v]) => [k, String(v)])))),
      ),
  },
  snapshots: {
    fromSource: (source: string) =>
      req<{ snapshot_id: string; width: number; height: number; url: string }>(apiUrl('/api/snapshots/from-source'), { method: 'POST', ...json({ source }) }),
    upload: async (file: File) => {
      const fd = new FormData()
      fd.append('file', file)
      const r = await fetch(apiUrl('/api/uploads'), { method: 'POST', body: fd, headers: EXTRA_HEADERS })
      if (!r.ok) throw new ApiError(r.status, (await r.json()).detail)
      return (await r.json()) as { kind: string; path: string; filename: string; snapshot_id: string; width: number; height: number; url: string; info: any }
    },
  },
  segment: {
    status: () => req<{ backend: string; text_prompt: boolean; point_prompt: boolean; box_prompt: boolean }>(apiUrl('/api/segment/status')),
    text: (b: { camera_id?: number; snapshot_id?: string; text: string; conf?: number }) => req<MaskOut>(apiUrl('/api/segment/text'), { method: 'POST', ...json(b) }),
    prompt: (b: { camera_id?: number; snapshot_id?: string; points?: number[][]; labels?: number[]; boxes?: number[][] }) =>
      req<MaskOut>(apiUrl('/api/segment/prompt'), { method: 'POST', ...json(b) }),
  },
  cameras: {
    list: () => req<Camera[]>(apiUrl('/api/cameras')),
    get: (id: number) => req<Camera>(apiUrl(`/api/cameras/${id}`)),
    create: (b: Record<string, unknown>) => req<Camera>(apiUrl('/api/cameras'), { method: 'POST', ...json(b) }),
    update: (id: number, b: Record<string, unknown>) => req<Camera>(apiUrl(`/api/cameras/${id}`), { method: 'PATCH', ...json(b) }),
    remove: (id: number) => req<void>(apiUrl(`/api/cameras/${id}`), { method: 'DELETE' }),
    putDirections: (id: number, directions: Direction[]) => req<Camera>(apiUrl(`/api/cameras/${id}/directions`), { method: 'PUT', ...json(directions) }),
    putMask: (id: number, mask_png_base64: string, directions: Direction[]) =>
      req<Camera>(apiUrl(`/api/cameras/${id}/mask`), { method: 'PUT', ...json({ mask_png_base64, directions }) }),
    start: (id: number) => req<Camera>(apiUrl(`/api/cameras/${id}/start`), { method: 'POST' }),
    stop: (id: number) => req<Camera>(apiUrl(`/api/cameras/${id}/stop`), { method: 'POST' }),
    refreshSnapshot: (id: number) => req<Camera>(apiUrl(`/api/cameras/${id}/snapshot`), { method: 'POST' }),
    analyze: (id: number, stride = 5) => req<Job>(apiUrl(`/api/cameras/${id}/analyze?frame_stride=${stride}`), { method: 'POST' }),
    jobs: (id: number) => req<Job[]>(apiUrl(`/api/cameras/${id}/jobs`)),
    clearSamples: (id: number, source?: string) => req<void>(apiUrl(`/api/cameras/${id}/samples${source ? `?source=${source}` : ''}`), { method: 'DELETE' }),
  },
  captures: {
    list: (id: number) => req<Capture[]>(apiUrl(`/api/cameras/${id}/captures`)),
    create: (id: number, mode: string, note = '') => req<Capture>(apiUrl(`/api/cameras/${id}/captures?mode=${mode}&note=${encodeURIComponent(note)}`), { method: 'POST' }),
    remove: (id: number, name: string) => req<void>(apiUrl(`/api/cameras/${id}/captures/${name}`), { method: 'DELETE' }),
    url: (u: string) => `${API_BASE}${u}`,
  },
  demo: () => req<{ camera_id: number; created: boolean }>(apiUrl('/api/system/demo'), { method: 'POST' }),
  stream: {
    live: (id: number) => req<LiveState>(apiUrl(`/api/stream/${id}/live`)),
    frameUrl: (id: number, mode: string, hud = true) => `${API_BASE}/api/stream/${id}/frame.jpg?mode=${mode}&hud=${hud}&_=${Date.now()}`,
    mjpegUrl: (id: number, mode: string, hud = true, maxFps = 8) => `${API_BASE}/api/stream/${id}/mjpeg?mode=${mode}&hud=${hud}&max_fps=${maxFps}`,
  },
  metrics: {
    live: () => req<LiveAll>(apiUrl('/api/metrics/live')),
    timeline: (minutes: number, bucket: number, direction = 0) =>
      req<{ bucket: number; buckets: string[]; cameras: Record<string, (number | null)[]>; vehicles: Record<string, (number | null)[]>; thresholds: number[] }>(
        apiUrl(`/api/metrics/timeline?minutes=${minutes}&bucket=${bucket}&direction=${direction}`),
      ),
    alerts: (p: { minutes?: number; camera_id?: number; min_level?: number; min_duration?: number } = {}) =>
      req<{ threshold: number; count: number; episodes: AlertEpisode[] }>(
        apiUrl(`/api/metrics/alerts?` + new URLSearchParams(Object.fromEntries(Object.entries(p).filter(([, v]) => v != null).map(([k, v]) => [k, String(v)])))),
      ),
    heatmap: (camera_id: number, days = 7, direction = 0) => req<{ weekdays: string[]; grid: (number | null)[][] }>(apiUrl(`/api/metrics/heatmap?camera_id=${camera_id}&days=${days}&direction=${direction}`)),
    historyCsvUrl: (camera_id: number, minutes: number, source = 'live') => `${API_BASE}/api/metrics/history.csv?camera_id=${camera_id}&minutes=${minutes}&source=${source}`,
    summaryCsvUrl: (by: string, minutes: number) => `${API_BASE}/api/metrics/summary.csv?by=${by}&minutes=${minutes}`,
    history: (p: { camera_id: number; direction?: number; minutes?: number; bucket?: number; source?: string }) =>
      req<{ points: HistoryPoint[]; source: string }>(apiUrl(`/api/metrics/history?` + new URLSearchParams(Object.fromEntries(Object.entries(p).filter(([, v]) => v != null).map(([k, v]) => [k, String(v)])))),
      ),
    summary: (by: string, minutes: number) => req<any>(apiUrl(`/api/metrics/summary?by=${by}&minutes=${minutes}`)),
  },
  apps: {
    hanRiver: () => req<{ name: string; bridges: { name: string; lat: number; lon: number }[] }>(apiUrl('/api/apps/presets/han-river')),
    groups: () => req<Group[]>(apiUrl('/api/apps/groups')),
    group: (id: number) => req<Group>(apiUrl(`/api/apps/groups/${id}`)),
    createGroup: (b: Record<string, unknown>) => req<Group>(apiUrl('/api/apps/groups'), { method: 'POST', ...json(b) }),
    updateGroup: (id: number, b: Record<string, unknown>) => req<Group>(apiUrl(`/api/apps/groups/${id}`), { method: 'PATCH', ...json(b) }),
    deleteGroup: (id: number) => req<void>(apiUrl(`/api/apps/groups/${id}`), { method: 'DELETE' }),
    report: (id: number, minutes: number, bucket = 300) => req<any>(apiUrl(`/api/apps/groups/${id}/report?minutes=${minutes}&bucket=${bucket}`)),
    daily: (id: number, days = 7) => req<any>(apiUrl(`/api/apps/groups/${id}/daily?days=${days}`)),
  },
}
