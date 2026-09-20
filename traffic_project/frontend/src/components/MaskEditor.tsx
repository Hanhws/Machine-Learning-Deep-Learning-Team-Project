import { Brush, Eraser, Hand, Minus, MousePointerClick, Pentagon, Plus, Redo2, Scissors, Sparkles, SquareDashedMousePointer, Undo2, Wand2, ZoomIn, ZoomOut } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { API_BASE, api, imageSrc, type Direction } from '../api/client'
import { DIRECTION_PALETTE } from '../lib/format'
import { Spinner } from './ui'

/**
 * 픽셀 단위 도로 마스크 편집기.
 * 상태: 프레임 크기의 Uint8Array 라벨맵 (0=도로 아님, 1..N=방향).
 * 도구: SAM 점/박스/텍스트/자동 제안 · 브러시 · 지우개 · 다각형 · 분할선 · 이동(확대/축소).
 */

export type Tool = 'sam-point' | 'sam-box' | 'brush' | 'eraser' | 'polygon' | 'split' | 'pan'

export interface MaskEditorHandle {
  exportPng: () => string
  label: Uint8Array
  stats: () => Record<number, number>
}

interface Props {
  imageUrl: string
  width: number
  height: number
  snapshotId?: string
  cameraId?: number
  directions: Direction[]
  onDirectionsChange: (d: Direction[]) => void
  initialMaskUrl?: string
  onReady?: (h: MaskEditorHandle) => void
  maxHeight?: number
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

async function decodeMaskPng(b64OrUrl: string, w: number, h: number): Promise<Uint8Array> {
  const img = new Image()
  img.crossOrigin = 'anonymous'
  img.src = b64OrUrl.startsWith('data:') ? b64OrUrl : b64OrUrl.startsWith('http') ? await imageSrc(b64OrUrl) : b64OrUrl.startsWith('/') ? await imageSrc(API_BASE + b64OrUrl) : `data:image/png;base64,${b64OrUrl}`
  await img.decode()
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(img, 0, 0, w, h)
  const d = ctx.getImageData(0, 0, w, h).data
  const out = new Uint8Array(w * h)
  for (let i = 0; i < w * h; i++) out[i] = d[i * 4]
  return out
}

function pointInPoly(x: number, y: number, poly: number[][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** 폴리라인 기준 좌/우 판정: 가장 가까운 선분의 외적 부호 */
function sideOfPolyline(x: number, y: number, line: number[][]): number {
  let best = Infinity
  let sign = 0
  for (let i = 0; i < line.length - 1; i++) {
    const [ax, ay] = line[i]
    const [bx, by] = line[i + 1]
    const dx = bx - ax, dy = by - ay
    const len2 = dx * dx + dy * dy || 1
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2))
    const px = ax + t * dx, py = ay + t * dy
    const d2 = (x - px) ** 2 + (y - py) ** 2
    if (d2 < best) {
      best = d2
      sign = dx * (y - ay) - dy * (x - ax) >= 0 ? 1 : -1
    }
  }
  return sign
}

const TOOL_KEYS: Record<string, Tool> = { s: 'sam-point', x: 'sam-box', b: 'brush', e: 'eraser', p: 'polygon', l: 'split', h: 'pan' }

export default function MaskEditor({ imageUrl, width, height, snapshotId, cameraId, directions, onDirectionsChange, initialMaskUrl, onReady, maxHeight = 640 }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const drawRef = useRef<HTMLCanvasElement>(null)
  const labelRef = useRef<Uint8Array>(new Uint8Array(width * height))
  const undoRef = useRef<Uint8Array[]>([])
  const redoRef = useRef<Uint8Array[]>([])
  const [, force] = useState(0)
  const rerender = () => force((v) => v + 1)

  const [tool, setTool] = useState<Tool>('sam-point')
  const [current, setCurrent] = useState(1)
  const [brush, setBrush] = useState(30)
  const [mergeMode, setMergeMode] = useState<'add' | 'remove'>('add')
  const [points, setPoints] = useState<{ x: number; y: number; label: number }[]>([])
  const [box, setBox] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
  const [poly, setPoly] = useState<number[][]>([])
  const [split, setSplit] = useState<number[][]>([])
  const [splitDirs, setSplitDirs] = useState<[number, number]>([1, 2])
  const [text, setText] = useState('road')
  const [busy, setBusy] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [samStatus, setSamStatus] = useState<{ backend: string; text_prompt: boolean } | null>(null)
  const [opacity, setOpacity] = useState(0.45)
  const [pending, setPending] = useState<Uint8Array | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null)
  const [bgSrc, setBgSrc] = useState('')
  useEffect(() => { imageSrc(imageUrl.startsWith('/') ? API_BASE + imageUrl : imageUrl).then(setBgSrc).catch(() => setBgSrc('')) }, [imageUrl])
  // 뷰 변환
  const [fit, setFit] = useState(1)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const dragging = useRef<{ x: number; y: number; mode: 'paint' | 'box' | 'pan'; sx?: number; sy?: number; px?: number; py?: number } | null>(null)
  const spaceRef = useRef(false)

  // 초기화
  useEffect(() => {
    labelRef.current = new Uint8Array(width * height)
    undoRef.current = []
    redoRef.current = []
    if (initialMaskUrl) decodeMaskPng(initialMaskUrl, width, height).then((m) => { labelRef.current = m; rerender() }).catch(() => rerender())
    else rerender()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height, initialMaskUrl])
  useEffect(() => { api.segment.status().then(setSamStatus).catch(() => setSamStatus(null)) }, [])

  // 컨테이너 크기에 맞춤
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth
      setFit(Math.min(w / width, maxHeight / height))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [width, height, maxHeight])
  const scale = fit * zoom
  const viewH = Math.min(maxHeight, height * fit)

  const stats = useCallback(() => {
    const lab = labelRef.current
    const counts: Record<number, number> = {}
    for (let i = 0; i < lab.length; i++) if (lab[i]) counts[lab[i]] = (counts[lab[i]] || 0) + 1
    return counts
  }, [])
  const exportPng = useCallback(() => {
    const c = document.createElement('canvas')
    c.width = width
    c.height = height
    const ctx = c.getContext('2d')!
    const img = ctx.createImageData(width, height)
    const lab = labelRef.current
    for (let i = 0; i < lab.length; i++) { img.data[i * 4] = lab[i]; img.data[i * 4 + 1] = lab[i]; img.data[i * 4 + 2] = lab[i]; img.data[i * 4 + 3] = 255 }
    ctx.putImageData(img, 0, 0)
    return c.toDataURL('image/png').split(',')[1]
  }, [width, height])
  useEffect(() => { onReady?.({ exportPng, get label() { return labelRef.current }, stats }) }, [exportPng, onReady, stats])

  const colors = useMemo(() => {
    const m: Record<number, [number, number, number]> = {}
    directions.forEach((d) => (m[d.index] = hexToRgb(d.color || DIRECTION_PALETTE[(d.index - 1) % DIRECTION_PALETTE.length])))
    return m
  }, [directions])

  // 오버레이
  useEffect(() => {
    const cv = overlayRef.current
    if (!cv) return
    const ctx = cv.getContext('2d')!
    const img = ctx.createImageData(width, height)
    const lab = labelRef.current
    const a = Math.round(opacity * 255)
    for (let i = 0; i < lab.length; i++) {
      const v = lab[i]
      if (v > 0) { const c = colors[v] ?? [255, 255, 255]; img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = a }
      if (pending && pending[i]) { img.data[i * 4] = 255; img.data[i * 4 + 1] = 230; img.data[i * 4 + 2] = 0; img.data[i * 4 + 3] = 170 }
    }
    ctx.putImageData(img, 0, 0)
  })

  // 가이드
  useEffect(() => {
    const cv = drawRef.current
    if (!cv) return
    const ctx = cv.getContext('2d')!
    ctx.clearRect(0, 0, width, height)
    const lw = 2.5 / scale
    ctx.lineWidth = lw
    points.forEach((p) => { ctx.beginPath(); ctx.arc(p.x, p.y, 8 / scale, 0, Math.PI * 2); ctx.fillStyle = p.label ? '#2fbf71' : '#e5484d'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.stroke() })
    if (box) { ctx.strokeStyle = '#4f8cff'; ctx.setLineDash([8 / scale, 6 / scale]); ctx.strokeRect(Math.min(box.x1, box.x2), Math.min(box.y1, box.y2), Math.abs(box.x2 - box.x1), Math.abs(box.y2 - box.y1)); ctx.setLineDash([]) }
    const drawLine = (pts: number[][], color: string, closed: boolean) => {
      if (!pts.length) return
      ctx.strokeStyle = color
      ctx.beginPath()
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
      if (closed && pts.length > 2) ctx.closePath()
      ctx.stroke()
      pts.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 5 / scale, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill() })
    }
    drawLine(poly, '#ffe600', true)
    drawLine(split, '#ff6ad5', false)
    if (cursor && (tool === 'brush' || tool === 'eraser')) { ctx.beginPath(); ctx.arc(cursor.x, cursor.y, brush / 2, 0, Math.PI * 2); ctx.strokeStyle = tool === 'brush' ? '#fff' : '#e5484d'; ctx.stroke() }
  }, [points, box, poly, split, scale, width, height, cursor, tool, brush])

  const pushUndo = () => { undoRef.current.push(labelRef.current.slice()); if (undoRef.current.length > 30) undoRef.current.shift(); redoRef.current = [] }
  const undo = () => { const p = undoRef.current.pop(); if (p) { redoRef.current.push(labelRef.current); labelRef.current = p; rerender() } }
  const redo = () => { const p = redoRef.current.pop(); if (p) { undoRef.current.push(labelRef.current); labelRef.current = p; rerender() } }

  const toImg = (e: { clientX: number; clientY: number }) => {
    const r = wrapRef.current!.getBoundingClientRect()
    return { x: (e.clientX - r.left - pan.x) / scale, y: (e.clientY - r.top - pan.y) / scale }
  }
  const clampImg = (p: { x: number; y: number }) => ({ x: Math.max(0, Math.min(width - 1, p.x)), y: Math.max(0, Math.min(height - 1, p.y)) })

  const paint = (x: number, y: number, value: number) => {
    const lab = labelRef.current
    const r = brush / 2
    const x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(width - 1, Math.ceil(x + r))
    const y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(height - 1, Math.ceil(y + r))
    for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) if ((xx - x) ** 2 + (yy - y) ** 2 <= r * r) lab[yy * width + xx] = value
  }

  const onPointerDown = (e: React.PointerEvent) => {
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    const raw = toImg(e)
    const p = clampImg(raw)
    if (tool === 'pan' || spaceRef.current || e.button === 1) { dragging.current = { x: 0, y: 0, mode: 'pan', sx: e.clientX, sy: e.clientY, px: pan.x, py: pan.y }; return }
    if (e.button === 2 && tool === 'sam-point') { setPoints((ps) => [...ps, { ...p, label: 0 }]); return }
    if (tool === 'sam-point') setPoints((ps) => [...ps, { ...p, label: e.shiftKey ? 0 : 1 }])
    else if (tool === 'sam-box') { dragging.current = { ...p, mode: 'box' }; setBox({ x1: p.x, y1: p.y, x2: p.x, y2: p.y }) }
    else if (tool === 'brush' || tool === 'eraser') { pushUndo(); dragging.current = { ...p, mode: 'paint' }; paint(p.x, p.y, tool === 'brush' ? current : 0); rerender() }
    else if (tool === 'polygon') setPoly((ps) => [...ps, [p.x, p.y]])
    else if (tool === 'split') setSplit((ps) => [...ps, [raw.x, raw.y]])
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const p = clampImg(toImg(e))
    setCursor(p)
    const d = dragging.current
    if (!d) return
    if (d.mode === 'pan') { setPan({ x: d.px! + (e.clientX - d.sx!), y: d.py! + (e.clientY - d.sy!) }); return }
    if (d.mode === 'box') setBox((b) => (b ? { ...b, x2: p.x, y2: p.y } : b))
    else if (d.mode === 'paint') {
      const n = Math.max(1, Math.ceil(Math.hypot(p.x - d.x, p.y - d.y) / (brush / 4)))
      for (let i = 1; i <= n; i++) paint(d.x + ((p.x - d.x) * i) / n, d.y + ((p.y - d.y) * i) / n, tool === 'brush' ? current : 0)
      dragging.current = { ...d, x: p.x, y: p.y }
      rerender()
    }
  }
  const onPointerUp = () => { dragging.current = null }
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const r = wrapRef.current!.getBoundingClientRect()
    const mx = e.clientX - r.left, my = e.clientY - r.top
    const nz = Math.max(1, Math.min(8, zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)))
    const k = nz / zoom
    let nx = mx - (mx - pan.x) * k, ny = my - (my - pan.y) * k
    if (nz === 1) { nx = 0; ny = 0 }
    setZoom(nz)
    setPan({ x: nx, y: ny })
  }
  const zoomTo = (nz: number) => { nz = Math.max(1, Math.min(8, nz)); if (nz === 1) setPan({ x: 0, y: 0 }); setZoom(nz) }

  // ---- SAM ----
  const ref = () => (snapshotId ? { snapshot_id: snapshotId } : { camera_id: cameraId })
  const toPending = async (b64: string) => {
    const m = await decodeMaskPng(b64, width, height)
    for (let i = 0; i < m.length; i++) m[i] = m[i] > 127 ? 1 : 0
    setPending(m)
  }
  const runSam = async (kind: 'point' | 'box' | 'text' | 'auto') => {
    setErr(null)
    setBusy({ point: 'SAM 점 추론 중…', box: 'SAM 박스 추론 중…', text: 'SAM3 텍스트 추론 중…', auto: '도로 자동 제안 중… (수 초)' }[kind])
    try {
      if (kind === 'text') await toPending((await api.segment.text({ ...ref(), text })).mask_png_base64)
      else if (kind === 'point') await toPending((await api.segment.prompt({ ...ref(), points: points.map((p) => [p.x, p.y]), labels: points.map((p) => p.label) })).mask_png_base64)
      else if (kind === 'box' && box) await toPending((await api.segment.prompt({ ...ref(), boxes: [[Math.min(box.x1, box.x2), Math.min(box.y1, box.y2), Math.max(box.x1, box.x2), Math.max(box.y1, box.y2)]] })).mask_png_base64)
      else if (kind === 'auto') {
        if (samStatus?.text_prompt) await toPending((await api.segment.text({ ...ref(), text: 'road' })).mask_png_base64)
        else {
          // SAM2 대체: 화면 하단 여러 지점에 점 프롬프트를 각각 주고 합집합
          const spots = [[0.3, 0.72], [0.5, 0.8], [0.7, 0.72], [0.4, 0.55], [0.6, 0.55]]
          const union = new Uint8Array(width * height)
          for (const [fx, fy] of spots) {
            const r = await api.segment.prompt({ ...ref(), points: [[fx * width, fy * height]], labels: [1] })
            const m = await decodeMaskPng(r.mask_png_base64, width, height)
            const cov = m.reduce((a, v) => a + (v > 127 ? 1 : 0), 0) / m.length
            if (cov > 0.6) continue // 화면 대부분을 잡은 결과(하늘/전체)는 버림
            for (let i = 0; i < m.length; i++) if (m[i] > 127) union[i] = 1
          }
          setPending(union)
        }
      }
    } catch (e: any) { setErr(e.message) } finally { setBusy(null) }
  }
  const applyPending = (dir?: number) => {
    if (!pending) return
    pushUndo()
    const lab = labelRef.current
    const v = mergeMode === 'remove' ? 0 : (dir ?? current)
    for (let i = 0; i < lab.length; i++) if (pending[i]) lab[i] = v
    setPending(null); setPoints([]); setBox(null)
    rerender()
  }
  const applyPolygon = (onlyRoad: boolean) => {
    if (poly.length < 3) return
    pushUndo()
    const lab = labelRef.current
    const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1])
    const x0 = Math.max(0, Math.floor(Math.min(...xs))), x1 = Math.min(width - 1, Math.ceil(Math.max(...xs)))
    const y0 = Math.max(0, Math.floor(Math.min(...ys))), y1 = Math.min(height - 1, Math.ceil(Math.max(...ys)))
    const v = mergeMode === 'add' ? current : 0
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) { const i = y * width + x; if (onlyRoad && lab[i] === 0) continue; if (pointInPoly(x + 0.5, y + 0.5, poly)) lab[i] = v }
    setPoly([])
    rerender()
  }
  const applySplit = () => {
    if (split.length < 2) return
    pushUndo()
    const lab = labelRef.current
    const [a, b] = splitDirs
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) { const i = y * width + x; if (lab[i] === 0) continue; lab[i] = sideOfPolyline(x + 0.5, y + 0.5, split) > 0 ? a : b }
    setSplit([])
    rerender()
  }
  const clearAll = () => { pushUndo(); labelRef.current.fill(0); rerender() }
  const fillHoles = () => {
    pushUndo()
    const lab = labelRef.current
    const n = directions.length
    const rad = 4
    const pass = (src: Uint8Array, val: number, grow: boolean) => {
      const out = src.slice()
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const i = y * width + x
        if (grow ? src[i] !== val : src[i] !== val) continue
        let hit = false
        for (let dy = -rad; dy <= rad && !hit; dy++) for (let dx = -rad; dx <= rad; dx++) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= width || yy >= height) { if (!grow) { hit = true; break } continue }; const s = src[yy * width + xx]; if (grow ? s === 0 : s !== val) { if (grow) out[yy * width + xx] = val; else { hit = true; break } } }
        if (!grow && hit) out[i] = 0
      }
      return out
    }
    let cur = lab
    for (let v = 1; v <= n; v++) cur = pass(pass(cur, v, true), v, false)
    labelRef.current = cur
    rerender()
  }
  const addDirection = () => { const idx = directions.length + 1; if (idx > 8) return; onDirectionsChange([...directions, { index: idx, name: `방향 ${idx}`, color: DIRECTION_PALETTE[(idx - 1) % DIRECTION_PALETTE.length] }]); setCurrent(idx) }
  const removeDirection = () => { if (directions.length <= 1) return; const idx = directions.length; pushUndo(); const lab = labelRef.current; for (let i = 0; i < lab.length; i++) if (lab[i] === idx) lab[i] = 0; onDirectionsChange(directions.slice(0, -1)); setCurrent(Math.min(current, idx - 1)); rerender() }

  // 단축키
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') return
      if (e.code === 'Space') { spaceRef.current = true; e.preventDefault(); return }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return }
      const k = e.key.toLowerCase()
      if (TOOL_KEYS[k]) setTool(TOOL_KEYS[k])
      else if (k === '[') setBrush((b) => Math.max(4, b - 6))
      else if (k === ']') setBrush((b) => Math.min(200, b + 6))
      else if (/^[1-8]$/.test(k) && directions.some((d) => d.index === +k)) setCurrent(+k)
      else if (k === 'enter') { if (pending) applyPending(); else if (tool === 'polygon' && poly.length >= 3) applyPolygon(false); else if (tool === 'split' && split.length >= 2) applySplit(); else if (tool === 'sam-point' && points.length) runSam('point'); else if (tool === 'sam-box' && box) runSam('box') }
      else if (k === 'escape') { setPending(null); setPoly([]); setSplit([]); setPoints([]); setBox(null) }
    }
    const up = (e: KeyboardEvent) => { if (e.code === 'Space') spaceRef.current = false }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up) }
  })

  const st = stats()
  const total = width * height
  const coverage = Object.values(st).reduce((a, b) => a + b, 0) / total
  const warnings: string[] = []
  if (coverage === 0) warnings.push('도로 영역이 비어 있습니다. SAM 자동 제안이나 다각형으로 도로를 지정하세요.')
  else if (coverage < 0.03) warnings.push('도로 영역이 화면의 3% 미만입니다. 분모가 너무 작으면 점유율이 크게 요동칩니다.')
  directions.forEach((d) => { if (coverage > 0 && !(st[d.index] > 0)) warnings.push(`"${d.name}" 에 할당된 픽셀이 없습니다. 분할선 도구로 나누거나 이 방향을 삭제하세요.`) })
  const cursorStyle = tool === 'pan' ? 'grab' : tool === 'brush' || tool === 'eraser' ? 'none' : tool === 'sam-box' ? 'crosshair' : 'crosshair'
  const TB = ({ t, icon, label, hint }: { t: Tool; icon: React.ReactNode; label: string; hint: string }) => (
    <button className={`sm ${tool === t ? 'active' : ''}`} onClick={() => setTool(t)} title={`${label} (${hint})`}>{icon}{label}</button>
  )

  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="toolbar" style={{ marginBottom: 0 }}>
        <div className="group">
          <TB t="sam-point" icon={<MousePointerClick />} label="SAM 점" hint="S · 클릭 포함, Shift/우클릭 제외" />
          <TB t="sam-box" icon={<SquareDashedMousePointer />} label="SAM 박스" hint="X" />
          <TB t="brush" icon={<Brush />} label="브러시" hint="B" />
          <TB t="eraser" icon={<Eraser />} label="지우개" hint="E" />
          <TB t="polygon" icon={<Pentagon />} label="다각형" hint="P" />
          <TB t="split" icon={<Scissors />} label="분할선" hint="L" />
          <TB t="pan" icon={<Hand />} label="이동" hint="H · Space+드래그" />
        </div>
        <div className="group">
          <span className="muted">브러시</span>
          <input type="range" min={4} max={200} value={brush} onChange={(e) => setBrush(+e.target.value)} style={{ width: 90 }} />
          <span className="muted num" style={{ width: 36 }}>{brush}px</span>
        </div>
        <div className="group">
          <button className="sm icon" onClick={() => zoomTo(zoom / 1.3)} title="축소"><ZoomOut /></button>
          <span className="muted num" style={{ width: 40, textAlign: 'center' }}>{Math.round(zoom * 100)}%</span>
          <button className="sm icon" onClick={() => zoomTo(zoom * 1.3)} title="확대 (휠)"><ZoomIn /></button>
        </div>
        <div className="group">
          <span className="muted">투명도</span>
          <input type="range" min={0.1} max={0.9} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} style={{ width: 70 }} />
        </div>
        <div className="group">
          <button className="sm icon" onClick={undo} disabled={!undoRef.current.length} title="실행취소 (⌘Z)"><Undo2 /></button>
          <button className="sm icon" onClick={redo} disabled={!redoRef.current.length} title="다시실행 (⇧⌘Z)"><Redo2 /></button>
          <button className="sm" onClick={fillHoles} title="작은 구멍/틈을 메웁니다">구멍 메우기</button>
          <button className="sm danger" onClick={clearAll}>전체 지우기</button>
        </div>
      </div>

      <div className="toolbar" style={{ marginBottom: 0 }}>
        <div className="group">
          <button className="sm primary" onClick={() => runSam('auto')} disabled={!!busy} title={samStatus?.text_prompt ? 'SAM3 텍스트 "road"' : 'SAM2: 화면 하단 여러 점으로 추론해 합칩니다'}><Sparkles />도로 자동 제안</button>
          {samStatus?.text_prompt && (<><input value={text} onChange={(e) => setText(e.target.value)} style={{ width: 90 }} /><button className="sm" onClick={() => runSam('text')} disabled={!!busy}><Wand2 />텍스트</button></>)}
        </div>
        <div className="group">
          <span className="muted">점 {points.length}</span>
          <button className="sm" onClick={() => runSam('point')} disabled={!!busy || !points.length}>점으로 추론 <span className="kbd">↵</span></button>
          <button className="sm ghost" onClick={() => setPoints([])} disabled={!points.length}>지우기</button>
        </div>
        <div className="group">
          <button className="sm" onClick={() => runSam('box')} disabled={!!busy || !box}>박스로 추론</button>
          <button className="sm ghost" onClick={() => setBox(null)} disabled={!box}>지우기</button>
        </div>
        <div className="group">
          <span className="muted">다각형 {poly.length}점</span>
          <button className="sm" onClick={() => applyPolygon(false)} disabled={poly.length < 3}>채우기</button>
          <button className="sm" onClick={() => applyPolygon(true)} disabled={poly.length < 3} title="이미 도로인 픽셀만 현재 방향으로 재할당">도로만 재할당</button>
          <button className="sm ghost" onClick={() => setPoly([])} disabled={!poly.length}>취소</button>
        </div>
        <div className="group">
          <span className="muted">분할선 {split.length}점</span>
          <select value={splitDirs[0]} onChange={(e) => setSplitDirs([+e.target.value, splitDirs[1]])} style={{ padding: '2px 22px 2px 6px', fontSize: 12 }}>{directions.map((d) => <option key={d.index} value={d.index}>왼쪽→{d.name}</option>)}</select>
          <select value={splitDirs[1]} onChange={(e) => setSplitDirs([splitDirs[0], +e.target.value])} style={{ padding: '2px 22px 2px 6px', fontSize: 12 }}>{directions.map((d) => <option key={d.index} value={d.index}>오른쪽→{d.name}</option>)}</select>
          <button className="sm" onClick={applySplit} disabled={split.length < 2} title="선을 중앙분리대를 따라 그린 뒤 적용: 도로 픽셀을 선 양쪽 방향으로 나눕니다">적용</button>
          <button className="sm ghost" onClick={() => setSplit([])} disabled={!split.length}>취소</button>
        </div>
        <div className="group">
          <span className="muted">합치기</span>
          <button className={`sm ${mergeMode === 'add' ? 'active' : ''}`} onClick={() => setMergeMode('add')}>추가</button>
          <button className={`sm ${mergeMode === 'remove' ? 'active' : ''}`} onClick={() => setMergeMode('remove')}>제외</button>
        </div>
        {busy && <span className="row muted" style={{ gap: 6 }}><Spinner />{busy}</span>}
        {err && <span className="error">{err}</span>}
      </div>

      {pending && (
        <div className="banner warn" style={{ alignItems: 'center' }}>
          <span className="grow">노란 미리보기 영역({((pending.reduce((a, v) => a + v, 0) / total) * 100).toFixed(1)}%)을 어디에 넣을까요?</span>
          {mergeMode === 'add' ? directions.map((d) => <button key={d.index} className="sm primary" onClick={() => applyPending(d.index)}><span className="dot" style={{ background: d.color, width: 8, height: 8 }} />{d.name} 에 추가</button>) : <button className="sm danger" onClick={() => applyPending()}>도로에서 제외</button>}
          <button className="sm ghost" onClick={() => setPending(null)}>버리기</button>
        </div>
      )}

      <div className="row" style={{ gap: 6 }}>
        <span className="muted">방향 (칠할 라벨, 숫자키로 선택)</span>
        {directions.map((d) => (
          <div key={d.index} className="group row" style={{ gap: 4, padding: '3px 6px', background: 'var(--panel-2)', border: `1px solid ${current === d.index ? 'var(--accent)' : 'var(--border)'}`, borderRadius: 8, cursor: 'pointer' }} onClick={() => setCurrent(d.index)}>
            <span className="kbd">{d.index}</span>
            <input type="color" value={d.color} onChange={(e) => onDirectionsChange(directions.map((x) => (x.index === d.index ? { ...x, color: e.target.value } : x)))} onClick={(e) => e.stopPropagation()} style={{ width: 26, height: 26 }} />
            <input value={d.name} onChange={(e) => onDirectionsChange(directions.map((x) => (x.index === d.index ? { ...x, name: e.target.value } : x)))} onClick={(e) => e.stopPropagation()} style={{ width: 120, padding: '3px 6px' }} />
            <span className="muted num">{(((st[d.index] || 0) / total) * 100).toFixed(1)}%</span>
          </div>
        ))}
        <button className="sm icon" onClick={addDirection} disabled={directions.length >= 8} title="방향 추가"><Plus /></button>
        <button className="sm icon" onClick={removeDirection} disabled={directions.length <= 1} title="마지막 방향 삭제"><Minus /></button>
        <span className="muted" style={{ marginLeft: 'auto' }}>도로 {(coverage * 100).toFixed(1)}% · {width}×{height}</span>
      </div>

      <div
        ref={wrapRef}
        className="editor-wrap"
        style={{ width: '100%', height: viewH, cursor: cursorStyle }}
        onContextMenu={(e) => e.preventDefault()}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => { onPointerUp(); setCursor(null) }}
      >
        <div style={{ position: 'absolute', left: 0, top: 0, transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`, transformOrigin: '0 0', width, height }}>
          <img src={bgSrc} alt="" width={width} height={height} draggable={false} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }} />
          <canvas ref={overlayRef} width={width} height={height} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }} />
          <canvas ref={drawRef} width={width} height={height} style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }} />
        </div>
      </div>
      {warnings.map((w, i) => <div key={i} className="banner warn">{w}</div>)}
      <div className="muted">
        순서: ① <b>도로 자동 제안</b> 또는 SAM 점/박스로 도로를 찾아 미리보기를 방향에 추가 → ② 방향이 2개면 <b>분할선</b>을 중앙분리대를 따라 그리고 적용 → ③ 브러시/지우개로 보정. 휠로 확대, Space+드래그로 이동.
      </div>
    </div>
  )
}
