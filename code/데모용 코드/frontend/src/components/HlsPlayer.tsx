import Hls from 'hls.js'
import { useEffect, useRef } from 'react'

/** ITS 원본 HLS 재생 (hls.js). http URL 은 https 페이지에서 차단될 수 있어 개발(http) 환경 기준. */
export default function HlsPlayer({ url, style }: { url: string; style?: React.CSSProperties }) {
  const ref = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    const v = ref.current
    if (!v || !url) return
    let hls: Hls | null = null
    if (Hls.isSupported()) {
      hls = new Hls({ lowLatencyMode: true })
      hls.loadSource(url)
      hls.attachMedia(v)
    } else if (v.canPlayType('application/vnd.apple.mpegurl')) {
      v.src = url
    }
    return () => {
      hls?.destroy()
    }
  }, [url])
  return <video ref={ref} controls autoPlay muted playsInline style={{ width: '100%', background: '#000', borderRadius: 8, ...style }} />
}
