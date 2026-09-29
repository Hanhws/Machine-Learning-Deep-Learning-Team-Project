import { useEffect, useRef, useState } from 'react'

/** interval(ms) 마다 fn 을 호출해 결과를 상태로 유지. */
export function usePolling<T>(fn: () => Promise<T>, interval: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fnRef = useRef(fn)
  fnRef.current = fn
  useEffect(() => {
    let alive = true
    let timer: number | undefined
    const tick = async () => {
      try {
        const d = await fnRef.current()
        if (alive) {
          setData(d)
          setError(null)
        }
      } catch (e: any) {
        if (alive) setError(e.message)
      }
      if (alive) timer = window.setTimeout(tick, interval)
    }
    tick()
    return () => {
      alive = false
      if (timer) clearTimeout(timer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [interval, ...deps])
  return { data, error, setData }
}
