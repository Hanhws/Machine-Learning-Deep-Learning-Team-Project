/**
 * 브라우저 E2E: 업로드 → 마스크 편집기(SAM 점 프롬프트, 다각형, 브러시) → 저장 → 상세 페이지.
 *   node e2e/register.mjs [baseUrl] [videoPath]
 * 설치된 Chrome 을 사용한다 (puppeteer-core).
 */
import puppeteer from 'puppeteer-core'
import path from 'node:path'
import fs from 'node:fs'

const base = process.argv[2] || 'http://localhost:8000'
const video = process.argv[3] || path.resolve('../data/uploads/sample_highway.mp4')
const outDir = process.env.SHOT_DIR || path.resolve('e2e/shots')
fs.mkdirSync(outDir, { recursive: true })
const chrome = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ['--window-size=1440,1100'] })
const page = await browser.newPage()
await page.setViewport({ width: 1440, height: 1100 })
page.on('pageerror', (e) => console.log('PAGE ERROR:', e.message))
page.on('console', (m) => { if (m.type() === 'error') console.log('CONSOLE ERROR:', m.text()) })
const shot = (n) => page.screenshot({ path: path.join(outDir, n + '.png') })
const clickText = async (text, tag = 'button') => {
  const ok = await page.evaluate((t, tg) => {
    const el = [...document.querySelectorAll(tg)].find((b) => b.textContent.trim().startsWith(t))
    if (!el) return false
    el.click()
    return true
  }, text, tag)
  if (!ok) throw new Error('button not found: ' + text)
}

await page.goto(base + '/register', { waitUntil: 'networkidle0' })
await clickText('영상/이미지 업로드')
const input = await page.waitForSelector('input[type=file]')
await input.uploadFile(video)
await page.waitForSelector('.editor-wrap', { timeout: 60000 })
await new Promise((r) => setTimeout(r, 800))
await shot('01_editor_empty')

// SAM 점 프롬프트: 캔버스 하단 중앙 클릭
const canvas = await page.$('.editor-wrap canvas:last-of-type')
const box = await canvas.boundingBox()
await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.85)
await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.6)
await clickText('점으로 추론')
await page.waitForFunction(() => [...document.querySelectorAll('button')].some((b) => b.textContent.trim() === '버리기'), { timeout: 120000 })
await shot('02_sam_pending')
await page.evaluate(() => { const b = [...document.querySelectorAll('button')].find((x) => x.textContent.trim() === '버리기'); b.previousElementSibling.click() }) // "<방향> 에 추가"
await new Promise((r) => setTimeout(r, 300))

// 다각형으로 오른쪽 절반을 방향 2 로 재할당 (먼저 방향 2 선택)
await clickText('다각형')
await page.evaluate(() => { const g = [...document.querySelectorAll('.group')].filter((x) => x.querySelector('input[type=color]'))[1]; g && g.click() })
// (뷰포트 안쪽 좌표만 사용)
for (const [fx, fy] of [[0.55, 0.05], [0.98, 0.05], [0.98, 0.55], [0.55, 0.55]]) await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy)
await clickText('채우기')
await new Promise((r) => setTimeout(r, 300))

// 브러시로 왼쪽 아래 조금 칠하기 (방향 1 선택)
await page.evaluate(() => { const g = [...document.querySelectorAll('.group')].filter((x) => x.querySelector('input[type=color]'))[0]; g && g.click() })
await clickText('브러시')
await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.8)
await page.mouse.down()
await page.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.9, { steps: 10 })
await page.mouse.up()
// 분할선: 화면 가운데 세로선 → 왼쪽 방향1 / 오른쪽 방향2
await clickText('분할선')
await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.1)
await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.6)
await clickText('적용')
await new Promise((r) => setTimeout(r, 300))
await shot('03_editor_edited')

await clickText('다음: 정보 입력')
await page.waitForSelector('.form')
await page.evaluate(() => {
  const inputs = document.querySelectorAll('.form input')
  const set = (el, v) => { const s = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set; s.call(el, v); el.dispatchEvent(new Event('input', { bubbles: true })) }
  set(inputs[0], 'E2E 업로드 카메라'); set(inputs[1], '호남선'); set(inputs[2], '전북 정읍'); set(inputs[3], '태인 졸음쉼터'); set(inputs[4], '126.95'); set(inputs[5], '35.65')
})
await shot('04_form')
await clickText('저장하고 모니터링 시작')
await page.waitForFunction(() => location.pathname.match(/^\/cameras\/\d+$/), { timeout: 60000 })
await new Promise((r) => setTimeout(r, 6000))
await shot('05_detail')
console.log('OK ->', page.url())
await browser.close()
