import { useEffect } from 'react'
import UnicornScene from 'unicornstudio-react'

function removeBadge() {
  // Target by exact img src (what the user confirmed in DevTools)
  document.querySelectorAll('img[src*="assets.unicorn.studio"]').forEach(el => {
    const link = el.closest('a')
    if (link) link.remove()
    else el.remove()
  })
  document.querySelectorAll('a[href*="unicorn.studio"]').forEach(el => el.remove())
}

export default function SolarSystemBackground() {
  useEffect(() => {
    // Inject a <style> tag into <head> at runtime — highest priority, always wins
    const style = document.createElement('style')
    style.id = 'hide-unicorn-badge'
    style.textContent = `
      img[src*="assets.unicorn.studio"],
      img[alt*="Made with unicorn"],
      a[href*="unicorn.studio"],
      a[href*="unicorn.studio"] * {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        pointer-events: none !important;
        position: absolute !important;
        left: -9999px !important;
      }
    `
    document.head.appendChild(style)

    // JS removal as belt-and-suspenders — never stop, SDK may re-inject
    removeBadge()
    const interval = setInterval(removeBadge, 500)
    const observer = new MutationObserver(removeBadge)
    observer.observe(document.body, { childList: true, subtree: true })

    return () => {
      style.remove()
      observer.disconnect()
      clearInterval(interval)
    }
  }, [])

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 0,
      width: '100vw', height: '100vh',
      overflow: 'hidden', pointerEvents: 'none',
      /* clip-path cuts the bottom 60px — hides the badge even inside iframes */
      clipPath: 'inset(0 0 60px 0)',
    }}>
      <UnicornScene
        projectId="YM5lYQzjUaOJu1aL0Drg"
        width="100%"
        height="100%"
        scale={1}
        dpi={1.5}
        sdkUrl="https://cdn.jsdelivr.net/gh/hiunicornstudio/unicornstudio.js@2.1.1/dist/unicornStudio.umd.js"
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  )
}
