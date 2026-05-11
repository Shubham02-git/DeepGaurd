import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function Header() {
  const [scrolled, setScrolled]     = useState(false)
  const [hoveredLink, setHoveredLink] = useState(null)
  const [apiOnline, setApiOnline]   = useState(null)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/health', { signal: AbortSignal.timeout(3000) })
        setApiOnline(res.ok)
      } catch {
        setApiOnline(false)
      }
    }
    check()
    const id = setInterval(check, 8000)
    return () => clearInterval(id)
  }, [])

  const navLinks = [
    { label: 'DETECT',       href: '#upload' },
    { label: 'HOW IT WORKS', href: '#how' },
    { label: 'MODEL',        href: '#model' },
  ]

  return (
    <motion.header
      initial={{ opacity: 0, y: -30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 50,
        background: scrolled ? 'rgba(3,5,12,0.96)' : 'rgba(3,5,12,0.72)',
        backdropFilter: 'blur(28px)',
        borderBottom: scrolled
          ? '1px solid rgba(6,182,212,0.12)'
          : '1px solid rgba(255,255,255,0.05)',
        transition: 'background 0.35s ease, border-color 0.35s ease',
      }}
    >
      {/* Top accent line — magenta core, purple flanks */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '1px',
        background: 'linear-gradient(90deg, transparent 0%, #7c3aed 15%, #c026d3 40%, #ffffff 50%, #c026d3 60%, #7c3aed 85%, transparent 100%)',
        opacity: 0.95,
        boxShadow: '0 0 12px rgba(192,38,211,0.6)',
      }} />

      <div style={{
        maxWidth: '1200px', margin: '0 auto', padding: '0 24px',
        height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>

        {/* ── Logo ─────────────────────────────────────────────────────── */}
        <a href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '14px' }}>

          {/* Premium hexagon icon */}
          <div style={{ position: 'relative', width: '42px', height: '42px', flexShrink: 0 }}>
            {/* Ambient glow layer */}
            <div style={{
              position: 'absolute', inset: '-8px',
              background: 'radial-gradient(ellipse at center, rgba(168,85,247,0.35) 0%, rgba(192,38,211,0.15) 45%, transparent 72%)',
              borderRadius: '50%',
              filter: 'blur(4px)',
            }} />
            <svg viewBox="0 0 44 44" fill="none" style={{ width: '100%', height: '100%', position: 'relative', display: 'block' }}>
              <defs>
                {/* Outer hex fill — deep glass */}
                <linearGradient id="hexFill" x1="22" y1="2" x2="22" y2="42" gradientUnits="userSpaceOnUse">
                  <stop offset="0%"   stopColor="#2d1b69" stopOpacity="0.9" />
                  <stop offset="100%" stopColor="#0d0520" stopOpacity="0.95" />
                </linearGradient>
                {/* Outer hex stroke — white top → magenta bottom */}
                <linearGradient id="hexStroke" x1="22" y1="2" x2="22" y2="42" gradientUnits="userSpaceOnUse">
                  <stop offset="0%"   stopColor="#e2d9f3" stopOpacity="0.9" />
                  <stop offset="45%"  stopColor="#a855f7" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#c026d3" stopOpacity="0.6" />
                </linearGradient>
                {/* Inner hex stroke — cyan */}
                <linearGradient id="innerStroke" x1="22" y1="8" x2="22" y2="36" gradientUnits="userSpaceOnUse">
                  <stop offset="0%"   stopColor="#67e8f9" stopOpacity="0.6" />
                  <stop offset="100%" stopColor="#a855f7" stopOpacity="0.3" />
                </linearGradient>
                {/* Center eye / iris */}
                <radialGradient id="iris" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor="#f0abfc" stopOpacity="1"   />
                  <stop offset="40%"  stopColor="#c026d3" stopOpacity="0.9" />
                  <stop offset="100%" stopColor="#7c3aed" stopOpacity="0.5" />
                </radialGradient>
                {/* Soft glow filter */}
                <filter id="lglow" x="-30%" y="-30%" width="160%" height="160%">
                  <feGaussianBlur stdDeviation="1.4" result="blur" />
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="sglow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="0.8" result="blur" />
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </defs>

              {/* ── Outer hexagon ── */}
              <polygon
                points="22,2 39,11.5 39,32.5 22,42 5,32.5 5,11.5"
                fill="url(#hexFill)"
                stroke="url(#hexStroke)"
                strokeWidth="1.4"
                filter="url(#lglow)"
              />

              {/* ── Inner hexagon (smaller, rotated 30°) ── */}
              <polygon
                points="22,9 32,14.5 32,29.5 22,35 12,29.5 12,14.5"
                fill="none"
                stroke="url(#innerStroke)"
                strokeWidth="0.8"
                filter="url(#sglow)"
              />

              {/* ── Corner tick marks — premium detail ── */}
              {[
                [22, 2,  22, 5.5],
                [39, 11.5, 35.9, 13.3],
                [39, 32.5, 35.9, 30.7],
                [22, 42, 22, 38.5],
                [5,  32.5, 8.1, 30.7],
                [5,  11.5, 8.1, 13.3],
              ].map(([x1,y1,x2,y2], i) => (
                <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke="#c084fc" strokeWidth="1.5" strokeLinecap="round" opacity="0.7" />
              ))}

              {/* ── Scan line — horizontal bar across icon ── */}
              <line x1="12" y1="22" x2="32" y2="22"
                stroke="#67e8f9" strokeWidth="0.5" strokeOpacity="0.35" />

              {/* ── Center iris / eye ── */}
              <circle cx="22" cy="22" r="5.5"
                fill="url(#iris)" filter="url(#lglow)" />
              {/* iris ring */}
              <circle cx="22" cy="22" r="5.5"
                fill="none" stroke="#f0abfc" strokeWidth="0.8" strokeOpacity="0.6" />
              {/* pupil */}
              <circle cx="22" cy="22" r="2.2"
                fill="#0d0520" />
              {/* pupil glint */}
              <circle cx="23.2" cy="20.8" r="0.7"
                fill="#ffffff" opacity="0.8" />
            </svg>
          </div>

          {/* Wordmark */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0px' }}>
              <span style={{
                fontWeight: 900, fontSize: '18px', letterSpacing: '4px',
                color: '#ffffff',
                fontFamily: "'Arial Black', Arial, sans-serif",
                lineHeight: 1, textTransform: 'uppercase',
                textShadow: '0 0 20px rgba(255,255,255,0.25)',
              }}>DEEP</span>
              <span style={{
                fontWeight: 900, fontSize: '18px', letterSpacing: '4px',
                color: '#e879f9',
                fontFamily: "'Arial Black', Arial, sans-serif",
                lineHeight: 1, textTransform: 'uppercase',
                textShadow: '0 0 18px rgba(232,121,249,0.7)',
              }}>GUARD</span>
            </div>
          </div>
        </a>

        {/* ── Nav ──────────────────────────────────────────────────────── */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: '2px' }}>
          {navLinks.map((link) => (
            <a
              key={link.label}
              href={link.href}
              onMouseEnter={() => setHoveredLink(link.label)}
              onMouseLeave={() => setHoveredLink(null)}
              style={{
                position: 'relative',
                padding: '7px 16px',
                fontSize: '10.5px',
                fontWeight: 700,
                letterSpacing: '2px',
                fontFamily: 'monospace',
                color: hoveredLink === link.label ? '#e879f9' : 'rgba(255,255,255,0.38)',
                textDecoration: 'none',
                transition: 'color 0.2s ease',
                borderRadius: '4px',
                background: hoveredLink === link.label ? 'rgba(192,38,211,0.08)' : 'transparent',
                textShadow: hoveredLink === link.label ? '0 0 12px rgba(232,121,249,0.5)' : 'none',
              }}
            >
              {link.label}
              {/* Animated underline — cyan */}
              <motion.div
                animate={{ scaleX: hoveredLink === link.label ? 1 : 0, opacity: hoveredLink === link.label ? 1 : 0 }}
                transition={{ duration: 0.18 }}
                style={{
                  position: 'absolute', bottom: '2px', left: '16px', right: '16px', height: '1px',
                  background: 'linear-gradient(90deg, #c026d3, #7c3aed)',
                  transformOrigin: 'left',
                  boxShadow: '0 0 8px rgba(192,38,211,0.8)',
                }}
              />
            </a>
          ))}
        </nav>

        {/* ── Right — status + GitHub ───────────────────────────────────── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>

          {/* Status pill */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: '7px',
            padding: '5px 13px',
            border: `1px solid ${apiOnline === false ? 'rgba(239,68,68,0.3)' : 'rgba(34,197,94,0.3)'}`,
            borderRadius: '6px',
            background: apiOnline === false ? 'rgba(239,68,68,0.06)' : 'rgba(34,197,94,0.07)',
            transition: 'all 0.4s ease',
          }}>
            <motion.div
              animate={{ opacity: [1, 0.15, 1], scale: [1, 0.85, 1] }}
              transition={{ duration: apiOnline === false ? 0.55 : 0.9, repeat: Infinity, ease: 'easeInOut' }}
              style={{
                width: '7px', height: '7px', borderRadius: '50%',
                background: apiOnline === false ? '#ef4444' : '#22c55e',
                boxShadow: apiOnline === false
                  ? '0 0 8px #ef4444, 0 0 16px rgba(239,68,68,0.4)'
                  : '0 0 8px #22c55e, 0 0 18px rgba(34,197,94,0.5)',
              }}
            />
            <span style={{
              fontSize: '9.5px', fontWeight: 700, letterSpacing: '2px', fontFamily: 'monospace',
              color: apiOnline === false ? 'rgba(239,68,68,0.85)' : 'rgba(34,197,94,0.9)',
            }}>
              {apiOnline === false ? 'SYS OFFLINE' : 'SYS ONLINE'}
            </span>
          </div>

          {/* GitHub */}
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: '36px', height: '36px', borderRadius: '8px',
              border: '1px solid rgba(255,255,255,0.07)',
              background: 'rgba(255,255,255,0.03)',
              color: 'rgba(255,255,255,0.4)',
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.color = '#e879f9'
              e.currentTarget.style.borderColor = 'rgba(192,38,211,0.4)'
              e.currentTarget.style.background = 'rgba(168,85,247,0.1)'
              e.currentTarget.style.boxShadow = '0 0 16px rgba(168,85,247,0.2)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.color = 'rgba(255,255,255,0.4)'
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)'
              e.currentTarget.style.background = 'rgba(255,255,255,0.03)'
              e.currentTarget.style.boxShadow = 'none'
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
          </a>
        </div>

      </div>
    </motion.header>
  )
}
