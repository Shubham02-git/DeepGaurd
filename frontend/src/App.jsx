import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import ResultCard from './components/ResultCard'
import HowItWorks from './components/HowItWorks'
import Footer from './components/Footer'
import SolarSystemBackground from './components/SolarSystemBackground'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''

export default function App() {
  const [file, setFile]       = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [model, setModel]     = useState('xception')

  const handleFile = (accepted) => {
    if (!accepted.length) return
    const f = accepted[0]
    setFile(f)
    setResult(null)
    setError(null)

    // Generate preview
    const url = URL.createObjectURL(f)
    setPreview({ url, type: f.type.startsWith('video') ? 'video' : 'image' })
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', file)

    try {
      const { data } = await axios.post(`${API_BASE}/predict?model=${model}`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120_000,
      })
      setResult(data)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Something went wrong'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (preview?.url) URL.revokeObjectURL(preview.url)
  }

  return (
    <div className="min-h-screen flex flex-col" style={{ position: 'relative', zIndex: 1 }}>
      <SolarSystemBackground />
      {/* Dark vignette overlay — deepens edges so UI cards pop */}
      <div style={{
        position: 'fixed', inset: 0, zIndex: 1, pointerEvents: 'none',
        background: 'radial-gradient(ellipse at center, rgba(4,1,10,0.35) 0%, rgba(4,1,10,0.72) 100%)',
      }} />
      {/* All UI */}
      <div style={{ position: 'relative', zIndex: 2, display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Header />

      <main id="upload" className="flex-1 container mx-auto px-4 py-12 max-w-5xl" style={{ scrollMarginTop: '80px' }}>

        {/* Upload + Result */}
        <AnimatePresence mode="wait">
          {result ? (
            <motion.div
              key="result"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.4 }}
            >
              <ResultCard
                result={result}
                preview={preview}
                onReset={handleReset}
              />
            </motion.div>
          ) : (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4 }}
            >
              <UploadZone
                onFile={handleFile}
                preview={preview}
                file={file}
                loading={loading}
                error={error}
                onAnalyze={handleAnalyze}
                onReset={handleReset}
                model={model}
                onModelChange={setModel}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* How it works section */}
        <HowItWorks />
      </main>

      <Footer />

      {/* Drag hint */}
      </div>
    </div>
  )
}
