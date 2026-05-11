import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, ImageIcon, Video, X, Zap, AlertCircle } from 'lucide-react'

const MAX_MB = 200

export default function UploadZone({ onFile, preview, file, loading, error, onAnalyze, onReset, model, onModelChange }) {
  const onDrop = useCallback((accepted) => {
    onFile(accepted)
  }, [onFile])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
    },
    maxFiles: 1,
    maxSize: MAX_MB * 1024 * 1024,
    disabled: loading,
  })

  return (
    <div className="space-y-6">
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`
          relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer overflow-hidden
          ${isDragActive  ? 'border-fuchsia-400 bg-fuchsia-500/10 drop-active' : 'border-purple-500/20 hover:border-purple-400/50 bg-white/[0.02]'}
          ${preview       ? 'border-purple-500/15 cursor-default' : ''}
          ${loading       ? 'opacity-50 pointer-events-none' : ''}
        `}
        style={{ minHeight: preview ? 'auto' : '320px' }}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {preview ? (
            /* Preview */
            <motion.div
              key="preview"
              className="relative"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {preview.type === 'image' ? (
                <img
                  src={preview.url}
                  alt="Preview"
                  className="w-full object-contain rounded-xl"
                  style={{ maxHeight: '420px' }}
                />
              ) : (
                <video
                  src={preview.url}
                  controls
                  className="w-full rounded-xl"
                  style={{ maxHeight: '420px' }}
                />
              )}

              {/* Overlay info */}
              <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent rounded-b-xl">
                <div className="flex items-center gap-2 text-sm text-white/80">
                  {preview.type === 'image' ? <ImageIcon size={14} /> : <Video size={14} />}
                  <span className="truncate">{file?.name}</span>
                  <span className="ml-auto text-white/40 shrink-0">
                    {(file?.size / 1024 / 1024).toFixed(1)} MB
                  </span>
                </div>
              </div>

              {/* Remove button */}
              <button
                onClick={(e) => { e.stopPropagation(); onReset() }}
                className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/60 hover:bg-black/80 flex items-center justify-center transition-colors"
              >
                <X size={14} />
              </button>
            </motion.div>
          ) : (
            /* Empty state */
            <motion.div
              key="empty"
              className="flex flex-col items-center justify-center py-20 px-8 text-center"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <motion.div
                className="relative mb-6"
                animate={isDragActive ? { scale: 1.15 } : { scale: 1 }}
                transition={{ type: 'spring', stiffness: 300 }}
              >
                <div className="w-16 h-16 rounded-2xl bg-purple-500/15 border border-fuchsia-400/25 flex items-center justify-center"
                  style={{ boxShadow: '0 0 30px rgba(192,38,211,0.2)' }}>
                  <Upload size={24} className="text-fuchsia-400" />
                </div>
                {/* Orbit ring */}
                <motion.div
                  className="absolute -inset-3 rounded-3xl border border-fuchsia-500/20"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 10, ease: 'linear', repeat: Infinity }}
                />
                <motion.div
                  className="absolute -inset-6 rounded-3xl border border-purple-500/10"
                  animate={{ rotate: -360 }}
                  transition={{ duration: 18, ease: 'linear', repeat: Infinity }}
                />
              </motion.div>

              <p className="text-lg font-semibold text-white/90 mb-2">
                {isDragActive ? '⚡ Drop it here!' : 'Drag & drop your file'}
              </p>
              <p className="text-sm text-white/40 mb-5">
                or <span className="text-fuchsia-400 underline underline-offset-2 cursor-pointer" style={{ textShadow: '0 0 10px rgba(232,121,249,0.5)' }}>browse</span> to choose a file
              </p>

              {/* Accepted types */}
              <div className="flex items-center gap-3">
                {[['Images', 'JPG PNG BMP WebP', ImageIcon], ['Videos', 'MP4 MOV AVI MKV', Video]].map(
                  ([label, exts, Icon]) => (
                    <div key={label} className="flex items-center gap-2 px-3 py-1.5 rounded-lg glass text-xs text-white/40">
                      <Icon size={12} className="text-white/30" />
                      <span>{label}</span>
                      <span className="font-mono text-white/20">{exts}</span>
                    </div>
                  )
                )}
              </div>

              <p className="mt-4 text-xs text-white/20">Max {MAX_MB}MB</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Drag overlay */}
        <AnimatePresence>
          {isDragActive && (
            <motion.div
              className="absolute inset-0 flex items-center justify-center rounded-2xl"
              style={{ background: 'rgba(168,85,247,0.12)', border: '2px solid rgba(192,38,211,0.6)' }}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div style={{ color: '#e879f9', fontSize: '20px', fontWeight: 700, textShadow: '0 0 20px rgba(232,121,249,0.8)', fontFamily: 'monospace', letterSpacing: '3px' }}>DROP TO ANALYZE</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <AlertCircle size={16} className="shrink-0 mt-0.5" />
            <span>{error}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Model Selector */}
      <div className="flex flex-col gap-2">
        <span className="text-xs text-white/35 font-medium uppercase tracking-widest" style={{ fontFamily: 'monospace' }}>Detection Model</span>
        <div className="flex gap-2">
          {[
            { key: 'xception',       label: 'Xception',        desc: 'Best accuracy' },
            { key: 'ucf',            label: 'UCF',             desc: 'Multi-task'    },
            { key: 'efficientnetb4', label: 'EfficientNetB4',  desc: 'Efficient'     },
          ].map(({ key, label, desc }) => (
            <button
              key={key}
              onClick={() => onModelChange(key)}
              disabled={loading}
              style={model === key ? {
                border: '1px solid rgba(192,38,211,0.55)',
                background: 'rgba(168,85,247,0.15)',
                color: '#e879f9',
                boxShadow: '0 0 16px rgba(168,85,247,0.2)',
              } : {
                border: '1px solid rgba(168,85,247,0.1)',
                background: 'rgba(255,255,255,0.02)',
                color: 'rgba(255,255,255,0.35)',
              }}
              className="flex-1 py-2 px-3 rounded-xl text-xs font-medium transition-all duration-200 hover:border-purple-400/30 hover:text-white/60"
            >
              <div className="font-semibold text-[11px]">{label}</div>
              <div className="opacity-60 text-[10px] mt-0.5">{desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Analyze button */}
      <AnimatePresence>
        {file && !loading && (
          <motion.button
            onClick={onAnalyze}
            className="w-full py-4 rounded-2xl font-semibold text-base relative overflow-hidden group btn-shimmer"
            style={{
              background: 'linear-gradient(135deg, #a855f7 0%, #c026d3 50%, #7c3aed 100%)',
              boxShadow: '0 0 30px rgba(168,85,247,0.4), 0 0 60px rgba(192,38,211,0.15)',
              letterSpacing: '1px',
            }}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
          >
            {/* Shimmer */}
            <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/15 to-transparent -skew-x-12 translate-x-[-150%] group-hover:translate-x-[250%] transition-transform duration-700 shimmer-inner" />
            <span className="relative flex items-center justify-center gap-2">
              <Zap size={18} />
              ANALYZE NOW
            </span>
          </motion.button>
        )}
      </AnimatePresence>

      {/* Loading */}
      <AnimatePresence>
        {loading && (
          <motion.div
            className="w-full py-4 rounded-2xl border flex items-center justify-center gap-3"
            style={{ background: 'rgba(168,85,247,0.08)', borderColor: 'rgba(168,85,247,0.3)' }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <ScanAnimation />
            <span style={{ color: '#d946ef', fontWeight: 600, fontFamily: 'monospace', letterSpacing: '2px', fontSize: '13px' }}>SCANNING…</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function ScanAnimation() {
  return (
    <div className="w-5 h-5 relative">
      <motion.div
        style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: '2px solid transparent', borderTopColor: '#e879f9', boxShadow: '0 0 8px rgba(232,121,249,0.6)' }}
        animate={{ rotate: 360 }}
        transition={{ duration: 0.7, ease: 'linear', repeat: Infinity }}
      />
      <motion.div
        style={{ position: 'absolute', inset: '4px', borderRadius: '50%', border: '1px solid transparent', borderTopColor: '#a855f7' }}
        animate={{ rotate: -360 }}
        transition={{ duration: 1.2, ease: 'linear', repeat: Infinity }}
      />
    </div>
  )
}
