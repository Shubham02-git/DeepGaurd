import { motion } from 'framer-motion'
import { CheckCircle, AlertTriangle, RotateCcw, Video, ImageIcon, Info } from 'lucide-react'

export default function ResultCard({ result, preview, onReset }) {
  const { label, confidence, prob_real: probReal, prob_fake: probFake, file_type, frames_analyzed: framesCount, model } = result
  const isFake       = label === 'fake'
  const isVideo      = file_type === 'video'

  const accent = isFake
    ? { text: 'text-red-400',  bg: 'bg-red-500/10',  border: 'border-red-500/20',  glow: 'glow-red',    bar: 'bg-red-500',  shadow: 'rgba(239,68,68,0.3)'  }
    : { text: 'text-cyan-300', bg: 'bg-cyan-500/10', border: 'border-cyan-400/20', glow: 'glow-cyan',   bar: 'bg-cyan-400', shadow: 'rgba(6,182,212,0.3)'  }

  return (
    <div className="space-y-6">
      {/* Main verdict card */}
      <motion.div
        className={`rounded-2xl glass border ${accent.border} p-6 relative overflow-hidden`}
        style={{ boxShadow: `0 0 50px ${accent.shadow}, inset 0 0 30px rgba(168,85,247,0.03)` }}
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ type: 'spring', stiffness: 200, damping: 20 }}
      >
        {/* Background glow blob */}
        <div
          className={`absolute -top-16 -right-16 w-64 h-64 rounded-full blur-3xl opacity-20 ${isFake ? 'bg-red-500' : 'bg-green-500'}`}
        />

        <div className="relative flex flex-col md:flex-row gap-6 items-start">
          {/* Preview thumbnail */}
          {preview && (
            <div className="shrink-0 w-full md:w-48 aspect-video rounded-xl overflow-hidden bg-black/30">
              {preview.type === 'image' ? (
                <img src={preview.url} alt="Analyzed" className="w-full h-full object-cover" />
              ) : (
                <video src={preview.url} className="w-full h-full object-cover" muted />
              )}
            </div>
          )}

          {/* Result */}
          <div className="flex-1 space-y-4">
            {/* Verdict */}
            <div className="flex items-center gap-3">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 300, delay: 0.2 }}
              >
                {isFake
                  ? <AlertTriangle size={32} className="text-red-400" />
                  : <CheckCircle   size={32} className="text-green-400" />
                }
              </motion.div>
              <div>
                <motion.p
                  className={`text-4xl font-black tracking-widest ${accent.text}`}
                  style={{ fontFamily: 'monospace', textShadow: isFake ? '0 0 20px rgba(239,68,68,0.5)' : '0 0 20px rgba(6,182,212,0.5)' }}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.25 }}
                >
                  {isFake ? 'DEEPFAKE' : 'AUTHENTIC'}
                </motion.p>
                <p className="text-sm text-white/40 mt-0.5">
                  {isFake
                    ? 'This content appears to be AI-generated or manipulated.'
                    : 'This content appears to be genuine.'}
                </p>
              </div>
            </div>

            {/* Confidence meter */}
            <div>
              <div className="flex justify-between text-xs text-white/50 mb-1.5">
                <span>Confidence</span>
                <span className={`font-mono font-semibold ${accent.text}`}>{confidence}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-white/5 overflow-hidden">
                <motion.div
                  className={`h-full rounded-full ${accent.bar}`}
                  style={{ width: 0 }}
                  animate={{ width: `${confidence}%` }}
                  transition={{ duration: 1, ease: 'easeOut', delay: 0.4 }}
                />
              </div>
            </div>

            {/* Meta */}
            <div className="flex items-center gap-2 flex-wrap">
              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${accent.bg} ${accent.text} border ${accent.border}`}>
                {isVideo ? <Video size={11} /> : <ImageIcon size={11} />}
                {isVideo ? `Video · ${framesCount} frames` : 'Image'}
              </span>
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs glass text-white/40">
                {model ? model.charAt(0).toUpperCase() + model.slice(1) : 'Xception'}
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Probability breakdown */}
      <motion.div
        className="grid grid-cols-2 gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
      >
        {[
          { label: 'Real Probability',  value: probReal, color: 'bg-green-500', text: 'text-green-400', textDim: 'text-green-500/60' },
          { label: 'Fake Probability',  value: probFake, color: 'bg-red-500',   text: 'text-red-400',   textDim: 'text-red-500/60' },
        ].map(({ label, value, color, text, textDim }) => (
          <div key={label} className="rounded-2xl glass p-5 space-y-3">
            <div className="flex justify-between items-baseline">
              <span className="text-xs text-white/40">{label}</span>
              <motion.span
                className={`text-2xl font-black font-mono ${text}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
              >
                {value}%
              </motion.span>
            </div>
            <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${color}`}
                initial={{ width: 0 }}
                animate={{ width: `${value}%` }}
                transition={{ duration: 1, ease: 'easeOut', delay: 0.5 }}
              />
            </div>
          </div>
        ))}
      </motion.div>

      {/* Disclaimer */}
      <motion.div
        className="flex items-start gap-2 p-3 rounded-xl bg-white/[0.02] border border-white/5 text-xs text-white/30"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <Info size={13} className="shrink-0 mt-0.5" />
        AI predictions are not 100% accurate. Always use human judgment for critical decisions.
      </motion.div>

      {/* Reset button */}
      <motion.button
        onClick={onReset}
        className="w-full py-3 rounded-xl border text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2"
        style={{ border: '1px solid rgba(168,85,247,0.2)', background: 'rgba(168,85,247,0.06)', color: 'rgba(255,255,255,0.5)', fontFamily: 'monospace', letterSpacing: '1px' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        whileHover={{ scale: 1.01, borderColor: 'rgba(168,85,247,0.5)', color: '#e879f9', boxShadow: '0 0 20px rgba(168,85,247,0.2)' }}
        whileTap={{ scale: 0.99 }}
      >
        <RotateCcw size={14} />
        ANALYZE ANOTHER FILE
      </motion.button>
    </div>
  )
}
