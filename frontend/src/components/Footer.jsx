import { motion } from 'framer-motion'
import { Shield } from 'lucide-react'

export default function Footer() {
  return (
    <motion.footer
      className="border-t border-white/5 mt-20 py-8"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
    >
      <div className="container mx-auto px-4 max-w-5xl flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-white/25">
        <div className="flex items-center gap-2">
          <Shield size={14} style={{ color: '#a855f7' }} />
          <span>DeepGuard — Deepfake Detection</span>
        </div>
        <p><span className="text-white/40">Xception · EfficientNetB4 · DFD</span></p>
        <p>For research purposes only</p>
      </div>
    </motion.footer>
  )
}
