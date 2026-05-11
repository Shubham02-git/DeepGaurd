import { Upload, Cpu, BarChart2 } from 'lucide-react'
import BlurText from './BlurText'

const steps = [
  {
    icon: Upload,
    title: 'Upload',
    desc: 'Drop any image or video file. We support JPG, PNG, MP4, MOV and more.',
    color: 'from-cyan-500 to-blue-600',
    glow: 'rgba(6,182,212,0.3)',
    borderColor: 'rgba(6,182,212,0.25)',
    shadow: 'rgba(6,182,212,0.08)',
  },
  {
    icon: Cpu,
    title: 'Analyze',
    desc: 'Xception and EfficientNetB4 models process each frame, detecting subtle manipulation artifacts.',
    color: 'from-purple-500 to-fuchsia-600',
    glow: 'rgba(168,85,247,0.3)',
    borderColor: 'rgba(168,85,247,0.25)',
    shadow: 'rgba(168,85,247,0.08)',
  },
  {
    icon: BarChart2,
    title: 'Result',
    desc: 'Get a clear verdict — Real or Fake — with confidence score and probability breakdown.',
    color: 'from-fuchsia-500 to-pink-600',
    glow: 'rgba(192,38,211,0.3)',
    borderColor: 'rgba(192,38,211,0.25)',
    shadow: 'rgba(192,38,211,0.08)',
  },
]

const modelStats = [
  ['Architecture',    'Xception · EfficientNetB4'],
  ['Dataset',         'DFD — 3,433 videos'],
  ['Training Frames', '~110,000'],
  ['Framework',       'PyTorch · CUDA 12.4'],
]

export default function HowItWorks() {
  return (
    <section id="how" className="mt-24" style={{ scrollMarginTop: '80px' }}>

      {/* Section header */}
      <div className="text-center mb-10">
        <BlurText
          text="HOW IT WORKS"
          delay={80}
          direction="top"
          stepDuration={0.3}
          animateBy="words"
          className="justify-center text-xs font-semibold tracking-widest text-white/30 uppercase mb-3"
        />
        <BlurText
          text="Three simple steps"
          delay={60}
          direction="top"
          stepDuration={0.4}
          animateBy="words"
          className="justify-center text-3xl font-bold text-white"
        />
      </div>

      {/* Step cards */}
      <div className="grid md:grid-cols-3 gap-5">
        {steps.map(({ icon: Icon, title, desc, color, glow, borderColor, shadow }, i) => (
          <div key={title} style={{ borderRadius: '16px', border: `1px solid ${borderColor}`, boxShadow: `0 0 24px ${shadow}` }}>
              <div className="rounded-2xl glass p-6 relative overflow-hidden">
                {/* Step number */}
                <span className="absolute top-4 right-4 text-4xl font-black text-white/[0.04] font-mono select-none">
                  {String(i + 1).padStart(2, '0')}
                </span>

                {/* Icon */}
                <div
                  className={`w-11 h-11 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center mb-4 shadow-lg`}
                  style={{ boxShadow: `0 8px 24px ${glow}` }}
                >
                  <Icon size={20} className="text-white" />
                </div>

                <BlurText
                  text={title}
                  delay={60}
                  direction="top"
                  stepDuration={0.35}
                  animateBy="words"
                  className="font-bold text-white mb-2"
                />
                <BlurText
                  text={desc}
                  delay={40}
                  direction="bottom"
                  stepDuration={0.3}
                  animateBy="words"
                  className="text-sm text-white/40 leading-relaxed"
                />
              </div>
          </div>
        ))}
      </div>

      {/* Model info */}
      <div id="model" className="mt-10" style={{ scrollMarginTop: '80px', borderRadius: '16px', border: '1px solid rgba(168,85,247,0.22)', boxShadow: '0 0 30px rgba(168,85,247,0.08)' }}>
          <div className="rounded-2xl glass p-6">
            <BlurText
              text="MODEL DETAILS"
              delay={60}
              direction="top"
              stepDuration={0.3}
              animateBy="words"
              className="text-xs font-semibold tracking-widest text-white/30 uppercase mb-4"
            />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {modelStats.map(([k, v], idx) => (
                <div key={k}>
                  <BlurText
                    text={k}
                    delay={50}
                    direction="top"
                    stepDuration={0.28}
                    animateBy="words"
                    className="text-xs text-white/30 mb-1"
                  />
                  <BlurText
                    text={v}
                    delay={50}
                    direction="bottom"
                    stepDuration={0.32}
                    animateBy="words"
                    className="font-semibold text-white/80 text-sm"
                  />
                </div>
              ))}
            </div>
          </div>
      </div>

      {/* Tagline */}
      <div className="mt-10 text-center">
        <BlurText
          text="Powered by deep learning — built to expose what the eye cannot see."
          delay={120}
          direction="bottom"
          stepDuration={0.4}
          animateBy="words"
          className="justify-center text-sm text-white/30 tracking-wide"
        />
      </div>
    </section>
  )
}
