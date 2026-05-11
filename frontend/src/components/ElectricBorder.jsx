import { useEffect, useRef, useCallback } from 'react';

function hexToRgba(hex, alpha = 1) {
  if (!hex) return `rgba(0,0,0,${alpha})`;
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map(c => c + c).join('');
  const int = parseInt(h, 16);
  const r = (int >> 16) & 255;
  const g = (int >> 8) & 255;
  const b = int & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

const ElectricBorder = ({
  children,
  color = '#5227FF',
  speed = 1,
  chaos = 0.12,
  borderRadius = 24,
  className,
  style,
}) => {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const animationRef = useRef(null);
  const timeRef = useRef(0);
  const lastFrameTimeRef = useRef(0);

  const random = useCallback(x => (Math.sin(x * 12.9898) * 43758.5453) % 1, []);

  const noise2D = useCallback((x, y) => {
    const i = Math.floor(x), j = Math.floor(y);
    const fx = x - i, fy = y - j;
    const a = random(i + j * 57), b = random(i + 1 + j * 57);
    const c = random(i + (j + 1) * 57), d = random(i + 1 + (j + 1) * 57);
    const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
    return a * (1 - ux) * (1 - uy) + b * ux * (1 - uy) + c * (1 - ux) * uy + d * ux * uy;
  }, [random]);

  const octavedNoise = useCallback((x, octaves, lacunarity, gain, baseAmplitude, baseFrequency, time, seed, baseFlatness) => {
    let y = 0, amplitude = baseAmplitude, frequency = baseFrequency;
    for (let i = 0; i < octaves; i++) {
      let oa = amplitude;
      if (i === 0) oa *= baseFlatness;
      y += oa * noise2D(frequency * x + seed * 100, time * frequency * 0.3);
      frequency *= lacunarity;
      amplitude *= gain;
    }
    return y;
  }, [noise2D]);

  const getCornerPoint = useCallback((cx, cy, r, startAngle, arcLength, progress) => {
    const angle = startAngle + progress * arcLength;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  }, []);

  const getRoundedRectPoint = useCallback((t, left, top, width, height, radius) => {
    const sw = width - 2 * radius, sh = height - 2 * radius;
    const ca = (Math.PI * radius) / 2;
    const perim = 2 * sw + 2 * sh + 4 * ca;
    const dist = t * perim;
    let acc = 0;

    if (dist <= acc + sw) { const p = (dist - acc) / sw; return { x: left + radius + p * sw, y: top }; } acc += sw;
    if (dist <= acc + ca) { const p = (dist - acc) / ca; return getCornerPoint(left + width - radius, top + radius, radius, -Math.PI / 2, Math.PI / 2, p); } acc += ca;
    if (dist <= acc + sh) { const p = (dist - acc) / sh; return { x: left + width, y: top + radius + p * sh }; } acc += sh;
    if (dist <= acc + ca) { const p = (dist - acc) / ca; return getCornerPoint(left + width - radius, top + height - radius, radius, 0, Math.PI / 2, p); } acc += ca;
    if (dist <= acc + sw) { const p = (dist - acc) / sw; return { x: left + width - radius - p * sw, y: top + height }; } acc += sw;
    if (dist <= acc + ca) { const p = (dist - acc) / ca; return getCornerPoint(left + radius, top + height - radius, radius, Math.PI / 2, Math.PI / 2, p); } acc += ca;
    if (dist <= acc + sh) { const p = (dist - acc) / sh; return { x: left, y: top + height - radius - p * sh }; } acc += sh;
    const p = (dist - acc) / ca;
    return getCornerPoint(left + radius, top + radius, radius, Math.PI, Math.PI / 2, p);
  }, [getCornerPoint]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const octaves = 10, lacunarity = 1.6, gain = 0.7;
    const amplitude = chaos, frequency = 10, baseFlatness = 0;
    const displacement = 60, borderOffset = 60;

    const updateSize = () => {
      const rect = container.getBoundingClientRect();
      const w = rect.width + borderOffset * 2, h = rect.height + borderOffset * 2;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
      ctx.scale(dpr, dpr);
      return { width: w, height: h };
    };

    let { width, height } = updateSize();

    const draw = currentTime => {
      const delta = (currentTime - lastFrameTimeRef.current) / 1000;
      timeRef.current += delta * speed;
      lastFrameTimeRef.current = currentTime;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.scale(dpr, dpr);

      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      const left = borderOffset, top = borderOffset;
      const bw = width - 2 * borderOffset, bh = height - 2 * borderOffset;
      const maxR = Math.min(bw, bh) / 2;
      const r = Math.min(borderRadius, maxR);
      const approxPerim = 2 * (bw + bh) + 2 * Math.PI * r;
      const samples = Math.floor(approxPerim / 2);

      ctx.beginPath();
      for (let i = 0; i <= samples; i++) {
        const progress = i / samples;
        const pt = getRoundedRectPoint(progress, left, top, bw, bh, r);
        const xn = octavedNoise(progress * 8, octaves, lacunarity, gain, amplitude, frequency, timeRef.current, 0, baseFlatness);
        const yn = octavedNoise(progress * 8, octaves, lacunarity, gain, amplitude, frequency, timeRef.current, 1, baseFlatness);
        const dx = pt.x + xn * displacement, dy = pt.y + yn * displacement;
        if (i === 0) ctx.moveTo(dx, dy); else ctx.lineTo(dx, dy);
      }
      ctx.closePath();
      ctx.stroke();
      animationRef.current = requestAnimationFrame(draw);
    };

    const ro = new ResizeObserver(() => { const s = updateSize(); width = s.width; height = s.height; });
    ro.observe(container);
    animationRef.current = requestAnimationFrame(draw);
    return () => { if (animationRef.current) cancelAnimationFrame(animationRef.current); ro.disconnect(); };
  }, [color, speed, chaos, borderRadius, octavedNoise, getRoundedRectPoint]);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-visible isolate ${className ?? ''}`}
      style={{ '--electric-border-color': color, borderRadius, ...style }}
    >
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none z-[2]">
        <canvas ref={canvasRef} className="block" />
      </div>
      <div className="absolute inset-0 rounded-[inherit] pointer-events-none z-0">
        <div className="absolute inset-0 rounded-[inherit] pointer-events-none"
          style={{ border: `2px solid ${hexToRgba(color, 0.6)}`, filter: 'blur(1px)' }} />
        <div className="absolute inset-0 rounded-[inherit] pointer-events-none"
          style={{ border: `2px solid ${color}`, filter: 'blur(4px)' }} />
        <div className="absolute inset-0 rounded-[inherit] pointer-events-none -z-[1] scale-110 opacity-30"
          style={{ filter: 'blur(32px)', background: `linear-gradient(-30deg, ${color}, transparent, ${color})` }} />
      </div>
      <div className="relative rounded-[inherit] z-[1]">{children}</div>
    </div>
  );
};

export default ElectricBorder;
