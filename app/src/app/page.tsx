"use client";

import { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom";
import Link from "next/link";
import dynamic from "next/dynamic";
import { motion, useInView, useScroll, useTransform, useSpring, AnimatePresence } from "framer-motion";
import { ArrowRight, Menu, X, Moon, Sun } from "lucide-react";

// Start fetching the hero GLB in parallel with the JS bundle instead of after
// Three.js parses. React 19 hoists this to <head> during SSR.
ReactDOM.preload("/sentinel.glb", { as: "fetch", crossOrigin: "anonymous" });

// Defer the 3D hero scene, it imports Three.js (~500KB) and blocks first paint
// if bundled in the initial chunk. Loads as soon as the main thread is idle.
const HeroScene = dynamic(() => import("@/components/HeroScene"), {
  ssr: false,
  loading: () => null,
});

// Scroll Progress Bar
function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 100, damping: 30, restDelta: 0.001 });

  return (
    <motion.div
      style={{ scaleX }}
      className="fixed top-0 left-0 right-0 h-[2px] bg-orange-500 origin-left z-[60]"
    />
  );
}

// Custom Cursor - CSS-based for performance
function CustomCursor() {
  const cursorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Skip on touch devices
    if (window.matchMedia("(pointer: coarse)").matches) {
      return;
    }

    const cursor = cursorRef.current;
    if (!cursor) return;

    const onMouseMove = (e: MouseEvent) => {
      cursor.style.setProperty("--x", `${e.clientX}px`);
      cursor.style.setProperty("--y", `${e.clientY}px`);
    };

    const onMouseEnter = () => cursor.style.opacity = "1";
    const onMouseLeave = () => cursor.style.opacity = "0";

    window.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseenter", onMouseEnter);
    document.addEventListener("mouseleave", onMouseLeave);

    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseenter", onMouseEnter);
      document.removeEventListener("mouseleave", onMouseLeave);
    };
  }, []);

  // Hide on touch devices via CSS
  return (
    <div
      ref={cursorRef}
      className="fixed pointer-events-none z-[100] mix-blend-difference hidden md:block"
      style={{
        left: "var(--x, -100px)",
        top: "var(--y, -100px)",
        transform: "translate(-50%, -50%)",
        opacity: 0,
      }}
    >
      <div className="w-3 h-3 bg-white rounded-full" />
    </div>
  );
}

// Dark Mode Toggle
function DarkModeToggle() {
  const [isDark, setIsDark] = useState(true); // Default to dark

  useEffect(() => {
    // Set dark mode as default on mount
    document.documentElement.classList.add("dark");
    setIsDark(true);
  }, []);

  const toggleDarkMode = () => {
    document.documentElement.classList.toggle("dark");
    setIsDark(!isDark);
  };

  return (
    <button
      onClick={toggleDarkMode}
      className="p-2 text-black/40 hover:text-black dark:text-white/40 dark:hover:text-white transition-colors"
      aria-label="Toggle dark mode"
    >
      {isDark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}

// Parallax Image wrapper
function ParallaxImage({
  src,
  alt,
  className = "",
}: {
  src: string;
  alt: string;
  className?: string;
}) {
  const ref = useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start end", "end start"],
  });
  const y = useTransform(scrollYProgress, [0, 1], ["-5%", "5%"]);

  return (
    <div ref={ref} className="overflow-hidden w-full h-full">
      <motion.img
        src={src}
        alt={alt}
        style={{ y }}
        className={`w-full h-[110%] object-cover ${className}`}
      />
    </div>
  );
}

// Tick mark section divider
function TickDivider() {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });
  const tickCount = 80;

  // Generate random heights and delays with seeded randomness for consistency
  const ticks = Array.from({ length: tickCount }).map((_, i) => {
    const seed = Math.sin(i * 12.9898) * 43758.5453;
    const random = seed - Math.floor(seed);
    const height = 10 + random * 20; // 10-30px height variation
    const delayOffset = (Math.sin(i * 0.5) * 0.5 + 0.5) * 0.3; // Wave-like stagger
    return { height, delayOffset };
  });

  return (
    <div ref={ref} className="w-full py-16 overflow-hidden">
      <div className="relative h-8 flex items-center justify-between">
        {ticks.map((tick, i) => (
          <motion.div
            key={i}
            initial={{ scaleY: 0, opacity: 0 }}
            animate={isInView ? { scaleY: 1, opacity: 0.15 } : { scaleY: 0, opacity: 0 }}
            transition={{
              duration: 0.4,
              delay: (i / tickCount) * 0.6 + tick.delayOffset,
              ease: [0.25, 0.1, 0.25, 1],
            }}
            style={{ height: tick.height }}
            className="w-px bg-orange-500 origin-center"
          />
        ))}
      </div>
    </div>
  );
}

// Staggered text animation - reveals words one by one
function AnimatedText({
  children,
  className = "",
  delay = 0,
  stagger = 0.03
}: {
  children: string;
  className?: string;
  delay?: number;
  stagger?: number;
}) {
  const words = children.split(" ");

  return (
    <span className={className}>
      {words.map((word, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: 0.4,
            delay: delay + (i * stagger),
            ease: [0.25, 0.1, 0.25, 1]
          }}
          className="inline-block mr-[0.25em]"
        >
          {word}
        </motion.span>
      ))}
    </span>
  );
}

// Scroll-triggered staggered text
function AnimatedTextOnScroll({
  children,
  className = "",
  stagger = 0.02
}: {
  children: string;
  className?: string;
  stagger?: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });
  const words = children.split(" ");

  return (
    <span ref={ref} className={className}>
      {words.map((word, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, y: 8 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 8 }}
          transition={{
            duration: 0.35,
            delay: i * stagger,
            ease: [0.25, 0.1, 0.25, 1]
          }}
          className="inline-block mr-[0.25em]"
        >
          {word}
        </motion.span>
      ))}
    </span>
  );
}

// Counter animation for numbers
function Counter({
  value,
  className = ""
}: {
  value: string;
  className?: string;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });
  const [displayValue, setDisplayValue] = useState("00");
  const targetNum = parseInt(value, 10);

  useEffect(() => {
    if (!isInView) return;

    let current = 0;
    const duration = 600; // ms
    const steps = 15;
    const increment = targetNum / steps;
    const stepDuration = duration / steps;

    const timer = setInterval(() => {
      current += increment;
      if (current >= targetNum) {
        setDisplayValue(value);
        clearInterval(timer);
      } else {
        setDisplayValue(String(Math.floor(current)).padStart(2, "0"));
      }
    }, stepDuration);

    return () => clearInterval(timer);
  }, [isInView, targetNum, value]);

  return (
    <span ref={ref} className={className}>
      {displayValue}
    </span>
  );
}

// Typewriter/Scramble effect for section labels (scroll-triggered)
function ScrambleText({
  children,
  className = "",
  delay = 0
}: {
  children: string;
  className?: string;
  delay?: number;
}) {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });
  const [displayText, setDisplayText] = useState("");
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

  useEffect(() => {
    if (!isInView) return;

    const text = children;
    let iteration = 0;
    const totalIterations = text.length * 3; // 3 scramble cycles per character

    const timeout = setTimeout(() => {
      const interval = setInterval(() => {
        setDisplayText(
          text
            .split("")
            .map((char, index) => {
              if (char === " ") return " ";
              if (index < iteration / 3) {
                return text[index];
              }
              return chars[Math.floor(Math.random() * chars.length)];
            })
            .join("")
        );

        iteration++;

        if (iteration > totalIterations) {
          setDisplayText(text);
          clearInterval(interval);
        }
      }, 30);

      return () => clearInterval(interval);
    }, delay * 1000);

    return () => clearTimeout(timeout);
  }, [isInView, children, delay]);

  return (
    <span ref={ref} className={className}>
      {displayText || children.split("").map(() => " ").join("")}
    </span>
  );
}

// Applications section with image switcher
function ApplicationsSection() {
  const [activeIndex, setActiveIndex] = useState(0);
  const sectors = [
    {
      sector: "Energy",
      examples: "Pipeline networks, power transmission, substations, solar and wind farms",
      image: "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?w=1600&q=80"
    },
    {
      sector: "Transportation",
      examples: "Bridges, rail corridors, highway systems, port infrastructure",
      image: "https://images.unsplash.com/photo-1545558014-8692077e9b5c?w=1600&q=80"
    },
    {
      sector: "Industrial",
      examples: "Refineries, tank farms, manufacturing facilities, storage terminals",
      image: "https://images.unsplash.com/photo-1586953208448-b95a79798f07?w=1600&q=80"
    },
    {
      sector: "Utilities",
      examples: "Water systems, telecom towers, distribution networks",
      image: "https://images.unsplash.com/photo-1558346490-a72e53ae2d4f?w=1600&q=80"
    },
  ];

  return (
    <section className="min-h-[70vh] relative overflow-hidden flex flex-col">
      {/* Background Images */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeIndex}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          className="absolute inset-0"
        >
          <img
            src={sectors[activeIndex].image}
            alt={sectors[activeIndex].sector}
            className="w-full h-full object-cover"
          />
          {/* Gradient overlay */}
          <div className="absolute inset-0 bg-gradient-to-t from-white via-white/80 to-white/40 dark:from-[#1F2124] dark:via-[#1F2124]/80 dark:to-[#1F2124]/40" />
        </motion.div>
      </AnimatePresence>

      {/* Left-centered content */}
      <div className="relative z-10 flex-1 flex items-center px-4 md:px-8">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="cursor-none text-left"
        >
          <p className="text-xs text-orange-500 mb-4 font-mono tracking-widest uppercase font-medium">
            <ScrambleText>Applications</ScrambleText>
          </p>
          <h2 className="text-2xl md:text-4xl font-medium tracking-tight mb-4">
            <AnimatedTextOnScroll>
              Where we operate
            </AnimatedTextOnScroll>
          </h2>
          {/* Active sector description */}
          <AnimatePresence mode="wait">
            <motion.p
              key={activeIndex}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="text-base text-black/60 dark:text-white/60 leading-relaxed max-w-md"
            >
              {sectors[activeIndex].examples}
            </motion.p>
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Bottom sector switcher - full width */}
      <div className="relative z-10 border-t border-black/10 dark:border-white/10">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-4 cursor-none">
            {sectors.map((item, i) => (
              <button
                key={i}
                onClick={() => setActiveIndex(i)}
                className={`py-6 text-sm font-medium transition-all duration-300 cursor-none relative border-r border-black/10 dark:border-white/10 last:border-r-0 ${
                  activeIndex === i
                    ? "text-black dark:text-white bg-black/5 dark:bg-white/5"
                    : "text-black/40 dark:text-white/40 hover:text-black/70 dark:hover:text-white/70 hover:bg-black/[0.02] dark:hover:bg-white/[0.02]"
                }`}
              >
                {item.sector}
                {activeIndex === i && (
                  <motion.span
                    layoutId="activeSector"
                    className="absolute bottom-0 left-0 right-0 h-[2px] bg-orange-500"
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                  />
                )}
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

// Scan reveal button - hover triggered
function GlitchButton({
  children,
  href,
  className = ""
}: {
  children: React.ReactNode;
  href: string;
  className?: string;
}) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1.1, duration: 0.5 }}
      className="relative inline-block overflow-hidden"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Scan line effect on hover */}
      <motion.div
        initial={{ x: "-100%" }}
        animate={{ x: isHovered ? "100%" : "-100%" }}
        transition={{ duration: 0.5, ease: [0.25, 0.1, 0.25, 1] }}
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent z-10 pointer-events-none"
      />
      <a
        href={href}
        className={`relative inline-flex items-center gap-2 ${className}`}
      >
        {children}
      </a>
    </motion.div>
  );
}

type ModalKey = "problem" | "solution";

export default function LandingPage() {
  const [activeModal, setActiveModal] = useState<ModalKey | null>(null);

  // ESC closes the modal
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setActiveModal(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);


  return (
    <div className="min-h-screen bg-white dark:bg-[#1F2124] text-black dark:text-white transition-colors duration-300">
      {/* Scroll Progress */}

      {/* Noise Overlay */}
      <div className="fixed inset-0 pointer-events-none z-[1] opacity-[0.015] dark:opacity-[0.03]">
        <div className="absolute inset-0 bg-noise" />
      </div>

      {/* Grid Background */}
      <div className="fixed inset-0 pointer-events-none z-[0] opacity-[0.03] dark:opacity-[0.05]">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
              linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px)
            `,
            backgroundSize: "60px 60px",
          }}
        />
      </div>

      {/* Minimal nav */}
      <nav className="fixed top-6 left-0 right-0 z-50 px-4 md:px-10 flex items-center justify-between">
        <Link href="/" className="text-sm font-semibold tracking-tight text-white">
          GalamseyWatch
        </Link>
        <div className="flex items-center gap-3 md:gap-6">
          {(["problem", "solution"] as ModalKey[]).map((key) => (
            <button
              key={key}
              onClick={() => setActiveModal(key)}
              className="relative text-xs uppercase tracking-wider text-white/60 hover:text-white transition-colors capitalize after:content-[''] after:absolute after:left-0 after:-bottom-1 after:h-[2px] after:w-0 hover:after:w-full after:transition-all after:duration-300 after:bg-gradient-to-r after:from-[#FFE88C] after:via-[#F5C518] after:to-[#B8860B]"
            >
              {key}
            </button>
          ))}
        </div>
      </nav>

      {/* Hero */}
      <section className="min-h-screen relative overflow-hidden pt-14">
        {/* Background image with slow zoom (Ken Burns) */}
        <div className="absolute inset-0 z-0 overflow-hidden">
          <motion.img
            src="https://images.unsplash.com/photo-1534996858221-380b92700493?q=80&w=2662&auto=format&fit=crop"
            alt="Earth from space"
            className="absolute inset-0 w-full h-full object-cover"
            initial={{ scale: 1 }}
            animate={{ scale: 1.15 }}
            transition={{ duration: 30, ease: "easeInOut", repeat: Infinity, repeatType: "reverse" }}
          />
          {/* Dark overlay */}
          <div className="absolute inset-0 bg-black/60" />
        </div>

        {/* Full-bleed 3D satellite */}
        <div className="absolute inset-0 z-[1]">
          <HeroScene />
        </div>

        {/* Headline anchored to very bottom-left */}
        <div className="absolute bottom-28 md:bottom-8 left-4 md:left-8 right-4 md:right-auto z-10 max-w-4xl">
          <h1 className="text-[clamp(2.25rem,6vw,4.5rem)] font-medium tracking-tight leading-[1.05] text-white mb-6">
            <AnimatedText delay={0.2} stagger={0.06}>
              Illegal mining,
            </AnimatedText>
            <br />
            <AnimatedText delay={0.5} stagger={0.06}>
              caught from space
            </AnimatedText>
          </h1>
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.6 }}
            className="text-sm md:text-base text-white/70 leading-relaxed max-w-2xl"
          >
            We fine-tuned Liquid AI&apos;s LFM2.5-VL-450M to spot illegal
            gold-mining pits in 10 m/px Sentinel-2 imagery. It runs entirely
            in your browser on WebGPU.
          </motion.p>
        </div>

        {/* Start mission button, far right */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.5, duration: 0.6 }}
          className="absolute bottom-8 left-4 right-4 md:left-auto md:right-8 z-10"
        >
          <Link
            href="/dashboard"
            className="inline-flex items-center justify-center md:justify-start gap-2 text-sm font-medium text-black w-full md:w-auto px-8 py-4 bg-gradient-to-br from-[#FFE88C] via-[#F5C518] to-[#B8860B] hover:from-[#FFF4B3] hover:via-[#FFD94A] hover:to-[#D4A017] transition-all shadow-[0_4px_20px_rgba(245,197,24,0.35)]"
          >
            Start mission
            <ArrowRight size={14} />
          </Link>
        </motion.div>
      </section>

      <Modal activeKey={activeModal} onClose={() => setActiveModal(null)} />
    </div>
  );
}

// Modal + content
const MODAL_CONTENT: Record<ModalKey, { title: string; body: React.ReactNode }> = {
  problem: {
    title: "The problem",
    body: (
      <>
        <p>
          <strong className="text-white">Galamsey</strong>, illegal small-scale
          gold mining, has stripped{" "}
          <strong className="text-white">4,700+ hectares</strong> of
          Ghana&apos;s forest reserves in the last decade. Rivers that supply
          drinking water to millions run turbid with mercury. Cocoa farms, the
          country&apos;s second-largest export, are being abandoned as topsoil
          is dug up for pits.
        </p>
        <p>
          The imagery to find these sites already exists. Sentinel-2 covers all
          of Ghana every five days, free of charge. The bottleneck is people:
          trained analysts with GIS tooling, reviewing tile after tile by hand.
          Sites are remote, move fast, and hide in petabytes of unlabeled scenes.
        </p>
        <p className="text-white/90">
          We want to put a galamsey detector in every enforcement
          officer&apos;s laptop. Nothing to install. No cloud to pay for. No
          imagery leaving the device.
        </p>
      </>
    ),
  },
  solution: {
    title: "How it works",
    body: (
      <>
        <p>
          Click a point on the Ghana map. We pull the clearest Sentinel-2 tile
          around that point (1.28 km wide, 10 m/px) and feed it, as paired RGB
          and SWIR false-color composites, to a fine-tuned vision-language
          model. The model returns both a natural-language description of the
          scene and a JSON list of bounding boxes for every mining pit it
          identifies.
        </p>
        <p>
          The base model is <strong className="text-white">LFM2.5-VL-450M</strong>,
          Liquid AI&apos;s open-weights 450M-parameter VLM. We fine-tuned it on{" "}
          <em>SmallMinesDS</em> (Ofori-Ampofo et al., 2025), 4,270 labeled
          Ghana patches, using two-image RGB + SWIR inputs, 8× D4 geometric
          augmentation, and three epochs of full fine-tuning on a single H100.
          Final weights exported to ONNX (fp16) for browser runtime.
        </p>
        <p>
          At inference, the model runs{" "}
          <strong className="text-white">entirely in your browser</strong> via
          WebGPU and transformers.js. A ~1 GB one-time download, then cached
          for every subsequent visit. Tiles stream from our SimSat wrapper over
          the public STAC Sentinel-2 archive. After that first fetch, nothing
          leaves your device.
        </p>
      </>
    ),
  },
};

function Modal({
  activeKey,
  onClose,
}: {
  activeKey: ModalKey | null;
  onClose: () => void;
}) {
  const content = activeKey ? MODAL_CONTENT[activeKey] : null;
  return (
    <AnimatePresence>
      {content && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-[100] bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.98 }}
            transition={{ duration: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
            className="bg-[#17191c] border border-white/10 rounded-lg max-w-2xl w-full max-h-[85vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between px-8 pt-8 pb-4">
              <h2 className="text-2xl font-medium tracking-tight text-white">
                {content.title}
              </h2>
              <button
                onClick={onClose}
                aria-label="Close"
                className="text-white/40 hover:text-white transition-colors p-1 -mr-1"
              >
                <X size={20} />
              </button>
            </div>
            <div className="px-8 pb-8 space-y-4 text-sm leading-relaxed text-white/75">
              {content.body}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
