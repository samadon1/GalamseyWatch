"use client";

import { useRef, useEffect } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

export default function HeroScene() {
  const containerRef = useRef<HTMLDivElement>(null);
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Pre-check WebGL before touching Three.js. Some browsers (Brave w/ fingerprinting
    // protection, Chrome w/ hardware accel off) refuse WebGL contexts, letting
    // Three.js fail internally floods the console with 3-4 errors per mount.
    // Probe silently returns null when GL is unavailable.
    const probe = document.createElement("canvas");
    if (!probe.getContext("webgl2") && !probe.getContext("webgl")) {
      // Ken Burns bg serves as the fallback hero, exit quietly.
      return;
    }

    let disposed = false;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      35,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.8;
    container.appendChild(renderer.domElement);

    // Environment map for PBR reflections, makes the satellite materials come alive
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    // Lighting, boosted for the dark-MLI Sentinel model
    const ambientLight = new THREE.AmbientLight(0xffffff, 2.2);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 5);
    keyLight.position.set(5, 5, 5);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xbfd4ff, 2.5);
    fillLight.position.set(-5, 3, 5);
    scene.add(fillLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 2);
    backLight.position.set(0, 5, -5);
    scene.add(backLight);

    // Rim light from the front to catch the satellite's silhouette edges
    const rimLight = new THREE.DirectionalLight(0x88bbff, 3);
    rimLight.position.set(0, 0, 8);
    scene.add(rimLight);

    // Load drone model
    let drone: THREE.Group | null = null;
    let droneTargetScale = 1;
    let droneCurrentScale = 0;
    let loadTime = 0;
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);

    loader.load(
      "/sentinel.glb",
      (gltf) => {
        if (disposed) return;
        drone = gltf.scene;

        // Center and scale the model
        const box = new THREE.Box3().setFromObject(drone);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());

        // Scale to fit the wider of the solar-panel wingspan within the viewport.
        // Use the maximum of width/height so the satellite always fits regardless of aspect ratio.
        const maxDim = Math.max(size.x, size.y, size.z);
        droneTargetScale = 4.5 / maxDim;
        drone.scale.setScalar(0); // start invisible; animate in
        droneCurrentScale = 0;

        // Center the model relative to target scale so the entrance tween keeps it centered
        drone.position.x = -center.x * droneTargetScale;
        drone.position.y = -center.y * droneTargetScale; // true vertical center
        drone.position.z = -center.z * droneTargetScale;

        drone.rotation.x = 0.2;
        drone.rotation.y = -1.1;

        loadTime = performance.now() * 0.001;
        scene.add(drone);
      },
      undefined,
      (error) => {
        console.error("Error loading drone model:", error);
      }
    );

    // Mouse tracking
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = (e.clientX / window.innerWidth - 0.5) * 2;
      mouseRef.current.y = (e.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener("mousemove", handleMouseMove);

    // Animation
    let animationId: number;
    const animate = () => {
      const time = performance.now() * 0.001;

      if (drone) {
        // Entrance: scale 0 → target over ~1.2s with ease-out
        if (droneCurrentScale < droneTargetScale) {
          const t = Math.min((time - loadTime) / 1.2, 1);
          const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
          droneCurrentScale = droneTargetScale * eased;
          drone.scale.setScalar(droneCurrentScale);
        }

        // Float, very gentle bob, nudged slightly up and right
        drone.position.y = -0.5 + Math.sin(time * 0.4) * 0.04;
        drone.position.x = 0.4 + Math.sin(time * 0.25) * 0.02;

        // Continuous slow Y rotation, full turn every ~90s
        const baseRotY = -1.1 + time * (Math.PI * 2) / 90;

        // Very subtle roll/pitch oscillation
        const baseRotX = 0.2 + Math.sin(time * 0.3) * 0.03;
        const baseRotZ = Math.sin(time * 0.2) * 0.02;

        // Mouse adds a small modulation on top of the base motion
        const targetRotY = baseRotY + -mouseRef.current.x * 0.08;
        const targetRotX = baseRotX + mouseRef.current.y * 0.05;

        drone.rotation.y += (targetRotY - drone.rotation.y) * 0.03;
        drone.rotation.x += (targetRotX - drone.rotation.x) * 0.03;
        drone.rotation.z += (baseRotZ - drone.rotation.z) * 0.03;
      }

      renderer.render(scene, camera);
      animationId = requestAnimationFrame(animate);
    };
    animate();

    const handleResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", handleResize);

    return () => {
      disposed = true;
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationId);
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      pmrem.dispose();
      renderer.dispose();
      // Explicit context release so HMR/StrictMode remounts don't accumulate
      // GL contexts past the browser's limit (~8–16).
      renderer.forceContextLoss();
    };
  }, []);

  return <div ref={containerRef} className="absolute inset-0" />;
}
