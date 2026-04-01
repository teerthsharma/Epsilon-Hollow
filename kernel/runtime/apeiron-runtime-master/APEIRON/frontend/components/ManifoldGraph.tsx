'use client';

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';

const PointCloud = () => {
    const pointsRef = useRef<THREE.Points>(null!);

    // Generate random manifold points
    const count = 2000;
    const [positions, colors] = useMemo(() => {
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);
        const color = new THREE.Color();

        for (let i = 0; i < count; i++) {
            // Sphere distribution
            const r = 4 * Math.cbrt(Math.random());
            const theta = Math.random() * 2 * Math.PI;
            const phi = Math.acos(2 * Math.random() - 1);

            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);

            positions[i * 3] = x;
            positions[i * 3 + 1] = y;
            positions[i * 3 + 2] = z;

            // Gradient colors (Blue to Cyan)
            color.setHSL(0.6 + Math.random() * 0.1, 0.8, 0.5);
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        return [positions, colors];
    }, [count]);

    useFrame((state) => {
        const time = state.clock.getElapsedTime();
        // Slow rotation ("Thinking")
        pointsRef.current.rotation.y = time * 0.05;
        pointsRef.current.rotation.z = time * 0.02;
    });

    return (
        <points ref={pointsRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={count}
                    array={positions}
                    itemSize={3}
                    args={[positions, 3]}
                />
                <bufferAttribute
                    attach="attributes-color"
                    count={count}
                    array={colors}
                    itemSize={3}
                    args={[colors, 3]}
                />
            </bufferGeometry>
            <pointsMaterial
                size={0.05}
                vertexColors
                transparent
                opacity={0.8}
                sizeAttenuation={true}
                depthWrite={false}
                blending={THREE.AdditiveBlending}
            />
        </points>
    );
};

export const ManifoldGraph = () => {
    return (
        <div className="h-full w-full bg-black relative">
            {/* Overlay Title */}
            <div className="absolute top-4 left-4 z-10 pointer-events-none">
                <h2 className="text-blue-500 font-mono text-[10px] tracking-widest uppercase mb-1">Topological Visualizer</h2>
                <div className="text-white/40 text-[9px]">VIEW: HYPER_MANIFOLD_R128</div>
            </div>

            <Canvas camera={{ position: [0, 0, 8], fov: 60 }}>
                <fog attach="fog" args={['black', 5, 15]} />
                <ambientLight intensity={0.5} />
                <PointCloud />
                <OrbitControls autoRotate autoRotateSpeed={0.5} enableZoom={false} />
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
            </Canvas>
        </div>
    );
};
