// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

'use client';

import React, { useEffect, useState } from 'react';
import { Activity, Database, Cpu, Zap, Radio } from 'lucide-react';
import { cn } from '@/lib/utils';

export const SystemStatus = () => {
    const [metrics, setMetrics] = useState({
        plasticity: 0,
        clusters: 4021,
        status: 'IDLE',
        latency: 12
    });

    // Simulation loop for "Bio-Metrics"
    useEffect(() => {
        const interval = setInterval(() => {
            setMetrics(prev => ({
                ...prev,
                plasticity: Math.floor(Math.random() * 20),
                status: Math.random() > 0.7 ? 'OPTIMIZING' : (Math.random() > 0.5 ? 'DREAMING' : 'IDLE'),
                latency: 10 + Math.floor(Math.random() * 5),
                clusters: prev.clusters + (Math.random() > 0.9 ? 1 : 0)
            }));
        }, 800);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-full bg-black/90 border-l border-white/10 p-6 font-mono text-xs flex flex-col relative overflow-hidden">
            {/* Background Decor */}
            <div className="absolute top-0 right-0 p-2 opacity-20 pointer-events-none">
                <Radio className="text-green-500 animate-pulse" size={48} />
            </div>

            <h2 className="text-zinc-500 mb-8 uppercase tracking-[0.2em] text-[10px] border-b border-white/5 pb-2">
                Apeiron // Kernel Telemetry
            </h2>

            {/* Metric 1: Neuroplasticity */}
            <div className="mb-8 group">
                <div className="flex items-center justify-between mb-2 text-zinc-300">
                    <div className="flex items-center gap-2">
                        <Zap size={14} className="text-yellow-500" />
                        <span className="group-hover:text-yellow-400 transition-colors">PLASTICITY_RATE</span>
                    </div>
                    <span className="text-yellow-500/50">{metrics.plasticity} ops/s</span>
                </div>
                <div
                    className="w-full bg-zinc-900 h-1.5 rounded-full overflow-hidden"
                    role="progressbar"
                    aria-label="Plasticity Rate"
                    aria-valuenow={metrics.plasticity}
                    aria-valuemin={0}
                    aria-valuemax={20}
                >
                    <div
                        className="bg-yellow-500 h-full transition-all duration-300 shadow-[0_0_10px_rgba(234,179,8,0.5)]"
                        style={{ width: `${Math.min((metrics.plasticity / 20) * 100, 100)}%` }}
                    />
                </div>
            </div>

            {/* Metric 2: Akashic Memory */}
            <div className="mb-8 group">
                <div className="flex items-center gap-2 mb-2 text-zinc-300">
                    <Database size={14} className="text-blue-500" />
                    <span className="group-hover:text-blue-400 transition-colors">MANIFOLD_CLUSTERS</span>
                </div>
                <div className="text-2xl font-bold text-white tracking-tight tabular-nums">
                    {metrics.clusters.toLocaleString()}
                    <span className="text-[10px] font-normal text-zinc-600 ml-2">BETTI-0</span>
                </div>
            </div>

            {/* Metric 3: Daemon State */}
            <div className={cn(
                "p-4 rounded border transition-all duration-500 mb-8",
                metrics.status === 'OPTIMIZING'
                    ? "border-green-500/30 bg-green-500/5 shadow-[0_0_20px_rgba(34,197,94,0.1)]"
                    : "border-white/5 bg-zinc-900/50"
            )}>
                <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                        <Cpu size={14} className={metrics.status === 'OPTIMIZING' ? "text-green-400" : "text-zinc-500"} />
                        <span className="font-bold text-zinc-300">DAEMON_THREAD</span>
                    </div>
                    {metrics.status === 'DREAMING' && <Activity size={12} className="text-purple-500 animate-bounce" />}
                </div>
                <div className={cn(
                    "text-sm tracking-widest",
                    metrics.status === 'OPTIMIZING' ? "text-green-400 animate-pulse" :
                        metrics.status === 'DREAMING' ? "text-purple-400" : "text-zinc-600"
                )}>
                    [{metrics.status}]
                </div>
            </div>

            {/* Footer / Raw Data */}
            <div className="mt-auto border-t border-white/5 pt-4 space-y-1 text-zinc-700">
                <div className="flex justify-between">
                    <span>KERNEL_LATENCY</span>
                    <span className={metrics.latency < 15 ? "text-green-600" : "text-red-600"}>{metrics.latency}ms</span>
                </div>
                <div className="flex justify-between">
                    <span>UPTIME</span>
                    <span>00:42:12</span>
                </div>
                <div className="flex justify-between">
                    <span>HBM_USAGE</span>
                    <span>0.5% (HOT)</span>
                </div>
            </div>
        </div>
    );
};
