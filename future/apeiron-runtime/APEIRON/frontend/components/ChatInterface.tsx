// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

import React, { useState, useRef, useEffect } from 'react';
import { useApeiron } from '../hooks/useApeiron';
import { Send, Zap, Cpu } from 'lucide-react';

export default function ChatInterface() {
    const { messages, sendMessage, isLearning, pulseType, thoughts } = useApeiron();
    const [input, setInput] = useState('');
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, thoughts]);

    const getPulseColor = () => {
        switch (pulseType) {
            case 'green': return 'text-green-400 border-green-500/50 shadow-[0_0_20px_rgba(74,222,128,0.5)]';
            case 'blue': return 'text-blue-400 border-blue-500/50 shadow-[0_0_20px_rgba(96,165,250,0.5)]';
            case 'red': return 'text-red-500 border-red-500/50 shadow-[0_0_20px_rgba(248,113,113,0.5)]';
            default: return 'text-gray-300 border-gray-800';
        }
    };

    return (
        <div className="flex h-screen bg-black text-gray-100 font-mono overflow-hidden selection:bg-green-500/30">

            {/* LEFT: System Telemetry (The "Pulse" Visualizer) */}
            <div className={`w-1/4 min-w-[250px] border-r border-gray-800 p-6 transition-all duration-500 hidden md:block ${isLearning ? 'bg-gray-900/30' : 'bg-black'}`}>
                <div className={`flex items-center gap-2 mb-8 transition-colors duration-300 ${isLearning ? 'text-white' : 'text-green-500'}`}>
                    <Cpu className={isLearning ? 'animate-spin' : ''} />
                    <span className="tracking-widest font-bold">APEIRON KERNEL</span>
                </div>

                <div className="space-y-6">
                    <div className={`p-4 rounded border transition-all duration-300 ${getPulseColor()} bg-gray-900/50`}>
                        <div className="text-xs text-gray-500 mb-1">KERNEL STATE</div>
                        <div className="text-lg font-bold animate-pulse">
                            {pulseType === 'green' && 'NEUROPLASTICITY ACTIVE'}
                            {pulseType === 'blue' && 'AKASHIC RECALL'}
                            {pulseType === 'red' && 'COGNITIVE DISSONANCE'}
                            {pulseType === 'none' && 'IDLE'}
                        </div>
                    </div>

                    <div className="p-4 rounded border border-gray-800 bg-gray-900/50 h-64 overflow-hidden flex flex-col">
                        <div className="text-xs text-gray-500 mb-2 border-b border-gray-800 pb-1">THOUGHT STREAM</div>
                        <div className="flex-1 overflow-y-auto font-mono text-xs space-y-2 text-gray-400 scrollbar-hide">
                            {thoughts.map((t, i) => (
                                <div key={i} className="opacity-80">
                                    {t}
                                </div>
                            ))}
                            <div ref={scrollRef} />
                        </div>
                    </div>

                    <div className="p-4 rounded border border-gray-800 bg-gray-900/50">
                        <div className="text-xs text-gray-500 mb-1">CONTEXT DEPTH</div>
                        <div className="text-lg font-bold text-blue-400">
                            {messages.length} Clusters
                        </div>
                    </div>
                </div>
            </div>

            {/* RIGHT: The Chat Stream */}
            <div className="flex-1 flex flex-col relative bg-black">
                {/* Visual Overlay for States */}
                <div className={`absolute top-0 left-0 w-full h-1 transition-all duration-300 pointer-events-none 
                    ${pulseType === 'green' ? 'bg-green-500 opacity-100 shadow-[0_0_30px_rgba(34,197,94,0.8)]' :
                        pulseType === 'blue' ? 'bg-blue-500 opacity-100 shadow-[0_0_30px_rgba(59,130,246,0.8)]' :
                            pulseType === 'red' ? 'bg-red-500 opacity-100 shadow-[0_0_30px_rgba(239,68,68,0.8)]' : 'opacity-0'}`} />

                <div
                    className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6"
                    role="log"
                    aria-live="polite"
                    aria-label="Chat history"
                >
                    {messages.map((msg) => (
                        <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-2xl p-4 rounded-lg border ${msg.isPlasticityEvent
                                ? 'border-green-500/50 bg-green-900/20 text-green-100 shadow-[0_0_15px_rgba(34,197,94,0.2)]'
                                : msg.sender === 'user'
                                    ? 'border-gray-700 bg-gray-800'
                                    : 'border-blue-900/30 bg-blue-900/10'
                                }`}>
                                {msg.isPlasticityEvent && (
                                    <div className="flex items-center gap-2 text-xs text-green-400 mb-2 uppercase tracking-wide">
                                        <Zap size={12} fill="currentColor" />
                                        <span>Weights Updated</span>
                                    </div>
                                )}
                                <p className="whitespace-pre-wrap">{msg.text}</p>
                            </div>
                        </div>
                    ))}
                    <div ref={scrollRef} />
                </div>

                {/* Input Zone */}
                <div className="p-6 border-t border-gray-800 bg-black">
                    {/* Mobile Thought Stream (Optional, minimal version) */}
                    {thoughts.length > 0 && (
                        <div className="md:hidden text-[10px] text-gray-500 mb-2 font-mono truncate">
                            {thoughts[thoughts.length - 1]}
                        </div>
                    )}

                    <form
                        onSubmit={(e) => { e.preventDefault(); sendMessage(input); setInput(''); }}
                        className="flex gap-4"
                    >
                        <input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Inject knowledge into the kernel..."
                            aria-label="Message input"
                            className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 focus:outline-none focus:border-green-500 transition-colors text-white"
                        />
                        <button
                            type="submit"
                            disabled={!input.trim()}
                            aria-label="Send message"
                            title="Send message"
                            className="p-3 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 rounded-lg text-green-500 transition-colors"
                        >
                            <Send size={20} />
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}
