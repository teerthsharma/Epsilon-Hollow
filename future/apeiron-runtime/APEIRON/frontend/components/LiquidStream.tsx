// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

type Message = {
    id: number;
    role: 'user' | 'apeiron';
    content: string;
    isHot?: boolean; // Did this update weights?
};

export const LiquidStream = () => {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<Message[]>([
        { id: 1, role: 'apeiron', content: 'Cognitive Runtime Online. Akashic Link Established.' }
    ]);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSend = () => {
        if (!input.trim()) return;

        const userMsg: Message = { id: Date.now(), role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');

        // Simulate Response
        setTimeout(() => {
            const isLearning = Math.random() > 0.5; // Simulate learning trigger
            const responseMsg: Message = {
                id: Date.now() + 1,
                role: 'apeiron',
                content: isLearning ? `Acknowledged. Updating Manifold with new axiom: "${input}".` : `Query resolved from Cluster #42F.`,
                isHot: isLearning
            };
            setMessages(prev => [...prev, responseMsg]);
        }, 1000);
    };

    return (
        <div className="h-full flex flex-col bg-black relative">
            {/* Header */}
            <div className="h-14 border-b border-white/10 flex items-center justify-between px-6">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    <span className="font-mono text-sm text-white tracking-wide">LIQUID_STREAM</span>
                </div>
                <div className="text-zinc-600 text-[10px] font-mono">v1.0 GENESIS</div>
            </div>

            {/* Chat Area */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-6 space-y-6"
                role="log"
                aria-live="polite"
                aria-label="Thought stream"
            >
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={cn(
                            "flex flex-col max-w-[80%]",
                            msg.role === 'user' ? "ml-auto items-end" : "mr-auto items-start"
                        )}
                    >
                        <div className={cn(
                            "p-4 rounded-2xl text-sm leading-relaxed backdrop-blur-sm relative transition-all duration-300",
                            msg.role === 'user'
                                ? "bg-zinc-900 border border-white/10 text-white rounded-tr-sm"
                                : "bg-black border border-green-500/20 text-green-100 rounded-tl-sm shadow-[0_0_15px_rgba(34,197,94,0.05)]"
                        )}>
                            {msg.content}

                            {/* Synapse Firing Effect */}
                            {msg.isHot && (
                                <div className="absolute -top-1 -right-1 w-3 h-3">
                                    <span className="absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75 animate-ping"></span>
                                    <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></span>
                                </div>
                            )}
                        </div>

                        {/* Meta Data */}
                        <div className="flex gap-2 mt-2 text-[10px] items-center text-zinc-600 font-mono uppercase">
                            <span>{new Date().toLocaleTimeString()}</span>
                            {msg.isHot && (
                                <span className="flex items-center gap-1 text-yellow-600">
                                    <Sparkles size={10} />
                                    WEIGHT_UPDATE
                                </span>
                            )}
                        </div>
                    </div>
                ))}

                {/* Ghost Text / Daemon Suggestion */}
                <div className="text-center py-8 opacity-20 hover:opacity-100 transition-opacity duration-700">
                    <h3 className="text-xs text-green-500 font-mono mb-2">[DAEMON_SUGGESTION]</h3>
                    <p className="text-sm text-zinc-400">&quot;Should we optimize the &apos;History&apos; cluster? It appears fragmented.&quot;</p>
                </div>
            </div>

            {/* Input Area */}
            <div className="p-6 pt-0">
                <div className="relative group">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="inject_thought( ... )"
                        aria-label="Inject thought"
                        className="w-full bg-zinc-900/50 border border-white/10 rounded-xl py-4 pl-6 pr-14 text-white placeholder:text-zinc-600 focus:outline-none focus:border-green-500/50 focus:ring-1 focus:ring-green-500/50 transition-all font-mono text-sm"
                    />
                    <button
                        onClick={handleSend}
                        disabled={!input.trim()}
                        aria-label="Send thought"
                        title="Send thought"
                        className="absolute right-2 top-2 p-2 bg-gradient-to-br from-green-600 to-green-900 rounded-lg text-white opacity-80 hover:opacity-100 transition-all hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                    >
                        <Send size={16} />
                    </button>

                    {/* Decorative Line */}
                    <div className="absolute bottom-0 left-6 right-6 h-[1px] bg-gradient-to-r from-transparent via-green-500/20 to-transparent group-focus-within:via-green-500/50 transition-all" />
                </div>
            </div>
        </div>
    );
};
