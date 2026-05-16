// Epsilon-Hollow - Copyright (c) 2024 Teerth Sharma
// SPDX-License-Identifier: Epsilon-Hollow

import { create } from 'zustand';

interface Message {
    id: string;
    sender: 'user' | 'apeiron';
    text: string;
    plasticity?: number;
}

interface ApeironState {
    messages: Message[];
    isConnected: boolean;
    plasticityScore: number;
    pulseTrigger: boolean;
    socket: WebSocket | null;

    connect: () => void;
    sendMessage: (text: string) => void;
    addMessage: (msg: Message) => void;
}

export const useApeironStore = create<ApeironState>((set, get) => ({
    messages: [],
    isConnected: false,
    plasticityScore: 0,
    pulseTrigger: false,
    socket: null,

    connect: () => {
        if (get().socket) return;

        const ws = new WebSocket('ws://127.0.0.1:8000/ws');

        ws.onopen = () => {
            console.log("Connected to APEIRON Kernel");
            set({ isConnected: true });
        };

        ws.onclose = () => {
            console.log("Disconnected. Retrying in 2s...");
            set({ isConnected: false, socket: null });
            setTimeout(() => get().connect(), 2000);
        };

        ws.onerror = (e) => {
            console.error("WS Error", e);
            ws.close(); // Trigger onclose
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                // data matches backend WsOutput { text, plasticity_score, pulse }

                set((state) => ({
                    messages: [...state.messages, {
                        id: Date.now().toString(),
                        sender: 'apeiron',
                        text: data.text,
                        plasticity: data.plasticity_score
                    }],
                    plasticityScore: data.plasticity_score,
                    pulseTrigger: data.pulse
                }));

                // Reset pulse after animation duration
                if (data.pulse) {
                    setTimeout(() => set({ pulseTrigger: false }), 2000);
                }

            } catch (e) {
                console.error("Failed to parse APEIRON message", e);
            }
        };

        set({ socket: ws });
    },

    sendMessage: (text: string) => {
        const { socket, addMessage } = get();
        if (socket && socket.readyState === WebSocket.OPEN) {
            const payload = JSON.stringify({ text, user: "User" });
            socket.send(payload);
            addMessage({ id: Date.now().toString(), sender: 'user', text });
        }
    },

    addMessage: (msg: Message) => set((state) => ({ messages: [...state.messages, msg] })),
}));
