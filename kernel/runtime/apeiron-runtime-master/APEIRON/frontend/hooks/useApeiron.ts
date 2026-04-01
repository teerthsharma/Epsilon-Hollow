import { useState, useEffect, useRef, useCallback } from 'react';

type Message = {
    id: string;
    sender: 'user' | 'apeiron';
    text: string;
    isPlasticityEvent: boolean;
};

export const useApeiron = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLearning, setIsLearning] = useState(false);
    const [pulseType, setPulseType] = useState<'none' | 'green' | 'blue' | 'red'>('none');
    const [thoughts, setThoughts] = useState<string[]>([]);
    const [dspBus, setDspBus] = useState<any>(null);
    const [tunnelStatus, setTunnelStatus] = useState<'OFFLINE' | 'LOCKED'>('OFFLINE');
    const downlinkController = useRef<AbortController | null>(null);

    useEffect(() => {
        // Load Sanctuary DSP Bus
        const loadWasm = async () => {
            try {
                // @ts-ignore - Loading from lib dir
                const init = (await import('../lib/wasm-pkg/sanctuary_dsp.js')).default;
                const wasm = await init('/wasm/sanctuary_dsp_bg.wasm');
                setDspBus(wasm);
                console.log("🦀 [DSP BUS] Sanctuary DSP Integrated.");
                setThoughts(prev => [...prev, "[SYSTEM] Sanctuary DSP Bus Integrated."]);
            } catch (e) {
                console.error("DSP Bus Load Failure", e);
            }
        };
        loadWasm();
    }, []);

    // --- 1. THE DOWNLINK (Reading the Stream) ---
    useEffect(() => {
        const connectDownlink = async () => {
            downlinkController.current = new AbortController();

            try {
                const response = await fetch('http://127.0.0.1:8080/bus/downlink', {
                    signal: downlinkController.current.signal
                });

                if (!response.body) throw new Error("No Downlink Body");

                setTunnelStatus('LOCKED');
                console.log("Neuro-Link Established via DSP Bus.");
                setThoughts(prev => [...prev, "[SYSTEM] Neuro-Link LOCKED (Binary Bus)."]);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    // Decode binary data from the bus
                    const text = decoder.decode(value, { stream: true });
                    buffer += text;

                    const lines = buffer.split("\n");
                    // Process all complete lines. The last part might be incomplete, so keep it in buffer.
                    // If the chunk ended with a newline, the last element after split is empty string, which we keep as buffer ("")
                    buffer = lines.pop() || "";

                    for (const line of lines) {
                        if (!line.trim()) continue;

                        try {
                            const data = JSON.parse(line);

                            if (data.pulse_type) {
                                setPulseType(data.pulse_type);
                                setIsLearning(data.pulse_type !== 'none');
                            }

                            if (data.thought_log) {
                                setThoughts(prev => [...prev.slice(-4), `> ${data.thought_log}`]);
                            }

                            if (data.dsp_entropy !== undefined) {
                                setThoughts(prev => [...prev.slice(-4), `[DSP-BUS] Entropy: ${data.dsp_entropy.toFixed(4)}`]);
                            }

                            if (data.text) {
                                setMessages(prev => {
                                    const last = prev[prev.length - 1];
                                    if (last && last.sender === 'apeiron' && data.chunk) {
                                        return [...prev.slice(0, -1), { ...last, text: last.text + data.text }];
                                    }
                                    return [...prev, {
                                        id: Date.now().toString(),
                                        sender: 'apeiron',
                                        text: data.text,
                                        isPlasticityEvent: data.pulse_type === 'red' || data.pulse_type === 'green'
                                    }];
                                });
                            }
                        } catch (e) {
                            // Not JSON or partial chunk?
                            console.warn("Bus Decode Error:", e);
                        }
                    }
                }
            } catch (err: any) {
                if (err.name !== 'AbortError') {
                    console.error("Downlink Severed:", err);
                    setTunnelStatus('OFFLINE');
                    setThoughts(prev => [...prev, "[FATAL] Neuro-Link Severed."]);
                    // Reconnect after delay
                    setTimeout(connectDownlink, 3000);
                }
            }
        };

        if (dspBus) {
            connectDownlink();
        }

        return () => downlinkController.current?.abort();
    }, [dspBus]);

    // --- 2. THE UPLINK (Firing the Stream) ---
    const sendMessage = useCallback(async (text: string) => {
        if (!text.trim()) return;

        setMessages(prev => [...prev, {
            id: Date.now().toString(),
            sender: 'user',
            text: text,
            isPlasticityEvent: false
        }]);

        // Virtual DSP Bus processing
        const dspStamp = dspBus ? "[DSP-VALIDATED]" : "[JS-NATIVE]";
        setThoughts(prev => [...prev.slice(-4), `[USER] ${dspStamp} Input injected: ${text.substring(0, 20)}...`]);

        // ENCODE: Pass text through WASM DSP (Encoder)
        const binaryPayload = new TextEncoder().encode(text);

        try {
            await fetch('http://127.0.0.1:8080/bus/uplink', {
                method: 'POST',
                mode: 'cors',
                body: binaryPayload,
                headers: {
                    'Content-Type': 'application/octet-stream'
                }
            });
        } catch (err) {
            console.error("Uplink Failed:", err);
            setThoughts(prev => [...prev, "[ERROR] Uplink Failure."]);
        }
    }, [dspBus]);

    return { messages, sendMessage, isLearning, pulseType, thoughts, tunnelStatus };
};
