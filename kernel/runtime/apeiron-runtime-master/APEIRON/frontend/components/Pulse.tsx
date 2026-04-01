import { useApeironStore } from '../lib/store';
import { useEffect, useState } from 'react';

export default function Pulse() {
    const { pulseTrigger, plasticityScore } = useApeironStore();
    const [active, setActive] = useState(false);

    useEffect(() => {
        if (pulseTrigger) {
            setActive(true);
            const timer = setTimeout(() => setActive(false), 2000);
            return () => clearTimeout(timer);
        }
    }, [pulseTrigger]);

    return (
        <div className="fixed top-6 right-6 flex items-center gap-3">
            <div className="text-right">
                <p className="text-xs text-gray-400 font-mono">NEUROPLASTICITY</p>
                <p className="text-sm font-bold text-white font-mono">{plasticityScore.toFixed(2)}</p>
            </div>
            <div
                className={`w-4 h-4 rounded-full transition-all duration-500 border border-white/20
                ${active ? 'bg-green-500 shadow-[0_0_20px_#22c55e]' : 'bg-gray-800'}
            `}
            />
        </div>
    );
}
