import { useState, useEffect } from 'react';
import { ShieldAlert, X, Phone, Users, Volume2 } from 'lucide-react';

interface EmergencyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function EmergencyModal({ isOpen, onClose }: EmergencyModalProps) {
  const [countdown, setCountdown] = useState<number>(5);
  const [callActive, setCallActive] = useState<boolean>(false);

  useEffect(() => {
    if (!isOpen) {
      setCountdown(5);
      setCallActive(false);
      return;
    }

    if (countdown > 0 && !callActive) {
      const timer = setTimeout(() => {
        setCountdown(countdown - 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else if (countdown === 0 && !callActive) {
      setCallActive(true);
    }
  }, [isOpen, countdown, callActive]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/75 backdrop-blur-md flex items-center justify-center z-[100] p-6 animate-fade-in">
      
      <div className="bg-surface max-w-md w-full rounded-[2rem] border-2 border-secondary overflow-hidden shadow-2xl relative flex flex-col justify-between p-8 space-y-6">
        
        {/* Header warnings */}
        <div className="flex items-center gap-3.5 pb-4 border-b border-outline-variant/30">
          <div className="bg-secondary p-2.5 rounded-2xl text-on-secondary animate-pulse">
            <ShieldAlert className="w-6 h-6" />
          </div>
          <div>
            <h2 className="font-sans text-lg font-black text-secondary uppercase tracking-wider">CRITICAL RESPONSE UNIT</h2>
            <p className="text-[10px] text-on-surface-variant font-bold">EMERGENCY MULTI-DIAL PROTOCOL</p>
          </div>
        </div>

        {/* Dynamic Countdown State */}
        {!callActive ? (
          <div className="text-center space-y-6 py-4">
            <div className="space-y-1">
              <p className="text-sm font-semibold text-primary">Emergency call initiating in...</p>
              <div className="font-mono text-7xl font-black text-secondary animate-bounce">
                {countdown}
              </div>
              <p className="text-xs text-on-surface-variant max-w-xs mx-auto">
                First responders, local emergency medical teams, and Sarah Jenkins (Primary Circle Lead) will be dispatched.
              </p>
            </div>

            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="flex-1 py-3.5 bg-surface-container-high hover:bg-surface-container-highest text-primary rounded-xl font-sans text-xs font-bold uppercase tracking-wider transition-all cursor-pointer focus:outline-none"
              >
                Cancel Call
              </button>
              <button
                onClick={() => setCallActive(true)}
                className="flex-1 py-3.5 bg-secondary text-on-secondary hover:bg-secondary-container rounded-xl font-sans text-xs font-bold uppercase tracking-wider transition-all cursor-pointer focus:outline-none shadow-sm"
              >
                Trigger Now
              </button>
            </div>
          </div>
        ) : (
          /* Active Call State */
          <div className="space-y-6 py-4">
            <div className="bg-secondary/10 border border-secondary/20 p-4 rounded-2xl text-center space-y-1">
              <span className="w-2.5 h-2.5 rounded-full bg-secondary inline-block pulsing-dot mr-2"></span>
              <span className="font-sans text-xs font-bold text-secondary uppercase tracking-widest">Active Connection Secure</span>
              <h3 className="font-sans text-lg font-black text-primary">First Responders Notified</h3>
              <p className="text-xs text-on-surface-variant">Live audio channel successfully connected to Residence Alpha.</p>
            </div>

            {/* Audio Waveform simulator */}
            <div className="h-16 bg-surface-container-low border border-outline-variant/30 rounded-2xl flex items-center justify-center p-3 overflow-hidden gap-1">
              <div className="w-1 bg-secondary h-[40%] rounded-full animate-pulse"></div>
              <div className="w-1 bg-secondary h-[70%] rounded-full animate-pulse delay-75"></div>
              <div className="w-1 bg-secondary h-[95%] rounded-full animate-pulse delay-150"></div>
              <div className="w-1 bg-secondary h-[50%] rounded-full animate-pulse delay-75"></div>
              <div className="w-1 bg-secondary h-[80%] rounded-full animate-pulse"></div>
              <div className="w-1 bg-secondary h-[30%] rounded-full animate-pulse delay-100"></div>
            </div>

            {/* Simulated Speakers logs */}
            <div className="space-y-2 bg-surface-container-low p-4 rounded-xl border border-outline-variant/20 text-[11px] font-sans text-on-surface-variant max-h-[110px] overflow-y-auto">
              <p className="text-primary font-bold">● Residence Alpha Hub Speaker:</p>
              <p className="italic">"We are connecting you... Stay calm, help is already on the way."</p>
              <p className="text-secondary font-bold pt-1">● Dr. Emily Stone notified:</p>
              <p className="italic">"Receiving telemetry. Commencing caregiver dispatch line."</p>
            </div>

            <button
              onClick={onClose}
              className="w-full py-3.5 bg-primary text-on-primary hover:bg-primary-container rounded-xl font-sans text-xs font-bold uppercase tracking-wider text-center cursor-pointer active:scale-95 transition-transform"
            >
              Terminate Emergency Call
            </button>
          </div>
        )}

      </div>
    </div>
  );
}
