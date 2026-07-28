import { HeartPulse, MapPin, Eye, Mic, ArrowRight, ShieldAlert, Heart, Activity } from 'lucide-react';
import { MainTab } from '../types';

interface MobileAppPromoProps {
  onNavigate: (tab: MainTab) => void;
  onTriggerEmergency: () => void;
}

export default function MobileAppPromo({ onNavigate, onTriggerEmergency }: MobileAppPromoProps) {
  return (
    <div className="w-full bg-surface text-on-surface animate-fade-in">
      
      {/* Hero Section: The Phone & Alert */}
      <section className="max-w-7xl mx-auto px-6 md:px-16 pt-28 pb-16 grid md:grid-cols-2 gap-12 lg:gap-16 items-center">
        <div className="space-y-8">
          {/* Active Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 bg-secondary-container/10 text-secondary font-sans text-xs font-semibold uppercase tracking-wider rounded-full border border-secondary/20">
            <span className="relative flex h-3 w-3">
              <span className="pulsing-dot absolute inline-flex h-full w-full rounded-full bg-secondary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-secondary animate-pulse"></span>
            </span>
            CRITICAL ALERT SYSTEM ACTIVE
          </div>

          <h1 className="font-sans text-4xl lg:text-5xl font-extrabold text-primary leading-tight tracking-tight">
            Vigilance That Travels <br className="hidden lg:inline" /> With You.
          </h1>

          <p className="font-sans text-base md:text-lg text-on-surface-variant max-w-lg leading-relaxed">
            The Safewatch app bridges the distance between you and your loved ones. Experience real-time fall detection notifications and seamless health monitoring from the palm of your hand.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 pt-4">
            <button
              onClick={() => onNavigate('dashboard')}
              className="px-8 py-4 bg-primary text-on-primary rounded-xl font-sans text-sm font-semibold tracking-wide shadow-md hover:shadow-xl hover:bg-primary-container active:scale-95 transition-all cursor-pointer text-center"
            >
              Protect Your Loved Ones
            </button>
            <button
              onClick={() => onNavigate('setup_guide')}
              className="px-8 py-4 border-2 border-primary text-primary rounded-xl font-sans text-sm font-semibold tracking-wide hover:bg-primary/5 active:scale-95 transition-all cursor-pointer text-center"
            >
              See How It Works
            </button>
          </div>

          {/* Store badges */}
          <div className="flex flex-wrap gap-4 pt-6">
            <div className="h-10 px-5 bg-black rounded-xl flex items-center gap-2 hover:opacity-90 active:scale-95 transition-transform cursor-pointer">
              <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.8-.91.65.03 2.47.26 3.64 1.98l-.09.06c-.22.15-2.17 1.29-2.14 3.84.04 3.02 2.6 4.07 2.63 4.09-.03.07-.41 1.42-1.34 2.69M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z"/></svg>
              <div><div className="text-white text-[8px] font-medium leading-none">Download on the</div><div className="text-white text-sm font-bold leading-tight">App Store</div></div>
            </div>
            <div className="h-10 px-5 bg-black rounded-xl flex items-center gap-2 hover:opacity-90 active:scale-95 transition-transform cursor-pointer">
              <svg className="w-5 h-5" viewBox="0 0 24 24"><path fill="#EA4335" d="m3 3.5 9.3 9.3L3 21.2V3.5z"/><path fill="#FBBC04" d="M3 21.2 13.3 11l2.7 2.7L3 21.2z"/><path fill="#34A853" d="m15.9 13.7 2.6-1.7-2.6-2.7-2.6 1.7 2.6 2.7z"/><path fill="#4285F4" d="M3 3.5 13.3 11l2.7-2.7L3 3.5z"/></svg>
              <div><div className="text-white text-[8px] font-medium leading-none">GET IT ON</div><div className="text-white text-sm font-bold leading-tight">Google Play</div></div>
            </div>
          </div>
        </div>

        {/* Right side: Mockup Phone & Overlay Alert */}
        <div className="relative flex justify-center items-center mt-8 md:mt-0">
          {/* Decorative Glows */}
          <div className="absolute -z-10 w-80 h-80 bg-primary/5 rounded-full blur-3xl"></div>
          <div className="absolute -top-10 -right-10 -z-10 w-60 h-60 bg-secondary/5 rounded-full blur-3xl"></div>

          {/* Main Mobile Mockup */}
          <div className="relative w-full max-w-[320px] aspect-[9/19] bg-primary rounded-[3rem] p-3 shadow-2xl border-[8px] border-outline-variant">
            {/* Notch */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/3 h-5 bg-primary rounded-b-2xl z-20"></div>
            
            <div className="relative h-full w-full bg-surface-container-lowest rounded-[2.3rem] overflow-hidden flex flex-col justify-between p-5 pt-7">
              {/* Internal Mockup Content */}
              <div className="space-y-5">
                {/* Internal App Navbar */}
                <div className="flex justify-between items-center">
                  <div className="w-6 h-1 bg-primary rounded-full relative after:content-[''] after:absolute after:top-1.5 after:left-0 after:w-4 after:h-1 after:bg-primary after:rounded-full"></div>
                  <div className="w-8 h-8 rounded-full bg-primary-container flex items-center justify-center text-xs font-bold text-on-primary-container">
                    RA
                  </div>
                </div>

                {/* Live Status Header */}
                <div className="space-y-1">
                  <h3 className="font-sans text-xl font-bold text-primary">Live Status</h3>
                  <p className="text-[10px] text-on-surface-variant font-medium">Connected to Home Hub 01</p>
                </div>

                {/* App Potential Fall Card */}
                <div className="p-3 bg-secondary-container/10 border border-secondary/20 rounded-2xl space-y-2 animate-pulse">
                  <div className="flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4 text-secondary" />
                    <span className="font-sans text-[11px] font-bold text-secondary uppercase tracking-wider">POTENTIAL FALL</span>
                  </div>
                  <div className="text-[10px] text-on-surface-variant font-medium">Kitchen Area • 2 mins ago</div>
                  <button 
                    onClick={onTriggerEmergency}
                    className="w-full py-2 bg-secondary text-on-primary rounded-xl text-[10px] font-bold uppercase tracking-wider hover:opacity-95 active:scale-95 transition-all"
                  >
                    EMERGENCY CALL
                  </button>
                </div>

                {/* Mini Stats strip */}
                <div className="grid grid-cols-2 gap-2.5">
                  <div className="p-2.5 bg-surface-container-low rounded-xl border border-outline-variant/40">
                    <div className="text-[9px] text-on-surface-variant uppercase font-bold tracking-wider">Heart Rate</div>
                    <div className="text-base font-extrabold text-primary flex items-baseline gap-0.5">
                      72 <span className="text-[9px] font-normal text-on-surface-variant">BPM</span>
                    </div>
                  </div>
                  <div className="p-2.5 bg-surface-container-low rounded-xl border border-outline-variant/40">
                    <div className="text-[9px] text-on-surface-variant uppercase font-bold tracking-wider">Mobility</div>
                    <div className="text-base font-extrabold text-primary">High</div>
                  </div>
                </div>
              </div>

              {/* Bottom Nav Mock */}
              <div className="w-full h-1 bg-outline-variant/50 rounded-full mx-auto max-w-[80px]"></div>
            </div>

            {/* Float Overlay Alert Notification (Exactly like visual image) */}
            <div className="absolute top-24 -left-4 right-4 z-30 bg-surface/95 backdrop-blur-md p-4 rounded-2xl shadow-xl border border-outline-variant/80 transform translate-y-1 animate-bounce">
              <div className="flex items-start gap-3">
                <div className="bg-secondary p-1.5 rounded-xl text-on-secondary flex-shrink-0">
                  <ShieldAlert className="w-4 h-4" />
                </div>
                <div className="flex-1 space-y-1 min-w-0">
                  <div className="flex justify-between items-center">
                    <span className="font-sans text-[10px] font-bold tracking-wider text-primary uppercase">SAFEWATCH</span>
                    <span className="text-[9px] text-on-surface-variant uppercase font-medium">Now</span>
                  </div>
                  <div className="font-extrabold text-xs text-secondary">Alert: Fall Detected</div>
                  <p className="text-[10px] leading-tight text-on-surface-variant line-clamp-2">
                    Unusual motion detected in the Living Room. Confirm status now.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Bento Grid: Designed for Peace of Mind */}
      <section className="bg-surface-container-low py-20 px-6 md:px-16 border-t border-outline-variant/10">
        <div className="max-w-7xl mx-auto space-y-12">
          
          <div className="text-center max-w-2xl mx-auto space-y-4">
            <h2 className="font-sans text-3xl md:text-4xl font-extrabold text-primary tracking-tight">
              Designed for Peace of Mind
            </h2>
            <p className="text-on-surface-variant text-sm md:text-base leading-relaxed">
              Complex technology made simple for the moments that matter most. Every notification is a commitment to care.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Fall Alerts Card (Span 2) */}
            <div className="md:col-span-2 bg-surface-container-lowest border border-outline-variant/30 rounded-[2rem] p-8 md:p-10 flex flex-col md:flex-row gap-8 items-center shadow-sm hover:shadow-md transition-all">
              <div className="space-y-4 flex-1">
                <div className="w-12 h-12 bg-secondary-container/10 border border-secondary/15 rounded-2xl flex items-center justify-center">
                  <ShieldAlert className="w-6 h-6 text-secondary" />
                </div>
                <h3 className="font-sans text-xl md:text-2xl font-bold text-primary">
                  Intelligent Fall Alerts
                </h3>
                <p className="text-on-surface-variant text-sm leading-relaxed">
                  Using edge-AI vision, our system distinguishes between a falling person and a falling object, reducing false alarms by 98% while ensuring rapid response when it's real.
                </p>
              </div>
              <div className="flex-1 w-full h-full min-h-[160px] max-h-[220px] rounded-2xl overflow-hidden shadow-sm">
                <img
                  alt="AI Smart home layout"
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover rounded-2xl transition-transform hover:scale-105 duration-500"
                  src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'%3E%3Ccircle cx='40' cy='40' r='40' fill='%23bbdefb'/%3E%3Ccircle cx='40' cy='30' r='14' fill='%231565c0' opacity='0.7'/%3E%3Cellipse cx='40' cy='65' rx='22' ry='18' fill='%231565c0' opacity='0.5'/%3E%3C/svg%3E"
                />
              </div>
            </div>

            {/* Smart Zones Card (Span 1, dark blue color theme) */}
            <div className="bg-primary text-on-primary rounded-[2rem] p-8 flex flex-col justify-between shadow-sm hover:shadow-md transition-all">
              <div className="space-y-4">
                <MapPin className="w-10 h-10 text-on-primary-container" />
                <h3 className="font-sans text-xl md:text-2xl font-bold">
                  Smart Zones
                </h3>
                <p className="text-on-primary-container text-xs md:text-sm leading-relaxed">
                  Create safe perimeters. Get notified instantly if your loved one leaves a predefined safe zone during night hours.
                </p>
              </div>
              
              <div className="mt-8 bg-surface-container-low/10 p-4 rounded-2xl border border-white/10">
                <div className="flex items-center gap-3">
                  <div className="w-3.5 h-3.5 rounded-full bg-secondary pulsing-dot"></div>
                  <span className="text-xs font-bold uppercase tracking-widest text-on-primary">
                    ACTIVE TRACKING
                  </span>
                </div>
              </div>
            </div>

            {/* Vitals Card (Span 1) */}
            <div className="bg-surface-container-lowest border border-outline-variant/30 rounded-[2rem] p-8 flex flex-col justify-between shadow-sm hover:shadow-md transition-all">
              <div className="space-y-4">
                <div className="w-12 h-12 bg-primary-container/10 border border-primary-container/15 rounded-2xl flex items-center justify-center">
                  <Activity className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-sans text-xl font-bold text-primary">
                  Vitals at a Glance
                </h3>
                <p className="text-on-surface-variant text-sm leading-relaxed">
                  Monitor sleep patterns, heart rate variability, and daily activity levels through beautiful, easy-to-read charts.
                </p>
              </div>

              {/* Graphical simulation of pillars (Exactly like image) */}
              <div className="pt-6 h-20 flex items-end gap-1.5 w-full">
                <div className="bg-primary/20 w-full h-[30%] rounded-t-md transition-all duration-500 hover:h-[35%]"></div>
                <div className="bg-primary/40 w-full h-[55%] rounded-t-md transition-all duration-500 hover:h-[60%]"></div>
                <div className="bg-primary/20 w-full h-[40%] rounded-t-md transition-all duration-500 hover:h-[45%]"></div>
                <div className="bg-primary/60 w-full h-[70%] rounded-t-md transition-all duration-500 hover:h-[75%]"></div>
                <div className="bg-secondary w-full h-[95%] rounded-t-md transition-all duration-500 hover:h-[100%] cursor-pointer" title="Peak activity warning!"></div>
                <div className="bg-primary/40 w-full h-[60%] rounded-t-md transition-all duration-500 hover:h-[65%]"></div>
              </div>
            </div>

            {/* Two-Way Voice (Span 2) */}
            <div className="md:col-span-2 bg-surface-container-lowest border border-outline-variant/30 rounded-[2rem] p-8 md:p-10 flex flex-col md:flex-row-reverse gap-8 items-center shadow-sm hover:shadow-md transition-all">
              <div className="space-y-4 flex-1">
                <div className="w-12 h-12 bg-tertiary-container/10 border border-tertiary-container/15 rounded-2xl flex items-center justify-center">
                  <Mic className="w-6 h-6 text-tertiary-container" />
                </div>
                <h3 className="font-sans text-xl md:text-2xl font-bold text-primary">
                  Two-Way Voice
                </h3>
                <p className="text-on-surface-variant text-sm leading-relaxed">
                  Speak directly through the Safewatch home hub using your smartphone. Comfort them instantly while help is on the way.
                </p>
                <button
                  onClick={() => onNavigate('features')}
                  className="inline-flex items-center gap-1.5 text-primary hover:text-primary-container font-bold text-sm transition-all focus:outline-none"
                >
                  Watch how it works <ArrowRight className="w-4 h-4" />
                </button>
              </div>
              <div className="flex-1 w-full h-full min-h-[160px] max-h-[220px] rounded-2xl overflow-hidden shadow-sm">
                <img
                  alt="Caregiver checking phone"
                  referrerPolicy="no-referrer"
                  className="w-full h-full object-cover rounded-2xl transition-transform hover:scale-105 duration-500"
                  src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'%3E%3Ccircle cx='40' cy='40' r='40' fill='%23f3e5f5'/%3E%3Ccircle cx='40' cy='30' r='14' fill='%236a1b9a' opacity='0.7'/%3E%3Cellipse cx='40' cy='65' rx='22' ry='18' fill='%236a1b9a' opacity='0.5'/%3E%3C/svg%3E"
                />
              </div>
            </div>

          </div>

        </div>
      </section>

      {/* CTA: Care without boundaries */}
      <section className="max-w-7xl mx-auto px-6 md:px-16 py-16">
        <div className="bg-surface-container-high rounded-[2.5rem] p-10 md:p-16 text-center relative overflow-hidden shadow-sm">
          <div className="absolute -top-24 -left-24 w-60 h-60 bg-primary/5 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-24 -right-24 w-60 h-60 bg-secondary/5 rounded-full blur-3xl"></div>

          <div className="relative z-10 space-y-6 max-w-2xl mx-auto">
            <h2 className="font-sans text-3xl md:text-4xl font-extrabold text-primary">
              Care without boundaries.
            </h2>
            <p className="text-on-surface-variant text-sm md:text-base leading-relaxed">
              Download the Safewatch app today and join thousands of families who have found peace of mind through empathetic technology.
            </p>

            <div className="flex flex-wrap justify-center gap-4 pt-4">
              <img
                alt="App Store"
                referrerPolicy="no-referrer"
                className="h-12 hover:opacity-90 active:scale-95 transition-transform cursor-pointer"
                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Crect width='120' height='120' rx='16' fill='%23e8f5e9'/%3E%3Ccircle cx='60' cy='42' r='20' fill='%232e7d32' opacity='0.7'/%3E%3Cellipse cx='60' cy='95' rx='32' ry='26' fill='%232e7d32' opacity='0.5'/%3E%3C/svg%3E"
              />
              <img
                alt="Play Store"
                referrerPolicy="no-referrer"
                className="h-12 hover:opacity-90 active:scale-95 transition-transform cursor-pointer"
                src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Crect width='120' height='120' rx='16' fill='%23e3f2fd'/%3E%3Ccircle cx='60' cy='42' r='20' fill='%230d47a1' opacity='0.7'/%3E%3Cellipse cx='60' cy='95' rx='32' ry='26' fill='%230d47a1' opacity='0.5'/%3E%3C/svg%3E"
              />
            </div>
          </div>
        </div>
      </section>

    </div>
  );
}
