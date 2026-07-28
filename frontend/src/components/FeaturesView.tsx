import { ShieldAlert, Flame, Phone, LineChart, CheckCircle, ArrowRight, Activity, Eye } from 'lucide-react';
import { MainTab } from '../types';

interface FeaturesViewProps {
  onNavigate: (tab: MainTab) => void;
}

export default function FeaturesView({ onNavigate }: FeaturesViewProps) {
  return (
    <div className="w-full bg-surface text-on-surface animate-fade-in pt-24">
      {/* Hero Header Section */}
      <section className="max-w-7xl mx-auto px-6 md:px-16 py-12 grid md:grid-cols-2 gap-12 items-center">
        <div className="space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-container text-on-primary-container text-xs font-semibold uppercase tracking-wider">
            <Eye className="w-4 h-4 text-on-primary-container" />
            Empathetic Vigilance
          </div>

          <h1 className="font-sans text-3xl lg:text-5xl font-extrabold text-primary leading-tight tracking-tight">
            Intelligent Care for the People Who Raised You.
          </h1>

          <p className="font-sans text-sm md:text-base text-on-surface-variant leading-relaxed max-w-lg">
            Advanced computer vision that watches over your loved ones without invading their privacy. Real-time protection, clinical reliability, and domestic warmth.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 pt-2">
            <button
              onClick={() => onNavigate('dashboard')}
              className="px-8 h-14 bg-primary text-on-primary rounded-xl font-sans text-sm font-semibold shadow-md hover:shadow-xl hover:bg-primary-container active:scale-95 transition-all cursor-pointer text-center flex items-center justify-center"
            >
              Protect Your Loved Ones
            </button>
            <button
              onClick={() => onNavigate('setup_guide')}
              className="px-8 h-14 border-2 border-primary text-primary rounded-xl font-sans text-sm font-semibold hover:bg-primary/5 active:scale-95 transition-all cursor-pointer text-center flex items-center justify-center"
            >
              See How It Works
            </button>
          </div>
        </div>

        {/* Right side graphic */}
        <div className="relative">
          <div className="aspect-square rounded-[2.5rem] overflow-hidden shadow-2xl relative border-8 border-surface-container-lowest">
            <img
              alt="Senior living safely"
              className="w-full h-full object-cover transition-transform hover:scale-105 duration-[2000ms]"
              src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='600' viewBox='0 0 600 600'%3E%3Crect width='600' height='600' fill='%23e8f5e9'/%3E%3Ccircle cx='300' cy='160' r='60' fill='%23a5d6a7'/%3E%3Cellipse cx='300' cy='340' rx='80' ry='120' fill='%2381c784'/%3E%3Cellipse cx='220' cy='430' rx='30' ry='90' fill='%23a5d6a7'/%3E%3Cellipse cx='380' cy='430' rx='30' ry='90' fill='%23a5d6a7'/%3E%3Crect x='240' y='240' width='120' height='20' rx='10' fill='%234caf50' opacity='0.4'/%3E%3Ctext x='300' y='560' text-anchor='middle' font-family='sans-serif' font-size='20' fill='%234caf50' opacity='0.7'%3EAI Monitoring Active%3C/text%3E%3C/svg%3E"
            />
            {/* UI Overlay Card */}
            <div className="absolute bottom-6 left-6 right-6 bg-surface/90 backdrop-blur-md p-4 rounded-2xl shadow-lg border border-outline-variant/60 flex items-center gap-4">
              <div className="w-11 h-11 rounded-full bg-secondary-container/15 flex items-center justify-center text-secondary">
                <CheckCircle className="w-6 h-6" />
              </div>
              <div className="space-y-0.5">
                <p className="font-sans text-[13px] font-bold text-primary">Vital Signs: Optimal</p>
                <p className="text-[10px] text-on-surface-variant italic">Last checked: 2 mins ago</p>
              </div>
            </div>
          </div>
          
          {/* Blur blobs */}
          <div className="absolute -top-12 -right-12 w-64 h-64 bg-primary-container/10 rounded-full blur-3xl -z-10"></div>
          <div className="absolute -bottom-12 -left-12 w-64 h-64 bg-secondary-fixed/20 rounded-full blur-3xl -z-10"></div>
        </div>
      </section>

      {/* Precision Monitoring Section */}
      <section className="bg-surface-container-low py-20 px-6 md:px-16 border-t border-outline-variant/10">
        <div className="max-w-7xl mx-auto space-y-12">
          
          <div className="text-center max-w-2xl mx-auto space-y-3">
            <h2 className="font-sans text-3xl font-extrabold text-primary tracking-tight">
              Precision Monitoring
            </h2>
            <p className="text-on-surface-variant text-sm md:text-base leading-relaxed">
              Our AI engine is trained on millions of data points to provide instant alerts for life-critical events while maintaining absolute privacy.
            </p>
          </div>

          {/* Bento grid */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 auto-rows-[220px]">
            
            {/* Fall Detection (Span 8) */}
            <div className="md:col-span-8 group relative overflow-hidden bg-surface-container-lowest rounded-[2rem] p-8 border border-outline-variant/35 hover:shadow-lg transition-all flex flex-col justify-between">
              <div className="flex justify-between items-start gap-4">
                <div className="max-w-sm space-y-3">
                  <span className="inline-flex p-3 bg-secondary-container/10 rounded-2xl text-secondary">
                    <ShieldAlert className="w-6 h-6" />
                  </span>
                  <h3 className="font-sans text-xl font-bold text-primary">Smart Fall Detection</h3>
                  <p className="text-on-surface-variant text-xs md:text-sm leading-relaxed">
                    Instant computer vision analysis identifies falls with 99.8% accuracy, notifying caregivers in seconds.
                  </p>
                </div>
                {/* mesh mesh graphic */}
                <div className="hidden lg:block w-40 h-40 bg-secondary-fixed rounded-2xl rotate-3 group-hover:rotate-6 transition-transform shadow-sm flex-shrink-0 flex items-center justify-center overflow-hidden">
                  <svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" className="w-full h-full opacity-70">
                    <rect width="160" height="160" fill="#e3f2fd"/>
                    {/* Skeleton joints */}
                    <circle cx="80" cy="28" r="12" fill="#1565c0" opacity="0.7"/>
                    <line x1="80" y1="40" x2="80" y2="90" stroke="#1565c0" strokeWidth="4" opacity="0.7"/>
                    <line x1="80" y1="55" x2="50" y2="75" stroke="#1565c0" strokeWidth="4" opacity="0.7"/>
                    <line x1="80" y1="55" x2="110" y2="75" stroke="#1565c0" strokeWidth="4" opacity="0.7"/>
                    <circle cx="50" cy="75" r="6" fill="#42a5f5" opacity="0.7"/>
                    <circle cx="110" cy="75" r="6" fill="#42a5f5" opacity="0.7"/>
                    <line x1="65" y1="90" x2="55" y2="125" stroke="#1565c0" strokeWidth="4" opacity="0.7"/>
                    <line x1="95" y1="90" x2="105" y2="125" stroke="#1565c0" strokeWidth="4" opacity="0.7"/>
                    <circle cx="55" cy="125" r="6" fill="#42a5f5" opacity="0.7"/>
                    <circle cx="105" cy="125" r="6" fill="#42a5f5" opacity="0.7"/>
                    <line x1="55" y1="125" x2="48" y2="148" stroke="#1565c0" strokeWidth="3" opacity="0.7"/>
                    <line x1="105" y1="125" x2="112" y2="148" stroke="#1565c0" strokeWidth="3" opacity="0.7"/>
                  </svg>
                </div>
              </div>
            </div>

            {/* Smoke / Fire (Span 4, colored orange-red) */}
            <div className="md:col-span-4 bg-secondary text-on-secondary rounded-[2rem] p-8 flex flex-col justify-between shadow-sm hover:shadow-lg transition-all">
              <div className="space-y-3">
                <span className="inline-flex p-3 bg-white/10 rounded-2xl text-white">
                  <Flame className="w-6 h-6 animate-pulse" />
                </span>
                <h3 className="font-sans text-xl font-bold">Smoke &amp; Fire</h3>
              </div>
              <p className="font-sans text-xs md:text-sm leading-relaxed opacity-90">
                Visual confirmation of fire hazards before traditional alarms even trigger.
              </p>
            </div>

            {/* Mobile Watch (Span 4, colored navy dark) */}
            <div className="md:col-span-4 bg-primary text-on-primary rounded-[2rem] p-8 flex flex-col justify-between shadow-sm hover:shadow-lg transition-all">
              <span className="font-sans text-[10px] font-bold uppercase tracking-widest text-on-primary-container">
                Always Connected
              </span>
              <div className="space-y-2">
                <h3 className="font-sans text-xl font-bold">Mobile Watch</h3>
                <p className="font-sans text-xs md:text-sm leading-relaxed text-on-primary-container">
                  Full live feed access and health analytics on your smartphone.
                </p>
              </div>
            </div>

            {/* Health streams (Span 8) */}
            <div className="md:col-span-8 bg-surface-container-highest rounded-[2rem] p-8 border border-outline-variant/35 flex items-center gap-8 group hover:shadow-lg transition-all justify-between">
              <div className="flex-1 space-y-4">
                <h3 className="font-sans text-xl font-bold text-primary">Health Data Streams</h3>
                <p className="text-on-surface-variant text-xs md:text-sm leading-relaxed max-w-md">
                  Continuous vitals monitoring including sleep patterns and mobility trends.
                </p>
                <div className="flex flex-wrap gap-2.5">
                  <span className="px-3 py-1 bg-surface-container-low rounded-full text-primary font-sans text-xs font-semibold border border-outline-variant/30">
                    Mobility
                  </span>
                  <span className="px-3 py-1 bg-surface-container-low rounded-full text-primary font-sans text-xs font-semibold border border-outline-variant/30">
                    Vitals
                  </span>
                  <span className="px-3 py-1 bg-surface-container-low rounded-full text-primary font-sans text-xs font-semibold border border-outline-variant/30">
                    Sleep
                  </span>
                </div>
              </div>
              
              <div className="hidden sm:flex p-4 bg-surface-container-low rounded-full border border-outline-variant/20 flex-shrink-0">
                <LineChart className="w-12 h-12 text-primary opacity-70 group-hover:scale-110 transition-transform" />
              </div>
            </div>

          </div>

        </div>
      </section>

      {/* Domestic Warmth, Clinical Precision section */}
      <section className="py-20 px-6 md:px-16 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          
          {/* Overlays / Status cards simulation */}
          <div className="order-2 lg:order-1 relative flex flex-col gap-6">
            
            {/* Status card 1: Normal Activity */}
            <div className="bg-surface-container-lowest rounded-2xl overflow-hidden border border-outline-variant/30 shadow-sm">
              <div className="h-1 w-full bg-teal-500"></div>
              <div className="p-5 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-surface-container-low rounded-xl text-primary border border-outline-variant/10">
                    <CheckCircle className="w-5 h-5" />
                  </div>
                  <div className="space-y-0.5">
                    <p className="font-sans text-sm font-bold text-primary">Normal Activity</p>
                    <p className="text-on-surface-variant text-xs">Living Room • Updated Just Now</p>
                  </div>
                </div>
                <span className="px-3 py-0.5 bg-green-100 text-green-700 rounded-full text-[10px] font-bold uppercase tracking-wider">
                  Secured
                </span>
              </div>
            </div>

            {/* Status card 2: Warning Alert offset (Exactly like image) */}
            <div className="bg-surface-container-lowest rounded-2xl overflow-hidden border border-outline-variant/50 shadow-lg relative lg:-translate-x-6">
              <div className="h-1 w-full bg-secondary"></div>
              <div className="p-5 flex items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className="p-3 bg-error-container text-secondary rounded-xl">
                    <ShieldAlert className="w-5 h-5" />
                  </div>
                  <div className="space-y-0.5">
                    <p className="font-sans text-sm font-bold text-primary">Stove left unattended</p>
                    <p className="text-on-surface-variant text-xs">Kitchen • Notification Sent</p>
                  </div>
                </div>
                <button className="px-4 py-1.5 bg-secondary text-on-secondary rounded-lg text-xs font-bold tracking-wider uppercase flex-shrink-0 shadow-sm">
                  Alert Sent
                </button>
              </div>
            </div>

          </div>

          {/* Value Props checklist */}
          <div className="order-1 lg:order-2 space-y-6">
            <h2 className="font-sans text-3xl font-extrabold text-primary leading-tight tracking-tight">
              Domestic Warmth, Clinical Precision.
            </h2>
            <p className="text-on-surface-variant text-sm md:text-base leading-relaxed">
              We don't just watch; we understand. Safewatch learns the unique rhythms of a home—recognizing the difference between a nap and a fall, or a tea kettle and a fire.
            </p>
            
            <ul className="space-y-4 pt-2">
              {[
                'Zero wearables required for constant monitoring',
                'HIPAA-compliant, end-to-end encrypted data',
                '24/7 human-verified emergency response'
              ].map((text, idx) => (
                <li key={idx} className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-secondary flex-shrink-0 mt-0.5" />
                  <span className="font-sans text-sm font-medium text-on-surface">
                    {text}
                  </span>
                </li>
              ))}
            </ul>
          </div>

        </div>
      </section>

      {/* Ready for a safer tomorrow section */}
      <section className="bg-primary-container text-on-primary-container py-20 px-6 md:px-16 relative overflow-hidden text-center border-t border-white/5">
        <div className="max-w-4xl mx-auto space-y-8 relative z-10">
          <h2 className="font-sans text-2xl md:text-4xl font-extrabold text-white tracking-tight">
            Ready for a safer tomorrow?
          </h2>
          <p className="text-on-primary-container text-sm md:text-base leading-relaxed max-w-2xl mx-auto opacity-90">
            Join thousands of families who have found peace of mind through Safewatch. Professional installation included with every subscription.
          </p>

          <div className="flex flex-col sm:flex-row justify-center gap-4 pt-4">
            <button
              onClick={() => onNavigate('dashboard')}
              className="px-8 h-14 bg-white text-primary font-sans text-sm font-bold rounded-xl hover:scale-105 active:scale-95 shadow-lg transition-all cursor-pointer flex items-center justify-center"
            >
              Get Started Today
            </button>
            <button
              onClick={() => onNavigate('setup_guide')}
              className="px-8 h-14 border-2 border-on-primary-container text-white font-sans text-sm font-bold rounded-xl hover:bg-white/10 active:scale-95 transition-all cursor-pointer flex items-center justify-center"
            >
              Speak with a Care Expert
            </button>
          </div>
        </div>
        
        {/* Particle back pattern */}
        <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(#ffffff 1.2px, transparent 1.2px)', backgroundSize: '36px 36px' }}></div>
      </section>
    </div>
  );
}
