import { useState } from 'react';
import { Check, Wifi, MapPin, Users, Package, ArrowRight, ShieldCheck, RefreshCw } from 'lucide-react';
import { MainTab } from '../types';

interface SetupGuideViewProps {
  onNavigate: (tab: MainTab) => void;
}

export default function SetupGuideView({ onNavigate }: SetupGuideViewProps) {
  const [currentStep, setCurrentStep] = useState<number>(2); // Starts at step 2 as step 1 is unboxed.
  
  // Wi-Fi settings
  const [wifiSelected, setWifiSelected] = useState<string>('Home_Fiber_Secure_5G');
  const [wifiPass, setWifiPass] = useState<string>('');
  const [isSyncing, setIsSyncing] = useState<boolean>(false);
  const [syncDone, setSyncDone] = useState<boolean>(false);

  // Invite circles
  const [contacts, setContacts] = useState<{ name: string; email: string; role: string }[]>([
    { name: 'Sarah Jenkins', email: 'sarah.j@family.com', role: 'Daughter' }
  ]);
  const [inviteName, setInviteName] = useState('');
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState('Relative');

  const progressPercentages: { [key: number]: number } = {
    1: 25,
    2: 50,
    3: 75,
    4: 90,
    5: 100 // Success
  };

  const handleAddContact = () => {
    if (inviteName.trim() && inviteEmail.trim()) {
      setContacts([...contacts, { name: inviteName, email: inviteEmail, role: inviteRole }]);
      setInviteName('');
      setInviteEmail('');
    }
  };

  const handleSyncWifi = () => {
    setIsSyncing(true);
    setTimeout(() => {
      setIsSyncing(false);
      setSyncDone(true);
      setTimeout(() => {
        setCurrentStep(4);
      }, 1000);
    }, 2000);
  };

  return (
    <div className="w-full bg-surface text-on-surface pt-24 pb-16 min-h-screen">
      <div className="max-w-7xl mx-auto px-6 md:px-16 space-y-12">
        
        {/* Header Hero */}
        <div className="text-center max-w-2xl mx-auto space-y-4">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary-container text-on-primary-container text-xs font-semibold uppercase tracking-wider">
            Step-by-step guidance
          </span>
          <h1 className="font-sans text-3xl md:text-5xl font-extrabold text-primary leading-tight tracking-tight">
            Empathetic Setup Experience
          </h1>
          <p className="text-on-surface-variant text-sm md:text-base leading-relaxed">
            We're here to guide you through a seamless integration of Safewatch into your home. Ensure your loved one is comfortable before we begin.
          </p>
        </div>

        {/* Progress Tracker Card (Exactly like Screen 3) */}
        <div className="glass-card rounded-3xl p-6 md:p-8 shadow-md border border-outline-variant/30 grid grid-cols-1 md:grid-cols-12 gap-8 items-center max-w-4xl mx-auto">
          
          {/* Percentage */}
          <div className="md:col-span-3 text-center space-y-1 md:border-r md:border-outline-variant/20 pr-4">
            <div className="font-mono text-5xl font-black text-primary">
              {progressPercentages[currentStep]}%
            </div>
            <p className="text-[10px] uppercase font-bold tracking-widest text-on-surface-variant">
              SETUP COMPLETED
            </p>
          </div>

          {/* Stepper checklist status bar */}
          <div className="md:col-span-9 grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { num: 1, label: 'Unboxing', icon: Package },
              { num: 2, label: 'Placement', icon: MapPin },
              { num: 3, label: 'Network', icon: Wifi },
              { num: 4, label: 'Care Circle', icon: Users }
            ].map((step) => {
              const isFinished = currentStep > step.num;
              const isCurrent = currentStep === step.num;
              const Icon = step.icon;
              return (
                <div 
                  key={step.num}
                  className={`flex items-center gap-3 p-3 rounded-xl border transition-all ${
                    isFinished 
                      ? 'bg-green-50 border-green-200 text-green-700' 
                      : isCurrent 
                        ? 'bg-primary/5 border-primary/40 text-primary font-bold shadow-sm' 
                        : 'bg-surface-container-low border-outline-variant/30 text-on-surface-variant'
                  }`}
                >
                  <span className="p-1 rounded-lg bg-white shadow-sm shrink-0">
                    {isFinished ? <Check className="w-4 h-4 text-green-600" /> : <Icon className="w-4 h-4" />}
                  </span>
                  <div>
                    <p className="text-[9px] uppercase font-bold tracking-wide opacity-80 leading-none">Step 0{step.num}</p>
                    <p className="text-[11px] font-bold leading-tight">{step.label}</p>
                  </div>
                </div>
              );
            })}
          </div>

        </div>

        {/* Dynamic setup guides content */}
        <div className="max-w-4xl mx-auto bg-surface-container-low rounded-3xl p-8 md:p-10 border border-outline-variant/20 shadow-sm min-h-[420px] flex flex-col justify-between">
          
          {/* STEP 2: HARDWARE PLACEMENT */}
          {currentStep === 2 && (
            <div className="grid md:grid-cols-2 gap-10 items-center">
              <div className="space-y-6">
                <span className="text-[10px] uppercase font-bold tracking-widest text-secondary">HARDWARE ASSEMBLY</span>
                <h3 className="font-sans text-xl md:text-2xl font-bold text-primary">Optimal Camera Placement</h3>
                
                <div className="space-y-4 text-xs md:text-sm text-on-surface-variant leading-relaxed">
                  <p>
                    For the most accurate computer vision and fall detection tracking, place the Safewatch Smart Hub at a height of <strong>6.5ft to 8.0ft</strong> from the floor.
                  </p>
                  <ul className="space-y-2.5">
                    <li className="flex gap-2 items-start">
                      <Check className="w-4 h-4 text-green-600 shrink-0 mt-0.5" />
                      <span>Ensure a clear view of key areas (e.g., bedside or kitchen sink)</span>
                    </li>
                    <li className="flex gap-2 items-start">
                      <Check className="w-4 h-4 text-green-600 shrink-0 mt-0.5" />
                      <span>Avoid placing behind mirrors, bookshelves, or hanging objects</span>
                    </li>
                  </ul>
                </div>

                <button
                  onClick={() => setCurrentStep(3)}
                  className="px-6 py-3 bg-primary text-on-primary rounded-xl font-sans text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 hover:bg-primary-container active:scale-95 transition-all cursor-pointer"
                >
                  Confirm Placement &amp; Proceed <ArrowRight className="w-4 h-4" />
                </button>
              </div>

              {/* Placement image */}
              <div className="relative aspect-video md:aspect-square bg-black rounded-2xl overflow-hidden shadow-md">
                <img
                  alt="Living room setup camera angle view"
                  className="w-full h-full object-cover opacity-85"
                  src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 640 480'%3E%3Crect width='640' height='480' fill='%231a1a2e'/%3E%3Crect x='80' y='100' width='480' height='280' rx='12' fill='%2316213e' stroke='%230f3460' stroke-width='2'/%3E%3Ccircle cx='320' cy='200' r='50' fill='%230f3460' opacity='0.7'/%3E%3Cpath d='M280 200 l40-30 v60 z' fill='%2353d8fb' opacity='0.8'/%3E%3Crect x='220' y='320' width='200' height='40' rx='8' fill='%230f3460'/%3E%3Ctext x='320' y='345' text-anchor='middle' font-family='monospace' font-size='12' fill='%2353d8fb'%3E● LIVE CAMERA FEED%3C/text%3E%3C/svg%3E"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                <div className="absolute bottom-4 left-4 text-white text-[11px] font-bold uppercase tracking-widest flex items-center gap-1.5">
                  <span className="w-2 h-2 bg-secondary rounded-full pulsing-dot"></span>
                  Live Setup Preview
                </div>
              </div>
            </div>
          )}

          {/* STEP 3: NETWORK SYNC */}
          {currentStep === 3 && (
            <div className="space-y-6 max-w-xl mx-auto w-full">
              <div className="text-center space-y-2">
                <span className="text-[10px] uppercase font-bold tracking-widest text-secondary">HUB SYSTEM SETUP</span>
                <h3 className="font-sans text-xl md:text-2xl font-bold text-primary">Connect to Your Secure Network</h3>
                <p className="text-xs text-on-surface-variant">The hub will synchronize security credentials with the cloud database.</p>
              </div>

              <div className="space-y-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] uppercase font-bold tracking-wider text-on-surface-variant">Select Wi-Fi Network</label>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {['Home_Fiber_Secure_5G', 'Safewatch_Ext_Guest', 'Netgear_488_Default', 'Linksys_Router_Home'].map((ssid) => (
                      <button
                        key={ssid}
                        onClick={() => setWifiSelected(ssid)}
                        className={`p-3 rounded-xl border text-left text-xs font-bold transition-all focus:outline-none cursor-pointer flex items-center gap-2 ${
                          wifiSelected === ssid
                            ? 'bg-primary border-primary text-on-primary'
                            : 'bg-surface border-outline-variant/40 hover:bg-surface-container-high text-on-surface'
                        }`}
                      >
                        <Wifi className="w-4 h-4 shrink-0" />
                        {ssid}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="text-[10px] uppercase font-bold tracking-wider text-on-surface-variant">Wi-Fi Password</label>
                  <input
                    type="password"
                    placeholder="••••••••••••"
                    value={wifiPass}
                    onChange={(e) => setWifiPass(e.target.value)}
                    className="w-full px-4 h-12 bg-surface border border-outline-variant/60 rounded-xl font-mono text-sm focus:outline-none focus:border-primary"
                  />
                </div>

                {isSyncing ? (
                  <div className="py-4 text-center space-y-3">
                    <RefreshCw className="w-8 h-8 text-primary animate-spin mx-auto" />
                    <p className="font-sans text-xs font-bold text-primary">Synchronizing hub with security keys...</p>
                  </div>
                ) : syncDone ? (
                  <div className="p-3 bg-green-50 text-green-700 rounded-xl border border-green-200 text-center font-sans text-xs font-bold">
                    Network pairing successful! Going to Care Circle Invite...
                  </div>
                ) : (
                  <button
                    onClick={handleSyncWifi}
                    disabled={!wifiPass}
                    className="w-full py-3.5 bg-primary text-on-primary disabled:opacity-50 rounded-xl font-sans text-xs font-bold uppercase tracking-wider text-center cursor-pointer hover:bg-primary-container transition-transform active:scale-95"
                  >
                    Sync Device
                  </button>
                )}
              </div>
            </div>
          )}

          {/* STEP 4: CARE CIRCLE INVITE */}
          {currentStep === 4 && (
            <div className="grid md:grid-cols-2 gap-10 items-center">
              <div className="space-y-6">
                <span className="text-[10px] uppercase font-bold tracking-widest text-secondary">SAFETY NETWORK</span>
                <h3 className="font-sans text-xl md:text-2xl font-bold text-primary">Build Your Care Circle</h3>
                <p className="text-xs text-on-surface-variant leading-relaxed">
                  Add trusted family members, medical specialists, or neighbors to receive real-time fall detection alerts.
                </p>

                {/* Form */}
                <div className="space-y-3 p-4 bg-surface rounded-2xl border border-outline-variant/30">
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="text"
                      placeholder="Contact Name"
                      value={inviteName}
                      onChange={(e) => setInviteName(e.target.value)}
                      className="px-3 py-2 bg-surface-container-low border border-outline-variant/50 rounded-lg text-xs font-semibold focus:outline-none focus:border-primary"
                    />
                    <select
                      value={inviteRole}
                      onChange={(e) => setInviteRole(e.target.value)}
                      className="px-3 py-2 bg-surface-container-low border border-outline-variant/50 rounded-lg text-xs font-semibold focus:outline-none focus:border-primary"
                    >
                      <option value="Daughter">Daughter</option>
                      <option value="Son">Son</option>
                      <option value="Doctor">Doctor</option>
                      <option value="Neighbor">Neighbor</option>
                    </select>
                  </div>
                  <div className="flex gap-2">
                    <input
                      type="email"
                      placeholder="email@address.com"
                      value={inviteEmail}
                      onChange={(e) => setInviteEmail(e.target.value)}
                      className="flex-1 px-3 py-2 bg-surface-container-low border border-outline-variant/50 rounded-lg text-xs font-semibold focus:outline-none focus:border-primary"
                    />
                    <button
                      onClick={handleAddContact}
                      className="px-4 py-2 bg-primary text-on-primary rounded-lg text-xs font-bold uppercase tracking-wider hover:opacity-95 cursor-pointer"
                    >
                      Add
                    </button>
                  </div>
                </div>

                <button
                  onClick={() => setCurrentStep(5)}
                  className="w-full py-3.5 bg-primary text-on-primary rounded-xl font-sans text-xs font-bold uppercase tracking-wider text-center cursor-pointer active:scale-95 transition-all"
                >
                  Complete Setup Walkthrough
                </button>
              </div>

              {/* Elder and caregiver image */}
              <div className="space-y-4">
                <div className="aspect-video rounded-2xl overflow-hidden shadow-md">
                  <img
                    alt="Elder caregiver smiling connection"
                    referrerPolicy="no-referrer"
                    className="w-full h-full object-cover"
                    src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 360'%3E%3Crect width='480' height='360' fill='%230a0a1a'/%3E%3Crect x='60' y='60' width='360' height='240' rx='8' fill='%23111133' stroke='%2300b4d8' stroke-width='1.5'/%3E%3Ccircle cx='240' cy='150' r='40' fill='%2300b4d8' opacity='0.15' stroke='%2300b4d8' stroke-width='1'/%3E%3Ccircle cx='240' cy='150' r='8' fill='%2300b4d8'/%3E%3Cline x1='240' y1='110' x2='240' y2='180' stroke='%2300b4d8' stroke-width='2' opacity='0.7'/%3E%3Cline x1='200' y1='140' x2='240' y2='170' stroke='%2300b4d8' stroke-width='2' opacity='0.7'/%3E%3Cline x1='280' y1='140' x2='240' y2='170' stroke='%2300b4d8' stroke-width='2' opacity='0.7'/%3E%3Crect x='100' y='260' width='280' height='24' rx='6' fill='%23001a2e'/%3E%3Ctext x='240' y='277' text-anchor='middle' font-family='monospace' font-size='10' fill='%2300b4d8'%3EPOSE DETECTED — AI SCAN ACTIVE%3C/text%3E%3C/svg%3E"
                  />
                </div>

                <div className="space-y-1.5">
                  <p className="text-[10px] uppercase font-bold tracking-wider text-on-surface-variant font-mono">Current Care Circle</p>
                  <div className="space-y-1.5 max-h-[100px] overflow-y-auto">
                    {contacts.map((c, i) => (
                      <div key={i} className="flex justify-between items-center text-xs p-1.5 bg-surface rounded-lg border border-outline-variant/30">
                        <span className="font-bold text-primary">{c.name} ({c.role})</span>
                        <span className="text-on-surface-variant italic text-[10px]">{c.email}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STEP 5: SUCCESS LANDING */}
          {currentStep === 5 && (
            <div className="text-center py-8 space-y-6 max-w-md mx-auto">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center text-green-600 mx-auto">
                <ShieldCheck className="w-8 h-8" />
              </div>

              <div className="space-y-2">
                <h3 className="font-sans text-2xl font-extrabold text-primary">You're All Set!</h3>
                <p className="text-on-surface-variant text-sm">
                  Safewatch is now fully active, synchronized, and vigilantly safeguarding your home environment.
                </p>
              </div>

              <button
                onClick={() => onNavigate('dashboard')}
                className="w-full py-4 bg-primary text-on-primary rounded-xl font-sans text-sm font-bold tracking-wider hover:bg-primary-container active:scale-95 transition-all text-center cursor-pointer"
              >
                Launch Care Dashboard
              </button>
            </div>
          )}

        </div>

      </div>
    </div>
  );
}
